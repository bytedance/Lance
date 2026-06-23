# Tiled VAE encode for high-resolution editing

Companion to [`TILED_VAE_DECODE.md`](TILED_VAE_DECODE.md) (which covers the **decode** side). This
documents **encode-side** spatial tiling, added so editing / i2v tasks — which must
encode a source image/video — run at higher resolution on a low-VRAM box.

**Status:** implemented on branch `low-ram-sharded-load`
(`modeling/vae/wan/model.py`). Stitch arithmetic unit-tested off-GPU (exact to
~1e-16). In-container parity / seam validation pending (see below).

## Why

Editing (`image_edit`, `video_edit`) and `i2v` first **encode** the source media to a
latent before diffusion:

```
inference_lance.py validate_on_fixed_batch
  → make_padded_latent (common/val/utils.py)
    → WanVideoVAE.vae_encode (modeling/vae/wan/model.py)
      → Wan2_2_VAE.encode (modeling/vae/wan/vae2_2.py)
```

`Wan2_2_VAE.encode` already streams **temporally** (chunks of 4 frames), so the memory
peak is **spatial**: the first encoder layers process the full-resolution input pixels
before the 16× downsample. At higher `RESOLUTION` (e.g. `video_480p` → VAE target 640)
that peak overruns the dedicated VAE card (cuda:4) with a CUDA `illegal memory access`
surfacing async at `make_padded_latent`. Before this change, editing only worked at
`video_192p`. This is the exact spatial problem the decode tiling solved, on the
encode side.

(The model-parallel *device* bug in the editing generation path — a separate issue —
was fixed earlier; see `de46a51` and `VAE_ENCODE_TILING_FOLLOWUP.md`.)

## What it does

`WanVideoVAE._tiled_encode` + `_should_tile_encode` (modeling/vae/wan/model.py), the
mirror of `_tiled_decode` / `_should_tile`:

- **Tile unit = latent cells; slice in pixel space.** For each latent tile
  `[r0:r1, c0:c1]`, the pixel slice is `x[..., r0*f:r1*f, c0*f:c1*f]` where
  `f = vae_config.downsample_spatial` (16). Latent-cell alignment means tiles abut
  exactly — no half-cell seams.
- **Encode per tile** via the existing `self.vae.encode`, which resets its own temporal
  `feat_cache` per call, so each tile is a correct independent temporal stream (same
  property decode relies on). Returns `(mu, log_var)` per tile.
- **Feather-blend** `mu` and `log_var` separately into full-resolution latent canvases
  using the shared `_blend_ramp_1d` ramp + weight-sum normalization (`wsum`), reusing
  `_tile_starts` for coverage. Latent-space analogue of the decode's pixel-space blend.
- **Reparameterize once, after stitching.** `vae_encode` calls `reparameterize` on the
  stitched full `mu`/`log_var`, not per tile — so the sampling noise (`randn_like`) is
  drawn once over the whole latent and per-tile noise can never seam. (This is why
  `_tiled_encode` returns `mu`/`log_var` rather than a sampled latent.)

### Gating & knobs
- Auto-enables when the **latent** grid `max(H//f, W//f)` exceeds
  `_VAE_AUTO_TILE_THRESHOLD` (36 ≈ 512²) — the same threshold as decode.
- Same config as decode, no new flags: `vae_tile_size` (0 = auto, >0 = force at that
  tile size, <0 = disable), `vae_tile_overlap`; CLI `--VAE_TILE` / `--VAE_TILE_OVERLAP`.
  Defaults: tile 32 latent cells, overlap 8.

## Scope (important)

- **Lifts the spatial / resolution ceiling**, not the frame-count one. It engages for
  high-res editing (e.g. `video_480p` → 640 → 40 latent cells, which previously OOM'd).
- **Does NOT help the low-res, high-frame-count case** (`video_192p` + `--NUM_FRAMES
  50`): at 192p the latent grid is ~16–23 cells, *below* the tiling threshold, so the
  encode isn't tiled — and that OOM is temporal (`NUM_FRAMES` sets the target latent
  length via `get_thw`, validation_dataset.py:828), a separate axis. Keep `NUM_FRAMES`
  low for editing regardless; temporal handling is out of scope here.

## Validation

- **Off-GPU (done):** the tile coverage + feather + `wsum` stitch reconstructs an
  exact-tile encode to ~1e-16 across 768² / non-square / non-divisible latent sizes,
  with full coverage (`wsum ≥ 1`).
- **In-container encode parity (done) — PASS.** `test_encode_tiling.py` (VAE only),
  768² video, deterministic `mu` (`use_sample=False`), tiled vs untiled, relative mean
  error:

  | config | rel mean err |
  |---|---|
  | tile 16 / overlap 4 (harsh) | 2.35% |
  | **tile 32 / overlap 8 (auto / production default)** | **0.57%** |
  | tile 32 / overlap 16 | 0.43% |
  | tile 32 / overlap 24 | 0.43% (plateau) |

  Converges and plateaus at 0.43%, which is the **bf16 rounding floor** (8-bit mantissa
  ≈ 0.39% relative). So at overlap ≥ 16 the tiling is exact-to-bf16 — feather-blending
  fully resolves the receptive-field contamination; no halo-crop needed. The default
  (overlap 8) sits at 0.57%, a hair above floor (a tiny residual seam), which is fine
  for conditioning. The earlier 2.37% was the forced overlap-4 stress config, not the
  real path.
- **End-to-end (done) — PASS.** `video_edit` at `--RESOLUTION video_480p --NUM_FRAMES
  17`, single clip (`config/examples/video_edit_single.json`). Tiling engaged
  (752×560 → latent 47×35, max 47 > 36 threshold), the encode no longer OOMs, and the
  edit applied correctly (red car on a snowy road) with no visible tile seams. ~26 min
  for the clip on the 5×3060 (480p editing is just heavy).

  **Async-hazard fix (required):** the tiled encode fires many per-tile kernels on the
  VAE's non-default card (cuda:N-1) while the current device is cuda:0; without a
  flush, a downstream `torch.cuda.empty_cache()` in `make_padded_latent` raced them
  into an `illegal memory access` (only `CUDA_LAUNCH_BLOCKING=1` masked it). Fixed by a
  `torch.cuda.synchronize(self.device)` at the end of `_tiled_encode` — the run above
  completed in normal async mode (no blocking env var).

  **Known limit — multi-clip earlyoom:** running all 3 example clips at 480p got
  SIGTERM'd by earlyoom (system-RAM killer) entering clip 2 — per-clip CPU
  accumulation, a *separate* issue from the encode. Single-clip runs are fine; for
  batches, add swap or process clips one at a time.

## Open questions / risks
- **Overlap vs. encoder receptive field — RESOLVED.** Parity converges to the bf16 floor
  by overlap 16 and plateaus, so overlap fully covers the encoder receptive field;
  feather (no halo-crop) suffices. Default 8 → 0.57% (near floor); `--VAE_TILE_OVERLAP
  16` → bf16-exact if a render needs max fidelity.
- **log_var blending — OK.** Error tracks the overlap trend and converges to the bf16
  floor with no pathological `log_var`-concentrated drift, so feather-averaging
  `log_var` is fine; no need to blend `std = exp(0.5·log_var)` instead.
- Editing was never a core deliverable on this box — this is opt-in extra scope built
  on the validated decode-tiling design.

## Pointers
- Code: `modeling/vae/wan/model.py` — `_tiled_encode`, `_should_tile_encode`,
  `vae_encode`; shared helpers `_tile_starts`, `_blend_ramp_1d`.
- Decode counterpart & rationale: `TILED_VAE_DECODE.md`.
- Editing device-fix + scope history: `VAE_ENCODE_TILING_FOLLOWUP.md` (outside git).
- Test: `test_encode_tiling.py`.
- Branch: `low-ram-sharded-load` (commit `d1fc9a4`).
