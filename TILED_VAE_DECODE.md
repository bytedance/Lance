# Tiled VAE decode for high-resolution video

(The primary, implemented mechanism is spatial **tiling**; Approach B below —
distributing tiles across GPUs — is the optional "sharded" extension.)

> Companion: [`TILED_VAE_ENCODE.md`](TILED_VAE_ENCODE.md) covers the **encode** side
> (for editing / i2v, which encode a source latent). This doc is decode-only.

**Status:** Approach A (single-GPU spatial tiling) implemented **and validated**
in-container — see "Validation results" below. Approach B (multi-GPU tile
distribution) and the CPU fallbacks remain proposals.

**Implemented (Approach A):** `WanVideoVAE._tiled_decode` + `_should_tile`
(`modeling/vae/wan/model.py`) tile the latent spatially, decode each tile through
the existing `self.vae.decode` (own temporal `feat_cache` per tile), and
feather-blend into the output with weight-sum normalization. Config knobs
`vae_tile_size` / `vae_tile_overlap` (`InferenceArguments`) are plumbed through
`inference_lance.py` and `inference_lance.sh` (`--VAE_TILE` / `--VAE_TILE_OVERLAP`).
The blend arithmetic was unit-tested off-GPU: reconstructing exact-tile decodes
matches the source to ~1e-16 across divisible/non-divisible/768²-latent cases with
full coverage (`wsum ≥ 1`).

This document plans how to lift the VAE-decode resolution ceiling that currently
caps t2v at ~480²/17 frames on a 12 GB card (see the "Resolution limit" note in
[`SHARDED_LOAD.md`](SHARDED_LOAD.md)). The goal is to decode 768² and larger on
the 8 GB-RAM / 5×3060 host.

---

## 1. Problem

t2v generation works end-to-end, but the final `WanVideoVAE` decode OOMs for
anything larger than ~480²/17 frames, even on a GPU dedicated entirely to the
VAE. At 768² the single-chunk conv activations peak just above 12 GB.

### Why the obvious fixes don't work

- **More GPUs for the VAE, LLM-style.** Layer-sharding the decoder across cards
  (like we did for the LLM) does **not** help. The VAE weights are tiny (~0.5 GB);
  the OOM is an *activation* peak — one decoder layer's full-resolution feature
  map (the traceback dies in `RMS_norm`/`F.pad` at the head, at full H×W). Pinning
  different layers to different cards still requires one card to hold that whole
  activation. The bottleneck is spatial extent, not parameter count.

- **Lower precision.** The decoder already runs under bf16 autocast; the final
  `.float()` is a small fraction. Worth maybe ~10–20%, not the ~3× we need for
  768².

### What the decode actually does (and where the memory goes)

`Wan2_2_VAE.decode(z, scale)` (vae2_2.py:787):

```
z: [1, 48, t, h, w]                      # latent (h = H/16, w = W/16)
x = conv2(z)
for i in range(t):                       # ALREADY temporally streamed, 1 latent frame at a time
    out_i = decoder(x[:, :, i:i+1], feat_cache=..., first_chunk=(i==0))
    out = cat([out, out_i], dim=2)       # accumulate frames
out = unpatchify(out, patch_size=2)      # final 2x spatial; channels 12 -> 3
```

The decoder upsamples 8× spatially (3 `Resample` stages) + the 2× `unpatchify`
= 16× total. The causal temporal conv state is carried frame-to-frame in
`feat_cache` (a per-conv list, reset by `clear_cache()` at the top of `decode()`).

**Conclusion:** temporal cost is already bounded (streamed). The remaining peak
is a *single frame's* spatial activations at full resolution. The fix is to
bound the **spatial** extent processed at once — i.e. **spatial tiling** — and,
as a second step, distribute tiles across the idle cards for speed.

---

## 2. Goals & non-goals

**Goals**
- Decode arbitrary H×W with bounded per-card memory (target: 768²–1024² on 12 GB).
- Reuse the existing, validated `decode()` per tile — avoid rewriting `Decoder3d`.
- No visible tile seams; output matches the untiled decode within tolerance.
- Off by default; opt-in via config so current 480² behavior is untouched.

**Non-goals**
- Faster decode at sizes that already fit (tiling adds overhead — only engage it
  above a threshold).
- Training / encode-side tiling (only inference decode is in scope; encode is not
  on the t2v critical path).

---

## 3. Approach A — spatial tiling on a single GPU (primary)

Decode the latent in overlapping spatial tiles, each through the full (temporally
streamed) decoder, then crop the halo and feather-blend tiles into the output
canvas. Because `decode()` resets its own `feat_cache` per call, **each tile is a
correct independent temporal stream** — we can call the existing method per tile.

### Mechanism

```
z: [1, 48, t, h, w]
canvas = zeros([1, 3, T_out, H, W])         # H=16h, W=16w; keep on CPU if large
weight = zeros([T_out, H, W])               # for feather normalization
for (lh0, lh1, lw0, lw1) in latent_tiles(h, w, tile, overlap):
    # take tile + halo from the latent (real neighbor cells, zeros at borders)
    z_tile = z[:, :, :, lh0-halo : lh1+halo, lw0-halo : lw1+halo]
    out_tile = vae.decode(z_tile, scale)    # existing method, own feat_cache
    out_tile = crop_halo(out_tile, halo*16) # drop receptive-field-contaminated border
    feather = ramp_mask(out_tile.shape)     # linear 0->1 ramp across the overlap region
    place out_tile * feather into canvas[..., region]; weight[region] += feather
canvas /= weight.clamp(min=eps)
out = canvas
```

### Key design points

- **Tile in latent space.** A latent tile of `tile×tile` cells → `16·tile`²
  pixels. E.g. `tile=24, halo=4` at 768² (h=w=48) gives 4 tiles of 24² latent =
  384² pixels each + halo — comfortably under the per-tile memory of the 480²
  decode that already works.
- **Halo (overlap) for conv receptive field.** Decoding a tile in isolation pads
  borders with zeros instead of neighbor content, so border pixels differ from
  the untiled result. Take a `halo` of real neighbor latent cells on each side,
  decode, then **crop `halo·16` pixels** off each interior edge so only
  receptive-field-clean pixels are kept. `halo` must cover the decoder's spatial
  receptive field in latent cells (see open question O1).
- **Feather blend.** Even with halo-crop, residual low-frequency mismatch can
  show as seams. Overlap adjacent kept-regions by a few pixels and blend with a
  linear ramp (weight accumulation as above). Halo-crop + feather together are
  robust.
- **Output canvas.** `[1,3,17,768,768]` float ≈ 108 MB — trivial; can live on the
  VAE card or CPU. Not a bottleneck.

### Where to implement

- New method `Wan2_2_VAE.tiled_decode(z, scale, tile, overlap, halo)` in
  `modeling/vae/wan/vae2_2.py`, or a wrapper in `WanVideoVAE.vae_decode`
  (`modeling/vae/wan/model.py`) that slices the latent and calls the existing
  `self.vae.decode` per tile. Prefer the wrapper — zero changes to `Decoder3d`.
- `WanVideoVAE.vae_decode` decides tiled vs. plain based on a threshold / flag.

### Cost

Serial over tiles → ~`n_tiles`× the per-tile decode time. For 768² with 4 tiles,
~4× a 384² decode. Acceptable for correctness; Approach B parallelizes it.

---

## 4. Approach B — distribute tiles across GPUs ("sharded VAE", phase 2)

During VAE decode the LLM cards (cuda:0–3) are idle. Replicate the VAE weights
(~0.5 GB) on each participating card and decode different tiles on different
cards in parallel, then gather + blend on one card (or CPU).

### Mechanism
- At startup (generation tasks), in addition to the dedicated VAE on cuda:N-1,
  hold lightweight VAE replicas on the other cards (each has ~8 GB free during
  decode since the LLM is idle but resident).
- Round-robin latent tiles across the replicas; run decodes concurrently (CUDA
  is async across devices; use per-device streams or just issue and sync).
- Gather decoded tiles to the canvas device (or CPU) and feather-blend.

### Trade-offs
- **Speedup:** up to ~`min(n_tiles, n_cards)`× over Approach A's serial loop.
- **Complexity:** weight replication, per-device latent slices, cross-device
  gather, synchronization. Higher risk than A.
- **Memory:** each replica card needs `LLM-layer resident + 0.5 GB VAE + one
  tile's activations`. Validate the idle-LLM cards have room (they held ~3.7 GB
  of layers, leaving ~8 GB — a 384²-tile decode fits).

Recommend B only after A is correct and if decode wall-clock matters.

---

## 5. Approach C — fallbacks

- **CPU-offload the accumulating output.** Move each decoded frame/tile to CPU as
  produced; keeps GPU holding only the active tile. Cheap, complements A.
- **CPU decode.** Move the whole VAE to CPU. Correct but very slow (conv3d on
  CPU). Last-resort for sizes that even tiling can't fit; document, don't default.

---

## 6. Implementation phases

| Phase | Scope | Deliverable | Status |
|---|---|---|---|
| 0 | Instrument: log peak VAE-decode VRAM vs. resolution; measure per-tile cost. | A table that sizes `tile`/`overlap`. | pending |
| 1 | Approach A wrapper in `WanVideoVAE.vae_decode` + `_tiled_decode`. | 768²/17-frame t2v decodes on one 12 GB card. | **done (impl)**, validation pending |
| 2 | Config knobs + auto-enable threshold. | `--VAE_TILE` / `--VAE_TILE_OVERLAP` plumbed through launcher. | **done** |
| 3 | (optional) Approach B multi-GPU tile distribution. | Decode speedup proportional to free cards. | proposal |

### Config / flags (phase 2)
- `vae_tile_size` (latent cells, `0` = auto/off), `vae_tile_overlap`,
  `vae_tile_halo` in `InferenceArguments`.
- Auto-enable when `H*W` exceeds the measured single-card ceiling (~480²–512²);
  below that, decode plainly (no tiling overhead). Default off preserves current
  behavior exactly.

---

## 7. Validation plan

1. **Parity at a size that fits untiled (480²).** Decode with and without tiling;
   assert max abs pixel diff below a small tolerance and PSNR high. This is the
   correctness gate for halo/blend.
2. **Seam inspection.** Visually check 768² output and diff adjacent-tile borders;
   no step discontinuities.
3. **Memory ceiling.** Confirm 768² (and try 1024²) decode stays under 12 GB with
   `torch.cuda.max_memory_allocated()` logging.
4. **Temporal consistency.** Confirm per-tile independent `feat_cache` doesn't
   introduce temporal flicker vs. untiled (the streaming is per-tile but the
   latent it streams is identical, so it should match — verify).
5. **Frame-count / fps** unchanged (ffprobe: 17 frames @ 12 fps as today).

### Validation results (5 × 3060, 8 GB RAM)

- **Parity (1):** same-seed 480²/17-frame t2v, tiled (`--VAE_TILE 24
  --VAE_TILE_OVERLAP 8`) vs. plain. PSNR **39.1 dB**, mean |diff| 1.9/255. The
  diff map concentrates on the moving subject's edges/texture (h264 re-encode
  noise), with **no grid pattern at tile boundaries** — i.e. no seams. ✅
- **Seams (2):** 768²/17-frame frame inspection — fur, foam, and sky are
  continuous across the tile boundaries; no step discontinuities. ✅
- **Capacity (3):** 768²/17-frame t2v auto-tiled — decode that previously OOM'd
  even on a dedicated card now completes; output is a valid 768×768 h264 clip.
  ✅ (1024² and `max_memory_allocated` logging not yet measured.)
- **Frame-count (5):** ffprobe reports 768×768, 17 frames @ 12 fps (1.42 s),
  unchanged. ✅
- **Temporal flicker (4):** not separately quantified beyond the per-frame
  inspection; no obvious flicker. Spot-check pending.

---

## 8. Risks & open questions

- **O1 — halo width. [RESOLVED]** An overlap of 8 latent cells passed the 480²
  parity test (39.1 dB, no seam grid) and produced seamless 768² output, so the
  decoder's receptive field is adequately covered at `overlap=8`. Smaller
  overlaps untested; 8 is the validated default.
- **O2 — temporal `feat_cache` under tiling.** Each tile re-streams all frames
  with its own cache. This should match untiled (same latent, same causal
  recursion per spatial location), but the `"Rep"` first-chunk handling in
  `Resample.forward` (vae2_2.py:126) must be verified per tile — confirm
  `first_chunk` semantics hold when the spatial extent is a sub-tile.
- **O3 — seam quality on high-frequency content.** Feather may blur fine detail
  in overlap bands; tune overlap width vs. sharpness.
- **O4 — Approach B replica memory.** Verify idle-LLM cards truly have room for a
  VAE replica + tile activations during decode (LLM weights stay resident).
- **O5 — non-square / odd sizes.** Tile loop must handle remainders (last tile
  smaller) and H≠W. Use ceil-div tiling with clamped edges.

---

## 9. Recommendation

Implement **Approach A** (single-GPU spatial tiling as a `vae_decode` wrapper)
first — it's low-risk (reuses the validated `decode()` per tile), solves the
resolution ceiling outright, and is independently useful even on single-GPU
hosts. Add the config knobs (phase 2). Pursue **Approach B** only if decode
wall-clock becomes the bottleneck once correctness is proven — it's a speed
optimization, not a capability unlock.
