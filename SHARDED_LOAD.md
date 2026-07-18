# Sharded model-parallel inference

This change adds **single-process, model-parallel inference across N GPUs**.
It's the second half of the work that lets Lance run on an 8 GB system RAM
+ 5 × RTX 3060 host. The first half — getting the model loaded without
materializing fp32 weights on CPU — is in
[`LOW_RAM_LOAD.md`](LOW_RAM_LOAD.md) and must be in place first; this change
builds on the streaming loader and `init_empty_weights()` infrastructure
introduced there.

## Why

A 3B Lance model in bf16 is ~6 GB of weights, plus ViT (~1.2 GB) and the
WanVideoVAE (~2-3 GB) for generation tasks. None of that fits on a single
12 GB 3060 with room for activations. But it does fit comfortably across
the **60 GB aggregate VRAM** of five 3060s if the LLM's transformer layers
are sharded across cards. Doing that requires:

- A device map that puts the right layers on the right cards.
- Cross-card forward hooks (`accelerate.dispatch_model`).
- Source-side fixes everywhere Lance's existing code assumes everything
  lives on one device.

The previous launcher ran `accelerate launch --num_processes=$NUM_GPUS`,
which is **data-parallel**: each process gets its own full copy of the
model. That doesn't help here — each rank still needs to fit the whole
model, and CPU RAM pressure goes up N× (one copy materialized per process).
For inference we want model-parallel: one process, model split across cards.

## What changed

### 1. Device map

`_build_lance_device_map(model, num_gpus)` (in `inference_lance.py`) builds
a `{module_name: gpu_index}` map:

- LLM transformer layers split across `cuda:1..N-1`, with `cuda:0` getting
  a **reduced** share (about half of the even split) because cuda:0 also
  hosts the entry/exit modules and the WanVideoVAE.
- `embed_tokens`, `norm`, `norm_moe_gen` (MoT generation-branch sibling),
  `rotary_emb`, `lm_head` pinned to `cuda:0` — these are the token-flow
  boundaries.
- ViT pinned to `cuda:0` because `Lance.validation_video_to_text` combines
  ViT output with `embed_tokens` output via `masked_scatter` *inline*
  (lance.py around line 1010). That combine happens in parent-class Python,
  not inside a submodule's `forward()`, so accelerate's hooks don't get a
  chance to align devices.
- Connector / time_embedder / vae2llm / llm2vae / latent_pos_embed all on
  cuda:0 (small).
- Safety net: any parameter not covered by an explicit prefix lands on
  cuda:0. Without this, `dispatch_model` rejects the device_map with a
  hard error the first time someone adds a top-level MoT sibling we didn't
  anticipate.

### 2. `accelerate.dispatch_model`

Installs pre/post forward hooks on each dispatched submodule so activations
get moved to the right card before each `.forward()`. After this point the
model must **not** be `.to()`-d (that would collapse every shard onto one
card). The per-batch `fsdp_model.to(device, dtype=bf16)` call in
`validate_on_fixed_batch` was already removed in the low-RAM change.

The streaming loader from the low-RAM change already supports a non-empty
device_map — every tensor is routed onto the GPU dictated by
`_device_for_param(name, device_map)` at load time, so the model is on
its shards *before* hooks are attached.

### 3. Replace `flex_attention` with eager-SDPA dense masks

`flex_attention`'s `BlockMask` captures device-specific tensors when it's
built. Under model parallelism, a layer on `cuda:>0` calls `flex_attention`
with `q/k/v` on that shard and a mask whose captures live on `cuda:0`;
dynamo's tracer refuses to combine them with
`Unhandled FakeTensor Device Propagation`.

The fix uses a path that already exists in `qwen2_navit.py`: when
`attention_mask` is a `List`, the attention forward iterates per-sample and
runs `torch.nn.functional.scaled_dot_product_attention` instead of
`flex_attention`. SDPA has no dynamo trace and crosses devices cleanly via
the standard accelerate hooks.

`_flex_mask_to_dense_list(mask_fn, seqlen, device, dtype)` evaluates the
flex mask function on a meshgrid of `(q_idx, kv_idx)` to get a bool mask,
converts to additive float (`-inf` where masked), and returns it as a
single-element `List`. All three `create_block_mask` call sites in
`lance.py` (one in `process_attention_mask`, one in the main `forward`,
one in `validation_video_to_text`) route through this helper.

### 4. Parent-class device-alignment fixes

A few places in `lance.py` and `qwen2_navit.py` combine tensors from
different shards in parent-Python (not inside a submodule's `forward()`),
which accelerate's hooks cannot reach. Each was fixed locally:

- `qwen2_navit.py:619` — at the start of each layer's `forward_train`,
  `attention_mask.to(device=packed_sequence_.device)` now handles both
  the old single-Tensor BlockMask path and the new List-of-Tensors SDPA
  path.
- `qwen2_navit.py:901` — after the layer loop in `Qwen2Model.forward_train`,
  `packed_sequence` lives on whichever shard ran the last layer (e.g.
  `cuda:N-1`). The index tensors and the final `norm` live on `cuda:0`.
  Added one `packed_sequence.to(packed_und_token_indexes.device)` to
  consolidate before the indexing-based combine.

### 5. Launcher and config

- `inference_lance.sh`:
  - `NUM_GPUS=5` default (was 1). It now means "number of shards", not
    "number of data-parallel processes".
  - `accelerate launch --num_processes 1` always — model parallelism is
    inside one process.
  - Forwards `--shard_num_gpus $NUM_GPUS` to the Python side.
- `config/config_factory.py`: adds `shard_num_gpus: int = 0` to
  `InferenceArguments`. `0` means "use all visible GPUs"
  (`torch.cuda.device_count()`); >0 caps to that many.

### 6. Generation tasks (t2i / t2v): the diffusion + VAE-decode path

The generation tasks exercise code the understanding path doesn't, and each
needed a model-parallel fix:

- **`forward_inference` gen-mode norm** (`qwen2_navit.py`, the `mode=="gen"`
  branch after the decoder-layer loop). This is the inference-mode twin of the
  `forward_train` fix above: after the layer loop `packed_query_sequence` is on
  the last shard, but `packed_text_indexes`/`packed_vae_token_indexes` and the
  `norm`/`norm_moe_gen` modules are on cuda:0. Added a
  `.to(packed_text_indexes.device)` guard before the index-put. This runs on
  every diffusion timestep. (The KVcache generation path attends via
  `flash_attn_varlen_func`, *not* flex_attention, so the dense-mask change in §3
  isn't even exercised here — no regression risk.)

- **VAE placement** (`modeling/vae/wan/model.py` + `inference_lance.py`). The
  WanVideoVAE used to hard-code `get_device()` = cuda:0. `WanVideoVAE` now
  accepts a `device=` (default `get_device()`, so single-GPU is unchanged), and
  the launcher builds it on the **last** shard.

- **Dedicated VAE card** (`_build_lance_device_map(..., reserve_last_for_vae=True)`
  for generation tasks). The video VAE decode's conv activations are large
  enough (~9 GB at 480² / 17 frames) that they won't fit on a card also holding
  LLM layers. For generation, the LLM is sharded across the first `N-1` cards
  and the last card is left empty of LLM weights so the VAE decode has a
  near-full 12 GB to itself.

**Resolution limit (important):** even a dedicated 12 GB card can't decode a
768²×17-frame video — that single-chunk conv activation peaks just over 12 GB.
480²×17 frames fits with ~2 GB to spare. Larger frames/resolution use VAE
**decode tiling** (spatial patches with overlap-blend) — see
[`TILED_VAE_DECODE.md`](TILED_VAE_DECODE.md), implemented in `WanVideoVAE._tiled_decode`
(auto-enabled above ~512², or `--VAE_TILE`). Note the launcher's
`VIDEO_HEIGHT`/`VIDEO_WIDTH` default to
**768**, independent of `--RESOLUTION`; pass `--VIDEO_HEIGHT 480 --VIDEO_WIDTH 480`
for t2v on a 12 GB card.

## Memory profile

### Understanding (`x2t_image` / `x2t_video`), Lance_3B, 5 × 3060

| Card | Holds | VRAM |
|---|---|---|
| cuda:0 | ~3 LLM layers + embed + lm_head + ViT + VAE + latent_pos_embed + connectors + CUDA context | ~6 GB |
| cuda:1–4 | ~8 LLM layers each | ~3 GB each |

### Generation (`t2v`, reserve-VAE-card), Lance_3B_Video, 480² / 17 frames

| Card | Holds | VRAM |
|---|---|---|
| cuda:0 | ~4 LLM layers + embed + lm_head + ViT + connectors | ~5 GB |
| cuda:1–3 | ~10–11 LLM layers each | ~3.7 GB each |
| cuda:4 | VAE only (decode peaks here) | ~0.8 GB idle → ~5 GB during decode |

Smoke tests confirmed: `x2t_image`, `x2t_video`, `t2i`, and `t2v` (480², 17
frames, 1.42 s mp4) all complete on the 8 GB-RAM / 5×3060 host.

## Performance

About 67 s per understanding batch at 768 resolution on the 5×3060 rig.
This is *slow* because:

- Every layer's attention runs eager SDPA with a dense mask instead of
  `flex_attention`'s compiled kernel.
- Activations shuttle across PCIe between cards via `dispatch_model`'s
  hooks at each layer boundary.

The point of this change is **fitting** the model on this hardware, not
throughput. A single A100 40 GB (cloud fallback) is the right move if you
need real speed.

## What `main` users keep

`shard_num_gpus=0` (the default) defers to `torch.cuda.device_count()`,
so on a 1-GPU host the device map collapses to "everything on cuda:0"
and `dispatch_model` is skipped. The dense-mask SDPA replacement does
run unconditionally — if you want `flex_attention` back for a single-card
setup, that's the one piece that's worth gating behind a flag.

## File-by-file summary

| File | Change |
|---|---|
| `inference_lance.py` | New `_build_lance_device_map` (with `reserve_last_for_vae`); `dispatch_model` import + call when sharding > 1; `shard_num_gpus` arg threading; VAE built on the last shard. |
| `inference_lance.sh` | `NUM_GPUS=5` default, `--num_processes 1`, passes `--shard_num_gpus`. |
| `config/config_factory.py` | Adds `shard_num_gpus: int = 0` to `InferenceArguments`. |
| `modeling/lance/lance.py` | New `_flex_mask_to_dense_list`; all three `create_block_mask` sites route through it. |
| `modeling/lance/qwen2_navit.py` | Layer `attention_mask.to(device=…)` handles List; `Qwen2Model.forward_train` and the `forward_inference` gen-mode branch move the sequence back to the index device after the layer loop. |
| `modeling/vae/wan/model.py` | `WanVideoVAE` accepts a `device=` override (default `get_device()`); `configure_vae_model`/`vae_encode`/`vae_decode` use it, so the VAE can live on a card other than cuda:0. |
