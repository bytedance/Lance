# Low-RAM inference load

This change lets `inference_lance.py` run on hosts with **less system RAM than
the model needs to materialize in fp32 on CPU** (~12 GB for Lance_3B). It
removes the CPU-side memory spike from the load path. Multi-GPU model
parallelism is layered on top by a separate change — see
[`SHARDED_LOAD.md`](SHARDED_LOAD.md).

## Why

On `main`, the first stage of `main()`:

```python
language_model: Qwen2ForCausalLM = Qwen2ForCausalLM(llm_config)
```

allocates a freshly-init'd fp32 3B model on CPU (~12 GB). On an 8 GB host this
gets OOM-killed before any GPU code runs. The actual checkpoint load
(`load_file → load_state_dict`) makes things worse by holding the full state
dict on CPU as a second copy. Several smaller allocations downstream
(numpy fp64 sin-cos, full-file `safe_open()` mmap) also push past the ceiling.

## What changed

### 1. Meta-init the model skeleton

LLM, ViT, and the `Lance` wrapper are constructed inside
`accelerate.init_empty_weights()`. Every `nn.Parameter` becomes shape-only on
the `meta` device — zero storage. The fp32-on-CPU spike disappears.

`modeling/lance/modeling_utils.py`'s `PositionEmbedding{,3D}._init_weights`
now early-returns when its param is still meta, deferring sin-cos
materialization until after the load.

### 2. Stream the checkpoint, don't mmap it

`safetensors.safe_open()` mmaps the whole 12 GB checkpoint file. On a host
with strict commit accounting and no swap, the kernel refuses a 12 GB
file-backed mapping (`ENOMEM`). The streaming loader (`_stream_load_into`)
opens the file in plain binary mode, reads the 8-byte header length + JSON
header, and seeks to each tensor's data offset. Peak CPU RAM during load is
one tensor at a time — worst case ~1.2 GB for the embedding layer, briefly.

### 3. Load tensors directly to GPU at bf16

Each tensor is read into CPU, cast to bf16, and handed to
`accelerate.utils.set_module_tensor_to_device(model, name, device,
value=tensor, dtype=torch.bfloat16)`. **The `dtype=` argument is
load-bearing**: without it, accelerate silently casts the value to
`old_value.dtype` to match the meta tensor's nominal dtype (fp32 default).
That would both double VRAM and produce fp32 weights that the bf16 autocast
path then promotes back to fp32 mid-attention — eventually crashing on an
index-put dtype mismatch.

After the load loop, `_materialize_remaining_meta` walks the model for
parameters still on meta (e.g. `latent_pos_embed.pos_embed`, which the
original code popped from the checkpoint to recompute per-resolution),
allocates real storage on the target device, and re-runs `_init_weights()`.

### 4. Compute sin-cos position embeddings on GPU, not CPU

`get_3d_sincos_pos_embed` (numpy fp64) used to allocate three intermediate
arrays of shape `(t*h*w, ~D/3)` plus a concatenated copy — peaking around
**4 GB of CPU RAM** for Lance's defaults (`t=31, h=w=64, D=2048`).

Replaced with `_torch_3d_sincos` / `_torch_2d_sincos` that compute on the
parameter's device in torch fp32. CPU contribution is ~zero. Same change for
the 2D variant used by `PositionEmbedding`.

### 5. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

The per-tensor streaming pattern fragments the default CUDA caching
allocator enough that big tensors can fail to allocate even when total free
VRAM is plenty. `expandable_segments` coalesces freed regions and lets large
allocations grow into them. Set by default in
`benchmarks/sample_env.sh::lance_setup_common_env` (`${VAR:-default}` so a
user-set value still wins).

## Memory profile (Lance_3B on a single GPU)

| Stage | Peak CPU RSS | Notes |
|---|---|---|
| Meta-init LLM/ViT/Lance | ~few hundred MB | torch + python + dataclass overhead |
| ViT streaming load (1.2 GB safetensors) | ~1 GB | one fp32 tensor at a time |
| Lance streaming load (12.3 GB safetensors) | ~1.5 GB | embedding layer is the worst tensor |
| Materialize popped sin-cos pos_embed | tiny | computed on GPU |
| Tokenizer + resize | <500 MB | |

Peak CPU RSS during load stays under ~2 GB, comfortably below an 8 GB
ceiling. Total VRAM usage on the target card is ~6 GB (Lance_3B in bf16
+ ViT + VAE), which fits on a single 40 GB GPU but not a 12 GB one — for
that case, see [`SHARDED_LOAD.md`](SHARDED_LOAD.md).

## What `main` users keep

For hosts that *do* have enough RAM, this change is still net-positive:
the load is faster (no fp32 → bf16 conversion afterwards, no full state-dict
held on CPU) and uses half the VRAM (bf16 instead of fp32 at rest). The
launcher and config are unchanged in this commit; the behavior change is
transparent to the runner.

## File-by-file summary

| File | Change |
|---|---|
| `inference_lance.py` | `init_empty_weights()` for LLM/ViT/Lance; new streaming safetensors reader (`_read_safetensors_header`, `_read_safetensors_tensor`, `_stream_load_into`); `_materialize_remaining_meta`; `_resolve_lance_checkpoint`; passes `dtype=torch.bfloat16` to `set_module_tensor_to_device`; removed the per-batch `.to(device)` calls on the model. |
| `modeling/lance/modeling_utils.py` | New `_torch_2d_sincos` / `_torch_3d_sincos`; `_init_weights` early-returns on meta tensors and otherwise computes on the param's device. |
| `benchmarks/sample_env.sh` | Exports `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (user override respected). |
