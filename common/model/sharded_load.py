# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# coding: utf-8

"""Low-RAM, multi-GPU sharded loading for Lance.

Shared by both the CLI runner (`inference_lance.py`) and the Gradio app
(`lance_gradio.py`) so there is one source of truth for:

- meta-init + streaming safetensors load (keeps CPU RAM near zero; no fp32 copy,
  no mmap of the 12 GB checkpoint),
- a `device_map` that shards the LLM layers across N GPUs (with a card reserved
  for the VAE on generation tasks),
- materializing any params left on `meta` after the streaming load (the popped
  sin-cos position embeddings).

See LOW_RAM_LOAD.md and SHARDED_LOAD.md for the rationale behind each piece.
"""

import os.path as osp
import json
import struct
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from accelerate.utils import set_module_tensor_to_device


# Names of buffers/params that the original codepath intentionally popped from the
# checkpoint before load (they are fixed sin-cos embeddings rebuilt per resolution).
POPPED_FROM_CHECKPOINT = frozenset({"latent_pos_embed.pos_embed"})


def resolve_lance_checkpoint(model_path_dir: str) -> str:
    """Return the path of the Lance checkpoint to load (preferring model.safetensors)."""
    for fname in ("model.safetensors", "ema.safetensors"):
        cand = osp.join(model_path_dir, fname)
        if osp.exists(cand):
            return cand
    raise FileNotFoundError(
        f"No Lance checkpoint ('model.safetensors' or 'ema.safetensors') found in {model_path_dir}. "
        "Download the full Lance_3B (or Lance_3B_Video) weights with:\n"
        '  hf download bytedance-research/Lance --local-dir downloads --include "Lance_3B/*"'
    )


def build_lance_device_map(model, num_gpus: int, reserve_last_for_vae: bool = False) -> Dict[str, int]:
    """Spread Lance's LLM transformer layers across the available cards.

    cuda:0 is the "entry/exit" device for tokens and logits (embed + lm_head + norm)
    and also hosts the ViT. Those fixed-cost residents eat ~2-3 GB on cuda:0 before a
    single LLM layer lands there, so we give cuda:0 a *reduced* layer share.

    `reserve_last_for_vae`: when True (generation tasks), the LLM is sharded across
    only the first `num_gpus - 1` cards, leaving the last GPU empty of LLM weights so
    the WanVideoVAE (built on that card) has a near-full 12 GB for its decode. The
    video VAE decode's conv activations (~9 GB at 480p/17 frames) won't fit on a card
    that also holds LLM layers, so a dedicated card is the simplest robust fix.
    """
    num_layers = len(model.language_model.model.layers)
    num_gpus = max(1, num_gpus)

    # Number of cards the LLM may use. Reserve the last one for the VAE on generation.
    llm_gpus = num_gpus - 1 if (reserve_last_for_vae and num_gpus >= 2) else num_gpus
    llm_gpus = max(1, llm_gpus)

    device_map: Dict[str, int] = {}

    if llm_gpus == 1:
        # All LLM layers on cuda:0 (single-GPU, or 2-GPU generation with VAE on cuda:1).
        for i in range(num_layers):
            device_map[f"language_model.model.layers.{i}"] = 0
    else:
        # cuda:0 gets roughly half its even share; the remainder spreads across
        # cuda:1..llm_gpus-1. For 36 layers / 4 LLM cards that's 4 on cuda:0, ~11 each.
        gpu0_layer_count = max(1, num_layers // (2 * llm_gpus))
        remaining = num_layers - gpu0_layer_count
        other_gpus = llm_gpus - 1
        layers_per_other = (remaining + other_gpus - 1) // other_gpus  # ceil-div
        for i in range(num_layers):
            if i < gpu0_layer_count:
                device_map[f"language_model.model.layers.{i}"] = 0
            else:
                idx = i - gpu0_layer_count
                gpu = 1 + min(idx // layers_per_other, other_gpus - 1)
                device_map[f"language_model.model.layers.{i}"] = gpu

    # Token entry/exit and both MoT norms pinned to cuda:0. `norm_moe_gen` is the
    # generation-branch sibling of `norm`; it must be on the same device because the
    # forward path indexes a shared sequence and dispatches by token type.
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.model.norm"] = 0
    if hasattr(model.language_model.model, "norm_moe_gen"):
        device_map["language_model.model.norm_moe_gen"] = 0
    if hasattr(model.language_model.model, "rotary_emb"):
        device_map["language_model.model.rotary_emb"] = 0
    device_map["language_model.lm_head"] = 0

    # Lance heads. Small (a few MB each) except latent_pos_embed (~250 MB sin-cos);
    # keep them all near the embed/connector on cuda:0.
    for extra in ("connector", "time_embedder", "vae2llm", "llm2vae",
                  "latent_pos_embed", "task_embedding", "modality_embedding"):
        if hasattr(model, extra) and getattr(model, extra) is not None:
            device_map[extra] = 0

    # ViT must live on cuda:0. Lance.validation_video_to_text combines ViT outputs
    # with embed_tokens outputs via `masked_scatter` inline — that combine happens in
    # parent-class Python, not inside a submodule's forward(), so accelerate's hooks
    # don't get a chance to align devices. cuda:0 gets a reduced LLM-layer share
    # precisely so there's headroom for the ViT + embed + lm_head.
    if hasattr(model, "vit_model") and model.vit_model is not None:
        device_map["vit_model"] = 0

    # Safety net: any parameter not covered by an explicit prefix above (e.g. a future
    # top-level MoT sibling we didn't anticipate) lands on cuda:0. Without this,
    # accelerate.dispatch_model rejects the device_map with a hard error.
    covered_prefixes = list(device_map.keys())
    for param_name, _ in model.named_parameters():
        if not any(param_name == p or param_name.startswith(p + ".") for p in covered_prefixes):
            device_map[param_name] = 0

    return device_map


def device_for_param(param_name: str, device_map: Dict[str, int]) -> int:
    """Find the device assignment for `param_name` by walking up its dotted path."""
    parts = param_name.split(".")
    for i in range(len(parts), 0, -1):
        prefix = ".".join(parts[:i])
        if prefix in device_map:
            return device_map[prefix]
    return 0  # default to cuda:0 for any unmapped params (Lance has very few)


# safetensors dtype string -> (numpy dtype used to read raw bytes, optional torch view dtype)
# bf16 has no native numpy dtype, so we read as uint16 then bit-cast via tensor.view().
_SAFE_DTYPE_MAP = {
    "F64":  (np.float64, None),
    "F32":  (np.float32, None),
    "F16":  (np.float16, None),
    "BF16": (np.uint16,  torch.bfloat16),
    "I64":  (np.int64,   None),
    "I32":  (np.int32,   None),
    "I16":  (np.int16,   None),
    "I8":   (np.int8,    None),
    "U8":   (np.uint8,   None),
    "BOOL": (np.bool_,   None),
}


def read_safetensors_header(f) -> Tuple[Dict, int]:
    """Read the 8-byte length + JSON header. Returns (header_dict, data_section_offset)."""
    header_len_bytes = f.read(8)
    if len(header_len_bytes) != 8:
        raise ValueError(f"Truncated safetensors file: only {len(header_len_bytes)}/8 length bytes")
    (header_len,) = struct.unpack("<Q", header_len_bytes)
    header_bytes = f.read(header_len)
    if len(header_bytes) != header_len:
        raise ValueError(f"Truncated safetensors header: got {len(header_bytes)}/{header_len}")
    return json.loads(header_bytes), 8 + header_len


def read_safetensors_tensor(f, meta: dict, data_section_offset: int) -> torch.Tensor:
    """Read one tensor's bytes via plain seek+read (no mmap) and return a CPU torch tensor."""
    start, end = meta["data_offsets"]
    nbytes = end - start
    f.seek(data_section_offset + start)
    raw = f.read(nbytes)
    if len(raw) != nbytes:
        raise ValueError(f"Short read: got {len(raw)}/{nbytes} bytes")
    np_dtype, view_dtype = _SAFE_DTYPE_MAP[meta["dtype"]]
    # .copy() detaches from the read-only `raw` bytes so the buffer can be freed before
    # we keep the torch tensor around. Peak CPU memory: one tensor at a time.
    np_arr = np.frombuffer(raw, dtype=np_dtype).copy().reshape(meta["shape"])
    del raw
    tensor = torch.from_numpy(np_arr)
    if view_dtype is not None:
        tensor = tensor.view(view_dtype)
    return tensor


def stream_load_into(
    model: nn.Module,
    safetensors_path: str,
    device_map: Dict[str, int],
    key_prefix: str = "",
    skip_keys: frozenset = frozenset(),
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[List[str], List[str]]:
    """Stream safetensors into `model`, one tensor at a time, directly onto GPU shards.

    Uses plain `open() + seek() + read()` rather than `safetensors.safe_open()` because
    safe_open mmaps the whole file (12 GB for Lance_3B) — the kernel's overcommit policy
    rejects that on the 8 GB host. With direct IO peak CPU RAM is one tensor at a time
    (worst case ~1.2 GB for the embedding layer at fp32, briefly).

    `key_prefix` is prepended to each safetensors key when looking up the target
    parameter in `model` — the ViT file stores bare keys; they live under `vit_model.*`
    in the Lance wrapper.
    """
    loaded: List[str] = []
    unknown: List[str] = []
    model_keys = set(dict(model.named_parameters()).keys()) | set(dict(model.named_buffers()).keys())
    with open(safetensors_path, "rb") as f:
        header, data_section_offset = read_safetensors_header(f)
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            full_name = f"{key_prefix}{key}"
            if full_name in skip_keys:
                continue
            if full_name not in model_keys:
                unknown.append(full_name)
                continue
            tensor = read_safetensors_tensor(f, meta, data_section_offset).to(dtype)
            device = device_for_param(full_name, device_map)
            # Pass dtype= explicitly: without it, set_module_tensor_to_device casts
            # `value` to `old_value.dtype` to match the meta tensor's nominal dtype
            # (which is fp32 from init_empty_weights' default). That silently upcasts
            # our bf16 tensors back to fp32, doubling VRAM and breaking the autocast
            # path (fp32 weights * bf16 activations -> fp32 output, then index-put into
            # a bf16 destination crashes with a dtype-mismatch error).
            set_module_tensor_to_device(model, full_name, device, value=tensor, dtype=dtype)
            loaded.append(full_name)
            del tensor
    return loaded, unknown


def materialize_remaining_meta(model, device_map: Dict[str, int], dtype: torch.dtype):
    """Allocate any still-meta params on their target devices and re-init the
    fixed sin-cos position embeddings (which were popped from the checkpoint)."""
    from modeling.lance.modeling_utils import PositionEmbedding, PositionEmbedding3D

    materialized = []
    for name, param in list(model.named_parameters()):
        if not param.is_meta:
            continue
        device = device_for_param(name, device_map)
        # Walk to the owning module to swap the meta param for a real one.
        *mod_parts, attr = name.split(".")
        owner = model
        for m in mod_parts:
            owner = getattr(owner, m)
        new_param = torch.nn.Parameter(
            torch.zeros(param.shape, dtype=dtype, device=f"cuda:{device}"),
            requires_grad=param.requires_grad,
        )
        setattr(owner, attr, new_param)
        materialized.append(name)

    # Same for any buffers that ended up meta (rare; defensive).
    for name, buf in list(model.named_buffers()):
        if not buf.is_meta:
            continue
        device = device_for_param(name, device_map)
        *mod_parts, attr = name.split(".")
        owner = model
        for m in mod_parts:
            owner = getattr(owner, m)
        owner.register_buffer(
            attr, torch.zeros(buf.shape, dtype=buf.dtype, device=f"cuda:{device}")
        )
        materialized.append(name)

    # Re-run the sin-cos init now that the param tensors are real.
    for sub in model.modules():
        if isinstance(sub, (PositionEmbedding, PositionEmbedding3D)):
            sub._init_weights()

    return materialized
