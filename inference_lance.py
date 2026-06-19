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

import warnings
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers.models.transformers.transformer_2d")
import os
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import os.path as osp
from copy import deepcopy
import json
from typing import Tuple, cast, Optional, Dict, List
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, set_seed
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
import struct
import numpy as np
from safetensors.torch import load_file
from accelerate import init_empty_weights, dispatch_model
from accelerate.utils import set_module_tensor_to_device

from data.dataset_base import DataConfig, simple_custom_collate
from data.data_utils import add_special_tokens
from modeling.vae.wan.model import WanVideoVAE
from modeling.lance import LanceConfig, Lance, Qwen2ForCausalLM
from modeling.qwen2 import Qwen2Tokenizer
from modeling.qwen2.modeling_qwen2 import Qwen2Config
from modeling.vit.qwen2_5_vl_vit import Qwen2_5_VisionTransformerPretrainedModel
from common.utils.misc import tuple_mul, AutoEncoderParams
from common.utils.logging import get_logger
from common.val.utils import make_padded_latent, decode_video_tensor
from data.datasets_custom import ValidationDataset
from config.config_factory import (
    ModelArguments,
    DataArguments,
    InferenceArguments,
    get_model_path,
)

from tqdm import trange


# Constants
MAX_GENERATION_LENGTH = 256
PROMPT_JSON_FILENAME = "prompt.json"
RESULT_JSON_FILENAME = "result.json"
INTERNAL_VALIDATION_MAX_SAMPLES = 100000
TASK_T2V = "t2v"
TASK_T2I = "t2i"
TASK_I2V = "i2v"
TASK_X2T_IMAGE = "x2t_image"
TASK_X2T_VIDEO = "x2t_video"
TASK_IMAGE_EDIT = "image_edit"
TASK_VIDEO_EDIT = "video_edit"
GENERATION_TASKS = {
    TASK_T2V,
    TASK_T2I,
    TASK_I2V,
    TASK_IMAGE_EDIT,
    TASK_VIDEO_EDIT,
}
UNDERSTANDING_TASKS = {
    TASK_X2T_IMAGE,
    TASK_X2T_VIDEO,
}
TASK_DEFAULT_CONFIGS = {
    TASK_T2I: {
        "model_family": "image",
        "example_json": "config/examples/t2i_example.json",
        "save_path_prefix": "results/t2i_sample",
    },
    TASK_T2V: {
        "model_family": "video",
        "example_json": "config/examples/t2v_example.json",
        "save_path_prefix": "results/t2v_sample",
    },
    TASK_I2V: {
        "model_family": "video",
        "example_json": "config/examples/i2v_example.json",
        "save_path_prefix": "results/i2v_sample",
    },
    TASK_IMAGE_EDIT: {
        "model_family": "image",
        "example_json": "config/examples/image_edit_example.json",
        "save_path_prefix": "results/image_edit_sample",
    },
    TASK_VIDEO_EDIT: {
        "model_family": "video",
        "example_json": "config/examples/video_edit_example.json",
        "save_path_prefix": "results/video_edit_sample",
    },
    TASK_X2T_IMAGE: {
        "model_family": "image",
        "example_json": "config/examples/x2t_image_example.json",
        "save_path_prefix": "results/x2t_image_sample",
    },
    TASK_X2T_VIDEO: {
        "model_family": "video",
        "example_json": "config/examples/x2t_video_example.json",
        "save_path_prefix": "results/x2t_video_sample",
    },
}

# Names of buffers/params that the original codepath intentionally popped from the
# checkpoint before load (they are fixed sin-cos embeddings rebuilt per resolution).
_POPPED_FROM_CHECKPOINT = frozenset({"latent_pos_embed.pos_embed"})


def _resolve_lance_checkpoint(model_path_dir: str) -> str:
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


def _build_lance_device_map(model: "Lance", num_gpus: int, reserve_last_for_vae: bool = False) -> Dict[str, int]:
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
    # with embed_tokens outputs via `masked_scatter` inline (lance.py:1010) — that
    # combine happens in parent-class Python, not inside a submodule's forward(), so
    # accelerate's hooks don't get a chance to align devices. cuda:0 gets a reduced
    # LLM-layer share precisely so there's headroom for the ViT + embed + lm_head.
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


def _device_for_param(param_name: str, device_map: Dict[str, int]) -> int:
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


def _read_safetensors_header(f) -> Tuple[Dict, int]:
    """Read the 8-byte length + JSON header. Returns (header_dict, data_section_offset)."""
    header_len_bytes = f.read(8)
    if len(header_len_bytes) != 8:
        raise ValueError(f"Truncated safetensors file: only {len(header_len_bytes)}/8 length bytes")
    (header_len,) = struct.unpack("<Q", header_len_bytes)
    header_bytes = f.read(header_len)
    if len(header_bytes) != header_len:
        raise ValueError(f"Truncated safetensors header: got {len(header_bytes)}/{header_len}")
    return json.loads(header_bytes), 8 + header_len


def _read_safetensors_tensor(f, meta: dict, data_section_offset: int) -> torch.Tensor:
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


def _stream_load_into(
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
        header, data_section_offset = _read_safetensors_header(f)
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            full_name = f"{key_prefix}{key}"
            if full_name in skip_keys:
                continue
            if full_name not in model_keys:
                unknown.append(full_name)
                continue
            tensor = _read_safetensors_tensor(f, meta, data_section_offset).to(dtype)
            device = _device_for_param(full_name, device_map)
            # Pass dtype= explicitly: without it, set_module_tensor_to_device casts
            # `value` to `old_value.dtype` to match the meta tensor's nominal dtype
            # (which is fp32 from init_empty_weights' default). That silently upcasts
            # our bf16 tensors back to fp32, doubling VRAM and breaking the autocast
            # path (fp32 weights * bf16 activations → fp32 output, then index-put into
            # a bf16 destination crashes with a dtype-mismatch error).
            set_module_tensor_to_device(model, full_name, device, value=tensor, dtype=dtype)
            loaded.append(full_name)
            del tensor
    return loaded, unknown


def _materialize_remaining_meta(model: "Lance", device_map: Dict[str, int], dtype: torch.dtype):
    """Allocate any still-meta params on their target devices and re-init the
    fixed sin-cos position embeddings (which were popped from the checkpoint)."""
    from modeling.lance.modeling_utils import PositionEmbedding, PositionEmbedding3D

    materialized = []
    for name, param in list(model.named_parameters()):
        if not param.is_meta:
            continue
        device = _device_for_param(name, device_map)
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
        device = _device_for_param(name, device_map)
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


def clean_memory(*objects):
    """Clear temporary container references and release unused GPU allocator cache."""
    for obj in objects:
        if isinstance(obj, dict):
            obj.clear()
        elif isinstance(obj, (list, set)):
            obj.clear()
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def apply_inference_defaults(
    model_args: ModelArguments,
    data_args: DataArguments,
    inference_args: InferenceArguments,
) -> None:
    if inference_args.task not in TASK_DEFAULT_CONFIGS:
        raise ValueError(f"Unsupported inference task: {inference_args.task}")

    task_config = TASK_DEFAULT_CONFIGS[inference_args.task]
    default_inference_args = InferenceArguments()

    model_family = task_config.get("model_family", "")
    if not model_args.model_path and model_family:
        model_args.model_path = get_model_path(f"lance.{model_family}")
    if not getattr(model_args, "llm_path", ""):
        model_args.llm_path = model_args.model_path
    if not model_args.vit_path:
        model_args.vit_path = get_model_path("vit.qwen2_5_vl")

    if not data_args.val_dataset_config_file and task_config.get("example_json"):
        data_args.val_dataset_config_file = task_config["example_json"]

    if inference_args.save_path_gen == default_inference_args.save_path_gen and task_config.get("save_path_prefix"):
        inference_args.save_path_gen = task_config["save_path_prefix"]
    if inference_args.validation_max_samples == default_inference_args.validation_max_samples:
        inference_args.validation_max_samples = INTERNAL_VALIDATION_MAX_SAMPLES
    if inference_args.video_height == default_inference_args.video_height:
        inference_args.video_height = int(task_config.get("video_height", default_inference_args.video_height))
    if inference_args.video_width == default_inference_args.video_width:
        inference_args.video_width = int(task_config.get("video_width", default_inference_args.video_width))
    if inference_args.resolution == default_inference_args.resolution:
        inference_args.resolution = task_config.get("resolution", default_inference_args.resolution)
    if inference_args.text_template == default_inference_args.text_template:
        inference_args.text_template = bool(task_config.get("text_template", default_inference_args.text_template))


def save_prompt_results(prompt_data_dict, save_path_gen, logger):
    """Save validation results to a JSON file."""
    prompt_json_path = os.path.join(save_path_gen, PROMPT_JSON_FILENAME)
    with open(prompt_json_path, 'w', encoding='utf-8') as f:
        json.dump(prompt_data_dict, f, ensure_ascii=False, indent=2)


def normalize_understanding_answer(text: Optional[str]) -> str:
    """Normalize generated understanding text before exporting it."""
    if text is None:
        return ""
    return text.replace("<|im_end|>", "").strip()


def save_understanding_results(
    prompt_data_dict: dict,
    dataset_config_file: str,
    save_path_gen: str,
) -> None:
    """Save x2t results as a structured result.json file."""
    with open(dataset_config_file, "r", encoding="utf-8") as f:
        dataset_samples = json.load(f)

    result_entries = []
    for sample_key, sample in dataset_samples.items():
        interleave_array = sample.get("interleave_array", [])
        element_dtype_array = sample.get("element_dtype_array", [])
        if len(interleave_array) < 2 or not element_dtype_array:
            continue

        visual_path = interleave_array[0]
        text_payload = interleave_array[1]
        question = text_payload[1] if isinstance(text_payload, list) and len(text_payload) > 1 else ""
        modality = element_dtype_array[0]

        lookup_keys = [os.path.basename(visual_path), sample_key]
        generated_answer = ""
        for lookup_key in lookup_keys:
            if lookup_key in prompt_data_dict:
                generated_answer = prompt_data_dict[lookup_key]
                break

        result_entries.append(
            {
                modality: visual_path,
                "question": question,
                "answer": normalize_understanding_answer(generated_answer),
            }
        )

    result_json_path = os.path.join(save_path_gen, RESULT_JSON_FILENAME)
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(result_entries, f, ensure_ascii=False, indent=2)


def validate_on_fixed_batch(
    fsdp_model: Lance,
    vae_model: Optional[WanVideoVAE],
    tokenizer: Qwen2Tokenizer,
    val_data_cpu: dict,
    training_args: InferenceArguments,
    model_args: ModelArguments,
    inference_args: InferenceArguments,
    new_token_ids,
    image_token_id: int,
    device: int,
    save_source_video: bool = False,
    save_path_gen: str = "",
    save_path_gt: str = "",
):
    val_data = val_data_cpu.cuda(device).to_dict()
    # Do NOT call fsdp_model.to(device) here: the model is sharded across multiple GPUs
    # via accelerate.dispatch_model, and .to() would collapse all shards onto one card.
    # Weights are already bf16 from the streaming load.

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        # Compute padded_latent.
        if "padded_videos" in val_data.keys():
            val_data["padded_latent"] = make_padded_latent(val_data["padded_videos"], val_data["vae_data_mode"], vae_model)

        # -------------------- Generation branch --------------------
        if inference_args.task in GENERATION_TASKS:
            save_fps = int(val_data.get("save_fps", 12))
            params = {
                "val_packed_text_ids": val_data["packed_text_ids"],
                "val_packed_text_indexes": val_data["packed_text_indexes"],
                "val_sample_lens": val_data["sample_lens"],
                "val_packed_position_ids": val_data["packed_position_ids"],
                "val_split_lens": val_data["split_lens"],
                "val_attn_modes": val_data["attn_modes"],
                "val_sample_N_target": val_data["sample_N_target"],
                "val_packed_vae_token_indexes": val_data["packed_vae_token_indexes"],
                "timestep_shift": training_args.validation_timestep_shift,
                "num_timesteps": training_args.validation_num_timesteps,
                "val_mse_loss_indexes": val_data.get("mse_loss_indexes", None),
                "val_padded_latent": val_data["padded_latent"],
                "video_sizes": val_data["video_sizes"],
                "cfg_text_scale": model_args.cfg_text_scale,
                "cfg_interval": training_args.cfg_interval,
                "cfg_renorm_min": training_args.cfg_renorm_min,
                "cfg_renorm_type": training_args.cfg_renorm_type,
                "device": device,
                "dtype": torch.bfloat16,
                "new_token_ids": new_token_ids,
                "max_samples": training_args.validation_max_samples,
                "validation_noise_seed": training_args.validation_noise_seed,
                "apply_chat_template": training_args.apply_chat_template,
                "apply_qwen_2_5_vl_pos_emb": training_args.apply_qwen_2_5_vl_pos_emb,
                "image_token_id": image_token_id,
                "val_packed_vit_token_indexes": val_data.get("packed_vit_token_indexes", None),
                "val_packed_vit_tokens": val_data.get("packed_vit_tokens", None),
                "vit_video_grid_thw": val_data.get("vit_video_grid_thw", None),
                "vae_video_grid_thw": val_data["vae_video_grid_thw"],
                "video_grid_thw": val_data.get("video_grid_thw", None),
                "caption": val_data.get("caption", None),  # The dataset uses "caption" as the default caption field.
                "sample_task": val_data["sample_task"],
                "sample_modality": val_data["sample_modality"],
                "cfg_type": training_args.cfg_type,
                "cfg_uncond_token_id": training_args.cfg_uncond_token_id,
                "index": val_data["index"],
                "val_padded_videos": val_data["padded_videos"] if save_source_video else None,
            }
            if inference_args.use_KVcache:
                denoise_latent, captions, padded_videos, index = fsdp_model.validation_gen_KVcache(**params)
            else:
                denoise_latent, captions, padded_videos, index = fsdp_model.validation_gen(**params)

            # Decode.
            for i_val, latent in enumerate(denoise_latent):
                if inference_args.task in {TASK_I2V, TASK_IMAGE_EDIT, TASK_VIDEO_EDIT}:
                    target_latents = [latent[-1]]
                else:
                    target_latents = latent

                v_list = []
                for latent_ in target_latents:
                    v_list.append(vae_model.vae_decode([latent_])[0])

                save_item_name = f"{index:06d}" if isinstance(index, int) else index
                v_thwc = decode_video_tensor(v_list, save_path=save_path_gen, save_half=False, save_item_name=save_item_name, save_fps=save_fps)

                if v_thwc.shape[0] > 1:
                    prompt_data_path = f"{save_item_name}.mp4"
                else:
                    prompt_data_path = f"{save_item_name}.png"
                inference_args.prompt_data_dict[prompt_data_path] = captions[i_val]

                if save_source_video:
                    curr_padded_videos = padded_videos[i_val * 2 : (i_val + 1) * 2]
                    v_thwc_gt = decode_video_tensor(curr_padded_videos[-1:], save_path=save_path_gt, save_item_name=save_item_name, save_fps=save_fps)
                    del curr_padded_videos, v_thwc_gt

                del v_list, v_thwc, latent, target_latents
                clean_memory()

            del denoise_latent, captions, padded_videos, params
            clean_memory()

        elif inference_args.task in UNDERSTANDING_TASKS:
            params = {
                "val_packed_text_ids": val_data["packed_text_ids"],
                "val_packed_text_indexes": val_data["packed_text_indexes"],
                "val_packed_position_ids": val_data["packed_position_ids"],
                "val_sample_N_target": val_data["sample_N_target"],
                "val_split_lens": val_data["split_lens"],
                "val_attn_modes": val_data["attn_modes"],
                "val_sample_lens": val_data["sample_lens"],
                "val_sample_type": val_data["sample_type"],
                "val_packed_vit_tokens": val_data["packed_vit_tokens"],
                "val_vit_video_grid_thw": val_data["vit_video_grid_thw"],
                "val_ce_loss_indexes": val_data["ce_loss_indexes"],
                "max_samples": training_args.validation_max_samples,
                "max_length": MAX_GENERATION_LENGTH,
                "device": device,
                "dtype": torch.bfloat16,
                "new_token_ids": new_token_ids,
                "pad_token_id": tokenizer.pad_token_id,
                "vocab_size": len(tokenizer),
                "caption": val_data.get("caption_cn", None),
                "tokenizer": tokenizer,
                "apply_chat_template": training_args.apply_chat_template,
                "apply_qwen_2_5_vl_pos_emb": training_args.apply_qwen_2_5_vl_pos_emb,
                "do_sample": False,
                "image_token_id": image_token_id,
                "index": val_data["index"],
            }
            if inference_args.use_KVcache:
                generated_sequence_all, captions, index = fsdp_model.validation_und_KVcache(**params)
            else:
                generated_sequence_all, captions, index = fsdp_model.validation_video_to_text(**params)

            for i_val, generated_sequence in enumerate(generated_sequence_all):
                cap = tokenizer.decode(generated_sequence[:, 0])
                # inference_args.prompt_data_dict[index] = f"target_caption: {captions} /// generated_caption: {cap} "
                inference_args.prompt_data_dict[index] = f"{cap}"
                del generated_sequence

            del generated_sequence_all, captions, params
            clean_memory()

    del val_data
    clean_memory()


def main():
    # ========================= Env setup ==============================
    assert torch.cuda.is_available()
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        GLOBAL_RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()
    else:
        GLOBAL_RANK = 0
        WORLD_SIZE = 1

    LOCAL_RANK = GLOBAL_RANK % torch.cuda.device_count()
    DEVICE = LOCAL_RANK
    torch.cuda.set_device(DEVICE)

    # ========================= Args and logger setup ==============================
    parser = HfArgumentParser((ModelArguments, DataArguments, InferenceArguments))
    model_args, data_args, inference_args = cast(
        Tuple[ModelArguments, DataArguments, InferenceArguments],
        parser.parse_args_into_dataclasses(),
    )
    training_args = inference_args

    # ========================= Load task paths and example JSONs from defaults ==============================
    apply_inference_defaults(model_args, data_args, inference_args)
    training_args.validation_noise_seed = training_args.validation_data_seed

    logger = get_logger()
    log_rank0 = print if GLOBAL_RANK == 0 else (lambda *_: None)  # Only print on rank 0.

    def log_stage(stage_name: str, start_time: float, extra: str = ""):
        elapsed = time.perf_counter() - start_time
        suffix = f" | {extra}" if extra else ""
        log_rank0(f"[startup] {stage_name} done in {elapsed:.2f}s{suffix}")

    # Set seed:
    seed = training_args.global_seed * WORLD_SIZE + GLOBAL_RANK
    set_seed(seed)

    # ========================= LLM model setup ==============================
    stage_start = time.perf_counter()
    log_rank0(f"[startup] Loading LLM config: {osp.join(model_args.model_path, 'llm_config.json')}")
    llm_config: Qwen2Config = Qwen2Config.from_json_file(osp.join(model_args.model_path, "llm_config.json"))
    log_stage("LLM config load", stage_start)

    llm_config.layer_module = model_args.layer_module
    llm_config.qk_norm = model_args.llm_qk_norm
    llm_config.qk_norm_und = model_args.llm_qk_norm_und
    llm_config.qk_norm_gen = model_args.llm_qk_norm_gen

    llm_config.tie_word_embeddings = model_args.tie_word_embeddings
    llm_config.freeze_und = training_args.freeze_und
    llm_config.apply_qwen_2_5_vl_pos_emb = training_args.apply_qwen_2_5_vl_pos_emb

    # ===== Meta-init: build the module skeleton with zero CPU RAM. =====
    # The bare Qwen2ForCausalLM(llm_config) call used to materialize a full fp32 3B
    # model on CPU (~12 GB), which is the load step that OOM-killed an 8 GB box.
    # Under init_empty_weights() every nn.Parameter is created on the "meta" device
    # (shape only, no storage), so this whole block stays at near-zero RAM.
    stage_start = time.perf_counter()
    log_rank0(f"[startup] Meta-initializing LLM: {model_args.model_path}")
    with init_empty_weights():
        language_model: Qwen2ForCausalLM = Qwen2ForCausalLM(llm_config)
    log_stage("LLM meta-init", stage_start)

    vit_model = None
    vit_config = None
    if training_args.visual_und:
        if model_args.vit_type in ("qwen2_5_vl", "qwen_2_5_vl_original"):
            stage_start = time.perf_counter()
            log_rank0(f"[startup] Loading VIT config: {model_args.vit_path}")
            vit_config = Qwen2_5_VLVisionConfig.from_pretrained(model_args.vit_path)
            log_stage("VIT config load", stage_start)

            stage_start = time.perf_counter()
            log_rank0("[startup] Meta-initializing VIT (weights loaded later from vit.safetensors)")
            with init_empty_weights():
                vit_model = Qwen2_5_VisionTransformerPretrainedModel(vit_config)
            log_stage("VIT meta-init", stage_start)
        else:
            raise ValueError(f"Unsupported vit_type: {model_args.vit_type}")

    if training_args.visual_gen:
        # WanVideoVAE itself uses torch.device("meta") + assign-load internally, so it
        # doesn't contribute to the CPU RAM spike. Built eagerly so vae_config is real.
        # Place it on the lightest shard (the last GPU) when sharding across >1 card:
        # cuda:0 is the most crowded device and the video VAE decode's conv activations
        # OOM it. On a single GPU this resolves to cuda:0 (unchanged behavior).
        num_visible_gpus = torch.cuda.device_count()
        shard_n = inference_args.shard_num_gpus or num_visible_gpus
        shard_n = max(1, min(shard_n, num_visible_gpus))
        vae_device = torch.device("cuda", shard_n - 1)
        stage_start = time.perf_counter()
        log_rank0(f"[startup] Initializing VAE on {vae_device} "
                  f"(tile_size={inference_args.vae_tile_size}, tile_overlap={inference_args.vae_tile_overlap})")
        vae_model = WanVideoVAE(
            device=vae_device,
            tile_size=inference_args.vae_tile_size,
            tile_overlap=inference_args.vae_tile_overlap,
        )
        vae_config: AutoEncoderParams = deepcopy(vae_model.vae_config)
        log_stage("VAE init", stage_start)
    else:
        vae_model = None
        vae_config = None

    config = LanceConfig(
        visual_gen=training_args.visual_gen,
        visual_und=training_args.visual_und,
        llm_config=llm_config,
        vit_config=vit_config if training_args.visual_und else None,
        vae_config=vae_config if training_args.visual_gen else None,
        latent_patch_size=model_args.latent_patch_size,
        max_num_frames=model_args.max_num_frames,
        max_latent_size=model_args.max_latent_size,
        vit_max_num_patch_per_side=model_args.vit_max_num_patch_per_side,
        connector_act=model_args.connector_act,
        interpolate_pos=model_args.interpolate_pos,
        timestep_shift=training_args.timestep_shift,
    )

    stage_start = time.perf_counter()
    log_rank0("[startup] Meta-initializing Lance wrapper")
    with init_empty_weights():
        model: Lance = Lance(
            language_model=language_model,
            vit_model=vit_model if training_args.visual_und else None,
            vit_type=model_args.vit_type,
            config=config,
            training_args=training_args,
        )
    log_stage("Lance meta-init", stage_start)

    # ===== Decide how to shard across GPUs. =====
    num_visible_gpus = torch.cuda.device_count()
    shard_n = inference_args.shard_num_gpus or num_visible_gpus
    shard_n = max(1, min(shard_n, num_visible_gpus))
    # Generation tasks decode through the VAE, whose video decode needs a near-full
    # card to itself; reserve the last GPU for it (the VAE was built there above).
    reserve_vae = bool(training_args.visual_gen) and inference_args.task in GENERATION_TASKS and shard_n >= 2
    log_rank0(f"[startup] Sharding Lance across {shard_n} GPU(s) (visible: {num_visible_gpus}; "
              f"reserve last GPU for VAE: {reserve_vae})")
    device_map = _build_lance_device_map(model, shard_n, reserve_last_for_vae=reserve_vae)

    # ===== Stream-load weights directly onto each shard's GPU at bf16. =====
    # ViT weights live in a separate file; in the Lance wrapper they sit under vit_model.*.
    if training_args.visual_und:
        vit_safetensors = osp.join(model_args.vit_path, "vit.safetensors")
        stage_start = time.perf_counter()
        log_rank0(f"[startup] Streaming VIT weights from {vit_safetensors}")
        vit_loaded, vit_unknown = _stream_load_into(
            model, vit_safetensors, device_map, key_prefix="vit_model.", dtype=torch.bfloat16,
        )
        log_stage("VIT streaming load", stage_start,
                  extra=f"loaded={len(vit_loaded)} unknown={len(vit_unknown)}")
        if vit_unknown:
            log_rank0(f"[startup] WARNING: {len(vit_unknown)} ViT key(s) had no matching param "
                      f"(first few: {vit_unknown[:5]})")

    # The main Lance checkpoint: covers language_model.*, the connector / vae<->llm /
    # time_embedder / latent_pos_embed (popped) / etc. Skip the popped sin-cos buffer.
    lance_ckpt = _resolve_lance_checkpoint(model_args.model_path)
    stage_start = time.perf_counter()
    log_rank0(f"[startup] Streaming Lance checkpoint from {lance_ckpt}")
    main_loaded, main_unknown = _stream_load_into(
        model, lance_ckpt, device_map, skip_keys=_POPPED_FROM_CHECKPOINT, dtype=torch.bfloat16,
    )
    log_stage("Lance streaming load", stage_start,
              extra=f"loaded={len(main_loaded)} unknown={len(main_unknown)}")
    if main_unknown:
        # Many Lance training-time keys (optimizer state, etc.) may not exist on the
        # inference model; informational, not fatal.
        log_rank0(f"[startup] NOTE: {len(main_unknown)} checkpoint key(s) had no matching param "
                  f"(first few: {main_unknown[:5]})")

    # Anything still meta (the popped sin-cos pos_embed, any non-checkpointed buffer)
    # gets allocated on its target device and re-initialized to the right values.
    materialized = _materialize_remaining_meta(model, device_map, dtype=torch.bfloat16)
    if materialized:
        log_rank0(f"[startup] Materialized {len(materialized)} meta param/buffer(s) post-load "
                  f"(first few: {materialized[:5]})")

    # init_moe() copies UND weights into the moe_gen slots. For inference from a fully-
    # trained Lance checkpoint, the moe_gen weights are already loaded above — running
    # init_moe now would either no-op (good) or clobber them with sharded cross-device
    # state_dict() copies (bad). Skip unconditionally on the meta-init path.
    if training_args.copy_init_moe:
        log_rank0("[startup] Skipping init_moe(): full checkpoint already contains moe_gen weights.")

    # ===== Tokenizer + post-load patch-ups. =====
    stage_start = time.perf_counter()
    log_rank0(f"[startup] Loading tokenizer: {model_args.model_path}")
    tokenizer: Qwen2Tokenizer = Qwen2Tokenizer.from_pretrained(model_args.model_path)
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
    log_stage("tokenizer load and special token init", stage_start, extra=f"num_new_tokens={num_new_tokens}")

    if num_new_tokens > 0:
        # Embedding and lm_head are both pinned to cuda:0 in the device_map, so
        # resize_token_embeddings can do its in-place resize without crossing devices.
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    if model_args.vit_type.lower() == "qwen2_5_vl":
        from common.model.hacks import hack_qwen2_5_vl_config
        language_model = hack_qwen2_5_vl_config(language_model)

    image_token_id = language_model.config.video_token_id  # <|image_pad|>
    new_token_ids.update({"image_token_id": image_token_id})
    model.update_tokenizer(tokenizer=tokenizer)

    if model_args.tie_word_embeddings: # and training_args.load_from_lance_checkpoint is False:
        # HACK: Handle the tying logic manually.
        model.language_model.untie_lm_head() # NOTE: untied lm head weights
        model.language_model.copy_new_token_rows_to_lm_head(num_new_tokens) # NOTE: copy the new token rows into lm_head

        # Make sure this stays False.
        model_args.tie_word_embeddings = False
        llm_config.tie_word_embeddings = False
    else:
        assert model.language_model.get_input_embeddings().weight.data.data_ptr() != model.language_model.get_output_embeddings().weight.data.data_ptr(), 'tie_word_embeddings conflict'

    # ===== Attach cross-device hooks so activations flow between shards. =====
    # dispatch_model walks `device_map` and installs pre/post forward hooks that move
    # activations to the right card before each submodule runs. After this point, the
    # model must NOT be .to()'d as that would collapse the shards.
    if shard_n > 1:
        model = dispatch_model(model, device_map=device_map)
    model.eval()
    if vae_model is not None and hasattr(vae_model, "eval"):
        vae_model.eval()

    # Setup packed dataloader
    stage_start = time.perf_counter()
    log_rank0(f"[startup] Loading dataset config and validation set: {data_args.val_dataset_config_file}")
    dataset_config = DataConfig.from_yaml(data_args.val_dataset_config_file)

    # NOTE: This block performs in-place assignments. ⚠️
    if training_args.visual_und:
        dataset_config.vit_patch_size = model_args.vit_patch_size
        dataset_config.vit_patch_size_temporal = model_args.vit_patch_size_temporal # TODO: fix
        dataset_config.vit_max_num_patch_per_side = model_args.vit_max_num_patch_per_side
        # dataset_config.vit_downsample = vit_downsample # NOTE: need to update !
    if training_args.visual_gen:
        assert len(model_args.latent_patch_size) == 3, "len(latent_patch_size) must be 3"
        vae_downsample = tuple_mul(
            model_args.latent_patch_size, (vae_config.downsample_temporal, vae_config.downsample_spatial, vae_config.downsample_spatial)
        )  # NOTE: This already includes patch_size.
        dataset_config.latent_patch_size = model_args.latent_patch_size
        dataset_config.vae_downsample = vae_downsample  # NOTE: update !
        dataset_config.max_latent_size = model_args.max_latent_size  # NOTE: update!
        dataset_config.max_num_frames = model_args.max_num_frames  # NOTE: update!

    # Fix: share dropout settings.
    dataset_config.text_cond_dropout_prob = model_args.text_cond_dropout_prob
    dataset_config.vae_cond_dropout_prob = model_args.vae_cond_dropout_prob
    dataset_config.vit_cond_dropout_prob = model_args.vit_cond_dropout_prob

    # Load inference parameters.
    dataset_config.num_frames = inference_args.num_frames
    dataset_config.H = inference_args.video_height
    dataset_config.W = inference_args.video_width
    dataset_config.task = inference_args.task
    dataset_config.resolution = inference_args.resolution
    dataset_config.text_template = inference_args.text_template
    dataset_config.enhance_prompt = inference_args.enhance_prompt
    if inference_args.enhance_prompt:
        if inference_args.task not in {TASK_T2V, TASK_I2V}:
            log_rank0("[startup] enhance_prompt is enabled but only applies to t2v and i2v; skipping prompt rewrite for this task.")
        else:
            log_rank0(f"[startup] enhance_prompt is enabled for {inference_args.task} prompts. Configure API_KEY in common/utils/caption_rewrite.py.")
    val_dataset = ValidationDataset(
        jsonl_path= data_args.val_dataset_config_file,
        tokenizer=tokenizer,
        data_args=data_args,
        model_args=model_args,
        training_args=training_args,
        new_token_ids=new_token_ids,
        dataset_config=dataset_config,
        local_rank=GLOBAL_RANK,  # global rank, not local rank
        world_size=WORLD_SIZE,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        collate_fn=simple_custom_collate,     # Top-level function
        drop_last=True,
        prefetch_factor=None,
        persistent_workers=False,
        multiprocessing_context=None,
    )
    log_stage("validation set and DataLoader init", stage_start, extra=f"dataset_size={len(val_dataset)}")

    # Prepare the validation data loader iterator.
    val_loader_iter = iter(val_loader)

    # Initialize a local dictionary to avoid accumulating stale data.
    if not hasattr(inference_args, "prompt_data_dict"):
        inference_args.prompt_data_dict = {}

    if not os.path.exists(inference_args.save_path_gen):
        os.makedirs(inference_args.save_path_gen)

    for epoch in trange(len(val_loader), desc="Validating", unit="batch", leave=True, ncols=80, disable=(GLOBAL_RANK != 0)):
        try:
            val_data_cpu = next(val_loader_iter)
        except StopIteration:
            break

        validate_on_fixed_batch(
            fsdp_model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            val_data_cpu=val_data_cpu,
            training_args=training_args,
            model_args=model_args,
            inference_args=inference_args,
            new_token_ids=new_token_ids,
            image_token_id=image_token_id,
            device=DEVICE,
            save_source_video=False, # Whether to save the GT video
            save_path_gen=inference_args.save_path_gen, # Generated video path
            save_path_gt="", # GT video path
        )
        del val_data_cpu
        clean_memory()

    # Final gather after all generation loops
    if dist.is_initialized():
        dist.barrier()
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, inference_args.prompt_data_dict)

        if GLOBAL_RANK == 0:
            merged = {}
            for d in gathered:
                merged.update(d)
            inference_args.prompt_data_dict = merged
            save_prompt_results(inference_args.prompt_data_dict, inference_args.save_path_gen, logger)
            if inference_args.task in UNDERSTANDING_TASKS:
                save_understanding_results(
                    prompt_data_dict=inference_args.prompt_data_dict,
                    dataset_config_file=data_args.val_dataset_config_file,
                    save_path_gen=inference_args.save_path_gen,
                )

    elif GLOBAL_RANK == 0:
        save_prompt_results(inference_args.prompt_data_dict, inference_args.save_path_gen, logger)
        if inference_args.task in UNDERSTANDING_TASKS:
            save_understanding_results(
                prompt_data_dict=inference_args.prompt_data_dict,
                dataset_config_file=data_args.val_dataset_config_file,
                save_path_gen=inference_args.save_path_gen,
            )

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
