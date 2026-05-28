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
from typing import Tuple, cast, Optional
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, set_seed
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
from safetensors.torch import load_file

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
TASK_X2T_IMAGE = "x2t_image"
TASK_X2T_VIDEO = "x2t_video"
TASK_IMAGE_EDIT = "image_edit"
TASK_VIDEO_EDIT = "video_edit"
GENERATION_TASKS = {
    TASK_T2V,
    TASK_T2I,
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

def init_from_model_path_if_needed(model: Qwen2ForCausalLM, model_args: ModelArguments):
    # Always load the trained Lance checkpoint from model_path.
    path_dir = model_args.model_path
    ema_path = osp.join(path_dir, "ema.safetensors")
    model_path = osp.join(path_dir, "model.safetensors")


    model_path_ft = None
    if osp.exists(model_path):
        model_path_ft = model_path
    elif osp.exists(ema_path):
        model_path_ft = ema_path

    if model_path_ft:
        model_state_dict = load_file(model_path_ft, device="cpu")
    else:
        raise FileNotFoundError(
            f"Fine-tuning failed: No valid checkpoint ('ema.safetensors' or 'model.safetensors') found in {path_dir}"
        )

    # NOTE: position embeds are fixed sinusoidal embeddings, so we can just pop it off,
    # which makes it easier to adapt to different resolutions.
    if 'latent_pos_embed.pos_embed' in model_state_dict:
        model_state_dict.pop('latent_pos_embed.pos_embed')

    msg = model.load_state_dict(model_state_dict, strict=False)  # strict = True | False
    clean_memory(model_state_dict)

    return msg


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
    fsdp_model = fsdp_model.to(device=device, dtype=torch.bfloat16)

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
                if inference_args.task in {TASK_IMAGE_EDIT, TASK_VIDEO_EDIT}:
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

    stage_start = time.perf_counter()
    log_rank0(f"[startup] Initializing LLM weights: {model_args.model_path}")
    language_model: Qwen2ForCausalLM = Qwen2ForCausalLM(llm_config)
    log_stage("LLM weight init", stage_start)

    if training_args.visual_und:
        if model_args.vit_type in ("qwen2_5_vl", "qwen_2_5_vl_original"):
            stage_start = time.perf_counter()
            log_rank0(f"[startup] Loading VIT config: {model_args.vit_path}")
            vit_config = Qwen2_5_VLVisionConfig.from_pretrained(model_args.vit_path)
            log_stage("VIT config load", stage_start)

            stage_start = time.perf_counter()
            log_rank0(f"[startup] Loading VIT weights: {osp.join(model_args.vit_path, 'vit.safetensors')}")
            vit_model = Qwen2_5_VisionTransformerPretrainedModel(vit_config)
            vit_weights = load_file(osp.join(model_args.vit_path, "vit.safetensors"))
            vit_model.load_state_dict(vit_weights, strict=True)
            log_stage("VIT weight load", stage_start)
        else:
            raise ValueError(f"Unsupported vit_type: {model_args.vit_type}")

        clean_memory(vit_weights)

    if training_args.visual_gen:
        stage_start = time.perf_counter()
        log_rank0("[startup] Initializing VAE")
        vae_model = WanVideoVAE()
        vae_config: AutoEncoderParams = deepcopy(vae_model.vae_config)
        log_stage("VAE init", stage_start)
    else:
        vae_model = None
        vae_config = None

    # Lance configuration
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
    model: Lance = Lance(
        language_model=language_model,
        vit_model=vit_model if training_args.visual_und else None,
        vit_type=model_args.vit_type,
        config=config,
        training_args=training_args,
    )
    stage_start = time.perf_counter()
    log_rank0(f"[startup] Moving Lance model to GPU {DEVICE}")
    model = model.to(DEVICE)
    log_stage("Lance model move to GPU", stage_start)

    # Setup tokenizer for model:
    stage_start = time.perf_counter()
    log_rank0(f"[startup] Loading tokenizer: {model_args.model_path}")
    tokenizer: Qwen2Tokenizer = Qwen2Tokenizer.from_pretrained(model_args.model_path)

    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
    log_stage("tokenizer load and special token init", stage_start, extra=f"num_new_tokens={num_new_tokens}")

    # Initialize MoE before loading the checkpoint.
    if training_args.copy_init_moe:
        language_model.init_moe()

    init_from_model_path_if_needed(model, model_args)

    # Resize afterward to avoid checkpoint shape mismatches or overwritten weights.
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    if model_args.vit_type.lower() == "qwen2_5_vl":
        from common.model.hacks import hack_qwen2_5_vl_config
        language_model = hack_qwen2_5_vl_config(language_model)

    image_token_id = language_model.config.video_token_id # image_token_id # <|image_pad|>
    new_token_ids.update({"image_token_id": image_token_id})
    model.update_tokenizer(tokenizer=tokenizer)

    if model_args.tie_word_embeddings: # and training_args.finetune_from_hf is False:
        # HACK: Handle the tying logic manually.
        model.language_model.untie_lm_head() # NOTE: untied lm head weights
        model.language_model.copy_new_token_rows_to_lm_head(num_new_tokens) # NOTE: copy the new token rows into lm_head

        # Make sure this stays False.
        model_args.tie_word_embeddings = False
        llm_config.tie_word_embeddings = False
    else: # HACK!!!
        assert model.language_model.get_input_embeddings().weight.data.data_ptr() != model.language_model.get_output_embeddings().weight.data.data_ptr(), 'tie_word_embeddings conflict'

    model = model.to(device=DEVICE, dtype=torch.bfloat16)
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
