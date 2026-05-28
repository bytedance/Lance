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
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import os.path as osp
from copy import deepcopy
from typing import Tuple, cast, Optional
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, set_seed
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
from safetensors.torch import load_file
from PIL import Image
from torchvision.utils import make_grid
import numpy as np
from tqdm import trange

from data.dataset_base import DataConfig, simple_custom_collate
from data.data_utils import add_special_tokens
from modeling.vae.wan.model import WanVideoVAE
from modeling.lance import LanceConfig, Lance, Qwen2ForCausalLM
from modeling.qwen2 import Qwen2Tokenizer
from modeling.qwen2.modeling_qwen2 import Qwen2Config
from modeling.vit.qwen2_5_vl_vit import Qwen2_5_VisionTransformerPretrainedModel
from common.utils.misc import tuple_mul, AutoEncoderParams
from common.utils.logging import get_logger
from common.val.utils import make_padded_latent
from data.datasets_custom import ValidationDataset
from config.config_factory import ModelArguments, DataArguments, TrainingArguments, EvaluationArguments, get_model_path


def init_from_vlm_if_needed(model: Qwen2ForCausalLM, model_args: ModelArguments, log_rank0):
    # NOTE: VLM initialization loads through this path.
    def load_safetensors_state_dict(folder_path):
        # Select safetensors files only and sort by filename for deterministic order.
        safetensor_files = sorted(
            f for f in os.listdir(folder_path) if f.endswith(".safetensors")
        )
        state_dict = {}
        for filename in safetensor_files:
            file_path = osp.join(folder_path, filename)
            state_dict.update(load_file(file_path))
        return state_dict

    state_dict = load_safetensors_state_dict(model_args.llm_path)

    # Rename parameters to match Lance parameter names.
    for k in list(state_dict.keys()):
        if "visual" in k:  # ViT and connector
            state_dict[k.replace("visual", "vit_model")] = state_dict.pop(k)
        else:
            # Add the language_model prefix.
            state_dict["language_model." + k] = state_dict.pop(k)

    result = model.load_state_dict(state_dict, strict=False)

    clean_memory(state_dict)


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

    msg = model.load_state_dict(model_state_dict, strict=False)

    clean_memory(model_state_dict)

    return msg


def clean_memory(*objects):
    """清理内存并释放 GPU 缓存"""
    for obj in objects:
        del obj
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def decode_video_tensor_for_dpg(v_list):
    """
    专门为 DPG 解码视频张量，保持原有的保存格式
    """
    N_target = len(v_list)
    if N_target != 1:
        from einops import rearrange
        padded_videos_latent = [v.permute(1, 0, 2, 3) for v in v_list]
        v_tc_hw = rearrange(padded_videos_latent, "n t c h w -> t c h (n w)")
    else:
        v_tc_hw = v_list[0].permute(1, 0, 2, 3)

    v_tc_hw = v_tc_hw.float().clip(-1, 1).mul_(0.5).add_(0.5).mul_(255).round().clamp(0, 255).to(torch.uint8)
    return v_tc_hw


def resolve_dpg_paths(
    model_args: ModelArguments,
    data_args: DataArguments,
) -> None:
    if not model_args.model_path:
        raise ValueError("DPG requires --model_path to be provided explicitly.")

    if not model_args.llm_path:
        model_args.llm_path = model_args.model_path

    if not model_args.vit_path:
        model_args.vit_path = get_model_path("vit.qwen2_5_vl")

    if not data_args.val_dataset_config_file:
        data_args.val_dataset_config_file = get_model_path("dpg.data")


def validate_on_fixed_batch(
    fsdp_model: Lance,
    vae_model: Optional[WanVideoVAE],
    tokenizer: Qwen2Tokenizer,
    val_data_cpu: dict,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    data_args: DataArguments,
    inference_args: EvaluationArguments,
    curr_step: int,
    logger,
    new_token_ids,
    image_token_id: int,
    device: int,
    save_source_video: bool = False,
    save_path_gen: str = "",
    save_path_gt: str = "",
    sample_num_per_prompt: int = 1,
):
    """
    验证逻辑，保持与原文件相同的保存格式
    """
    # Check whether distributed execution has been initialized.
    if dist.is_initialized():
        is_rank0 = (dist.get_rank() == 0)
    else:
        is_rank0 = True

    log_rank0 = logger.info if is_rank0 else (lambda *_: None)
    val_data = val_data_cpu.cuda(device).to_dict()

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        # Compute padded_latent.
        if "padded_videos" in val_data.keys():
            val_data["padded_latent"] = make_padded_latent(val_data["padded_videos"], val_data["vae_data_mode"], vae_model)

        # -------------------- GEN branch --------------------
        tensor_list_for_grid = []
        loop_iterator = trange(sample_num_per_prompt) if is_rank0 else range(sample_num_per_prompt)

        # Support resumable generation.
        save_name = f"{save_path_gen}/{val_data['index']}.png"
        if os.path.exists(save_name):
            return None

        for sample_num_per_prompt_index in loop_iterator:
            # Sample generations with the original parameters.
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
                "validation_noise_seed": training_args.validation_noise_seed + sample_num_per_prompt_index,
                "apply_chat_template": training_args.apply_chat_template,
                "apply_qwen_2_5_vl_pos_emb": training_args.apply_qwen_2_5_vl_pos_emb,
                "image_token_id": image_token_id,
                "val_packed_vit_token_indexes": val_data.get("packed_vit_token_indexes", None),
                "val_packed_vit_tokens": val_data.get("packed_vit_tokens", None),
                "vit_video_grid_thw": val_data.get("vit_video_grid_thw", None),
                "vae_video_grid_thw": val_data["vae_video_grid_thw"],
                "video_grid_thw": val_data.get("video_grid_thw", None),
                "caption": val_data.get("caption", None),
                "sample_task": val_data["sample_task"],
                "sample_modality": val_data["sample_modality"],
                "cfg_type": training_args.cfg_type,
                "cfg_uncond_token_id": training_args.cfg_uncond_token_id,
                "index": val_data["index"],
                "val_padded_videos": val_data["padded_videos"] if save_source_video else None,
            }

            if training_args.use_KVcache:
                denoise_latent, captions, padded_videos, index = fsdp_model.validation_gen_KVcache(**params)
            else:
                denoise_latent, captions, padded_videos, index = fsdp_model.validation_gen(**params)

            # Decode and save.
            for i_val, latent in enumerate(denoise_latent):
                v_list = [vae_model.vae_decode([latent_])[0] for latent_ in latent]

                # Keep the original save format.
                v_thwc = decode_video_tensor_for_dpg(v_list)

                # Use frame 0 directly.
                if v_thwc.shape[0] == 1:
                    tensor_list_for_grid.append(v_thwc.squeeze(0).cpu())
                else:
                    raise NotImplementedError("需要保存图像")

    # Keep the original save format.
    grid_tensor = make_grid(tensor_list_for_grid, nrow=int(np.sqrt(sample_num_per_prompt)), padding=0, pad_value=255)
    grid_numpy = grid_tensor.permute(1, 2, 0).numpy()
    Image.fromarray(grid_numpy).save(save_name)


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
    parser = HfArgumentParser((ModelArguments, DataArguments, EvaluationArguments))
    model_args, data_args, inference_args = cast(
        Tuple[ModelArguments, DataArguments, EvaluationArguments],
        parser.parse_args_into_dataclasses(),
    )
    training_args = inference_args

    # ========================= DPG path resolution ==============================
    resolve_dpg_paths(model_args, data_args)

    # NOTE: validation_noise_seed matches validation_data_seed.
    training_args.validation_noise_seed = inference_args.evaluation_seed
    training_args.validation_data_seed = inference_args.evaluation_seed
    logger = get_logger()
    log_rank0 = print if GLOBAL_RANK == 0 else (lambda *_: None)

    # Set seed:
    seed = training_args.global_seed * WORLD_SIZE + GLOBAL_RANK
    set_seed(seed)

    # ========================= LLM model setup ==============================
    llm_config: Qwen2Config = Qwen2Config.from_json_file(osp.join(model_args.model_path, "llm_config.json"))

    llm_config.layer_module = model_args.layer_module
    llm_config.qk_norm = model_args.llm_qk_norm
    llm_config.qk_norm_und = model_args.llm_qk_norm_und
    llm_config.qk_norm_gen = model_args.llm_qk_norm_gen

    llm_config.tie_word_embeddings = model_args.tie_word_embeddings
    llm_config.freeze_und = training_args.freeze_und
    llm_config.apply_qwen_2_5_vl_pos_emb = training_args.apply_qwen_2_5_vl_pos_emb

    language_model: Qwen2ForCausalLM = Qwen2ForCausalLM(llm_config)

    if training_args.visual_und:
        if model_args.vit_type in ("qwen2_5_vl", "qwen_2_5_vl_original"):
            vit_config = Qwen2_5_VLVisionConfig.from_pretrained(model_args.vit_path)
            vit_model = Qwen2_5_VisionTransformerPretrainedModel(vit_config)
            vit_weights = load_file(osp.join(model_args.vit_path, "vit.safetensors"))
            vit_model.load_state_dict(vit_weights, strict=True)
        else:
            raise ValueError(f"Unsupported vit_type: {model_args.vit_type}")

        clean_memory(vit_weights)

    if training_args.visual_gen:
        vae_model = WanVideoVAE()
        vae_config: AutoEncoderParams = deepcopy(vae_model.vae_config)
    else:
        vae_model = None
        vae_config = None

    # Lance config.
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
    model = model.to(DEVICE)

    # Setup tokenizer for model:
    tokenizer: Qwen2Tokenizer = Qwen2Tokenizer.from_pretrained(model_args.model_path)

    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)

    # Initialize MoE before loading the checkpoint.
    if training_args.copy_init_moe:
        language_model.init_moe()

    init_from_model_path_if_needed(model, model_args)

    # Resize after loading the checkpoint.
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    if model_args.vit_type.lower() == "qwen2_5_vl":
        from common.model.hacks import hack_qwen2_5_vl_config
        language_model = hack_qwen2_5_vl_config(language_model)

    image_token_id = language_model.config.video_token_id
    new_token_ids.update({"image_token_id": image_token_id})
    model.update_tokenizer(tokenizer=tokenizer)

    if model_args.tie_word_embeddings:
        model.language_model.untie_lm_head()
        model.language_model.copy_new_token_rows_to_lm_head(num_new_tokens)

        model_args.tie_word_embeddings = False
        llm_config.tie_word_embeddings = False
    else:
        assert model.language_model.get_input_embeddings().weight.data.data_ptr() != model.language_model.get_output_embeddings().weight.data.data_ptr(), 'tie_world_embeddings 冲突'

    model = model.to(device=DEVICE, dtype=torch.bfloat16)
    model.eval()
    if vae_model is not None and hasattr(vae_model, "eval"):
        vae_model.eval()

    # Setup packed dataloader with a simple DataConfig instance.
    dataset_config = DataConfig(grouped_datasets={})

    # Configure basic parameters.
    dataset_config.num_frames = inference_args.num_frames
    dataset_config.H = inference_args.video_height
    dataset_config.W = inference_args.video_width
    dataset_config.task = inference_args.task
    dataset_config.resolution = inference_args.resolution
    dataset_config.text_template = inference_args.text_template

    # Configure VIT parameters.
    if training_args.visual_und:
        dataset_config.vit_patch_size = model_args.vit_patch_size
        dataset_config.vit_patch_size_temporal = model_args.vit_patch_size_temporal
        dataset_config.vit_max_num_patch_per_side = model_args.vit_max_num_patch_per_side

    # Configure VAE parameters.
    if training_args.visual_gen and vae_config:
        assert len(model_args.latent_patch_size) == 3, "len(latent_patch_size) must be 3"
        vae_downsample = tuple_mul(
            model_args.latent_patch_size, (vae_config.downsample_temporal, vae_config.downsample_spatial, vae_config.downsample_spatial)
        )
        dataset_config.latent_patch_size = model_args.latent_patch_size
        dataset_config.vae_downsample = vae_downsample
        dataset_config.max_latent_size = model_args.max_latent_size
        dataset_config.max_num_frames = model_args.max_num_frames

    # Share dropout settings.
    dataset_config.text_cond_dropout_prob = model_args.text_cond_dropout_prob
    dataset_config.vae_cond_dropout_prob = model_args.vae_cond_dropout_prob
    dataset_config.vit_cond_dropout_prob = model_args.vit_cond_dropout_prob

    # Create dataset.
    val_dataset = ValidationDataset(
        jsonl_path= data_args.val_dataset_config_file,
        tokenizer=tokenizer,
        data_args=data_args,
        model_args=model_args,
        training_args=training_args,
        new_token_ids=new_token_ids,
        dataset_config=dataset_config,
        local_rank=GLOBAL_RANK,
        world_size=WORLD_SIZE,
    )

    val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            collate_fn=simple_custom_collate,
            drop_last=True,
            prefetch_factor=None,
            persistent_workers=False,
            multiprocessing_context=None,
        )

    val_loader_iter = iter(val_loader)

    if not os.path.exists(inference_args.save_path_gen):
        os.makedirs(inference_args.save_path_gen, exist_ok=True)

    # Main loop.
    from tqdm import tqdm
    import time
    from datetime import datetime, timedelta

    total_batches = len(val_loader)
    pbar = tqdm(total=total_batches, desc="Validating", unit="batch", leave=True, ncols=120, disable=(GLOBAL_RANK != 0))
    start_time = time.time()

    for i in range(total_batches):
        val_data_cpu = next(val_loader_iter)

        validate_on_fixed_batch(
            fsdp_model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            val_data_cpu=val_data_cpu,
            training_args=training_args,
            model_args=model_args,
            data_args=data_args,
            inference_args=inference_args,
            curr_step=0,
            logger=logger,
            new_token_ids=new_token_ids,
            image_token_id=image_token_id,
            device=DEVICE,
            save_source_video=False,
            save_path_gen=inference_args.save_path_gen,
            save_path_gt="",
            sample_num_per_prompt=inference_args.sample_num_per_prompt,
        )

        if GLOBAL_RANK == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            eta_seconds = avg_time * (total_batches - i - 1)
            expected_finish = datetime.now() + timedelta(seconds=eta_seconds)
            finish_str = expected_finish.strftime('%Y-%m-%d %H:%M:%S')

            pbar.set_postfix_str(f"ETA: {timedelta(seconds=int(eta_seconds))} | Finish: {finish_str}")
            pbar.update(1)

    if GLOBAL_RANK == 0:
        pbar.close()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
