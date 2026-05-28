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
import json
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
from common.val.utils import make_padded_latent
from data.datasets_custom import ValidationDataset
from config.config_factory import ModelArguments, DataArguments, EvaluationArguments, get_model_path


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

    model.load_state_dict(model_state_dict, strict=False)

    clean_memory(model_state_dict)


def clean_memory(*objects):
    """清理内存并释放 GPU 缓存"""
    for obj in objects:
        del obj
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def decode_video_tensor_for_geneval(v_list):
    """
    专门为 GenEVAL 解码视频张量，保持原有的保存格式
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


def resolve_geneval_paths(
    model_args: ModelArguments,
    data_args: DataArguments,
) -> None:
    if not model_args.model_path:
        raise ValueError("GenEVAL requires --model_path to be provided explicitly.")

    if not model_args.vit_path:
        model_args.vit_path = get_model_path("vit.qwen2_5_vl")

    if not data_args.val_dataset_config_file:
        data_args.val_dataset_config_file = get_model_path("geneval.data")


def build_runtime_dataset_config(
    model_args: ModelArguments,
    inference_args: EvaluationArguments,
    vae_config: Optional[AutoEncoderParams],
) -> DataConfig:
    """
    当前推理链不再依赖 dataset_config_file，运行期 DataConfig 由显式参数拼装。
    """
    dataset_config = DataConfig()

    dataset_config.num_frames = inference_args.num_frames
    dataset_config.H = inference_args.video_height
    dataset_config.W = inference_args.video_width
    dataset_config.task = inference_args.task
    dataset_config.resolution = inference_args.resolution
    dataset_config.text_template = inference_args.text_template
    dataset_config.max_duration = inference_args.max_duration
    dataset_config.system_prompt_type = inference_args.system_prompt_type

    if inference_args.visual_und:
        dataset_config.vit_patch_size = model_args.vit_patch_size
        dataset_config.vit_patch_size_temporal = model_args.vit_patch_size_temporal
        dataset_config.vit_max_num_patch_per_side = model_args.vit_max_num_patch_per_side

    if inference_args.visual_gen and vae_config:
        assert len(model_args.latent_patch_size) == 3, "len(latent_patch_size) must be 3"
        dataset_config.latent_patch_size = model_args.latent_patch_size
        dataset_config.vae_downsample = tuple_mul(
            model_args.latent_patch_size,
            (vae_config.downsample_temporal, vae_config.downsample_spatial, vae_config.downsample_spatial),
        )
        dataset_config.max_latent_size = model_args.max_latent_size
        dataset_config.max_num_frames = model_args.max_num_frames

    dataset_config.text_cond_dropout_prob = model_args.text_cond_dropout_prob
    dataset_config.vae_cond_dropout_prob = model_args.vae_cond_dropout_prob
    dataset_config.vit_cond_dropout_prob = model_args.vit_cond_dropout_prob

    return dataset_config


def validate_on_fixed_batch(
    fsdp_model: Lance,
    vae_model: Optional[WanVideoVAE],
    val_data_cpu: dict,
    model_args: ModelArguments,
    inference_args: EvaluationArguments,
    new_token_ids,
    image_token_id: int,
    device: int,
    save_source_video: bool = False,
    save_path_gen: str = "",
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

    val_data = val_data_cpu.cuda(device).to_dict()

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        # Compute padded_latent.
        if "padded_videos" in val_data.keys():
            val_data["padded_latent"] = make_padded_latent(val_data["padded_videos"], val_data["vae_data_mode"], vae_model)

        # Create an output folder from val_data["index"] first.
        index_save = val_data["index"]
        index_save = f"{index_save:05d}"
        os.makedirs(os.path.join(save_path_gen, index_save), exist_ok=True)
        os.makedirs(os.path.join(save_path_gen, index_save, "samples"), exist_ok=True)

        # Save metadata.jsonl.
        metadata = val_data["additional_info"]
        with open(os.path.join(save_path_gen, index_save, "metadata.jsonl"), 'w') as f:
            f.write(json.dumps(metadata, ensure_ascii=False) + "\n")

        # -------------------- GEN branch --------------------
        tensor_list_for_grid = []
        loop_iterator = trange(sample_num_per_prompt) if is_rank0 else range(sample_num_per_prompt)

        for sample_num_per_prompt_index in loop_iterator:
            # Sample generations.
            params = {
                "val_packed_text_ids": val_data["packed_text_ids"],
                "val_packed_text_indexes": val_data["packed_text_indexes"],
                "val_sample_lens": val_data["sample_lens"],
                "val_packed_position_ids": val_data["packed_position_ids"],
                "val_split_lens": val_data["split_lens"],
                "val_attn_modes": val_data["attn_modes"],
                "val_sample_N_target": val_data["sample_N_target"],
                "val_packed_vae_token_indexes": val_data["packed_vae_token_indexes"],
                "timestep_shift": inference_args.validation_timestep_shift,
                "num_timesteps": inference_args.validation_num_timesteps,
                "val_mse_loss_indexes": val_data.get("mse_loss_indexes", None),
                "val_padded_latent": val_data["padded_latent"],
                "video_sizes": val_data["video_sizes"],
                "cfg_text_scale": model_args.cfg_text_scale,
                "cfg_interval": inference_args.cfg_interval,
                "cfg_renorm_min": inference_args.cfg_renorm_min,
                "cfg_renorm_type": inference_args.cfg_renorm_type,
                "device": device,
                "dtype": torch.bfloat16,
                "new_token_ids": new_token_ids,
                "max_samples": inference_args.validation_max_samples,
                "validation_noise_seed": inference_args.validation_noise_seed + sample_num_per_prompt_index,
                "apply_chat_template": inference_args.apply_chat_template,
                "apply_qwen_2_5_vl_pos_emb": inference_args.apply_qwen_2_5_vl_pos_emb,
                "image_token_id": image_token_id,
                "val_packed_vit_token_indexes": val_data.get("packed_vit_token_indexes", None),
                "val_packed_vit_tokens": val_data.get("packed_vit_tokens", None),
                "vit_video_grid_thw": val_data.get("vit_video_grid_thw", None),
                "vae_video_grid_thw": val_data["vae_video_grid_thw"],
                "video_grid_thw": val_data.get("video_grid_thw", None),
                "caption": val_data.get("caption", None),
                "sample_task": val_data["sample_task"],
                "sample_modality": val_data["sample_modality"],
                "cfg_type": inference_args.cfg_type,
                "cfg_uncond_token_id": inference_args.cfg_uncond_token_id,
                "index": val_data["index"],
                "val_padded_videos": val_data["padded_videos"] if save_source_video else None,
            }

            if inference_args.use_KVcache:
                denoise_latent, _, _, _ = fsdp_model.validation_gen_KVcache(**params)
            else:
                denoise_latent, _, _, _ = fsdp_model.validation_gen(**params)

            # Decode and save.
            for latent in denoise_latent:
                v_list = [vae_model.vae_decode([latent_])[0] for latent_ in latent]

                # Keep the original save format.
                v_thwc = decode_video_tensor_for_geneval(v_list)

                # Use frame 0 directly.
                if v_thwc.shape[0] == 1:
                    tensor_list_for_grid.append(v_thwc.squeeze(0).cpu())

                    # Save a single image.
                    save_name = f"{save_path_gen}/{index_save}/samples/{sample_num_per_prompt_index}.png"
                    Image.fromarray((v_thwc.squeeze(0).permute(1, 2, 0).cpu().numpy()).astype('uint8')).save(save_name)
                else:
                    raise NotImplementedError("需要保存图像")

        # Save the grid image.
        save_name = f"{save_path_gen}/{index_save}/grid.png"
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
    model_args, data_args, inference_args = cast(Tuple[ModelArguments, DataArguments, EvaluationArguments], parser.parse_args_into_dataclasses())

    # ========================= GenEVAL path resolution ==============================
    resolve_geneval_paths(model_args, data_args)

    # NOTE: validation_noise_seed matches validation_data_seed.
    inference_args.validation_noise_seed = inference_args.evaluation_seed
    inference_args.validation_data_seed = inference_args.evaluation_seed
    # Set seed:
    seed = inference_args.global_seed * WORLD_SIZE + GLOBAL_RANK
    set_seed(seed)
    log_rank0 = print if GLOBAL_RANK == 0 else (lambda *_: None)

    # ========================= LLM model setup ==============================
    llm_config: Qwen2Config = Qwen2Config.from_json_file(osp.join(model_args.model_path, "llm_config.json"))

    llm_config.layer_module = model_args.layer_module
    llm_config.qk_norm = model_args.llm_qk_norm
    llm_config.qk_norm_und = model_args.llm_qk_norm_und
    llm_config.qk_norm_gen = model_args.llm_qk_norm_gen

    llm_config.tie_word_embeddings = model_args.tie_word_embeddings
    llm_config.freeze_und = inference_args.freeze_und
    llm_config.apply_qwen_2_5_vl_pos_emb = inference_args.apply_qwen_2_5_vl_pos_emb

    language_model: Qwen2ForCausalLM = Qwen2ForCausalLM(llm_config)

    if inference_args.visual_und:
        if model_args.vit_type in ("qwen2_5_vl", "qwen_2_5_vl_original"):
            vit_config = Qwen2_5_VLVisionConfig.from_pretrained(model_args.vit_path)
            vit_model = Qwen2_5_VisionTransformerPretrainedModel(vit_config)
            vit_weights = load_file(osp.join(model_args.vit_path, "vit.safetensors"))
            vit_model.load_state_dict(vit_weights, strict=True)
        else:
            raise ValueError(f"Unsupported vit_type: {model_args.vit_type}")

        clean_memory(vit_weights)

    if inference_args.visual_gen:
        vae_model = WanVideoVAE()
        vae_config: AutoEncoderParams = deepcopy(vae_model.vae_config)
    else:
        vae_model = None
        vae_config = None

    # Lance config.
    config = LanceConfig(
        visual_gen=inference_args.visual_gen,
        visual_und=inference_args.visual_und,
        llm_config=llm_config,
        vit_config=vit_config if inference_args.visual_und else None,
        vae_config=vae_config if inference_args.visual_gen else None,
        latent_patch_size=model_args.latent_patch_size,
        max_num_frames=model_args.max_num_frames,
        max_latent_size=model_args.max_latent_size,
        vit_max_num_patch_per_side=model_args.vit_max_num_patch_per_side,
        connector_act=model_args.connector_act,
        interpolate_pos=model_args.interpolate_pos,
        timestep_shift=inference_args.timestep_shift,
    )
    model: Lance = Lance(
        language_model=language_model,
        vit_model=vit_model if inference_args.visual_und else None,
        vit_type=model_args.vit_type,
        config=config,
        training_args=inference_args,
    )
    model = model.to(DEVICE)

    # Setup tokenizer for model:
    tokenizer: Qwen2Tokenizer = Qwen2Tokenizer.from_pretrained(model_args.model_path)

    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)

    # Initialize MoE before loading the checkpoint.
    if inference_args.copy_init_moe:
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
    # Some VAE wrappers (e.g. `WanVideoVAE`) are plain helper objects rather
    # than `nn.Module`s, and their internal model is already switched to eval.
    if vae_model is not None and hasattr(vae_model, "eval"):
        vae_model.eval()

    dataset_config = build_runtime_dataset_config(
        model_args=model_args,
        inference_args=inference_args,
        vae_config=vae_config,
    )

    # Create dataset.
    val_dataset = ValidationDataset(
        jsonl_path= data_args.val_dataset_config_file,
        tokenizer=tokenizer,
        data_args=data_args,
        model_args=model_args,
        training_args=inference_args,
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
    for _ in trange(len(val_loader), desc="Validating", unit="batch", leave=True, ncols=80, disable=(GLOBAL_RANK != 0)):
        val_data_cpu = next(val_loader_iter)

        validate_on_fixed_batch(
            fsdp_model=model,
            vae_model=vae_model,
            val_data_cpu=val_data_cpu,
            model_args=model_args,
            inference_args=inference_args,
            new_token_ids=new_token_ids,
            image_token_id=image_token_id,
            device=DEVICE,
            save_source_video=False,
            save_path_gen=inference_args.save_path_gen,
            sample_num_per_prompt=inference_args.sample_num_per_prompt,
        )

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
