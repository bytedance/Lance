# coding: utf-8

# Standard library imports
import json
import os
import warnings
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from time import time
from typing import List, Optional

warnings.filterwarnings(
    "ignore",
    message=".*torch\\.cuda\\.amp\\.custom_(fwd|bwd).*deprecated.*",
    category=FutureWarning,
    module="deepspeed.runtime.zero.linear",
)

# Third-party package imports
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from transformers.optimization import (
    get_constant_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup,
)
from modeling.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig

# Local repository imports
from common.utils.basic import get_global_rank
from common.utils.fs import download, mkdir
from common.utils.misc import AutoEncoderParams, tuple_mul
from common.val.utils import (
    decode_text_interleave,
    decode_video_tensor,
    make_padded_latent,
    map_splits_to_samples,
)
from config.config_factory import ModelArguments, DataArguments, TrainingArguments
from data.dataset_base_train import DataConfig, PackedDataset, simple_custom_collate
from modeling.lance import Lance, Qwen2ForCausalLM
from modeling.qwen2 import Qwen2Tokenizer
from modeling.qwen2.modeling_qwen2 import Qwen2Config
from modeling.vae.wan.model import WanVideoVAE
from modeling.vit.qwen2_5_vl_vit import Qwen2_5_VisionTransformerPretrainedModel
from train.fsdp_utils import FSDPCheckpoint, FSDPConfig, fsdp_ema_setup, fsdp_ema_update


def setup_output_paths(
    training_args: TrainingArguments,
    logger,
    global_rank: int,
):
    run_output_dir = os.path.join(training_args.outputs_dir, training_args.wandb_name)
    training_args.run_output_dir = run_output_dir
    training_args.config_dir = os.path.join(run_output_dir, "configs")
    training_args.ckpt_dir = os.path.join(run_output_dir, "checkpoints")
    training_args.wandb_dir = os.path.join(run_output_dir, "wandb")

    if global_rank == 0:
        mkdir(training_args.config_dir)
        mkdir(training_args.ckpt_dir)
        if training_args.wandb_offline:
            mkdir(training_args.wandb_dir)
        logger.info(f"training_args.run_output_dir: {training_args.run_output_dir}")
        logger.info(f"training_args.config_dir: {training_args.config_dir}")
        logger.info(f"training_args.ckpt_dir: {training_args.ckpt_dir}")
        if training_args.wandb_offline:
            logger.info(f"training_args.wandb_dir: {training_args.wandb_dir}")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def setup_rank0_logging_and_wandb(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    logger,
    global_rank: int,
):
    if global_rank != 0:
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb_name = training_args.wandb_name[:64]
    wandb_id = f"{wandb_name}-{timestamp}"[:64]

    wandb_init_kwargs = {
        "project": training_args.wandb_project,
        "name": wandb_name,
        "id": wandb_id,
        "resume": training_args.wandb_resume,
        "mode": "offline" if training_args.wandb_offline else "online",
    }
    if training_args.wandb_offline:
        wandb_init_kwargs["dir"] = training_args.wandb_dir

    wandb.init(**wandb_init_kwargs)
    wandb.config.update({**vars(training_args), **vars(model_args), **vars(data_args)})


def prepare_model_paths(model_args: ModelArguments, training_args: TrainingArguments):
    if training_args.load_from_lance_checkpoint:
        model_args.model_path = download(model_args.model_path, add_hash_suffix=True)
        required_files = [
            "llm_config.json",
            "generation_config.json",
            "merges.txt",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
        ]
        missing_files = [
            filename
            for filename in required_files
            if not os.path.exists(os.path.join(model_args.model_path, filename))
        ]
        if missing_files:
            raise FileNotFoundError(
                "MODEL_PATH must contain all Lance LLM/tokenizer files when "
                f"load_from_lance_checkpoint=True. Missing in {model_args.model_path}: "
                f"{', '.join(missing_files)}"
            )
    else:
        model_args.llm_path = download(model_args.llm_path)

    model_args.vit_path = download(model_args.vit_path)


def freeze_model_components(model: Lance, training_args: TrainingArguments, log_rank0):
    if training_args.freeze_llm:
        model.language_model.eval()
        for param in model.language_model.parameters():
            param.requires_grad = False
        log_rank0("Freeze all LLM parameters")

    if training_args.freeze_llm_embed_tokens:
        model.language_model.freeze_embed_tokens()
        model.language_model.freeze_lm_head()
        log_rank0("Freeze LLM token embeddings and lm_head")

    if training_args.freeze_und_params:
        model.language_model.freeze_und_params()
        log_rank0("Freeze UND parameters")

    if training_args.freeze_vit and training_args.visual_und:
        model.vit_model.eval()
        for param in model.vit_model.parameters():
            param.requires_grad = False
        log_rank0("Freeze VIT parameters")

    if training_args.freeze_vit_connector:
        model.connector.eval()
        for param in model.connector.parameters():
            param.requires_grad = False
        log_rank0("Freeze VIT connector parameters")


def save_trainable_parameters(model: Lance, fsdp_model: torch.nn.Module, training_args: TrainingArguments, logger, global_rank: int):
    if global_rank != 0:
        return

    report_path = os.path.join(training_args.config_dir, "trainable_parameters.txt")
    sep = "=" * 40 + " check requires_grad " + "=" * 40
    lines = [
        sep,
        "fsdp_model:",
        str(fsdp_model),
        sep,
    ]
    for name, param in model.named_parameters():
        if param.requires_grad:
            lines.append(f"{name}: {param.requires_grad}")
    lines.append(sep)

    mkdir(os.path.dirname(report_path))
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")
    logger.info(f"Saved trainable parameter report to {report_path}")


def save_checkpoint_load_report(
    report_dir: Optional[str],
    report_name: str,
    title: str,
    matched: int,
    not_matched: int,
    missing,
    unexpected,
    log_rank0,
):
    log_rank0(f"{title}: matched={matched}, not_matched={not_matched}, missing={len(missing)}, unexpected={len(unexpected)}")
    if get_global_rank() != 0 or report_dir is None:
        return

    mkdir(report_dir)
    report_path = os.path.join(report_dir, report_name)
    lines = [
        title,
        f"matched: {matched}",
        f"not_matched: {not_matched}",
        f"missing_count: {len(missing)}",
        f"unexpected_count: {len(unexpected)}",
        "",
        "missing_keys:",
        *[str(item) for item in missing],
        "",
        "unexpected_keys:",
        *[str(item) for item in unexpected],
    ]
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")
    log_rank0(f"Saved checkpoint load report to {report_path}")


def setup_model_components(model_args: ModelArguments, training_args: TrainingArguments, log_rank0):
    if training_args.load_from_lance_checkpoint:
        llm_config: Qwen2Config = Qwen2Config.from_json_file(os.path.join(model_args.model_path, "llm_config.json"))
    else:
        llm_config: Qwen2Config = Qwen2Config.from_pretrained(model_args.llm_path)

    llm_config.layer_module = model_args.layer_module
    llm_config.qk_norm = model_args.llm_qk_norm
    llm_config.qk_norm_und = model_args.llm_qk_norm_und
    llm_config.qk_norm_gen = model_args.llm_qk_norm_gen
    log_rank0(f"llm_config.qk_norm: {llm_config.qk_norm}, llm_config.qk_norm_und: {llm_config.qk_norm_und}, llm_config.qk_norm_gen: {llm_config.qk_norm_gen}")

    llm_config.tie_word_embeddings = model_args.tie_word_embeddings
    llm_config.freeze_und = training_args.freeze_und
    llm_config.apply_qwen_2_5_vl_pos_emb = training_args.apply_qwen_2_5_vl_pos_emb

    if training_args.load_from_lance_checkpoint or training_args.init_from_vlm_checkpoint:
        language_model: Qwen2ForCausalLM = Qwen2ForCausalLM(llm_config)
    else:
        language_model: Qwen2ForCausalLM = Qwen2ForCausalLM.from_pretrained(model_args.llm_path, config=llm_config)

    vit_config, vit_model = None, None
    if training_args.visual_und:
        if model_args.vit_type in ("qwen2_5_vl", "qwen_2_5_vl_original"):
            vit_config = Qwen2_5_VLVisionConfig.from_pretrained(model_args.vit_path)
            vit_model = Qwen2_5_VisionTransformerPretrainedModel(vit_config)
            vit_weights = load_file(os.path.join(model_args.vit_path, "vit.safetensors"))
            msg = vit_model.load_state_dict(vit_weights, strict=True)
            log_rank0(f"Load vit model weights: {msg}, from {model_args.vit_path}")
        else:
            raise ValueError(f"Unsupported vit_type: {model_args.vit_type}")

        del vit_weights
        import gc; gc.collect(); torch.cuda.empty_cache()

    if training_args.visual_gen:
        if training_args.vae_model_type.lower() in ("wan", "wanvideo", "wan-video"):
            vae_model = WanVideoVAE()
        else:
            raise ValueError(f"Unsupported vae_model_type: {training_args.vae_model_type}")
        vae_config: AutoEncoderParams = deepcopy(vae_model.vae_config)
    else:
        vae_model = None
        vae_config = None

    return llm_config, language_model, vit_config, vit_model, vae_model, vae_config


def build_fsdp_config(training_args: TrainingArguments):
    return FSDPConfig(
        sharding_strategy=training_args.sharding_strategy,
        backward_prefetch=training_args.backward_prefetch,
        cpu_offload=training_args.cpu_offload,
        num_replicate=training_args.num_replicate,
        num_shard=training_args.num_shard,
        use_orig_params=True,
    )


def build_lr_scheduler(optimizer: torch.optim.Optimizer, training_args: TrainingArguments):
    if training_args.lr_scheduler == "cosine":
        return get_cosine_with_min_lr_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.total_steps,
            min_lr=training_args.min_lr,
            num_cycles=5,
        )

    if training_args.lr_scheduler == "constant":
        return get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=training_args.warmup_steps,
        )

    raise ValueError(f"Unknown lr_scheduler: {training_args.lr_scheduler}")


def load_training_state(
    optimizer: torch.optim.Optimizer,
    scheduler,
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    resume_from,
    resume_model_only: bool,
    fsdp_config: FSDPConfig,
    global_rank: int,
    world_size: int,
):
    if not resume_model_only:
        return FSDPCheckpoint.try_load_train_state(
            resume_from,
            optimizer,
            scheduler,
            fsdp_config,
        )

    train_step = 0
    data_status = None
    if not training_args.load_data_status:
        return optimizer, scheduler, train_step, data_status

    try:
        train_step = int(os.path.basename(os.path.normpath(model_args.model_path)).split(".")[0]) + 1
        data_status_path = os.path.join(model_args.model_path, "data_status.pt")
        data_status_all = torch.load(data_status_path, weights_only=True, map_location="cpu")
        assert world_size == len(data_status_all), f"WORLD_SIZE ({world_size}) must be equal to the length of data_status_all ({len(data_status_all)})"

        dataset_names, data_worker_ids = [], []
        for d_status in data_status_all:
            for d_name, d_worker_info in d_status.items():
                dataset_names.append(d_name)
                data_worker_ids.extend(d_worker_info.keys())
        dataset_names = list(set(dataset_names))
        data_worker_ids = list(set(data_worker_ids))

        assert data_args.num_workers == len(data_worker_ids), f"num_workers ({data_args.num_workers}) must be equal to the length of data_worker_ids ({len(data_worker_ids)})"
        data_status = data_status_all[global_rank]
        print(
            f"Successfully load train_step and data_status ***** \n"
            f"train_step: {train_step}, data_status_all: {data_status_all}, global_rank: {global_rank}, data_status: {data_status}\n"
        )
    except Exception:
        train_step = 0
        data_status = None
        print(
            f"Failed to load train_step and data_status ***** \n"
            f"train_step: {train_step}, data_status: {data_status}\n"
        )

    return optimizer, scheduler, train_step, data_status


def build_train_dataset_config(
    data_args: DataArguments,
    model_args: ModelArguments,
    training_args: TrainingArguments,
    vae_config: Optional[AutoEncoderParams],
):
    dataset_config = DataConfig.from_yaml(data_args.dataset_config_file)

    if training_args.visual_und:
        dataset_config.vit_patch_size = model_args.vit_patch_size
        dataset_config.vit_patch_size_temporal = model_args.vit_patch_size_temporal
        dataset_config.vit_max_num_patch_per_side = model_args.vit_max_num_patch_per_side

    if training_args.visual_gen:
        assert len(model_args.latent_patch_size) == 3, "len(latent_patch_size) must be 3"
        vae_downsample = tuple_mul(
            model_args.latent_patch_size,
            (
                vae_config.downsample_temporal,
                vae_config.downsample_spatial,
                vae_config.downsample_spatial,
            ),
        )
        dataset_config.latent_patch_size = model_args.latent_patch_size
        dataset_config.vae_downsample = vae_downsample
        dataset_config.max_latent_size = model_args.max_latent_size
        dataset_config.max_num_frames = model_args.max_num_frames

    dataset_config.text_cond_dropout_prob = model_args.text_cond_dropout_prob
    dataset_config.vae_cond_dropout_prob = model_args.vae_cond_dropout_prob
    dataset_config.vit_cond_dropout_prob = model_args.vit_cond_dropout_prob

    return dataset_config


def compute_training_loss(
    loss_dict: dict,
    data: dict,
    ce_loss_weights,
    training_args: TrainingArguments,
    device: int,
    world_size: int,
):
    loss = torch.tensor(0.0, device=device)

    ce = loss_dict["ce"]
    if ce is not None:
        total_ce_tokens = torch.tensor(len(data["ce_loss_indexes"]), device=device)
        dist.all_reduce(total_ce_tokens, op=dist.ReduceOp.SUM)
        if training_args.ce_loss_reweighting:
            ce = ce * ce_loss_weights
            total_ce_loss_weights = ce_loss_weights.sum()
            dist.all_reduce(total_ce_loss_weights, op=dist.ReduceOp.SUM)
            ce = ce.sum() * world_size / total_ce_loss_weights
        else:
            ce = ce.sum() * world_size / total_ce_tokens
        loss_dict["ce"] = ce.detach()
        loss = loss + ce * training_args.ce_weight
    else:
        loss_dict["ce"] = torch.tensor(0, device=device)
        total_ce_tokens = torch.tensor(0, device=device)

    total_mse_tokens = loss_dict.pop("total_mse_tokens")
    frame_mse = loss_dict.pop("frame_mse")
    if training_args.visual_gen:
        mse = loss_dict["mse"]
        if mse is not None:
            total_mse_tokens = torch.tensor(total_mse_tokens, device=device)
            dist.all_reduce(total_mse_tokens, op=dist.ReduceOp.SUM)
            mse = mse.mean(dim=-1).sum() * world_size / total_mse_tokens
            loss_dict["mse"] = mse.detach()
            loss = loss + mse * training_args.mse_weight
        else:
            loss_dict["mse"] = torch.tensor(0, device=device)
            total_mse_tokens = torch.tensor(0, device=device)

        if frame_mse is not None:
            total_frame_mse_tokens = torch.tensor(sum(data["key_frame_mask"] == 1), device=device)
            dist.all_reduce(total_frame_mse_tokens, op=dist.ReduceOp.SUM)
            frame_mse = frame_mse.mean(dim=-1).sum() * world_size / total_frame_mse_tokens
            loss_dict["frame_mse"] = frame_mse.detach()
            loss = loss + frame_mse * training_args.mse_weight
        else:
            loss_dict["frame_mse"] = torch.tensor(0, device=device)
    else:
        loss_dict["mse"] = torch.tensor(0, device=device)
        total_mse_tokens = torch.tensor(0, device=device)

    return loss, loss_dict, total_ce_tokens, total_mse_tokens


def optimizer_step_with_ema(
    loss,
    fsdp_model: torch.nn.Module,
    ema_model,
    optimizer: torch.optim.Optimizer,
    scheduler,
    training_args: TrainingArguments,
    curr_step: int,
    log_rank0,
):
    optimizer.zero_grad()
    loss.backward()
    total_norm = fsdp_model.clip_grad_norm_(training_args.max_grad_norm)

    jump_first_step = getattr(training_args, "jump_first_step", False) and (curr_step == 0)
    if not jump_first_step:
        optimizer.step()
        scheduler.step()
        if training_args.use_ema and ema_model is not None:
            if curr_step == training_args.ema_start_steps:
                fsdp_ema_update(ema_model, fsdp_model, decay=0.0)
                log_rank0(f"[EMA] initialized at step {curr_step}")
            elif curr_step > training_args.ema_start_steps:
                fsdp_ema_update(ema_model, fsdp_model, decay=training_args.ema)
    else:
        log_rank0(f"Jump step #{curr_step} without updating parameters.")

    return total_norm


def log_training_metrics(
    loss_dict: dict,
    total_mse_tokens,
    total_ce_tokens,
    total_norm,
    data: dict,
    optimizer: torch.optim.Optimizer,
    progress_bar,
    training_args: TrainingArguments,
    curr_step: int,
    start_time: float,
    device: int,
    world_size: int,
    global_rank: int,
):
    if curr_step % training_args.log_every != 0:
        return start_time

    total_samples = torch.tensor(len(data["sample_lens"]), device=device)
    dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

    torch.cuda.synchronize()
    end_time = time()
    steps_per_sec = training_args.log_every / (end_time - start_time)
    message = f"(step={curr_step:07d}) "
    wandb_log = {}
    for key, value in loss_dict.items():
        avg_loss = torch.tensor(value.item(), device=device)
        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss.item() / world_size
        message += f"Train Loss {key}: {avg_loss:.4f}, "
        wandb_log[key] = avg_loss
    message += f"Train Steps/Sec: {steps_per_sec:.2f}, "
    logs = {"loss": message, "lr": optimizer.param_groups[0]["lr"]}
    progress_bar.set_postfix(**logs, refresh=False)
    progress_bar.update(training_args.log_every)

    wandb_log["lr"] = optimizer.param_groups[0]["lr"]
    wandb_log["total_mse_tokens"] = total_mse_tokens.item()
    wandb_log["total_ce_tokens"] = total_ce_tokens.item()
    wandb_log["total_norm"] = total_norm.item()
    wandb_log["total_samples"] = total_samples.item()

    mem_allocated = torch.tensor(torch.cuda.max_memory_allocated() / 1024**2, device=device)
    dist.all_reduce(mem_allocated, op=dist.ReduceOp.MAX)
    wandb_log["mem_allocated"] = mem_allocated
    mem_cache = torch.tensor(torch.cuda.max_memory_reserved() / 1024**2, device=device)
    dist.all_reduce(mem_cache, op=dist.ReduceOp.MAX)
    wandb_log["mem_cache"] = mem_cache

    if global_rank == 0:
        wandb.log(wandb_log, step=curr_step)
    return time()


def setup_ema_and_load_checkpoint(
    model: Lance,
    training_args: TrainingArguments,
    fsdp_config: FSDPConfig,
    load_ckpt,
    resume_from=None,
    resume_model_only: bool = True,
    logger=None,
):
    ema_model = None
    if training_args.use_ema:
        ema_model = deepcopy(model)
        load_ckpt(ema_model, ema=True)

    load_ckpt(model, ema=False)

    if resume_from is not None and not resume_model_only:
        model, ema_model = FSDPCheckpoint.try_load_ckpt(
            resume_from=resume_from,
            logger=logger,
            model=model,
            ema_model=ema_model,
            resume_from_ema=training_args.finetune_from_ema,
            report_dir=training_args.config_dir,
        )

    if ema_model is not None:
        ema_model = fsdp_ema_setup(ema_model, fsdp_config)

    return ema_model


def prepare_checkpoint_loader(
    model_args: ModelArguments,
    training_args: TrainingArguments,
    llm_config: Qwen2Config,
    language_model: Qwen2ForCausalLM,
    tokenizer: Qwen2Tokenizer,
    num_new_tokens: int,
    log_rank0,
    report_dir: Optional[str] = None,
):
    if training_args.copy_init_moe:
        language_model.init_moe()
        log_rank0("Copy init moe params: copy llm weight to gen.")

    should_untie_lm_head = model_args.tie_word_embeddings
    if should_untie_lm_head:
        model_args.tie_word_embeddings = False
        llm_config.tie_word_embeddings = False

    def load_ckpt(model: Lance, ema: bool = False):
        if training_args.load_from_lance_checkpoint:
            load_from_lance_checkpoint(model, model_args, log_rank0, ema=ema, report_dir=report_dir)
        elif training_args.init_from_vlm_checkpoint:
            init_from_vlm_checkpoint(
                model,
                model_args,
                log_rank0,
                report_dir=report_dir,
                report_name=f"vlm_checkpoint_load_report_{'ema' if ema else 'model'}.txt",
            )

        if num_new_tokens > 0:
            model.language_model.resize_token_embeddings(len(tokenizer))
            model.config.llm_config.vocab_size = len(tokenizer)
            model.language_model.config.vocab_size = len(tokenizer)
            log_rank0(f"Note: {num_new_tokens} new tokens are added!")
        else:
            log_rank0("Note: NO new tokens!")

        if model_args.vit_type.lower() == "qwen2_5_vl":
            from common.model.hacks import hack_qwen2_5_vl_config
            hack_qwen2_5_vl_config(model.language_model)

        model.update_tokenizer(tokenizer=tokenizer)

        if should_untie_lm_head:
            model.language_model.untie_lm_head()
            model.language_model.copy_new_token_rows_to_lm_head(num_new_tokens)
            log_rank0("Note: copy embed tokens to lm_head and untie")
        else:
            assert (
                model.language_model.get_input_embeddings().weight.data.data_ptr()
                != model.language_model.get_output_embeddings().weight.data.data_ptr()
            ), "tie_world_embeddings 冲突"

        freeze_model_components(model, training_args, log_rank0)

    return load_ckpt


def get_image_token_id(language_model: Qwen2ForCausalLM):
    return language_model.config.video_token_id


# =============================================================================
# Checkpoint and resume utilities
# =============================================================================

def get_latest_ckpt(checkpoint_dir):
    step_dirs = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
    if len(step_dirs) == 0:
        return None
    step_dirs = sorted(step_dirs, key=lambda x: int(x))
    latest_step_dir = os.path.join(checkpoint_dir, step_dirs[-1])
    return latest_step_dir


def prepare_resume_and_finetune_settings(training_args: TrainingArguments):
    if training_args.auto_resume:
        resume_from = get_latest_ckpt(training_args.ckpt_dir)
        if resume_from is None:
            resume_from = training_args.resume_from
            resume_model_only = training_args.resume_model_only
        else:
            resume_model_only = False
    else:
        resume_from = training_args.resume_from
        resume_model_only = training_args.resume_model_only

    return resume_from, resume_model_only


# =============================================================================
# Distributed validation logging utilities
# =============================================================================

def _gather_for_rank0(local_list):
    """Gather arbitrary Python objects from all ranks to rank 0 and flatten."""
    world_size = dist.get_world_size()
    rank = get_global_rank()
    if rank == 0:
        gathered = [None] * world_size
    else:
        gathered = None
    dist.gather_object(local_list, gathered if rank == 0 else None, dst=0)
    if rank == 0:
        return [item for sub in gathered for item in sub]
    return None


def _log_media_across_ranks(local_media_data, tag, step, fps, logger=None):
    """
    local_media_data: List[Dict] with keys {"data", "caption", "is_image"}
    Gathers from all ranks and logs to wandb on rank 0.
    """
    flat = _gather_for_rank0(local_media_data)
    if flat is None:
        return
    all_media = []
    for i, item in enumerate(flat):  # TODO: 支持序列级的视频显示
        if item.get("validation_log_type", "") == "table":
            if i == 0:
                all_media = wandb.Table(columns=["id", "condition_image", "condition_text", "target_text", "target_image", "condition_video", "target_video"])
                tag = f"{tag}/step_{step:06d}"

            if item.get("data_vit", None) is not None:
                data_vit = item["data_vit"]
                data_vit_video = None
                if data_vit.ndim == 3:  # image
                    data_vit = wandb.Image(data_vit)
                else:
                    data_vit_video = wandb.Video(data_vit, fps=fps, format="mp4")
                    data_vit = None
                    if i==0:
                        data_vit = wandb.Image(np.zeros((1,224, 224, 3), dtype=np.uint8))

            else:
                data_vit, data_vit_video = None, None

            if item.get("data", None) is not None:
                data = item["data"]
                data_video = None
                if data.ndim == 3:  # image
                    data = wandb.Image(data)
                else:
                    data_video = wandb.Video(data, fps=fps, format="mp4")
                    data = None
                    if i==0:
                        data = wandb.Image(np.zeros((1,224, 224, 3), dtype=np.uint8))

            else:
                data, data_video = None, None

            all_media.add_data(
                i,
                data_vit,  # wandb.Image(item["data_vit"]) if item.get("data_vit", None) is not None else None,
                item.get("caption", ""),
                item.get("cap_target", ""),
                data,  # wandb.Image(item["data"]) if item.get("data", None) is not None else None,
                data_vit_video,
                data_video,
            )
        else:
            if item.get("is_image", False):
                data = item["data"] if item.get("data", None) is not None else item.get("data_vit", None)
                caption = f"cap_condition: {item.get('caption', '')}---cap_target: {item.get('cap_target', '')}"
                all_media.append(wandb.Image(data, caption=caption))
            else:
                data = item["data"] if item.get("data", None) is not None else item["data_vit"]
                caption = f"cap_condition: {item.get('caption', '')}---cap_target: {item.get('cap_target', '')}"
                all_media.append(wandb.Video(data, fps=fps, format="mp4", caption=caption))
    if all_media:
        wandb.log({tag: all_media}, step=step)
        if logger is not None:
            logger.info(f"Logged {len(flat)} items to {tag}.")


# =============================================================================
# Model checkpoint loading utilities
# =============================================================================

def init_from_vlm_checkpoint(model: Qwen2ForCausalLM, model_args: ModelArguments, log_rank0, report_dir: Optional[str] = None, report_name: str = "vlm_checkpoint_load_report.txt"):
    # NOTE: 初始化加载VLM模型走这里
    def load_safetensors_state_dict(folder_path):
        # 只选取safetensors文件，按文件名排序保证顺序
        safetensor_files = sorted(
            f for f in os.listdir(folder_path) if f.endswith(".safetensors")
        )
        state_dict = {}
        for filename in safetensor_files:
            file_path = os.path.join(folder_path, filename)
            state_dict.update(load_file(file_path))
        return state_dict

    state_dict = load_safetensors_state_dict(model_args.llm_path)

    # 参数名的更改以适配当前 Lance 模型的参数名
    for k in list(state_dict.keys()):
        if "visual" in k:  # ViT and connector
            state_dict[k.replace("visual", "vit_model")] = state_dict.pop(k)
        else:
            # 添加language_model前缀
            state_dict["language_model." + k] = state_dict.pop(k)

    result = model.load_state_dict(state_dict, strict=False)
    missing = result.missing_keys
    unexpected = result.unexpected_keys
    # 匹配的参数数量
    matched = len(state_dict) - len(unexpected)
    # 未匹配的参数数量
    not_matched = len(missing) + len(unexpected)

    save_checkpoint_load_report(
        report_dir=report_dir,
        report_name=report_name,
        title="Init from pretrained VLM checkpoint",
        matched=matched,
        not_matched=not_matched,
        missing=missing,
        unexpected=unexpected,
        log_rank0=log_rank0,
    )
    log_rank0(f"Loading form Pretrained VLM {model_args.llm_path} is finished.")

    del state_dict
    import gc; gc.collect(); torch.cuda.empty_cache()


def load_from_lance_checkpoint(model: Qwen2ForCausalLM, model_args: ModelArguments, log_rank0, ema=False, report_dir: Optional[str] = None):
    # NOTE: 微调模型走这里 优先级更高; 加载模型优先加载 ema.safetensors 其次才是 model.safetensors
    path_dir = model_args.model_path
    ema_path = os.path.join(path_dir, "ema.safetensors")
    model_path = os.path.join(path_dir, "model.safetensors")

    model_path_ft = None
    if ema and os.path.exists(ema_path):
        model_path_ft = ema_path
        log_rank0("Found preferred EMA checkpoint for fine-tuning.")
    elif os.path.exists(model_path):
        model_path_ft = model_path
        log_rank0("EMA checkpoint not found. Using fallback: 'model.safetensors'.")

    if model_path_ft:
        log_rank0(f"Loading fine-tune model from: {model_path_ft}")
        model_state_dict = load_file(model_path_ft, device="cpu")
    else:
        raise FileNotFoundError(
            f"Fine-tuning failed: No valid checkpoint ('ema.safetensors' or 'model.safetensors') found in {path_dir}"
        )

    # NOTE: position embeds are fixed sinusoidal embeddings, so we can just pop it off,
    # which makes it easier to adapt to different resolutions.
    if 'latent_pos_embed.pos_embed' in model_state_dict:
        model_state_dict.pop('latent_pos_embed.pos_embed')
        log_rank0(f"Pop `latent_pos_embed.pos_embed` from hf model")
    # model_state_dict.pop('vit_pos_embed.pos_embed')  # TODO: check vit_pos_embed.pos_embed 是否在里面

    msg = model.load_state_dict(model_state_dict, strict=False)  # strict = True | False
    missing = msg.missing_keys
    unexpected = msg.unexpected_keys
    matched = len(model_state_dict) - len(unexpected)  # 匹配的参数数量
    not_matched = len(missing) + len(unexpected)  # 未匹配的参数数量
    report_name = f"lance_checkpoint_load_report_{'ema' if ema else 'model'}.txt"
    save_checkpoint_load_report(
        report_dir=report_dir,
        report_name=report_name,
        title=f"Init from Lance checkpoint ({'ema' if ema else 'model'}): {model_args.model_path}",
        matched=matched,
        not_matched=not_matched,
        missing=missing,
        unexpected=unexpected,
        log_rank0=log_rank0,
    )

    del model_state_dict
    import gc; gc.collect(); torch.cuda.empty_cache()

    return msg


# =============================================================================
# Training config saving
# =============================================================================

def save_training_config(model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments, logger):
    import fsspec
    """Save training, model, and data arguments to a JSON file on rank 0."""
    if get_global_rank() == 0:
        logger.info(f"Training arguments {training_args}")
        logger.info(f"Model arguments {model_args}")
        logger.info(f"Data arguments {data_args}")

        all_args = {
            "model_args": model_args.__dict__,
            "data_args": data_args.__dict__,
            "training_args": training_args.__dict__,
        }
        mkdir(training_args.config_dir)
        config_path = os.path.join(training_args.config_dir, "training_config.json")
        try:
            with fsspec.open(config_path, "w") as f:
                json.dump(all_args, f, indent=4, default=str)
            logger.info(f"Saved training configuration to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save training configuration: {e}")


# =============================================================================
# Validation data and evaluation utilities
# =============================================================================

def _get_data_mode(val_data: dict) -> str:
    # 优先从 val_data 读取；若无，则根据键猜测（尽量不动原数据）
    if "data_mode" in val_data:
        return val_data["data_mode"]
    return "offline" if "padded_latent" in val_data else "online"


def get_fixed_validation_data(
    data_args: DataArguments,
    model_args: ModelArguments,
    training_args: TrainingArguments,
    tokenizer: Qwen2Tokenizer,
    new_token_ids,
    image_token_id: int,
    GLOBAL_RANK: int,
    WORLD_SIZE: int,
    DEVICE: int,
    log_rank0,
):
    """构造并返回一个固定的 validation 批次（与现有内联实现等价）。"""
    assert data_args.val_dataset_config_file is not None and os.path.exists(data_args.val_dataset_config_file)
    # 1) 加载独立的 val dataset 配置，并覆盖 dropout
    val_dataset_config = DataConfig.from_yaml(data_args.val_dataset_config_file)
    val_dataset_config.text_cond_dropout_prob = model_args.val_text_cond_dropout_prob
    val_dataset_config.vae_cond_dropout_prob = model_args.val_vae_cond_dropout_prob
    val_dataset_config.vit_cond_dropout_prob = model_args.val_vit_cond_dropout_prob
    val_dataset_config.latent_patch_size = model_args.latent_patch_size
    log_rank0(
        f"val_dataset_config.text_cond_dropout_prob: {val_dataset_config.text_cond_dropout_prob}, "
        f"val_dataset_config.vae_cond_dropout_prob: {val_dataset_config.vae_cond_dropout_prob}, "
        f"val_dataset_config.vit_cond_dropout_prob: {val_dataset_config.vit_cond_dropout_prob}"
    )

    val_loader = None
    val_dataset = None
    val_data_args = deepcopy(data_args)
    val_data_args.num_workers = min(val_data_args.num_workers, 1)
    try:
        log_rank0("Fetching a fixed batch for validation...")
        # 2) Dataset（参数与原实现保持一致）
        val_dataset = PackedDataset(
            val_dataset_config,
            tokenizer=tokenizer,
            special_tokens=new_token_ids,
            local_rank=GLOBAL_RANK,  # global rank, not local rank
            world_size=WORLD_SIZE,
            interpolate_pos=model_args.interpolate_pos,
            use_flex=training_args.use_flex,
            data_status=None,
            apply_chat_template=training_args.apply_chat_template,
            image_token_id=image_token_id,
            **asdict(val_data_args),
        )
        # 固定顺序/seed
        val_dataset.set_epoch(training_args.validation_data_seed)

        # 3) DataLoader（参数与原实现保持一致）
        val_num_workers = 0
        ctx = torch.multiprocessing.get_context("spawn") if val_num_workers > 0 else None
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            num_workers=val_num_workers,
            pin_memory=True,
            collate_fn=simple_custom_collate,     # 顶层函数
            drop_last=True,
            prefetch_factor=1 if val_num_workers > 0 else None,
            persistent_workers=True if val_num_workers > 0 else False,
            multiprocessing_context=ctx,
        )

        # 4) 取一个固定 batch 并转为 dict
        val_data_cpu = next(iter(val_loader))
        # val_data_cpu = val_data_cpu.cuda(DEVICE).to_dict()
        log_rank0("Fixed validation batch fetched, val_loader and val_dataset deleted.")
        return val_data_cpu
    finally:
        if val_loader is not None:
            del val_loader
        if val_dataset is not None:
            del val_dataset
        import gc; gc.collect()
        log_rank0("Temporary validation resources have been released.")


def validate_on_fixed_batch(
    fsdp_model: Lance,
    vae_model: Optional[WanVideoVAE],
    tokenizer: Qwen2Tokenizer,
    val_data_cpu: dict,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    data_args: DataArguments,
    curr_step: int,
    logger,
    new_token_ids,
    image_token_id: int,
    device: int,
):
    """
    抽取的验证逻辑。等价于原来 for-loop 里的整块 validation。
    """
    log_rank0 = (lambda msg: logger.info(msg)) if get_global_rank() == 0 else (lambda *_: None)
    val_data = val_data_cpu.cuda(device).to_dict()

    fsdp_model.eval()
    try:
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            log_rank0(f"Running validation on fixed batch at step {curr_step}...")

            # 解析文本（保持与原逻辑一致）
            # 如果 val dataset 在构造时没有套模板/pos_emb，这里 decode_text_* 会按开关再处理一遍
            val_texts_gen, val_texts_und = decode_text_interleave(
                tokenizer,
                val_data,
            )
            val_attn_modes = val_data["attn_modes"]
            sample_splits = map_splits_to_samples(val_data["sample_lens"], val_data["split_lens"])

            val_sample_N_target = val_data["sample_N_target"]
            val_sample_type = val_data["sample_type"]

            # -------------------- GEN 分支 --------------------
            if training_args.validation_type.lower() in ("gen", "und_gen", "gen_und") and len(val_texts_gen) > 0:
                data_mode = _get_data_mode(val_data)  # fix: 不再只在 step==0 时定义

                # 计算 video_sizes（不改变原推理配置）
                if data_mode == "offline":
                    if curr_step == 0:
                        # 第0步要打 GT，所以一次性 decode
                        val_padded_videos = vae_model.vae_decode(list(val_data["padded_latent"]))
                        first_shape = val_padded_videos[0].shape[1:]  # T,H,W
                    else:
                        # 后续仅取一个样本获取尺寸，避免每次都全量解码，减少额外开销
                        first_shape = vae_model.vae_decode([val_data["padded_latent"][0]])[0].shape[1:]
                    video_sizes = [first_shape for _ in (val_data["padded_latent"])]
                else:
                    video_sizes = [v.shape[1:] for v in val_data["padded_videos"]]
                    if curr_step == 0:
                        val_padded_videos = val_data["padded_videos"]

                # 第 0 步记录 GT
                if curr_step == 0:
                    if model_args.val_text_cond_dropout_prob > 0:
                        val_texts_gen = ["NULL"] * len(video_sizes)
                    local_gt_media_data = []
                    curr_sample, curr_video_tensor_index, curr_video_tensor_index_vit = 0, 0, 0
                    for i_gt, N_target in enumerate(val_sample_N_target[:-1]):  # 去除最后pad样本
                        left, right = sample_splits[i_gt][0], sample_splits[i_gt][-1] + 1
                        N_target_VIT = val_attn_modes[left:right].count("full")  # 仅在ti2i 中不为0

                        if val_sample_type[i_gt] != "gen":
                            curr_video_tensor_index_vit += N_target_VIT
                            continue
                        curr_sample += 1
                        if curr_sample > training_args.validation_max_samples:
                            break

                        video_tensor = val_padded_videos[curr_video_tensor_index : curr_video_tensor_index + N_target]  # [N], 每一项为[C T H W]
                        curr_video_tensor_index += N_target
                        v_thwc = decode_video_tensor(video_tensor)
                        is_image = v_thwc.shape[0] == 1

                        # ===== 对GEN 分支 VIT 条件特征的处理 =====
                        if N_target_VIT > 0:
                            video_tensor_vit = val_data["padded_videos_vit"][curr_video_tensor_index_vit : curr_video_tensor_index_vit + N_target_VIT]
                            curr_video_tensor_index_vit += N_target_VIT
                            v_thwc_vit = decode_video_tensor(video_tensor_vit, video_type="vit")
                            media_data_vit = v_thwc_vit[0] if is_image else np.ascontiguousarray(v_thwc_vit.transpose(0, 3, 1, 2))
                        else:
                            media_data_vit = None

                        media_data = v_thwc[0] if is_image else np.ascontiguousarray(v_thwc.transpose(0, 3, 1, 2))
                        cap = val_texts_gen[curr_sample-1] if (curr_sample-1) < len(val_texts_gen) else "GT_text_wrong"
                        caption = cap
                        local_gt_media_data.append(
                            {"data": media_data, "caption": caption, "is_image": is_image, "data_vit": media_data_vit, "validation_log_type": training_args.validation_log_type}
                        )
                    _log_media_across_ranks(local_gt_media_data, "validation_gt_samples_gen", curr_step, training_args.validation_video_saving_fps, logger)


                # 计算 padded_latent
                val_data["padded_latent"] = make_padded_latent(val_data['padded_videos'], val_data['vae_data_mode'], vae_model)

                # 采样生成（保持原参数）
                with fsdp_model.summon_full_params(fsdp_model, writeback=False, rank0_only=False):
                    denoise_latent = fsdp_model.validation_gen(
                        val_packed_text_ids=val_data["packed_text_ids"],
                        val_packed_text_indexes=val_data["packed_text_indexes"],
                        val_sample_lens=val_data["sample_lens"],
                        val_packed_position_ids=val_data["packed_position_ids"],
                        val_split_lens=val_data["split_lens"],
                        val_attn_modes=val_data["attn_modes"],
                        val_sample_N_target=val_data["sample_N_target"],                        val_packed_vae_token_indexes=val_data["packed_vae_token_indexes"],
                        timestep_shift=training_args.validation_timestep_shift,
                        num_timesteps=training_args.validation_num_timesteps,
                        val_mse_loss_indexes=val_data.get("mse_loss_indexes", None),
                        val_padded_latent=val_data["padded_latent"],
                        video_sizes=video_sizes,
                        cfg_text_scale=model_args.cfg_text_scale,
                        cfg_interval=training_args.cfg_interval,
                        cfg_renorm_min=training_args.cfg_renorm_min,
                        cfg_renorm_type=training_args.cfg_renorm_type,
                        device=device,
                        dtype=torch.bfloat16,
                        new_token_ids=new_token_ids,
                        max_samples=training_args.validation_max_samples,
                        validation_noise_seed=training_args.validation_noise_seed,
                        apply_chat_template=training_args.apply_chat_template,
                        apply_qwen_2_5_vl_pos_emb=training_args.apply_qwen_2_5_vl_pos_emb,
                        image_token_id=image_token_id,
                        val_packed_vit_token_indexes=val_data.get("packed_vit_token_indexes", None),
                        val_packed_vit_tokens=val_data.get("packed_vit_tokens", None),
                        vit_video_grid_thw=val_data.get("vit_video_grid_thw", None),
                        vae_video_grid_thw=val_data["vae_video_grid_thw"],
                        video_grid_thw=val_data.get("video_grid_thw", None),
                        sample_task=val_data["sample_task"],
                        sample_modality=val_data["sample_modality"],
                        cfg_type=training_args.cfg_type,
                        cfg_uncond_token_id=training_args.cfg_uncond_token_id,
                    )
                # 解码 + 日志
                local_media_data = []
                for i_val, latent in enumerate(denoise_latent):
                    v_list = []
                    for latent_ in latent:
                        v_list.append(vae_model.vae_decode([latent_])[0])
                    v_thwc = decode_video_tensor(v_list)

                    is_image = v_thwc.shape[0] == 1
                    media_data = v_thwc[0] if is_image else np.ascontiguousarray(v_thwc.transpose(0, 3, 1, 2))
                    cap = val_texts_gen[i_val] if i_val < len(val_texts_gen) else "GT_text_wrong"
                    caption = cap
                    local_media_data.append({"data": media_data, "caption": caption, "is_image": is_image, "validation_log_type": training_args.validation_log_type})

                if dist.is_available() and dist.is_initialized():
                    dist.barrier()

                _log_media_across_ranks(local_media_data, "validation_pre_samples_gen", curr_step, training_args.validation_video_saving_fps, logger)
                log_rank0(f"Validation(gen) at step {curr_step} finished.")
                log_rank0(f"cfg type is: {training_args.cfg_type}.")


            # -------------------- UND 分支 --------------------
            visual_first = 0
            if training_args.validation_type.lower() in ("und", "und_gen", "gen_und") and len(val_texts_und) > 0:
                # 准备可视化用的输入视频（每步都构造，避免依赖 step==0 的局部变量）  # fix
                vis_list = []
                curr_sample, curr_video_tensor_index = 0, 0
                for i_gt, N_target in enumerate(val_sample_N_target[:-1]):  # 去除最后pad样本
                    left, right = sample_splits[i_gt][0], sample_splits[i_gt][-1] + 1
                    N_target_VIT = val_attn_modes[left:right].count("full")
                    if val_sample_type[i_gt] != "und":
                        curr_video_tensor_index += N_target_VIT
                        continue
                    curr_sample += 1
                    if curr_sample > training_args.validation_max_samples:
                        break
                    if N_target_VIT != 0:
                        video_tensor = val_data["padded_videos_vit"][curr_video_tensor_index : curr_video_tensor_index + N_target_VIT]  # [N], 每一项为[C 2 H W] or [C T H W]
                        curr_video_tensor_index += N_target_VIT
                        v_thwc = decode_video_tensor(video_tensor, video_type="vit")

                        is_image = v_thwc.shape[0] == 2  # 与原注释保持一致：2帧则视作图像
                        media_data = v_thwc[0] if is_image else np.ascontiguousarray(v_thwc.transpose(0, 3, 1, 2))
                    else:
                        media_data, is_image = None, True

                    if media_data is None and vis_list == [] and "full" in val_attn_modes:  # 避免第一行图像为None 导致的显示退化
                        visual_first += 1
                        continue
                    vis_list.append((media_data, is_image))

                # 第 0 步记录 GT
                if curr_step == 0:
                    local_gt_media_data = []
                    for i_gt, (media_data, is_image) in enumerate(vis_list):
                        cap = val_texts_und[i_gt] if i_gt < len(val_texts_und) else "GT_text_wrong"

                        bos_token_id = tokenizer.decode(new_token_ids["bos_token_id"])
                        cap = cap.split(bos_token_id)
                        cap_target = cap[-1]  # 取出作为target
                        cap = (bos_token_id).join(cap[:-1])

                        caption = cap
                        local_gt_media_data.append(
                            {"data_vit": media_data, "caption": caption, "is_image": is_image, "cap_target": cap_target, "validation_log_type": training_args.validation_log_type}
                        )
                    _log_media_across_ranks(local_gt_media_data, "validation_gt_samples_und", curr_step, training_args.validation_video_saving_fps, logger)

                vocab_size = len(tokenizer)
                with fsdp_model.summon_full_params(fsdp_model, writeback=False, rank0_only=False):
                    generated_sequence_all = fsdp_model.validation_video_to_text(
                        val_packed_text_ids=val_data["packed_text_ids"],
                        val_packed_text_indexes=val_data["packed_text_indexes"],
                        val_packed_position_ids=val_data["packed_position_ids"],
                        val_sample_N_target=val_data["sample_N_target"],
                        val_split_lens=val_data["split_lens"],
                        val_attn_modes=val_data["attn_modes"],
                        val_sample_lens=val_data["sample_lens"],
                        val_sample_type=val_data["sample_type"],
                        val_packed_vit_tokens=val_data["packed_vit_tokens"],
                        val_vit_video_grid_thw=val_data["vit_video_grid_thw"],
                        val_ce_loss_indexes=val_data["ce_loss_indexes"],
                        max_samples=training_args.validation_max_samples,
                        max_length=256,
                        device=device,
                        dtype=torch.bfloat16,
                        new_token_ids=new_token_ids,
                        pad_token_id=tokenizer.pad_token_id,
                        vocab_size=vocab_size,
                        tokenizer=tokenizer,
                        apply_chat_template=training_args.apply_chat_template,
                        apply_qwen_2_5_vl_pos_emb=training_args.apply_qwen_2_5_vl_pos_emb,
                        do_sample=False,
                        image_token_id=image_token_id,
                    )
                print(f"generated_sequence_all end")
                local_media_data = []
                for i_val, generated_sequence in enumerate(generated_sequence_all):
                    if i_val < visual_first:
                        continue
                    try:
                        cap = tokenizer.decode(generated_sequence[:, 0])
                    except Exception:
                        log_rank0(f"Rank {get_global_rank()} failed to decode sequence {i_val}: {generated_sequence}")
                        continue
                    caption = cap
                    media_data, is_image = vis_list[i_val - visual_first]
                    local_media_data.append(
                        {
                            "data_vit": media_data,
                            "caption": val_texts_und[i_val],
                            "cap_target": caption,
                            "is_image": is_image,
                            "validation_log_type": training_args.validation_log_type,
                        }
                    )
                    print('local_media_data:  ',local_media_data)

                if dist.is_available() and dist.is_initialized():
                    dist.barrier()

                _log_media_across_ranks(local_media_data, "validation_pre_samples_und", curr_step, training_args.validation_video_saving_fps, logger)
                log_rank0(f"Validation(und) at step {curr_step} finished.")
    finally:
        # del val_data
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        # dist.barrier()
        fsdp_model.train()
