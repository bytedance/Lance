# coding: utf-8

import warnings
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers.models.transformers.transformer_2d")
warnings.filterwarnings(
    "ignore",
    message=".*torch\\.cuda\\.amp\\.custom_(fwd|bwd).*deprecated.*",
    category=FutureWarning,
    module="deepspeed.runtime.zero.linear",
)

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Standard library imports
import functools
import sys
import traceback
from dataclasses import asdict
from time import time
from typing import Tuple, cast

# Third-party package imports
import wandb
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)

from transformers import HfArgumentParser, set_seed

# Local repository imports
from common.utils.basic import get_global_rank, get_world_size
from common.utils.logging import get_logger
from common.val.utils import make_padded_latent

from data.dataset_base_train import PackedDataset,  simple_custom_collate
from data.data_utils import add_special_tokens

from modeling.lance import Lance, LanceConfig
from modeling.qwen2 import Qwen2Tokenizer

from config.config_factory import ModelArguments, DataArguments, TrainingArguments

from train.fsdp_utils import (
    FSDPCheckpoint,
    grad_checkpoint_check_fn,
    fsdp_wrapper,
)
from train.train_utils import (
    build_fsdp_config,
    build_lr_scheduler,
    build_train_dataset_config,
    compute_training_loss,
    get_fixed_validation_data,
    get_image_token_id,
    log_training_metrics,
    load_training_state,
    optimizer_step_with_ema,
    prepare_checkpoint_loader,
    prepare_model_paths,
    prepare_resume_and_finetune_settings,
    save_trainable_parameters,
    save_training_config,
    setup_output_paths,
    setup_ema_and_load_checkpoint,
    setup_model_components,
    setup_rank0_logging_and_wandb,
    validate_on_fixed_batch,
)



def main():
    # ========================= Env setup ==============================
    assert torch.cuda.is_available()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    dist.init_process_group("nccl")
    GLOBAL_RANK = dist.get_rank()
    LOCAL_RANK = GLOBAL_RANK % torch.cuda.device_count() # equal to get_local_rank() ??
    WORLD_SIZE = get_world_size()
    DEVICE = LOCAL_RANK # equal to global_rank % torch.cuda.device_count()
    torch.cuda.set_device(DEVICE)

    # ========================= Args, logger and wandb setup ==============================
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = cast(Tuple[ModelArguments, DataArguments, TrainingArguments], parser.parse_args_into_dataclasses())

    training_args.N_key_frame = data_args.N_key_frame
    training_args.incre_time_pro = data_args.incre_time_pro

    logger = get_logger()
    log_rank0 = (lambda msg: logger.info(msg)) if GLOBAL_RANK == 0 else (lambda *_: None)  # 只在rank0打印

    setup_output_paths(training_args, logger, GLOBAL_RANK)
    setup_rank0_logging_and_wandb(model_args, data_args, training_args, logger, GLOBAL_RANK)
    save_training_config(model_args, data_args, training_args, logger)

    # ========================= Resume and finetune setup ==============================
    resume_from, resume_model_only = prepare_resume_and_finetune_settings(training_args)

    # Set seed:
    seed = training_args.global_seed * WORLD_SIZE + GLOBAL_RANK
    set_seed(seed)

    prepare_model_paths(model_args, training_args)

    llm_config, language_model, vit_config, vit_model, vae_model, vae_config = setup_model_components(model_args, training_args, log_rank0)

    # Lance的配置
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

    # Setup tokenizer for model:
    if training_args.load_from_lance_checkpoint:
        tokenizer: Qwen2Tokenizer = Qwen2Tokenizer.from_pretrained(model_args.model_path)
    else:
        tokenizer: Qwen2Tokenizer = Qwen2Tokenizer.from_pretrained(model_args.llm_path)

    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)

    load_ckpt = prepare_checkpoint_loader(
        model_args=model_args,
        training_args=training_args,
        llm_config=llm_config,
        language_model=language_model,
        tokenizer=tokenizer,
        num_new_tokens=num_new_tokens,
        log_rank0=log_rank0,
        report_dir=training_args.config_dir,
    )

    fsdp_config = build_fsdp_config(training_args)
    ema_model = setup_ema_and_load_checkpoint(
        model=model,
        training_args=training_args,
        fsdp_config=fsdp_config,
        load_ckpt=load_ckpt,
        resume_from=resume_from,
        resume_model_only=resume_model_only,
        logger=logger,
    )
    image_token_id = get_image_token_id(language_model)


    fsdp_model: Lance = fsdp_wrapper(model, fsdp_config)
    apply_activation_checkpointing(
        fsdp_model,
        checkpoint_wrapper_fn=functools.partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
        check_fn=grad_checkpoint_check_fn,  # grad_checkpoint_check_fn 是自定义的检查函数，告诉 apply_activation_checkpointing 应该对哪些层或模块开启激活检查点
    )

    save_trainable_parameters(model, fsdp_model, training_args, logger, GLOBAL_RANK)

    # ========================= Optimizer and scheduler setup ==============================
    params_to_train = [p for p in fsdp_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params_to_train,  # <<< 关键：必须是过滤后的列表 能否显著减少训练参数？
        lr=training_args.lr,
        betas=(training_args.beta1, training_args.beta2),
        eps=training_args.eps,
        weight_decay=0,
    )
    scheduler = build_lr_scheduler(optimizer, training_args)

    optimizer, scheduler, train_step, data_status = load_training_state(
        optimizer=optimizer,
        scheduler=scheduler,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        resume_from=resume_from,
        resume_model_only=resume_model_only,
        fsdp_config=fsdp_config,
        global_rank=GLOBAL_RANK,
        world_size=WORLD_SIZE,
    )

    # Setup packed dataloader
    dataset_config = build_train_dataset_config(data_args, model_args, training_args, vae_config)

    if training_args.validation_step > 0:
        val_data_cpu = get_fixed_validation_data(
            data_args=data_args,
            model_args=model_args,
            training_args=training_args,
            tokenizer=tokenizer,
            new_token_ids=new_token_ids,
            image_token_id=image_token_id,
            GLOBAL_RANK=GLOBAL_RANK,
            WORLD_SIZE=WORLD_SIZE,
            DEVICE=DEVICE,
            log_rank0=log_rank0,
        )

    train_dataset = PackedDataset(
        dataset_config,
        tokenizer=tokenizer,
        special_tokens=new_token_ids,
        local_rank=GLOBAL_RANK,
        world_size=WORLD_SIZE,
        interpolate_pos=model_args.interpolate_pos,
        use_flex=training_args.use_flex,
        data_status=data_status,
        apply_chat_template=training_args.apply_chat_template,
        image_token_id=image_token_id,
        cfg_type=training_args.cfg_type,
        cfg_uncond_token_id=training_args.cfg_uncond_token_id,
        **asdict(data_args),
    )

    train_dataset.set_epoch(data_args.data_seed)

    ctx = torch.multiprocessing.get_context("spawn") if data_args.num_workers > 0 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # batch size is 1 for packed dataset
        num_workers=data_args.num_workers,
        pin_memory=True,
        collate_fn=simple_custom_collate,
        drop_last=True,
        prefetch_factor=data_args.prefetch_factor if data_args.num_workers > 0 else None,
        persistent_workers=True if data_args.num_workers > 0 else False,       # True的话避免跨 epoch 持有句柄
        multiprocessing_context=ctx,    # 关键：spawn 取代 fork
    )


    fsdp_model.train()
    if training_args.use_ema and ema_model is not None:
        ema_model.eval()

    # ========================= Training loop ==============================
    start_time = time()
    if GLOBAL_RANK == 0:
        logger.info(f"Training for {training_args.total_steps} steps, starting at {train_step}...")
    progress_bar = tqdm(total=training_args.total_steps, initial=train_step, disable=not GLOBAL_RANK == 0, desc="Training")

    for curr_step, data in enumerate(train_loader, start=train_step):
        try:
            data = data.cuda(DEVICE).to_dict()
            data_indexes = data.pop("batch_data_indexes", None)
            ce_loss_weights = data.pop("ce_loss_weights", None)
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                vae_data_mode = data.pop("vae_data_mode")
                padded_videos = data.pop("padded_videos")
                if training_args.visual_gen:
                    with torch.no_grad():
                        data["padded_latent"] = make_padded_latent(padded_videos, vae_data_mode, vae_model)
                if "padded_videos_vit" in data:
                    data.pop("padded_videos_vit")

                loss_dict = fsdp_model(**data)

            loss, loss_dict, total_ce_tokens, total_mse_tokens = compute_training_loss(
                loss_dict=loss_dict,
                data=data,
                ce_loss_weights=ce_loss_weights,
                training_args=training_args,
                device=DEVICE,
                world_size=WORLD_SIZE,
            )

            if not torch.isfinite(loss).all():
                print("Non-finite loss at step", curr_step)
                logger.info(f"Non-finite loss data: {data}")

            is_bad = torch.tensor(0.0, device=DEVICE)
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"bad data at step {curr_step}, rank {GLOBAL_RANK}, loss is nan or inf")
                is_bad = torch.tensor(1.0, device=DEVICE)
            dist.all_reduce(is_bad, op=dist.ReduceOp.SUM)
            if is_bad.item() > 0.5:
                logger.error(f"bad data at step {curr_step}, rank {GLOBAL_RANK}, sum of is_bad {is_bad.item()}, skip this step in all ranks")
                optimizer.zero_grad()
                continue

            total_norm = optimizer_step_with_ema(
                loss=loss,
                fsdp_model=fsdp_model,
                ema_model=ema_model,
                optimizer=optimizer,
                scheduler=scheduler,
                training_args=training_args,
                curr_step=curr_step,
                log_rank0=log_rank0,
            )

            start_time = log_training_metrics(
                loss_dict=loss_dict,
                total_mse_tokens=total_mse_tokens,
                total_ce_tokens=total_ce_tokens,
                total_norm=total_norm,
                data=data,
                optimizer=optimizer,
                progress_bar=progress_bar,
                training_args=training_args,
                curr_step=curr_step,
                start_time=start_time,
                device=DEVICE,
                world_size=WORLD_SIZE,
                global_rank=GLOBAL_RANK,
            )

            if data_status is None:
                data_status = {}
            for item in data_indexes:
                if item["dataset_name"] not in data_status.keys():
                    data_status[item["dataset_name"]] = {}
                data_status[item["dataset_name"]][item["worker_id"]] = item["data_indexes"]

            if  (curr_step == training_args.ckpt_debug_steps) or (curr_step > 0 and curr_step % training_args.save_every == 0):
                if curr_step == training_args.ckpt_debug_steps:
                    log_rank0(f"ckpt_debug_steps = {curr_step}, saving ckps just for debug...")

                import gc; gc.collect(); torch.cuda.empty_cache()
                if GLOBAL_RANK == 0:
                    gather_list = [None] * WORLD_SIZE
                else:
                    gather_list = None
                dist.gather_object(data_status, gather_list, dst=0)

                FSDPCheckpoint.fsdp_save_fsdp_ckpt(
                    ckpt_dir=training_args.ckpt_dir,
                    train_steps=curr_step,
                    model=fsdp_model,
                    ema_model=(
                        ema_model if (training_args.use_ema and curr_step >= training_args.ema_start_steps)
                        else None
                    ),
                    optimizer=optimizer,
                    scheduler=scheduler,
                    logger=logger,
                    fsdp_config=fsdp_config,
                    data_status=gather_list,
                    blocking=(curr_step >= (training_args.total_steps - 1)),
                    source_model_path=model_args.model_path,
                )
            if (training_args.validation_step > 0) and ((curr_step == training_args.ckpt_debug_steps) or (curr_step % training_args.validation_step == 0)):
                validate_on_fixed_batch(
                    fsdp_model=fsdp_model,
                    vae_model=vae_model,
                    tokenizer=tokenizer,
                    val_data_cpu=val_data_cpu,
                    training_args=training_args,
                    model_args=model_args,
                    data_args=data_args,
                    curr_step=curr_step,
                    logger=logger,
                    new_token_ids=new_token_ids,
                    image_token_id=image_token_id,
                    device=DEVICE,
                )

        except Exception:
            logger.error(f"[TRAINING EXCEPTION] Step {curr_step}, Rank {GLOBAL_RANK}, Error:")
            traceback.print_exc()
            # 清空梯度，避免脏数据影响下一轮
            optimizer.zero_grad()
            # 分布式多卡同步，确保所有卡一起跳过
            try:
                dist.barrier()
            except:
                pass
            # 跳过当前批次，继续训练
            continue

    if GLOBAL_RANK == 0:
        logger.info("Done!")
        wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
