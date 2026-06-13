# coding: utf-8

import functools
import os
import os.path as osp
import warnings
import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
    ShardedStateDictConfig,
    ShardedOptimStateDictConfig
)
from torch.distributed.checkpoint import save as dcp_save, load as dcp_load
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from safetensors.torch import load_file

from modeling.lance.modeling_utils import MLPconnector, TimestepEmbedder
from modeling.lance.modeling_utils import PositionEmbedding3D

from modeling.lance.qwen2_navit import (
    # Qwen2ForCausalLM,
    Qwen2DecoderLayer,
    Qwen2MoEDecoderLayer,
    Qwen2MoTDecoderLayer,
)
from common.utils.fs import mkdir, is_hdfs_path, copy, exists
from common.utils.save import get_local_dir, save

# 在文件顶部添加以下代码来忽略指定的FutureWarning
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="torch.distributed.fsdp.fully_sharded_data_parallel"
)


# -------------------------- helpers --------------------------
def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _world() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def __barrier__():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _copy_model_metadata_files(source_model_path, save_path, blocking=True):
    if not source_model_path or _rank() != 0:
        return

    metadata_filenames = [
        "generation_config.json",
        "llm_config.json",
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ]
    for filename in metadata_filenames:
        src = osp.join(source_model_path, filename)
        if exists(src):
            copy(src, osp.join(save_path, filename), blocking=blocking)


def _save_load_report(report_dir, report_name, title, msg, logger):
    missing = list(getattr(msg, "missing_keys", []))
    unexpected = list(getattr(msg, "unexpected_keys", []))
    logger.info(f"{title}: missing={len(missing)}, unexpected={len(unexpected)}")
    if report_dir is None or _rank() != 0:
        return

    mkdir(report_dir)
    report_path = osp.join(report_dir, report_name)
    lines = [
        title,
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
    logger.info(f"Saved checkpoint load report to {report_path}")


class FSDPConfig:
    def __init__(
        self,
        sharding_strategy,
        backward_prefetch,
        cpu_offload,
        num_replicate,
        num_shard=8,
        use_orig_params=False,
    ):
        self.sharding_strategy = sharding_strategy
        self.backward_prefetch = backward_prefetch
        self.cpu_offload = cpu_offload
        self.num_replicate = num_replicate
        self.num_shard = num_shard
        self.use_orig_params = use_orig_params


def fsdp_wrapper(original_model, fsdp_config: FSDPConfig, ignored_modules=[], mixed_precision_override=None):
    if fsdp_config.sharding_strategy == 'HYBRID_SHARD':
        device_mesh = init_device_mesh(
            "cuda",
            mesh_shape=(fsdp_config.num_replicate, fsdp_config.num_shard),
            mesh_dim_names=("replicate", "shard")
        )
    else:
        device_mesh = None

    mp = mixed_precision_override or MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        # reduce_dtype=torch.float32, # TODO: 改成torch.float32 会把bfloat16->float32
        buffer_dtype=torch.bfloat16,
    )

    return FSDP(
        original_model,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                # Qwen2ForCausalLM,
                Qwen2DecoderLayer,
                Qwen2MoEDecoderLayer,
                Qwen2MoTDecoderLayer,
                MLPconnector,
                TimestepEmbedder,
                # PositionEmbedding,  # NOTE: NOT USED
                PositionEmbedding3D,
            },
        ),
        ignored_modules=ignored_modules,
        mixed_precision=mp,
        device_id=dist.get_rank() % torch.cuda.device_count(),
        sharding_strategy=ShardingStrategy[fsdp_config.sharding_strategy],
        backward_prefetch=BackwardPrefetch[fsdp_config.backward_prefetch],
        cpu_offload=CPUOffload(offload_params=fsdp_config.cpu_offload),
        device_mesh=device_mesh,
        use_orig_params=fsdp_config.use_orig_params,
    )


class FSDPCheckpoint:
    @staticmethod
    def fsdp_save_fsdp_ckpt(ckpt_dir, train_steps, model, ema_model, optimizer, scheduler, data_status, logger, fsdp_config, blocking=True, **kwargs):
        save_path = osp.join(ckpt_dir, f"{train_steps:07d}")
        mkdir(save_path)
        logger.info(f"Begin saving checkpoint info to {save_path}")
        source_model_path = kwargs.get("source_model_path")

        local_save_dir = get_local_dir() if is_hdfs_path(save_path) else None
        rank = _rank()
        world = _world()

        if fsdp_config.sharding_strategy == "HYBRID_SHARD":
            assert world == fsdp_config.num_shard * fsdp_config.num_replicate, f"world={world} != shard({fsdp_config.num_shard})*replicate({fsdp_config.num_replicate})"

        # ---- 0) 若是 HDFS 目标，把 DCP 的输出先写到本地临时目录，再整体拷贝 ----
        dcp_root = save_path
        if is_hdfs_path(save_path):
            dcp_root = osp.join(get_local_dir(), osp.basename(save_path))
            os.makedirs(dcp_root, exist_ok=True)

        # ---- 1) save sharded via DCP ----
        if kwargs.get("flag_save_shard_model", False):  # 是否保存shard model
            # ---- 1.1) EMA (sharded via DCP) ----
            if ema_model is not None:  # NOTE: ema_model 暂时为 None
                with FSDP.state_dict_type(
                    ema_model,
                    StateDictType.SHARDED_STATE_DICT,
                    ShardedStateDictConfig(offload_to_cpu=True),
                ):
                    ema_model_state = ema_model.state_dict()
                    dcp_save(ema_model_state, checkpoint_id=osp.join(dcp_root, "ema"))
                    del ema_model_state
                    import gc; gc.collect(); torch.cuda.empty_cache()

                __barrier__()

            # ---- 1.2) Model (sharded via DCP) ----
            with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, ShardedStateDictConfig(offload_to_cpu=True)):
                model_state = model.state_dict()
                dcp_save(model_state, checkpoint_id=os.path.join(dcp_root, "model"))
                del model_state
                import gc; gc.collect(); torch.cuda.empty_cache()

            __barrier__()

        # ---- 2) Model FULL ----
        if kwargs.get("flag_save_full_model", True): # 是否保存full model
            # ---- 2.1) EMA Model FULL ----
            if ema_model is not None:
                with FSDP.state_dict_type(ema_model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(rank0_only=True, offload_to_cpu=True)):
                    sd = ema_model.state_dict()

                    __barrier__()  # 👈 关键：在 FULL 上下文内先同步一次

                    if rank == 0:
                        # （可选）contiguous 处理
                        for k, v in list(sd.items()):
                            if isinstance(v, torch.Tensor) and not v.is_contiguous():
                                sd[k] = v.contiguous()
                        save(sd, osp.join(save_path, "ema.safetensors"), blocking=blocking, local_dir=local_save_dir)

                    __barrier__()

                del sd
                import gc; gc.collect(); torch.cuda.empty_cache()

            # ---- 2.2) Model FULL ----
            # NOTE: 已确认保存的模型finetune loss正常，2025.08.21，✅
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(rank0_only=True, offload_to_cpu=True)):
                sd = model.state_dict()  # 注意：即使 rank0-only，所有 rank 也必须调 state_dict() 参与集体通信

                __barrier__()  # 👈 关键：在 FULL 上下文内先同步一次

                if rank == 0:
                    # （可选）contiguous 处理
                    for k, v in list(sd.items()):
                        if isinstance(v, torch.Tensor) and not v.is_contiguous():
                            sd[k] = v.contiguous()
                    save(sd, osp.join(save_path, "model.safetensors"), blocking=blocking, local_dir=local_save_dir)

                __barrier__() # 关键补丁：在 FULL 上下文内同步一次，避免非 rank0 过早退出 context

            del sd
            import gc; gc.collect(); torch.cuda.empty_cache()

        _copy_model_metadata_files(source_model_path, save_path, blocking=blocking)
        __barrier__()

        # --- 3) If target is HDFS, copy sharded dirs up ---
        if is_hdfs_path(save_path):  # NOTE: 因为用dcp存的，所有需要额外拷贝；若未保存切片，则不会进行拷贝
            for sub in ("model", "ema"):
                src = osp.join(dcp_root, sub)
                if os.path.exists(src):
                    copy(src, save_path, blocking=blocking)  # fix: directory copy
                    logger.info(f"Copy {src} to HDFS or Local path: {save_path} done.")

        # ---- 3) Optimizer (sharded via DCP) ----
        if kwargs.get("flag_save_optimizer", True): # 是否保存 optimizer，默认保存以支持完整 resume
            # Determine shard file name (keeps your original convention)
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                shard_index = rank
                total_shards = world
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                shard_index = rank % fsdp_config.num_shard
                total_shards = fsdp_config.num_shard
            else:
                raise NotImplementedError

            opt_path = osp.join(save_path, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt")

            # Export the *sharded* optimizer state dict (no rank0 aggregation; low memory)
            with FSDP.state_dict_type(
                model,
                StateDictType.SHARDED_STATE_DICT,
                optim_state_dict_config=ShardedOptimStateDictConfig(offload_to_cpu=True),  # torch 2.5.1: no `group` arg
            ):
                osd = FSDP.optim_state_dict(model, optimizer)

            # Save the shard (HYBRID: only first `num_shard` ranks write, matching your pattern)
            try:
                if fsdp_config.sharding_strategy == "FULL_SHARD":
                    save(osd, opt_path, blocking=blocking, local_dir=local_save_dir)
                elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                    if rank < fsdp_config.num_shard:
                        save(osd, opt_path, blocking=blocking, local_dir=local_save_dir)
            finally:
                del osd
                import gc; gc.collect(); torch.cuda.empty_cache()

        # ---- Scheduler (rank0) ----
        if rank == 0 and scheduler is not None:
            # torch.save(scheduler.state_dict(), osp.join(save_path, "scheduler.pt"))
            save(scheduler.state_dict(), osp.join(save_path, "scheduler.pt"), blocking=blocking, local_dir=local_save_dir)

        # ---- Data status (per-rank) ----
        if rank == 0 and data_status is not None:
            save(data_status, osp.join(save_path, "data_status.pt"), blocking=blocking, local_dir=local_save_dir)

            del data_status
            import gc; gc.collect(); torch.cuda.empty_cache()

        __barrier__()

        return

    @staticmethod
    def try_load_ckpt(resume_from, logger, model, ema_model=None, resume_from_ema=False, report_dir=None):
        # TODO: 待check！
        if resume_from is not None and osp.exists(resume_from):
            logger.info(f"Loading checkpoint from {resume_from}.")
            if resume_from_ema:
                model_state_dict_path = osp.join(resume_from, f"ema.safetensors")
            else:
                model_state_dict_path = osp.join(resume_from, f"model.safetensors")
            model_state_dict = load_file(model_state_dict_path, device="cpu")
            # NOTE position embeds are fixed sinusoidal embeddings, so we can just pop it off,
            # which makes it easier to adapt to different resolutions.
            for key in ["latent_pos_embed.pos_embed", "vit_pos_embed.pos_embed"]:
                if key in model_state_dict:
                    model_state_dict.pop(key)
            msg = model.load_state_dict(model_state_dict, strict=False)
            _save_load_report(report_dir, "resume_checkpoint_load_report_model.txt", f"Resume model checkpoint: {model_state_dict_path}", msg, logger)
            del model_state_dict

            if ema_model is not None:
                ema_state_dict_path = osp.join(resume_from, f"ema.safetensors")
                if not osp.exists(ema_state_dict_path):
                    logger.info(f"replicaing ema model from {model_state_dict_path}.")
                    ema_state_dict_path = model_state_dict_path
                ema_state_dict = load_file(ema_state_dict_path, device="cpu")
                # NOTE position embeds are fixed sinusoidal embeddings, so we can just pop it off,
                # which makes it easier to adapt to different resolutions.
                for key in ["latent_pos_embed.pos_embed", "vit_pos_embed.pos_embed"]:
                    if key in ema_state_dict:
                        ema_state_dict.pop(key)
                msg = ema_model.load_state_dict(ema_state_dict, strict=False)
                _save_load_report(report_dir, "resume_checkpoint_load_report_ema.txt", f"Resume EMA checkpoint: {ema_state_dict_path}", msg, logger)
                del ema_state_dict
        else:
            logger.info(f"Training from scratch.")
        return model, ema_model

    @staticmethod
    def try_load_fsdp_ckpt(resume_from, logger, model, ema_model=None, resume_from_ema=False, report_dir=None):
        # TODO: 待check！
        if resume_from is None or not os.path.exists(resume_from):
            logger.info("Training from scratch.")
            return model, ema_model

        logger.info(f"Loading checkpoint from {resume_from}.")

        # ---- Model (or EMA) via DCP ----
        load_dir = osp.join(resume_from, "ema" if resume_from_ema else "model")
        assert isinstance(model, FSDP)
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, ShardedStateDictConfig(offload_to_cpu=True)):
            model_state = model.state_dict()
            dcp_load(model_state, checkpoint_id=load_dir)
            for k in ["latent_pos_embed.pos_embed", "vit_pos_embed.pos_embed"]:
                if k in model_state:
                    model_state.pop(k)
            msg = model.load_state_dict(model_state, strict=False)
            _save_load_report(report_dir, "resume_fsdp_checkpoint_load_report_model.txt", f"Resume FSDP model checkpoint: {load_dir}", msg, logger)
            del model_state
            gc.collect()
            torch.cuda.empty_cache()

        # ---- EMA (optional) ----
        if ema_model is not None:
            ema_dir = osp.join(resume_from, "ema")
            assert isinstance(ema_model, FSDP)
            with FSDP.state_dict_type(ema_model, StateDictType.SHARDED_STATE_DICT, ShardedStateDictConfig(offload_to_cpu=True)):
                ema_state = ema_model.state_dict()
                dcp_load(ema_state, checkpoint_id=ema_dir)
                for k in ["latent_pos_embed.pos_embed", "vit_pos_embed.pos_embed"]:
                    if k in ema_state:
                        ema_state.pop(k)
                msg = ema_model.load_state_dict(ema_state, strict=False)
                _save_load_report(report_dir, "resume_fsdp_checkpoint_load_report_ema.txt", f"Resume FSDP EMA checkpoint: {ema_dir}", msg, logger)
                del ema_state
                import gc; gc.collect(); torch.cuda.empty_cache()

        return model, ema_model

    @staticmethod
    def try_load_train_state(resume_from, optimizer, scheduler, fsdp_config):
        if resume_from is not None and osp.exists(resume_from):
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                shard_index = dist.get_rank()
                total_shards = dist.get_world_size()
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                shard_index = dist.get_rank() % fsdp_config.num_shard
                total_shards = fsdp_config.num_shard
            else:
                raise NotImplementedError

            optimizer_state_dict_path = osp.join(resume_from, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt")
            optimizer_state_dict = torch.load(optimizer_state_dict_path, map_location="cpu", weights_only=True)
            optimizer.load_state_dict(optimizer_state_dict)
            del optimizer_state_dict

            scheduler_state_dict_path = osp.join(resume_from, "scheduler.pt")
            scheduler_state_dict = torch.load(scheduler_state_dict_path, weights_only=True, map_location="cpu")
            scheduler.load_state_dict(scheduler_state_dict)
            del scheduler_state_dict

            train_steps = int(osp.basename(osp.normpath(resume_from))) + 1
            """
            data_status = [
                {
                    dataset_name: {
                        worker_id: [parquet_idx, row_group_id, row_idx],
                    },
                },
            ]
            """
            data_status_path = osp.join(resume_from, "data_status.pt")
            if osp.exists(data_status_path):
                data_status = torch.load(data_status_path, weights_only=True, map_location="cpu")
                local_rank = dist.get_rank()
                if local_rank < len(data_status):
                    data_status = data_status[local_rank]
                else:
                    data_status = None
            else:
                data_status = None
        else:
            train_steps = 0
            data_status = None
        return optimizer, scheduler, train_steps, data_status


def grad_checkpoint_check_fn(module):
    module_options = (
        Qwen2DecoderLayer,
        Qwen2MoEDecoderLayer,
        Qwen2MoTDecoderLayer,
        MLPconnector,
    )
    return isinstance(module, module_options)


def fsdp_ema_setup(ema_model, fsdp_config, ignored_modules=[]):
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad = False

    ema_model = fsdp_wrapper(ema_model, fsdp_config, ignored_modules=ignored_modules)
    return ema_model


@torch.no_grad()
def fsdp_ema_update(ema_model, model, decay=0.9999):
    ema_handles = traversal_utils._get_fsdp_handles(ema_model)
    new_handles = traversal_utils._get_fsdp_handles(model)
    assert len(ema_handles) == len(new_handles)
    ema_params = []
    new_params = []

    for ema_handle, new_handle in zip(ema_handles, new_handles):
        if ema_handle.flat_param is not None and new_handle.flat_param.requires_grad:
            ema_params.append(ema_handle.flat_param.data)
            new_params.append(new_handle.flat_param.data.to(dtype=ema_handle.flat_param.dtype))

    torch._foreach_mul_(ema_params, decay)
    torch._foreach_add_(ema_params, new_params, alpha=1 - decay)


# =============================================== 在CPU做EMA的相关实现，待验证（未调试成功） =====================================================
def fsdp_ema_setup_v2(ema_model, fsdp_config, ignored_modules=[], backend="fsdp_cpu_offload"):
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad = False

    if backend == "none_cpu_plain":
        return ema_model.cpu()

    # 默认：FSDP 包裹，但强制 offload 到 CPU（显存≈0）
    ema_cfg = FSDPConfig(
        sharding_strategy=fsdp_config.sharding_strategy,
        backward_prefetch=fsdp_config.backward_prefetch,
        cpu_offload=True,  # 关键
        num_replicate=fsdp_config.num_replicate,
        num_shard=fsdp_config.num_shard,
        use_orig_params=fsdp_config.use_orig_params,
    )

    mp_ema = MixedPrecision(param_dtype=torch.float32, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
    ema_model = fsdp_wrapper(ema_model, ema_cfg, ignored_modules=ignored_modules, mixed_precision_override=mp_ema)
    return ema_model


@torch.no_grad()
def fsdp_ema_update_v2(ema_model, model, decay=0.9999):
    ema_handles = traversal_utils._get_fsdp_handles(ema_model)
    new_handles = traversal_utils._get_fsdp_handles(model)
    assert len(ema_handles) == len(new_handles)
    ema_params = []
    new_params = []

    for ema_handle, new_handle in zip(ema_handles, new_handles):
        if ema_handle.flat_param is not None and new_handle.flat_param.requires_grad:
            # EMA 常驻 CPU：确保 new_params 也在 CPU，再做 foreach
            ema_params.append(ema_handle.flat_param.data)  # CPU
            new_params.append(new_handle.flat_param.data.to(device="cpu", dtype=ema_handle.flat_param.dtype, non_blocking=True))

    torch._foreach_mul_(ema_params, decay)
    torch._foreach_add_(ema_params, new_params, alpha=1 - decay)
