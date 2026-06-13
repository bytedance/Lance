# coding: utf-8

import os
import os.path as osp
import uuid
from typing import Any, Optional

import torch
from safetensors.torch import save_file as save_safetensors, save_model as save_safetensors_model

from .basic import get_global_rank
from .fs import copy, is_hdfs_path, mkdir
from .logging import get_logger

logger = get_logger(__name__)

_local_dir = None


def get_local_dir():
    """
    Get a local directory for temporary storage for this process.
    """
    global _local_dir
    if _local_dir is None:
        _local_dir = os.path.join("local_save", "rank_" + str(get_global_rank()) + "_" + str(uuid.uuid4()))
        mkdir(_local_dir)
    return _local_dir


def set_local_dir(dirname):
    """
    Set a local directory for temporary storage for this process.
    """
    global _local_dir
    if dirname is None:
        return
    _local_dir = os.path.join(dirname, str(uuid.uuid4()))
    mkdir(_local_dir)


def get_local_path(path: str) -> str:
    """
    Get a local path for storing the file.
    If the path is already a local path, directly return.
    """
    if is_hdfs_path(path):
        path = os.path.join(get_local_dir(), os.path.basename(path))
    else:
        mkdir(os.path.dirname(path))
    return path


def convert_dtype(states: Any, dtype: Optional[torch.dtype] = None):
    """
    Recursively convert the state_dict to device and dtype.
    """
    if dtype is None:
        return states
    if torch.is_tensor(states):
        return states.to("cpu", dtype)
    if isinstance(states, dict):
        return {k: convert_dtype(v, dtype) for k, v in states.items()}
    if isinstance(states, list):
        return [convert_dtype(v, dtype) for v in states]
    return states


def save(data: Any, path: str, blocking: bool = True, local_dir: Optional[str] = None):
    """
    Safely save data to a local or HDFS path.
    """
    if not is_hdfs_path(path):
        if path.endswith(".safetensors"):
            if isinstance(data, torch.nn.Module):
                save_safetensors_model(data, path)
            else:
                save_safetensors(data, path)
        else:
            torch.save(data, path)

        logger.info(f"Early saved to local path: {path}")
        return

    if local_dir is None:
        local_dir = get_local_dir()

    local_path = osp.join(local_dir, osp.basename(path))
    if path.endswith(".safetensors"):
        if isinstance(data, torch.nn.Module):
            save_safetensors_model(data, local_path)
        else:
            save_safetensors(data, local_path)
    else:
        torch.save(data, local_path)
    logger.info(f"Saved to local path: {local_path}")

    copy(local_path, path, blocking=blocking)
    logger.info(f"Copy {local_path} to HDFS or Local path: {path} done.")


def dummy_indexes_searchsorted(packed_text_indexes: torch.LongTensor, ce_loss_indexes: torch.LongTensor) -> torch.LongTensor:
    """
    Find dummy indexes via searchsorted over sorted packed text indexes.
    """
    sorted_vals, sorted_pos = torch.sort(packed_text_indexes)
    loc = torch.searchsorted(sorted_vals, ce_loss_indexes)
    return sorted_pos[loc]
