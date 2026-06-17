# coding: utf-8

import os
import torch
import torch.distributed as dist

def get_global_rank() -> int:
    """
    Get the global rank, the global index of the GPU.
    """
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    """
    Get the local rank, the local index of the GPU.
    """
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_world_size() -> int:
    """
    Get the world size, the total amount of GPUs.
    """
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_master():
    """
    Check if the current process is the master process (rank 0).
    """
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_device() -> torch.device:
    """
    Get current rank device.
    """
    return torch.device("cuda", get_local_rank())


def barrier_if_distributed(*args, **kwargs):
    """
    Synchronizes all processes if under distributed context.
    """
    if dist.is_initialized():
        return dist.barrier(*args, **kwargs)
