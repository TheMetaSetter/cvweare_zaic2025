"""Helpers for coordinating DistributedDataParallel runs across the pipeline."""

from __future__ import annotations

import os
from typing import Tuple

import torch
import torch.distributed as dist


def setup_distributed_environment() -> Tuple[int, int, int, bool]:
    """Initialize process group so train_yolo/run_pipeline integrate cleanly with torchrun."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return 0, 0, 1, False

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    torch.cuda.set_device(local_rank)

    return dist.get_rank(), local_rank, world_size, True


def cleanup_distributed_environment() -> None:
    """Tear down the process group once the YOLO stage reaches global synchronization."""
    if dist.is_initialized():
        dist.destroy_process_group()

