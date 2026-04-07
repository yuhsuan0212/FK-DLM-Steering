"""
Distributed training / inference utilities.

Provides helpers for PyTorch distributed process-group management,
cross-GPU gathering, and reduction operations.
"""

import os
from typing import TypeVar

import torch
import torch.distributed as dist

T = TypeVar("T")


def get_distributed_info() -> dict[str, int | bool]:
    """Return rank, world_size, local_rank, and is_distributed flag from env vars."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return {
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "is_distributed": world_size > 1,
    }


def setup_distributed() -> dict[str, int | bool]:
    """Init the distributed process group and return the info dict."""
    info = get_distributed_info()
    if info["is_distributed"] and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    return info


def cleanup_distributed(info: dict[str, int | bool]) -> None:
    """Destroy the process group if distributed."""
    if info["is_distributed"] and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(info: dict[str, int | bool]) -> bool:
    """Return True if the current process is rank 0."""
    return int(info["rank"]) == 0


def gather_records(
    records: list[tuple[int, str]], info: dict[str, int | bool]
) -> list[tuple[int, str]]:
    """Gather indexed records from all ranks to rank 0."""
    if not info["is_distributed"]:
        return records

    world_size = int(info["world_size"])
    if is_main_process(info):
        gathered_records: list[list[tuple[int, str]] | None] = [
            None for _ in range(world_size)
        ]
        dist.gather_object(records, gathered_records, dst=0)
        merged: list[tuple[int, str]] = []
        for local_records in gathered_records:
            if local_records:
                merged.extend(local_records)
        return merged

    dist.gather_object(records, dst=0)
    return []


def reduce_sum_int(
    value: int, info: dict[str, int | bool], device: torch.device
) -> int:
    """Sum an integer across all ranks, returning the result on rank 0."""
    if not info["is_distributed"]:
        return int(value)
    tensor = torch.tensor(value, device=device, dtype=torch.long)
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
    if not is_main_process(info):
        return 0
    return int(tensor.item())


def barrier(info: dict[str, int | bool]) -> None:
    """Synchronise all processes. No-op when not distributed."""
    if info["is_distributed"] and dist.is_initialized():
        dist.barrier()


def broadcast_object(obj: object, info: dict[str, int | bool], src: int = 0) -> object:
    """Broadcast a Python object from *src* to all ranks."""
    if not info["is_distributed"]:
        return obj
    container = [obj]
    dist.broadcast_object_list(container, src=src)
    return container[0]


def all_reduce_sum(tensor: torch.Tensor, info: dict[str, int | bool]) -> torch.Tensor:
    """In-place all-reduce (sum) a tensor across all ranks and return it.

    All ranks receive the summed result.  No-op when not distributed.
    """
    if info["is_distributed"] and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def gather_tensors_to_main(
    tensors: dict[str, torch.Tensor],
    info: dict[str, int | bool],
) -> "dict[str, torch.Tensor] | None":
    """Gather a dict of CPU tensors from all ranks to rank 0.

    Returns the concatenated dict on rank 0; ``None`` on all other ranks.
    All tensors must already reside on CPU before calling this function.
    """
    if not info["is_distributed"]:
        return tensors

    world_size = int(info["world_size"])
    if is_main_process(info):
        gathered: list[dict[str, torch.Tensor] | None] = [None] * world_size
        dist.gather_object(tensors, gathered, dst=0)
        return {
            k: torch.cat([g[k] for g in gathered if g is not None], dim=0)
            for k in tensors
        }

    dist.gather_object(tensors, dst=0)
    return None


def shard_list(items: list[T], info: dict[str, int | bool]) -> list[tuple[int, T]]:
    """Deterministically shard *items* across ranks by global index.

    Returns a list of ``(global_index, item)`` tuples assigned to
    the current rank.
    """
    world_size = int(info["world_size"])
    rank = int(info["rank"])
    return [(idx, item) for idx, item in enumerate(items) if idx % world_size == rank]


def get_device(info: dict[str, int | bool], device_mode: str = "auto") -> torch.device:
    """Return the appropriate :class:`torch.device` for the current rank.

    Args:
        info: Distributed info dict from :func:`get_distributed_info`.
        device_mode: One of ``"auto"``, ``"cuda"``, ``"cpu"``.
            ``"auto"`` selects CUDA when available, otherwise CPU.
    """
    device_mode = device_mode.lower()
    if info["is_distributed"]:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Distributed mode is enabled but CUDA is unavailable. "
                "Please launch with GPU resources for multi-card inference."
            )
        local_rank = int(info["local_rank"])
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    if device_mode == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_mode)
