"""
Shared PyTorch helper utilities.

Small, pure-utility functions used across multiple sub-packages.
"""

from __future__ import annotations

import random

import torch


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    """Map a string dtype name to the corresponding :class:`torch.dtype`.

    Supported values: ``"bfloat16"``, ``"float16"``, ``"float32"``.
    """
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_name not in dtype_map:
        raise ValueError(
            f"Unsupported torch_dtype '{dtype_name}'. "
            f"Supported values: {list(dtype_map.keys())}"
        )
    return dtype_map[dtype_name]


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility (stdlib, torch, CUDA)."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
