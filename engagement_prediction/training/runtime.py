"""Runtime helpers shared by canonical model-training stages."""

from __future__ import annotations


def get_device(arg_device: str | None) -> str:
    """Return an explicit device or choose CUDA when it is available."""
    if arg_device is not None:
        return arg_device

    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def clear_cuda_memory() -> None:
    """Run garbage collection and release initialized CUDA cache memory."""
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def set_random_seeds(seed: int) -> None:
    """Seed Python, NumPy, PyTorch, and any initialized CUDA devices."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
