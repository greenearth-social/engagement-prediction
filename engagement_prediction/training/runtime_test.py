from __future__ import annotations

import random

import numpy as np
import torch

from engagement_prediction.training import runtime


def test_get_device_preserves_explicit_device(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert runtime.get_device("cpu") == "cpu"


def test_get_device_detects_available_runtime(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert runtime.get_device(None) == "cuda"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert runtime.get_device(None) == "cpu"


def test_set_random_seeds_reproduces_cpu_randomness(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    runtime.set_random_seeds(17)
    first = (random.random(), np.random.random(), torch.rand(1))
    runtime.set_random_seeds(17)
    second = (random.random(), np.random.random(), torch.rand(1))

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


def test_clear_cuda_memory_skips_uninitialized_cuda(monkeypatch) -> None:
    empty_cache_calls: list[None] = []
    synchronize_calls: list[None] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: empty_cache_calls.append(None))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: synchronize_calls.append(None))

    runtime.clear_cuda_memory()

    assert empty_cache_calls == []
    assert synchronize_calls == []


def test_clear_cuda_memory_releases_initialized_cuda(monkeypatch) -> None:
    empty_cache_calls: list[None] = []
    synchronize_calls: list[None] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: empty_cache_calls.append(None))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: synchronize_calls.append(None))

    runtime.clear_cuda_memory()

    assert empty_cache_calls == [None]
    assert synchronize_calls == [None]
