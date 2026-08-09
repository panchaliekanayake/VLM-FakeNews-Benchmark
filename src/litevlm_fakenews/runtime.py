"""Reproducibility and inference-time helpers."""

from __future__ import annotations

import random
import time
from collections.abc import Iterator
from contextlib import contextmanager


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def synchronize_accelerator() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


@contextmanager
def elapsed_seconds() -> Iterator[list[float]]:
    """Measure a block after accelerator synchronization."""
    result: list[float] = []
    synchronize_accelerator()
    started = time.perf_counter()
    try:
        yield result
    finally:
        synchronize_accelerator()
        result.append(time.perf_counter() - started)
