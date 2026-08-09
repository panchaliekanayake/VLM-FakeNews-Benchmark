"""Model-adapter interface and shared helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class ModelAdapter(ABC):
    """A loaded VLM that produces one response for an image-text prompt."""

    @abstractmethod
    def generate(self, image_path: Path, prompt: str) -> str:
        raise NotImplementedError


def require_cuda(torch: Any, model_name: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(f"{model_name} inference requires a CUDA-capable GPU.")
