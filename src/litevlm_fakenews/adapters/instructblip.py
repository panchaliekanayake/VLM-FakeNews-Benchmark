"""InstructBLIP Vicuna-7B adapter using the official LAVIS repository."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .base import ModelAdapter


class InstructBlipAdapter(ModelAdapter):
    def __init__(self, repo_path: str | None = None, gpu_id: int = 0, **_: object) -> None:
        if not repo_path:
            raise ValueError("InstructBLIP requires --repo-path pointing to an official LAVIS checkout.")
        import torch

        resolved = str(Path(repo_path).resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
        from lavis.models import load_model_and_preprocess

        self.torch = torch
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        previous = Path.cwd()
        try:
            os.chdir(resolved)
            self.model, processors, _ = load_model_and_preprocess(
                name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=self.device
            )
        finally:
            os.chdir(previous)
        self.processor = processors["eval"]

    def generate(self, image_path: Path, prompt: str) -> str:
        from PIL import Image

        with Image.open(image_path) as image:
            tensor = self.processor(image.convert("RGB")).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            answer = self.model.generate(
                {"image": tensor, "prompt": prompt}, use_nucleus_sampling=False,
                num_beams=1, max_length=256, min_length=1, top_p=0.9,
                repetition_penalty=1.5, length_penalty=1.0, num_captions=1, temperature=1.0,
            )
        return str(answer[0] if isinstance(answer, (list, tuple)) else answer).strip()
