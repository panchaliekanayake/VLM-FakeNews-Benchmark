"""MiniGPT-4 Vicuna-7B adapter using an official repository checkout."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from .base import ModelAdapter


def _values(value: object) -> list:
    if isinstance(value, dict):
        return list(value.values())
    try:
        return list(value.values())  # type: ignore[union-attr]
    except (AttributeError, TypeError):
        return []


class MiniGPT4Adapter(ModelAdapter):
    def __init__(self, model_path: str, repo_path: str | None = None, gpu_id: int = 0, **_: object) -> None:
        if not repo_path:
            raise ValueError("MiniGPT-4 requires --repo-path pointing to an official MiniGPT-4 checkout.")
        import torch

        resolved = str(Path(repo_path).resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
        for module in ("minigpt4", "minigpt4.datasets.builders", "minigpt4.models", "minigpt4.processors"):
            importlib.import_module(module)
        from minigpt4.common.config import Config
        from minigpt4.common.registry import registry
        from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0

        self.torch = torch
        cfg = Config(argparse.Namespace(cfg_path=model_path, options=None))
        model_cfg = cfg.model_cfg
        model_cls = registry.get_model_class(model_cfg.arch)
        device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        self.model = model_cls.from_config(model_cfg).to(device).eval()

        processor_cfg = None
        for dataset_cfg in _values(cfg.datasets_cfg):
            visual = getattr(dataset_cfg, "vis_processor", None)
            processor_cfg = getattr(visual, "train", None) or getattr(visual, "eval", None)
            if processor_cfg is not None:
                break
        if processor_cfg is None:
            raise ValueError("MiniGPT-4 config does not define a visual processor.")
        processor = registry.get_processor_class(processor_cfg.name).from_config(processor_cfg)
        self.chat = Chat(self.model, processor, device=device)
        self.conversation_template = CONV_VISION_Vicuna0

    def generate(self, image_path: Path, prompt: str) -> str:
        from PIL import Image

        with Image.open(image_path) as source:
            image = source.convert("RGB")
        state = self.conversation_template.copy()
        images: list = []
        self.chat.upload_img(image, state, images)
        self.chat.encode_img(images)
        self.chat.ask(prompt, state)
        with self.torch.inference_mode():
            answer = self.chat.answer(
                conv=state, img_list=images, num_beams=1, temperature=1e-5,
                max_new_tokens=64, max_length=2000,
            )
        if isinstance(answer, (list, tuple)):
            answer = answer[0]
        return str(answer).strip()
