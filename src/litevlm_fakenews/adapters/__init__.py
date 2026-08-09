"""Lazy registry for the six models evaluated in the paper."""

from __future__ import annotations

from .base import ModelAdapter

MODEL_DEFAULTS = {
    "llava-v1.6-vicuna-7b": "liuhaotian/llava-v1.6-vicuna-7b",
    "qwen-vl-qwen-7b": "Qwen/Qwen-VL",
    "otter-image-mpt-7b": "luodian/OTTER-Image-MPT7B",
    "minigpt4-vicuna-7b": None,
    "blip2-flan-t5-xl": "Salesforce/blip2-flan-t5-xl",
    "instructblip-vicuna-7b": "lavis-managed",
}

MODEL_REVISIONS = {
    "llava-v1.6-vicuna-7b": "deae57a8c0ccb0da4c2661cc1891cc9d06503d11",
    "qwen-vl-qwen-7b": "0547ed36a86561e2e42fecec8fd0c4f6953e33c4",
    "otter-image-mpt-7b": "af423fbd0fb44263e25227e3df13771aa3f300bc",
    "minigpt4-vicuna-7b": None,
    "blip2-flan-t5-xl": "0eb0d3b46c14c1f8c7680bca2693baafdb90bb28",
    "instructblip-vicuna-7b": None,
}


def create_adapter(
    name: str,
    model_path: str | None,
    repo_path: str | None,
    gpu_id: int,
    revision: str | None = None,
) -> ModelAdapter:
    resolved_path = model_path or MODEL_DEFAULTS[name]
    resolved_revision = revision or MODEL_REVISIONS[name]
    if resolved_path is None:
        raise ValueError(f"{name} requires --model-path; this release does not provide a machine-specific config.")
    if name == "llava-v1.6-vicuna-7b":
        from .llava import LlavaAdapter
        return LlavaAdapter(resolved_path, gpu_id=gpu_id, revision=resolved_revision)
    if name == "qwen-vl-qwen-7b":
        from .qwenvl import QwenVLAdapter
        return QwenVLAdapter(resolved_path, gpu_id=gpu_id, revision=resolved_revision)
    if name == "otter-image-mpt-7b":
        from .otter import OtterAdapter
        return OtterAdapter(resolved_path, gpu_id=gpu_id, revision=resolved_revision)
    if name == "minigpt4-vicuna-7b":
        from .minigpt4 import MiniGPT4Adapter
        return MiniGPT4Adapter(resolved_path, repo_path=repo_path, gpu_id=gpu_id)
    if name == "blip2-flan-t5-xl":
        from .blip2 import Blip2Adapter
        return Blip2Adapter(resolved_path, gpu_id=gpu_id, revision=resolved_revision)
    if name == "instructblip-vicuna-7b":
        from .instructblip import InstructBlipAdapter
        return InstructBlipAdapter(repo_path=repo_path, gpu_id=gpu_id)
    raise ValueError(f"Unsupported model: {name}")


__all__ = ["MODEL_DEFAULTS", "MODEL_REVISIONS", "ModelAdapter", "create_adapter"]
