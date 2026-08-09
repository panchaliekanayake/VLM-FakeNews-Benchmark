"""Unified, batch-size-one inference pipeline."""

from __future__ import annotations

import hashlib
import platform
import subprocess
from importlib import metadata as importlib_metadata
from pathlib import Path

from .adapters import MODEL_DEFAULTS, MODEL_REVISIONS, create_adapter
from .dataset import load_samples, missing_images
from .evaluation import evaluate_records, write_json
from .prompt import build_prompt
from .runtime import elapsed_seconds, set_seed

GENERATION_CONFIGS = {
    "llava-v1.6-vicuna-7b": {"do_sample": False, "temperature": 0.0, "num_beams": 1, "max_new_tokens": 512},
    "qwen-vl-qwen-7b": {"do_sample": False, "max_new_tokens": 128},
    "otter-image-mpt-7b": {"num_beams": 3, "no_repeat_ngram_size": 3, "max_new_tokens": 64},
    "minigpt4-vicuna-7b": {"temperature": 0.00001, "num_beams": 1, "max_new_tokens": 64, "max_length": 2000},
    "blip2-flan-t5-xl": {"num_beams": 1, "max_new_tokens": 32},
    "instructblip-vicuna-7b": {
        "use_nucleus_sampling": False, "num_beams": 1, "max_length": 256,
        "min_length": 1, "top_p": 0.9, "repetition_penalty": 1.5,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    versions = {}
    for package in ("accelerate", "huggingface-hub", "numpy", "Pillow", "torch", "transformers"):
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            continue
    return versions


def _git_revision(repo_path: str | None) -> str | None:
    if not repo_path:
        return None
    completed = subprocess.run(
        ["git", "-C", repo_path, "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() or None


def _accelerator_metadata(gpu_id: int) -> dict:
    try:
        import torch
    except ImportError:
        return {"available": False}
    result = {
        "available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "cudnn": (torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None),
    }
    if torch.cuda.is_available():
        result.update({"gpu_id": gpu_id, "gpu_name": torch.cuda.get_device_name(gpu_id)})
    return result


def run_inference(
    annotation_file: Path,
    data_root: Path,
    output_dir: Path,
    model_name: str,
    model_path: str | None = None,
    revision: str | None = None,
    repo_path: str | None = None,
    gpu_id: int = 0,
    seed: int = 42,
    limit: int | None = None,
    timing_samples: int = 100,
) -> dict:
    """Generate predictions and save predictions, metrics, and run metadata."""
    if timing_samples < 1:
        raise ValueError("timing_samples must be positive")
    samples = load_samples(annotation_file, data_root)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        samples = samples[:limit]
    absent = list(missing_images(samples))
    if absent:
        preview = "\n".join(str(path) for path in absent[:5])
        raise FileNotFoundError(f"{len(absent)} image(s) are missing. First paths:\n{preview}")

    set_seed(seed)
    resolved_model_path = model_path or MODEL_DEFAULTS[model_name]
    resolved_revision = revision or MODEL_REVISIONS[model_name]
    adapter = create_adapter(model_name, model_path, repo_path, gpu_id, revision=revision)
    predictions: list[dict] = []
    durations: list[float] = []
    for sample in samples:
        question = build_prompt(sample.text)
        with elapsed_seconds() as elapsed:
            answer = adapter.generate(sample.image_path, question)
        if len(durations) < timing_samples:
            durations.append(elapsed[0])
        predictions.append(
            {
                "sample_index": sample.index,
                "question": question,
                "answer": answer,
                "gt_answers": sample.ground_truth,
                "image_path": str(sample.image_path),
                "model_name": model_name,
                "multiple_gt_answers": sample.fake_class,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_file = output_dir / "predictions.json"
    metrics_file = output_dir / "metrics.json"
    metadata_file = output_dir / "run_metadata.json"
    metrics = evaluate_records(predictions)
    metadata = {
        "model_name": model_name,
        "model_path": resolved_model_path,
        "model_revision": resolved_revision,
        "external_repo_path": str(Path(repo_path).resolve()) if repo_path else None,
        "external_repo_revision": _git_revision(repo_path),
        "generation_config": GENERATION_CONFIGS[model_name],
        "annotation_file": str(annotation_file.resolve()),
        "data_root": str(data_root.resolve()),
        "seed": seed,
        "batch_size": 1,
        "samples": len(predictions),
        "timed_samples": len(durations),
        "mean_inference_ms_per_sample": (sum(durations) / len(durations) * 1000 if durations else None),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": _package_versions(),
        "accelerator": _accelerator_metadata(gpu_id),
        "annotation_sha256": _sha256(annotation_file),
    }
    write_json(predictions, prediction_file)
    write_json(metrics, metrics_file)
    write_json(metadata, metadata_file)
    return {"predictions": prediction_file, "metrics": metrics_file, "metadata": metadata_file}
