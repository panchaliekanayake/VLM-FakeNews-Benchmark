"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import __version__
from .adapters import MODEL_DEFAULTS
from .dataset import load_samples, missing_images
from .evaluation import evaluate_file, write_json
from .inference import run_inference


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="litevlm-fnd", description="Lightweight VLM fake-news benchmark")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate-data", help="Validate an annotation file and its image paths")
    validate.add_argument("--annotations", type=Path, required=True)
    validate.add_argument("--data-root", type=Path, required=True)

    evaluate = commands.add_parser("evaluate", help="Score an existing prediction JSON file")
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--output", type=Path)

    run = commands.add_parser("run", help="Run one paper model over one dataset split")
    run.add_argument("--model", choices=sorted(MODEL_DEFAULTS), required=True)
    run.add_argument("--model-path", help="Override the model checkpoint or MiniGPT-4 config")
    run.add_argument("--revision", help="Override the pinned Hugging Face checkpoint revision")
    run.add_argument("--repo-path", help="Official LAVIS or MiniGPT-4 checkout, when required")
    run.add_argument("--annotations", type=Path, required=True)
    run.add_argument("--data-root", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--gpu-id", type=int, default=0)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--limit", type=int)
    run.add_argument("--timing-samples", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate-data":
        samples = load_samples(args.annotations, args.data_root)
        absent = list(missing_images(samples))
        print(json.dumps({"samples": len(samples), "missing_images": len(absent)}, indent=2))
        return 1 if absent else 0
    if args.command == "evaluate":
        metrics = evaluate_file(args.predictions)
        if args.output:
            write_json(metrics, args.output)
        print(json.dumps(metrics, indent=2))
        return 0
    paths = run_inference(
        annotation_file=args.annotations, data_root=args.data_root, output_dir=args.output_dir,
        model_name=args.model, model_path=args.model_path, revision=args.revision, repo_path=args.repo_path,
        gpu_id=args.gpu_id, seed=args.seed, limit=args.limit, timing_samples=args.timing_samples,
    )
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
