"""Dataset schema loading and validation."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .labels import ground_truth_label


@dataclass(frozen=True)
class Sample:
    index: int
    text: str
    image_path: Path
    ground_truth: str
    fake_class: str | None


def load_samples(annotation_file: Path, data_root: Path) -> list[Sample]:
    """Load MMFakeBench-compatible JSON without mutating source records."""
    with annotation_file.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError("Annotation file must contain a JSON list.")

    samples: list[Sample] = []
    normalized_root = Path(os.path.abspath(data_root))
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} must be a JSON object.")
        missing = {"text", "image_path", "gt_answers"} - record.keys()
        if missing:
            raise ValueError(f"Record {index} is missing: {', '.join(sorted(missing))}")
        ground_truth_label(record["gt_answers"])
        relative_image = str(record["image_path"]).lstrip("/\\")
        image_path = Path(os.path.abspath(normalized_root / relative_image))
        if not image_path.is_relative_to(normalized_root):
            raise ValueError(f"Record {index} image_path escapes data_root: {record['image_path']!r}")
        samples.append(
            Sample(
                index=index,
                text=str(record["text"]),
                image_path=image_path,
                ground_truth=str(record["gt_answers"]),
                fake_class=(str(record["fake_cls"]) if record.get("fake_cls") is not None else None),
            )
        )
    return samples


def missing_images(samples: list[Sample]) -> Iterator[Path]:
    return (sample.image_path for sample in samples if not sample.image_path.is_file())
