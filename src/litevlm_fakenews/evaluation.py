"""Dependency-free binary classification evaluation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from .labels import FAKE, TRUE, ground_truth_label, parse_response


def _safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_records(records: Sequence[dict]) -> dict:
    """Evaluate predictions using fixed-label macro averaging.

    Macro precision, recall and F1 average the True and Fake class scores. This
    matches the original experiment implementation and avoids dropping a class
    when a model never predicts it.
    """
    if not records:
        raise ValueError("At least one prediction is required.")

    matrix = [[0, 0], [0, 0]]
    invalid = 0
    for index, record in enumerate(records):
        if "answer" not in record or "gt_answers" not in record:
            raise ValueError(f"Prediction {index} requires answer and gt_answers fields.")
        truth = ground_truth_label(record["gt_answers"])
        parsed = parse_response(record["answer"])
        matrix[truth][parsed.binary_label] += 1
        invalid += int(not parsed.valid)

    class_metrics: dict[str, dict[str, float | int]] = {}
    for label, name in ((TRUE, "true"), (FAKE, "fake")):
        tp = matrix[label][label]
        fp = sum(matrix[row][label] for row in (TRUE, FAKE)) - tp
        fn = sum(matrix[label]) - tp
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        class_metrics[name] = {"precision": precision, "recall": recall, "f1": f1, "support": sum(matrix[label])}

    accuracy = _safe_divide(matrix[0][0] + matrix[1][1], len(records))
    macro = {
        metric: sum(float(class_metrics[name][metric]) for name in ("true", "fake")) / 2
        for metric in ("precision", "recall", "f1")
    }
    return {
        "samples": len(records),
        "accuracy": accuracy,
        "macro": macro,
        "classes": class_metrics,
        "confusion_matrix": {"labels": ["true", "fake"], "values": matrix},
        "invalid_outputs": invalid,
        "invalid_output_rate": invalid / len(records),
    }


def evaluate_file(prediction_file: Path) -> dict:
    with prediction_file.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError("Prediction file must contain a JSON list.")
    return evaluate_records(records)


def write_json(payload: object, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
