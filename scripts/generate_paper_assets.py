#!/usr/bin/env python3
"""Regenerate the README trade-off figure from committed paper result CSVs."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "paper_classification_metrics.csv"
LATENCY = ROOT / "results" / "paper_inference_time.csv"
OUTPUT = ROOT / "assets" / "inference_tradeoff.svg"


def main() -> None:
    scores: dict[str, list[float]] = defaultdict(list)
    with METRICS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            scores[row["model"]].append(float(row["f1"]))
    with LATENCY.open(newline="", encoding="utf-8") as handle:
        latencies = {row["model"]: float(row["milliseconds_per_sample"]) for row in csv.DictReader(handle)}

    width, height = 860, 500
    left, right, top, bottom = 90, 35, 45, 75
    plot_w, plot_h = width - left - right, height - top - bottom
    maximum_latency = 12500.0
    minimum_score, maximum_score = 0.25, 0.46

    def x(value: float) -> float:
        return left + value / maximum_latency * plot_w

    def y(value: float) -> float:
        return top + (maximum_score - value) / (maximum_score - minimum_score) * plot_h

    colors = ["#2563eb", "#0891b2", "#7c3aed", "#ea580c", "#16a34a", "#dc2626"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#1f2937}.axis{stroke:#374151;stroke-width:1}.grid{stroke:#e5e7eb;stroke-width:1}.label{font-size:12px}.title{font-size:19px;font-weight:600}</style>',
        f'<text class="title" x="{width / 2}" y="26" text-anchor="middle">Inference time and mean F1-score</text>',
    ]
    for tick in (0, 2500, 5000, 7500, 10000, 12500):
        px = x(tick)
        lines.extend((f'<line class="grid" x1="{px}" y1="{top}" x2="{px}" y2="{top + plot_h}"/>',
                      f'<text class="label" x="{px}" y="{top + plot_h + 22}" text-anchor="middle">{tick}</text>'))
    for tick in (0.25, 0.30, 0.35, 0.40, 0.45):
        py = y(tick)
        lines.extend((f'<line class="grid" x1="{left}" y1="{py}" x2="{left + plot_w}" y2="{py}"/>',
                      f'<text class="label" x="{left - 12}" y="{py + 4}" text-anchor="end">{tick:.2f}</text>'))
    lines.extend((f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>',
                  f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>',
                  f'<text x="{left + plot_w / 2}" y="{height - 18}" text-anchor="middle">'
                  'Milliseconds per sample (lower is better)</text>',
                  f'<text transform="translate(22 {top + plot_h / 2}) rotate(-90)" text-anchor="middle">'
                  'Mean macro F1 (higher is better)</text>'))
    offsets = {
        "BLIP-2 Flan-T5-XL": (10, 20), "Qwen-VL with Qwen-7B": (10, -10),
        "LLaVA-v1.6-Vicuna-7B": (10, -10), "MiniGPT-4 with Vicuna-7B": (10, 20),
        "Otter-Image-MPT-7B": (10, -10), "InstructBLIP-Vicuna-7B": (-10, -10),
    }
    for color, model in zip(colors, latencies, strict=True):
        mean = sum(scores[model]) / len(scores[model])
        px, py = x(latencies[model]), y(mean)
        dx, dy = offsets[model]
        anchor = "end" if dx < 0 else "start"
        lines.extend((f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6" fill="{color}" stroke="white" stroke-width="2"/>',
                      f'<text class="label" x="{px + dx:.1f}" y="{py + dy:.1f}" text-anchor="{anchor}">{model}</text>'))
    lines.append("</svg>")
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
