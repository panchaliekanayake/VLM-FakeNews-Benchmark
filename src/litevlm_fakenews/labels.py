"""Deterministic parsing of free-form VLM responses."""

from __future__ import annotations

import re
from dataclasses import dataclass

TRUE = 0
FAKE = 1

_ANSWER_MARKERS = ("The answer is:", "GPT:<answer>", "GPT: Answer:", "GPT:")
_DECISIONS = ("text refutes", "image refutes", "mismatch", "original", "match")


@dataclass(frozen=True)
class ParsedLabel:
    binary_label: int
    decision: str
    valid: bool


def prediction_segment(response: object) -> str:
    """Remove an echoed prompt when a model includes an answer marker."""
    text = str(response)
    for marker in _ANSWER_MARKERS:
        if marker in text:
            text = text.rsplit(marker, 1)[1]
    return text.strip()


def parse_response(response: object) -> ParsedLabel:
    """Map the first valid decision to True/Fake; invalid output defaults to True.

    This fallback exactly follows the protocol described in the paper. Invalid
    generations remain observable through ``valid=False``.
    """
    normalized = re.sub(r"[^a-z0-9]+", " ", prediction_segment(response).lower())
    normalized = " ".join(word for word in normalized.split() if word not in {"a", "an", "the"})
    for candidate in _DECISIONS:
        if re.search(rf"\bfinish {candidate}\b", normalized):
            decision = candidate.replace(" ", "_")
            if decision in {"original", "match"}:
                return ParsedLabel(TRUE, "original", True)
            return ParsedLabel(FAKE, decision, True)
    return ParsedLabel(TRUE, "original", False)


def ground_truth_label(value: object) -> int:
    """Normalize accepted binary ground-truth labels."""
    normalized = str(value).strip().lower()
    if normalized in {"true", "real", "original", "0"}:
        return TRUE
    if normalized in {"fake", "false", "misinformation", "1"}:
        return FAKE
    raise ValueError(f"Unsupported ground-truth label: {value!r}")
