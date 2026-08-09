import json
from pathlib import Path

import pytest

from litevlm_fakenews.dataset import load_samples, missing_images


def test_load_schema_and_resolve_leading_slash(tmp_path: Path) -> None:
    image = tmp_path / "images" / "one.png"
    image.parent.mkdir()
    image.write_bytes(b"test")
    annotations = tmp_path / "split.json"
    annotations.write_text(json.dumps([
        {"text": "caption", "image_path": "/images/one.png", "gt_answers": "Fake", "fake_cls": "mismatch"}
    ]))
    samples = load_samples(annotations, tmp_path)
    assert samples[0].image_path == image
    assert list(missing_images(samples)) == []


def test_reject_path_traversal(tmp_path: Path) -> None:
    annotations = tmp_path / "split.json"
    annotations.write_text(json.dumps([
        {"text": "caption", "image_path": "../../outside.png", "gt_answers": "True"}
    ]))
    with pytest.raises(ValueError, match="escapes data_root"):
        load_samples(annotations, tmp_path / "data")
