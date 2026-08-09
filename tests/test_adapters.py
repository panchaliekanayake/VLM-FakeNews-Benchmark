import pytest

from litevlm_fakenews.adapters import MODEL_REVISIONS, create_adapter


def test_llava_receives_gpu_and_pinned_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    class FakeAdapter:
        def __init__(self, model_path: str, **kwargs: object) -> None:
            captured.update({"model_path": model_path, **kwargs})

    monkeypatch.setattr("litevlm_fakenews.adapters.llava.LlavaAdapter", FakeAdapter)
    create_adapter("llava-v1.6-vicuna-7b", None, None, gpu_id=3)
    assert captured["gpu_id"] == 3
    assert captured["revision"] == MODEL_REVISIONS["llava-v1.6-vicuna-7b"]


def test_minigpt4_requires_explicit_config() -> None:
    with pytest.raises(ValueError, match="requires --model-path"):
        create_adapter("minigpt4-vicuna-7b", None, "/tmp/minigpt4", gpu_id=0)
