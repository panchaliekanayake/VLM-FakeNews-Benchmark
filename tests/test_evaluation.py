import pytest

from litevlm_fakenews.evaluation import evaluate_records


def test_known_confusion_matrix_and_macro_scores() -> None:
    records = [
        {"answer": "Finish[ORIGINAL].", "gt_answers": "True"},
        {"answer": "Finish[MISMATCH].", "gt_answers": "True"},
        {"answer": "Finish[TEXT REFUTES].", "gt_answers": "Fake"},
        {"answer": "unparseable", "gt_answers": "Fake"},
    ]
    metrics = evaluate_records(records)
    assert metrics["confusion_matrix"]["values"] == [[1, 1], [1, 1]]
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["macro"]["f1"] == pytest.approx(0.5)
    assert metrics["invalid_outputs"] == 1


def test_empty_predictions_fail() -> None:
    with pytest.raises(ValueError, match="At least one"):
        evaluate_records([])
