from litevlm_fakenews.labels import FAKE, TRUE, parse_response, prediction_segment


def test_all_paper_decisions() -> None:
    assert parse_response("Finish[TEXT REFUTES].").binary_label == FAKE
    assert parse_response("Finish[IMAGE REFUTES].").binary_label == FAKE
    assert parse_response("Finish[MISMATCH].").binary_label == FAKE
    assert parse_response("Finish[ORIGINAL].").binary_label == TRUE
    assert parse_response("Finish[MATCH].").binary_label == TRUE


def test_invalid_defaults_to_true_and_is_flagged() -> None:
    parsed = parse_response("I cannot determine this.")
    assert parsed.binary_label == TRUE
    assert parsed.valid is False


def test_echoed_prompt_is_removed() -> None:
    response = "Finish[TEXT REFUTES] appears in prompt. The answer is: Finish[ORIGINAL]."
    assert prediction_segment(response) == "Finish[ORIGINAL]."
    assert parse_response(response).binary_label == TRUE


def test_parser_matches_original_priority_and_punctuation_tolerance() -> None:
    response = "Finish ORIGINAL, but correction: finish_mismatch."
    parsed = parse_response(response)
    assert parsed.binary_label == FAKE
    assert parsed.decision == "mismatch"
