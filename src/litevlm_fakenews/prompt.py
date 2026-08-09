"""Prompt used for every model and dataset in the paper."""

# Spelling and punctuation are preserved from the experiment prompt because
# prompt wording is part of the reproducibility contract.
STANDARD_PROMPT = "\n".join(
    (
        "Given a multimodal misinformation, it contains both news caption and news image. "
        "News caption is: [News caption]",
        "To make a accurate judgement of the multimodal misinformation, please follow the instructions bellow, ",
        "1 Is there any credible objective evidence refuting the news caption? If yes, please answer in the form: "
        "'Finish[TEXT REFUTES].'. If no, continue to step 2.",
        "2 Is there any credible objective evidence refuting the news image? If yes, please answer in the form: "
        "'Finish[IMAGE REFUTES].'. If no, continue to step 3.",
        "3 Does the news caption match the content of news image? If yes, please answer in the form: "
        "'Finish[ORIGINAL].'. If no, please answer in the form: 'Finish[MISMATCH].'.",
        "You should answer in the following form: 'Finish[TEXT REFUTES].' or 'Finish[IMAGE REFUTES].' or "
        "'Finish[ORIGINAL].' or 'Finish[MISMATCH].'.",
        "The answer is:",
    )
)


def build_prompt(caption: str) -> str:
    """Insert an unmodified dataset caption into the shared paper prompt."""
    return STANDARD_PROMPT.replace("[News caption]", caption)
