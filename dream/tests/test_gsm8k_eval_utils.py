from dream.src.gsm8k_eval_utils import (
    answers_match,
    extract_predicted_answer,
    reference_final_answer,
)


def test_reference_final_answer():
    ref = "Some reasoning.\n#### 42"
    assert reference_final_answer(ref) == "42"


def test_extract_predicted_answer_prefers_hash_line():
    text = "x = 1 + 1\n#### 2"
    assert extract_predicted_answer(text) == "2"


def test_answers_match_numeric():
    assert answers_match("42", "42")
    assert answers_match("42.0", "42")
