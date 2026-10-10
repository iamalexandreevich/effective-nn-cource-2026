import pytest
from transformers import AutoTokenizer

from longbench import build_prompt_ids, mean_score, pick_samples, truncate_middle

TOKENIZER = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-360M")


def test_qa_f1_counts_overlap_after_normalization():
    assert mean_score("multifieldqa_en", ["The cat sat"], [["a cat sat down"]]) == pytest.approx(80.0)


def test_qa_f1_is_zero_without_overlap():
    assert mean_score("musique", ["dog"], [["cat"]]) == 0.0


def test_score_takes_best_of_several_reference_answers():
    assert mean_score("musique", ["Paris"], [["London", "Paris"]]) == 100.0


def test_code_similarity_skips_fence_and_comment_lines():
    prediction = "```python\n# comment\nx = 1\n```"
    assert mean_score("lcc", [prediction], [["x = 1"]]) == 100.0


def test_rouge_l_is_perfect_for_identical_summary():
    summary = "the agency should improve oversight of the program"
    assert mean_score("gov_report", [summary], [[summary]]) == pytest.approx(100.0, abs=0.01)


def test_mean_score_averages_over_samples_and_rounds():
    assert mean_score("musique", ["cat", "dog"], [["cat"], ["bird"]]) == 50.0


def test_truncate_middle_keeps_both_ends():
    assert truncate_middle(list(range(10)), max_length=4) == [0, 1, 8, 9]


def test_truncate_middle_leaves_short_input_unchanged():
    assert truncate_middle([1, 2, 3], max_length=8) == [1, 2, 3]


def test_prompt_is_cut_to_max_length_and_keeps_question():
    sample = {"context": "word " * 5000, "input": "What is the answer?"}
    ids = build_prompt_ids(TOKENIZER, "Text: {context}\nQuestion: {input}\nAnswer:", sample, max_length=512)
    assert len(ids) == 512 and TOKENIZER.decode(ids).endswith("Question: What is the answer?\nAnswer:")


def test_pick_samples_is_reproducible_and_has_requested_size():
    rows = [{"id": index} for index in range(100)]
    assert pick_samples(rows, count=10, seed=3) == pick_samples(rows, count=10, seed=3)
    assert len(pick_samples(rows, count=10, seed=3)) == 10


def test_pick_samples_returns_everything_when_count_is_larger():
    rows = [{"id": index} for index in range(5)]
    assert len(pick_samples(rows, count=50, seed=0)) == 5
