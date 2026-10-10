import pytest
from transformers import AutoTokenizer

from niah import NeedleTrial, build_prompt_ids, is_correct, make_trials, needle_text

TOKENIZER = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-360M")
HAYSTACK_IDS = TOKENIZER(" ".join(f"Sentence number {index} is here." for index in range(2000)))["input_ids"]
CONTEXT_LENGTH = 1024


def prompt_text(depth):
    trial = NeedleTrial(key="quiet-river", value=4815162, depth=depth, haystack_offset=0)
    return TOKENIZER.decode(build_prompt_ids(TOKENIZER, HAYSTACK_IDS, trial, CONTEXT_LENGTH)), trial


def test_prompt_has_requested_length():
    trial = NeedleTrial(key="quiet-river", value=4815162, depth=0.5, haystack_offset=37)
    assert len(build_prompt_ids(TOKENIZER, HAYSTACK_IDS, trial, CONTEXT_LENGTH)) == CONTEXT_LENGTH


@pytest.mark.parametrize("depth", [0.0, 0.25, 0.5, 0.75, 1.0], ids=["d0", "d25", "d50", "d75", "d100"])
def test_needle_lands_near_requested_depth(depth):
    text, trial = prompt_text(depth)
    needle_start = text.index(needle_text(trial).strip())
    haystack_start = text.index("Sentence number")
    haystack_end = text.rindex("is here.") + len("is here.")
    relative_position = (needle_start - haystack_start) / (haystack_end - haystack_start)
    assert relative_position == pytest.approx(depth, abs=0.03)


def test_needle_follows_sentence_end():
    text, trial = prompt_text(0.5)
    assert text[: text.index(needle_text(trial).strip())].rstrip().endswith(".")


def test_prompt_ends_with_question_about_key():
    text, _ = prompt_text(0.5)
    assert text.endswith("The special magic number for quiet-river mentioned in the provided text is:")


@pytest.mark.parametrize(
    "generated, expected",
    [(" 4815162.\n", True), (" 4815163", False), (" 48151620", False)],
    ids=["exact", "other_number", "longer_number"],
)
def test_answer_must_contain_exact_value(generated, expected):
    assert is_correct(generated, 4815162) is expected


def test_trials_cover_every_depth_equally():
    trials = make_trials(depths=(0.0, 0.5, 1.0), trials_per_depth=4, haystack_length=10_000, context_length=1024, seed=0)
    assert sorted(trial.depth for trial in trials) == [0.0] * 4 + [0.5] * 4 + [1.0] * 4


def test_trials_are_reproducible():
    arguments = dict(depths=(0.0, 1.0), trials_per_depth=3, haystack_length=10_000, context_length=1024, seed=7)
    assert make_trials(**arguments) == make_trials(**arguments)
