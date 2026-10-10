import json
import random
import re
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

ESSAYS_API_URL = "https://api.github.com/repos/gkamradt/LLMTest_NeedleInAHaystack/contents/needlehaystack/PaulGrahamEssays"
HEADER = (
    "Some special magic numbers are hidden within the following text. "
    "Make sure to memorize it. I will quiz you about the numbers afterwards.\n"
)
NEEDLE = " One of the special magic numbers for {key} is: {value}."
QUESTION = (
    "\nWhat is the special magic number for {key} mentioned in the provided text? "
    "The special magic number for {key} mentioned in the provided text is:"
)
KEY_ADJECTIVES = (
    "whispering", "elite", "cloistered", "quiet", "amber", "brisk", "hollow", "lucid", "rustic", "silent",
    "tangled", "vivid", "wistful", "frozen", "gentle", "humble", "jagged", "mellow", "nimble", "proud",
)
KEY_NOUNS = (
    "workhorse", "butterfly", "colonization", "river", "lantern", "harbor", "meadow", "compass", "orchard", "falcon",
    "glacier", "violin", "canyon", "beacon", "thimble", "quarry", "saddle", "tundra", "walnut", "zephyr",
)
MAX_NEW_TOKENS = 12


@dataclass(frozen=True)
class NeedleTrial:
    key: str
    value: int
    depth: float
    haystack_offset: int


def make_trials(depths, trials_per_depth, haystack_length, context_length, seed):
    rng = random.Random(seed)
    return [
        NeedleTrial(
            key=f"{rng.choice(KEY_ADJECTIVES)}-{rng.choice(KEY_NOUNS)}",
            value=rng.randint(1_000_000, 9_999_999),
            depth=depth,
            haystack_offset=rng.randint(0, haystack_length - context_length),
        )
        for depth in depths
        for _ in range(trials_per_depth)
    ]


def needle_text(trial):
    return NEEDLE.format(key=trial.key, value=trial.value)


def build_prompt_ids(tokenizer, haystack_ids, trial, context_length):
    header = tokenizer(HEADER)["input_ids"]
    needle = tokenizer(needle_text(trial))["input_ids"]
    question = tokenizer(QUESTION.format(key=trial.key))["input_ids"]
    filler_length = context_length - len(header) - len(needle) - len(question)
    filler = haystack_ids[trial.haystack_offset : trial.haystack_offset + filler_length]
    insert_at = _sentence_boundary_before(tokenizer, filler, round(trial.depth * filler_length))
    return header + filler[:insert_at] + needle + filler[insert_at:] + question


def is_correct(generated, value):
    return re.search(rf"(?<!\d){value}(?!\d)", generated) is not None


def evaluate_niah(model, tokenizer, haystack_ids, trials, context_length, device, window=None):
    results = []
    for trial in trials:
        prompt_ids = build_prompt_ids(tokenizer, haystack_ids, trial, context_length)
        if window is not None:
            prompt_ids = prompt_ids[-window:]
        generated = _greedy_generate(model, tokenizer, prompt_ids, device)
        results.append({**asdict(trial), "generated": generated, "correct": is_correct(generated, trial.value)})
    return results


def load_haystack_ids(tokenizer, cache_path):
    cache_path = Path(cache_path)
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(_download_essays())
    return tokenizer(cache_path.read_text())["input_ids"]


def _greedy_generate(model, tokenizer, prompt_ids, device):
    input_ids = torch.tensor([prompt_ids], device=device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0, len(prompt_ids) :], skip_special_tokens=True)


def _sentence_boundary_before(tokenizer, token_ids, position):
    for index in range(position, 0, -1):
        if tokenizer.decode(token_ids[index - 1]).rstrip().endswith("."):
            return index
    return position


def _download_essays():
    with urllib.request.urlopen(ESSAYS_API_URL) as response:
        listing = json.load(response)
    essays = []
    for entry in sorted(listing, key=lambda item: item["name"]):
        with urllib.request.urlopen(entry["download_url"]) as response:
            essays.append(response.read().decode("utf-8"))
    return "\n\n".join(essays)
