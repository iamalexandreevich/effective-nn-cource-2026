import json
import random
import re
import string
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

from fuzzywuzzy import fuzz
from huggingface_hub import hf_hub_download
from rouge import Rouge

HW2_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = HW2_DIR / "artifacts" / "longbench"
CONFIG_URL = "https://raw.githubusercontent.com/THUDM/LongBench/main/LongBench/config/{name}.json"
DATASETS = ("multifieldqa_en", "musique", "gov_report", "lcc")


# Метрики воспроизводят LongBench/metrics.py (THUDM/LongBench, MIT)
def normalize_answer(text):
    text = text.lower()
    text = "".join(char for char in text if char not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def f1_score(prediction_tokens, truth_tokens):
    common = Counter(prediction_tokens) & Counter(truth_tokens)
    same = sum(common.values())
    if same == 0:
        return 0
    precision = same / len(prediction_tokens)
    recall = same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def qa_f1_score(prediction, truth):
    return f1_score(normalize_answer(prediction).split(), normalize_answer(truth).split())


def code_sim_score(prediction, truth):
    first_code_line = ""
    for line in prediction.lstrip("\n").split("\n"):
        if "`" not in line and "#" not in line and "//" not in line:
            first_code_line = line
            break
    return fuzz.ratio(first_code_line, truth) / 100


def rouge_score(prediction, truth):
    try:
        return Rouge().get_scores([prediction], [truth], avg=True)["rouge-l"]["f"]
    except Exception:
        return 0.0


METRICS = {"multifieldqa_en": qa_f1_score, "musique": qa_f1_score, "gov_report": rouge_score, "lcc": code_sim_score}


def mean_score(dataset, predictions, answers):
    total = 0.0
    for prediction, references in zip(predictions, answers):
        total += max(METRICS[dataset](prediction, reference) for reference in references)
    return round(100 * total / len(predictions), 2)


def truncate_middle(token_ids, max_length):
    if len(token_ids) <= max_length:
        return token_ids
    half = max_length // 2
    return token_ids[:half] + token_ids[-half:]


def build_prompt_ids(tokenizer, prompt_format, sample, max_length):
    return truncate_middle(tokenizer(prompt_format.format(**sample))["input_ids"], max_length)


def pick_samples(rows, count, seed):
    if count >= len(rows):
        return list(rows)
    return random.Random(seed).sample(rows, count)


def load_samples(dataset, count, seed):
    archive = hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset", local_dir=DATA_DIR)
    with zipfile.ZipFile(archive) as zipped:
        rows = [json.loads(line) for line in zipped.read(f"data/{dataset}.jsonl").decode("utf-8").splitlines()]
    return pick_samples(rows, count, seed)


def load_task_config(name):
    path = DATA_DIR / f"{name}.json"
    if not path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(CONFIG_URL.format(name=name)) as response:
            path.write_bytes(response.read())
    return json.loads(path.read_text())
