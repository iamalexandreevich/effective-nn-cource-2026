import math
from pathlib import Path

import torch
from datasets import load_dataset

EVAL_DATASET = "PrimeIntellect/fineweb-edu"
# data0 и data1 идут в рекалибровку, data3 не используется в обучении
EVAL_DATA_FILES = "data3/*.parquet"


def load_eval_sequences(tokenizer, sequence_length, count, cache_path, skip_tokens=0):
    cache_path = Path(cache_path)
    if cache_path.exists():
        return torch.load(cache_path)
    sequences = _tokenize_held_out_documents(tokenizer, sequence_length, count, skip_tokens)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sequences, cache_path)
    return sequences


def perplexity(model, sequences, device, window=None):
    nll_sum, token_count = 0.0, 0
    for sequence in sequences:
        sequence_nll_sum, sequence_tokens = _sequence_nll(model, sequence.unsqueeze(0).to(device), window)
        nll_sum += sequence_nll_sum
        token_count += sequence_tokens
    return math.exp(nll_sum / token_count)


def _sequence_nll(model, input_ids, window):
    length = input_ids.shape[1]
    window = window or length
    stride = max(window // 4, 1)
    nll_sum, token_count, scored_until = 0.0, 0, 0
    for begin in range(0, length, stride):
        end = min(begin + window, length)
        chunk = input_ids[:, begin:end]
        labels = chunk.clone()
        labels[:, : scored_until - begin] = -100
        with torch.no_grad():
            loss = model(chunk, labels=labels).loss
        scored_tokens = int((labels[:, 1:] != -100).sum())
        nll_sum += loss.item() * scored_tokens
        token_count += scored_tokens
        scored_until = end
        if end == length:
            break
    return nll_sum, token_count


def _tokenize_held_out_documents(tokenizer, sequence_length, count, skip_tokens):
    documents = load_dataset(EVAL_DATASET, data_files=EVAL_DATA_FILES, split="train", streaming=True)
    needed_tokens = sequence_length * count + skip_tokens
    tokens = []
    for document in documents:
        tokens.extend(tokenizer(document["text"])["input_ids"] + [tokenizer.eos_token_id])
        if len(tokens) >= needed_tokens:
            break
    return torch.tensor(tokens[skip_tokens:needed_tokens]).view(count, sequence_length)
