import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

from pretokenized_data import TokenBlockDataset, load_pretokenized_dataset, tokenize_parquet_files

TOKENIZER = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-360M")
DOCUMENTS = ["First document here.", "Second one follows.", "Third and last."]


def write_parquet(path):
    pq.write_table(pa.table({"text": DOCUMENTS}), path)
    return path


def test_tokens_are_documents_joined_by_eos(tmp_path):
    tokens_path = tokenize_parquet_files([write_parquet(tmp_path / "a.parquet")], TOKENIZER, tmp_path / "tokens.npy")
    expected = []
    for document in DOCUMENTS:
        expected += TOKENIZER(document)["input_ids"] + [TOKENIZER.eos_token_id]
    assert np.load(tokens_path).tolist() == expected


def test_blocks_are_consecutive_and_labels_match_inputs(tmp_path):
    tokens_path = tmp_path / "tokens.npy"
    np.save(tokens_path, np.arange(10, dtype=np.uint16))
    dataset = TokenBlockDataset(tokens_path, block_size=4)
    assert len(dataset) == 2
    assert dataset[1]["input_ids"].tolist() == [4, 5, 6, 7]
    assert torch.equal(dataset[1]["labels"], dataset[1]["input_ids"])


def test_loader_ignores_unrelated_config_keys(tmp_path):
    tokens_path = tmp_path / "tokens.npy"
    np.save(tokens_path, np.arange(10, dtype=np.uint16))
    datasets = load_pretokenized_dataset(tokens_path=str(tokens_path), max_seq_length=4, tokenizer=None, streaming=True)
    assert len(datasets["train_dataset"]) == 2 and datasets["eval_dataset"] is None
