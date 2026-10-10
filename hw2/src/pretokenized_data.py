from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import default_data_collator

TOKENIZE_BATCH_DOCUMENTS = 1000


class TokenBlockDataset(torch.utils.data.Dataset):
    def __init__(self, tokens_path, block_size):
        self._tokens = np.load(tokens_path, mmap_mode="r")
        self._block_size = block_size

    def __len__(self):
        return len(self._tokens) // self._block_size

    def __getitem__(self, index):
        start = index * self._block_size
        block = torch.from_numpy(self._tokens[start : start + self._block_size].astype(np.int64))
        return {"input_ids": block, "attention_mask": torch.ones_like(block), "labels": block.clone()}


# Ключи make_dataset_fn из data_cfg авторов приходят сюда вместе с нашими, лишние игнорируем
def load_pretokenized_dataset(tokens_path, max_seq_length, **unused_config):
    return {
        "train_dataset": TokenBlockDataset(tokens_path, max_seq_length),
        "eval_dataset": None,
        "data_collator": default_data_collator,
        "compute_metrics": None,
        "preprocess_logits_for_metrics": None,
    }


def tokenize_parquet_files(parquet_paths, tokenizer, output_path):
    chunks = []
    for parquet_path in parquet_paths:
        for batch in pq.ParquetFile(parquet_path).iter_batches(batch_size=TOKENIZE_BATCH_DOCUMENTS, columns=["text"]):
            chunks.append(_tokenize_documents(tokenizer, batch.column("text").to_pylist()))
    output_path = Path(output_path)
    np.save(output_path, np.concatenate(chunks))
    return output_path


def _tokenize_documents(tokenizer, documents):
    token_lists = tokenizer(documents)["input_ids"]
    joined = [token for tokens in token_lists for token in (*tokens, tokenizer.eos_token_id)]
    return np.asarray(joined, dtype=np.uint16)
