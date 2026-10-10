import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

HW2_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HW2_DIR / "src"))

from pretokenized_data import tokenize_parquet_files  # noqa: E402

DATASET = "PrimeIntellect/fineweb-edu"
TRAIN_FILES = tuple(f"data0/train-{index:05d}.parquet" for index in range(4))
DOWNLOAD_DIR = HW2_DIR / "artifacts" / "fineweb"
BUNDLE_DIR = HW2_DIR / "artifacts" / "kaggle" / "tokens_bundle"
TOKENIZER = "HuggingFaceTB/SmolLM-360M"


def main():
    parquet_paths = [hf_hub_download(DATASET, name, repo_type="dataset", local_dir=DOWNLOAD_DIR) for name in TRAIN_FILES]
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    tokenize_parquet_files(parquet_paths, AutoTokenizer.from_pretrained(TOKENIZER), BUNDLE_DIR / "fineweb_edu_tokens.npy")
    metadata = {"title": "drope-tokens", "id": "lild1tz/drope-tokens", "licenses": [{"name": "ODC-By-1.0"}]}
    (BUNDLE_DIR / "dataset-metadata.json").write_text(json.dumps(metadata))
    (BUNDLE_DIR / "source_files.json").write_text(json.dumps({"dataset": DATASET, "files": TRAIN_FILES, "tokenizer": TOKENIZER}))


if __name__ == "__main__":
    main()
