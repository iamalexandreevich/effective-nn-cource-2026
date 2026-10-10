import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi

HW2_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HW2_DIR / "src"))

from hf_release import build_release  # noqa: E402

SNAPSHOT_DIR = HW2_DIR / "artifacts" / "kaggle" / "main" / "snapshots" / "step-1000"
RELEASE_DIR = HW2_DIR / "artifacts" / "hf_release"


def main():
    args = parse_args()
    release_dir = build_release(SNAPSHOT_DIR, HW2_DIR / "results", RELEASE_DIR)
    if args.dry_run:
        print(f"Релиз собран в {release_dir}, загрузка не выполнялась")
        return
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(folder_path=str(release_dir), repo_id=args.repo_id, repo_type="model", commit_message="SmolLM-360M-DroPE, step 1000")
    print(f"https://huggingface.co/{args.repo_id}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id", help="например, username/SmolLM-360M-DroPE-66M")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
