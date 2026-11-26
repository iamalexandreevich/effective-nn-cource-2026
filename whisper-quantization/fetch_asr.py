import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode, urlparse
from urllib.request import urlopen

SERVER = "datasets-server.huggingface.co"


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize selected Hugging Face ASR rows as mono 16 kHz WAV")
    parser.add_argument("destination", type=Path)
    parser.add_argument("--dataset", default="openslr/librispeech_asr")
    parser.add_argument("--config", default="clean")
    parser.add_argument("--split", required=True)
    parser.add_argument("--offsets", type=int, nargs="+", required=True)
    parser.add_argument("--per-offset", type=int, default=8)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        pages = list(pool.map(lambda offset: _rows(args, offset), args.offsets))
        if any(not page for page in pages):
            raise ValueError("An offset returned no rows")
        rows = [page[index] for index in range(max(map(len, pages))) for page in pages if index < len(page)]
        if len({row["row_idx"] for row in rows}) != len(rows):
            raise ValueError("Selected row ranges overlap")
        records = list(pool.map(lambda row: _materialize(args, row), rows))
    manifest = args.destination / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
    print(f"{manifest}: {len(records)} utterances")


def _rows(args: argparse.Namespace, offset: int) -> list[dict]:
    query = urlencode(dict(dataset=args.dataset, config=args.config, split=args.split,
                           offset=offset, length=args.per_offset))
    with urlopen(f"https://{SERVER}/rows?{query}", timeout=45) as response:
        return json.load(response)["rows"]


def _materialize(args: argparse.Namespace, item: dict) -> dict:
    index, row = item["row_idx"], item["row"]
    text = next((row[key] for key in ("text", "transcription", "sentence", "normalized_text") if key in row), None)
    if not isinstance(text, str):
        raise ValueError(f"Missing transcript at row {index}")
    url = row["audio"][0]["src"]
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != SERVER:
        raise ValueError(f"Unexpected audio host at row {index}")
    audio = f"{index:08}.wav"
    target = args.destination / audio
    if not target.exists():
        temporary = target.with_suffix(".partial")
        try:
            with urlopen(url, timeout=60) as response:
                data = response.read()
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", "pipe:0",
                            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-f", "wav", str(temporary)],
                           input=data, check=True)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return {
        "id": str(row.get("id") or f"{args.dataset}/{args.config}/{args.split}/{index}"),
        "audio": audio,
        "text": text,
        "split": args.split,
        "dataset": args.dataset,
        "config": args.config,
        "row": index,
        "asset": parsed.path,
    }


if __name__ == "__main__":
    main()
