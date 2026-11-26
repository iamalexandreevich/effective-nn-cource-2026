import argparse
import json
import random
import subprocess
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(".runtime")
WORK = ROOT / "context-study"
COUNT = 200
SEED = {"libri": 20261125, "golos": 20261126}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", choices=SEED)
    corpus = parser.parse_args().corpus
    prepare(corpus)


def prepare(corpus):
    target = WORK / corpus
    target.mkdir(parents=True, exist_ok=True)
    old_manifest = ROOT / f"data/{corpus}-dev/manifest.jsonl"
    old = {row["row"]: row for row in map(json.loads, old_manifest.read_text().splitlines())}
    parquet_path = WORK / f"{corpus}-dev.parquet"
    source = pq.ParquetFile(parquet_path)
    source_size = source.metadata.num_rows
    chosen = random.Random(SEED[corpus]).sample([i for i in range(source_size) if i not in old], COUNT)
    columns = ["audio", "id", "text"] if corpus == "libri" else ["audio", "transcription"]
    wanted = set(chosen) | old.keys()
    results = {}
    offset = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for batch in source.iter_batches(columns=columns, batch_size=128):
            end = offset + batch.num_rows
            rows = [offset + index for index in range(batch.num_rows) if offset + index in wanted]
            selected = batch.take([index - offset for index in rows]).to_pylist() if rows else []
            tasks = []
            for index, row in zip(rows, selected, strict=True):
                if index in old:
                    assert row.get("text", row.get("transcription")) == old[index]["text"], (corpus, index)
                    if "id" in row:
                        assert row["id"] == old[index]["id"]
                else:
                    tasks.append(pool.submit(materialize, target, corpus, index, row))
            for task in tasks:
                record = task.result()
                results[record["row"]] = record
            offset = end
    assert len(results) == COUNT and set(results) == set(chosen)
    manifest = target / "manifest.jsonl"
    tmp = manifest.with_suffix(".partial")
    tmp.write_text("".join(json.dumps(results[index], ensure_ascii=False) + "\n" for index in chosen))
    tmp.replace(manifest)
    for band in ("short", "medium", "long"):
        count = sum(row["band"] == band for row in results.values())
        print(f"{corpus}: {band}={count}", flush=True)


def materialize(target, corpus, index, row):
    audio = target / f"{index:08}.wav"
    if not audio.exists():
        temporary = audio.with_suffix(".partial")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", "pipe:0",
                        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-f", "wav", str(temporary)],
                       input=row["audio"]["bytes"], check=True)
        temporary.replace(audio)
    with wave.open(str(audio), "rb") as stream:
        assert (stream.getnchannels(), stream.getsampwidth(), stream.getframerate()) == (1, 2, 16000)
        samples = stream.getnframes()
    band = "short" if samples <= 152000 else ("medium" if samples <= 312000 else "long")
    return {"id": row.get("id", f"golos/validation/{index}"), "row": index, "audio": str(audio.resolve()),
            "text": row.get("text", row.get("transcription")), "samples": samples, "band": band}


if __name__ == "__main__":
    main()
