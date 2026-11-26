import hashlib
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from time import perf_counter

ROOT = Path(".runtime")
WORK = ROOT / "context-study"
CLI = WORK / "whisper-cli-timed"
MODEL = ROOT / "models/small-q4_0.bin"
MODES = {
    "full_flash": (False, True),
    "full_no_flash": (False, False),
    "adaptive_flash": (True, True),
    "adaptive_no_flash": (True, False),
}
CONTEXT = {"short": 512, "medium": 1024, "long": 0}
LANGUAGE = {"libri": "en", "golos": "ru"}
BATCH = 25
MICROS = re.compile(r"^STUDY_CLIP_MICROS (\d+) (\d+)$", re.MULTILINE)


def main():
    samples = {corpus: read_sample(corpus) for corpus in LANGUAGE}
    model_hash = digest(MODEL)
    cli_hash = digest(CLI)
    for batch_index, start in enumerate(range(0, 200, BATCH)):
        corpora = ("golos", "libri") if batch_index % 2 == 0 else ("libri", "golos")
        for corpus in corpora:
            rows = samples[corpus][start:start + BATCH]
            names = tuple(MODES)
            rotation = (batch_index + int(corpus == "libri")) % len(names)
            for mode in names[rotation:] + names[:rotation]:
                adaptive, flash = MODES[mode]
                groups = defaultdict(list)
                for row in rows:
                    context = CONTEXT[row["band"]] if adaptive else 0
                    groups[context].append(row)
                wall = 0.0
                for context, group in sorted(groups.items()):
                    result = run_group(corpus, mode, batch_index, group, context, flash,
                                       (model_hash, cli_hash))
                    wall += result["wall_seconds"]
                print(f"{corpus} {start + len(rows):>3}/200 {mode} {wall:.2f} s", flush=True)


def read_sample(corpus):
    records = [json.loads(line) for line in (WORK / corpus / "manifest.jsonl").read_text().splitlines()]
    assert len(records) == len({row["id"] for row in records}) == 200
    return records


def run_group(corpus, mode, batch_index, rows, context, flash, hashes):
    work = WORK / "runs" / corpus / mode / f"batch{batch_index:02}-ctx{context}"
    work.mkdir(parents=True, exist_ok=True)
    summary = work / "group.json"
    identity = {
        "ids": [row["id"] for row in rows],
        "context": context,
        "flash_attention": flash,
        "model_sha256": hashes[0],
        "cli_sha256": hashes[1],
    }
    if summary.exists():
        result = json.loads(summary.read_text())
        assert result["identity"] == identity, work
        return result
    attempt = work / f"attempt-{len(list(work.glob('attempt-*'))):02}"
    attempt.mkdir()
    cmd = make_command(corpus, context, flash, rows, attempt)
    log = attempt / "run.log"
    started = perf_counter()
    with log.open("w") as stream:
        status = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
    elapsed = perf_counter() - started
    status.check_returncode()
    timings = {int(index): int(micros) for index, micros in MICROS.findall(log.read_text())}
    assert sorted(timings) == list(range(len(rows))), log
    predictions = []
    for index, row in enumerate(rows):
        path = attempt / f"{index:03}.json"
        raw = json.loads(path.read_text())
        text = "".join(item["text"] for item in raw["transcription"]).strip()
        predictions.append({**row, "prediction": text, "inference_us": timings[index], "native_json": str(path)})
    result = {"identity": identity, "wall_seconds": elapsed, "predictions": predictions, "command": cmd}
    temp = summary.with_suffix(".partial")
    temp.write_text(json.dumps(result, ensure_ascii=False) + "\n")
    temp.replace(summary)
    return result


def make_command(corpus, context, flash, rows, directory):
    command = [str(CLI), "-m", str(MODEL), "-l", LANGUAGE[corpus], "-t", "4", "-ng", "-nf",
               "-nt", "-bs", "1", "-bo", "1", "-tp", "0", "-oj", "-ac", str(context),
               "-fa" if flash else "-nfa"]
    for index, row in enumerate(rows):
        command.extend(["-f", row["audio"], "-of", str(directory / f"{index:03}")])
    return command


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


if __name__ == "__main__":
    main()
