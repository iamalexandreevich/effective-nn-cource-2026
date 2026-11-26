import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from jiwer import process_words

from benchmark import _normalizer
from run_study import BATCH, LANGUAGE, MODES, WORK, read_sample

BANDS = ("short", "medium", "long")
BASE = "full_flash"


def main():
    parser = argparse.ArgumentParser(description="Summarize matched CPU benchmark runs")
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    summary = {}
    for corpus, language in LANGUAGE.items():
        corpus_summary = summarize_corpus(corpus, language, args.partial)
        if corpus_summary:
            summary[corpus] = corpus_summary
    if args.partial:
        for corpus, result in summary.items():
            print(corpus, result["clips"], "matched recordings")
            for mode, row in result["modes"].items():
                overall = row["all"]
                print(f"  {mode:19} WER={overall['wer_pct']:5.2f}%  "
                      f"p50={overall['median_ms']:6.1f} ms  "
                      f"wall={row['wall_seconds']:7.2f} s")
        return
    path = WORK / "summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(path)


def summarize_corpus(corpus, language, partial):
    sample = read_sample(corpus)
    source_batches = [sample[index:index + BATCH] for index in range(0, len(sample), BATCH)]
    directories = WORK / "runs" / corpus
    all_groups = {
        mode: [json.loads(path.read_text()) for path in sorted((directories / mode).glob("*/group.json"))]
        for mode in MODES
    }
    shared = set.intersection(*[{row["id"] for group in groups for row in group["predictions"]}
                                for groups in all_groups.values()])
    if partial:
        sample = [row for batch in source_batches if {row["id"] for row in batch} <= shared for row in batch]
        if not sample:
            return {}
    targets = {row["id"] for row in sample}
    if not partial:
        assert targets == shared and len(shared) == 200, (corpus, len(shared))

    normalize = _normalizer(language)
    references = {row["id"]: normalize(row["text"]) for row in sample}
    modes = {}
    edits_by_mode = {}
    for mode, groups in all_groups.items():
        selected = [group for group in groups if set(group["identity"]["ids"]) <= targets]
        results = [record for group in selected for record in group["predictions"]]
        assert {row["id"] for row in results} == targets and len(results) == len(targets)
        result = score_records(results, normalize, references)
        edits_by_mode[mode] = {row["id"]: row for row in result}
        wall = sum(group["wall_seconds"] for group in selected)
        modes[mode] = {"wall_seconds": wall,
                       "subprocesses": len(selected),
                       "all": aggregate(result),
                       "bands": {band: aggregate([record for record in result if record["band"] == band])
                                 for band in BANDS}}
        modes[mode]["wall_rtf"] = wall / modes[mode]["all"]["audio_seconds"]

    baseline = edits_by_mode[BASE]
    extreme = {record["id"] for records in edits_by_mode.values() for record in records.values()
               if record["insertions"] >= 20}
    retained = targets - extreme
    for mode, result in modes.items():
        current = edits_by_mode[mode]
        result["insertion_outliers"] = [
            {field: row[field] for field in ("id", "samples", "words", "errors", "insertions")}
            for row in current.values() if row["insertions"] >= 20]
        if extreme:
            result["diagnostic_without_shared_outliers"] = aggregate([current[key] for key in sorted(retained)])
        for band in ("all", *BANDS):
            keys = [key for key, row in current.items() if band == "all" or row["band"] == band]
            if not keys:
                continue
            metric = result["all"] if band == "all" else result["bands"][band]
            metric["paired"] = paired_delta(corpus, baseline, current, keys, bootstrap=not partial)
        result["speedup_wall"] = modes[BASE]["wall_seconds"] / result["wall_seconds"]
        result["speedup_inference"] = modes[BASE]["all"]["inference_seconds"] / result["all"]["inference_seconds"]
    return {"clips": len(sample), "modes": modes}


def score_records(records, normalize, references):
    result = []
    for record in records:
        alignment = process_words(references[record["id"]], normalize(record["prediction"]))
        errors = alignment.substitutions + alignment.deletions + alignment.insertions
        words = alignment.hits + alignment.substitutions + alignment.deletions
        result.append({**record, "errors": errors, "words": words, "substitutions": alignment.substitutions,
                       "deletions": alignment.deletions, "insertions": alignment.insertions})
    return result


def aggregate(records):
    if not records:
        return {"clips": 0}
    words = sum(row["words"] for row in records)
    errors = sum(row["errors"] for row in records)
    latencies = np.asarray([row["inference_us"] for row in records]) / 1000
    audio_seconds = sum(row["samples"] for row in records) / 16000
    inference_seconds = float(latencies.sum() / 1000)
    return {
        "clips": len(records), "errors": errors, "words": words, "wer_pct": 100 * errors / words,
        "substitutions": sum(row["substitutions"] for row in records),
        "deletions": sum(row["deletions"] for row in records),
        "insertions": sum(row["insertions"] for row in records),
        "median_ms": float(np.median(latencies)), "p95_ms": float(np.percentile(latencies, 95)),
        "audio_seconds": audio_seconds, "inference_seconds": inference_seconds,
        "inference_rtf": inference_seconds / audio_seconds,
    }


def paired_delta(corpus, baseline, current, keys, *, bootstrap):
    differences = [current[key]["errors"] - baseline[key]["errors"] for key in keys]
    reference_words = sum(baseline[key]["words"] for key in keys)
    result = {
        "improved": sum(value < 0 for value in differences),
        "worsened": sum(value > 0 for value in differences),
        "unchanged": sum(value == 0 for value in differences),
        "delta_wer_pp": 100 * sum(differences) / reference_words,
        "largest_worsening": max(differences),
        "largest_improvement": min(differences),
    }
    if bootstrap:
        clusters = defaultdict(lambda: [0, 0])
        for key, delta in zip(keys, differences, strict=True):
            unit = key.split("-", 1)[0] if corpus == "libri" else key
            clusters[unit][0] += delta
            clusters[unit][1] += baseline[key]["words"]
        if len(clusters) >= 10:
            data = np.array(list(clusters.values()))
            rng = np.random.default_rng(116)
            totals = data[rng.integers(len(data), size=(6000, len(data)))].sum(axis=1)
            result["bootstrap_pct95_pp"] = np.percentile(100 * totals[:, 0] / totals[:, 1], (2.5, 97.5)).tolist()
            result["bootstrap_units"] = len(clusters)
    return result


if __name__ == "__main__":
    main()
