# Expanded matched CPU study
- 200 records from each pinned dev parquet split, 400 distinct total. Sample without replacement with seeds 20261125/20261126, excluding all 32 prior records per corpus. Confirm previous row indices against text and LibriSpeech IDs.
- Freeze duration policy from the exploratory pass: at most 9.5 seconds => context 512; at most 19.5 => 1024; above => full. Use the stock Q4_0 from the earlier study, CPU M3 Pro, 4 threads, same greedy temperature-zero no-fallback no-timestamps decoder.
- Four conditions: full/adaptive × flash enabled/disabled. For each preset process the same batches of 25, rotating condition order, alternating corpus order. In adaptive conditions start one CLI per present context per batch. Context stays fixed within a process.
- Capture per-utterance wall time around whisper_full_parallel in the exact CLI upstream source. Use the original libraries and compiler flags. Measure complete subprocess wall time separately.
- Report corpus/stratum WER, error counts, median and p95 inference time, total elapsed, paired WER deltas, and error concentration. No test splits. Choose a candidate only after reviewing measured speed and accuracy.
