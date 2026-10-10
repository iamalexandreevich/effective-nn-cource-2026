#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src:third_party/DroPE
snapshots_dir=${1:-artifacts/kaggle/main/snapshots}

for snapshot in $(ls -d "$snapshots_dir"/step-* | sort -t- -k2 -nr); do
  step=${snapshot##*-}
  tune_file=results/eval/beta_step${step}.json
  [ -f "$tune_file" ] || .venv/bin/python src/tune_beta.py --checkpoint "$snapshot" --output "$tune_file"
  for scale in 2 4; do
    coefficient=$(python3 -c "import json; print(json.load(open('$tune_file'))['best_coefficient']['$scale.0'])")
    result=results/eval/drope_step${step}_s${scale}.json
    [ -f "$result" ] || .venv/bin/python src/long_context_eval.py --method drope --checkpoint "$snapshot" \
      --beta-coefficient "$coefficient" --scales "$scale" --name "drope_step${step}"
  done
  [ -f "results/eval/drope_step${step}_s1.json" ] || .venv/bin/python src/long_context_eval.py --method drope \
    --checkpoint "$snapshot" --scales 1 --name "drope_step${step}"
done
