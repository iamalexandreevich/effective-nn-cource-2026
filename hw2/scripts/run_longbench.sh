#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src:third_party/DroPE
snapshot=${1:-artifacts/kaggle/main/snapshots/step-1000}
samples=${2:-60}

run() {
  .venv/bin/python src/longbench_eval.py --samples "$samples" "$@" 2>&1 | grep -E "^[a-z_0-9]+ [a-z_]+: |Error|Traceback"
}

run --method drope --scale 4 --beta-coefficient 0.2 --checkpoint "$snapshot" --name drope_step1000_s4
run --method yarn --scale 4 --name yarn_s4
run --method ntk_dynamic --scale 4 --name ntk_dynamic_s4
run --method ntk --scale 4 --name ntk_s4
run --method base --scale 1 --name base_s1
echo LONGBENCH_DONE
