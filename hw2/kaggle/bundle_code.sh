#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

bundle_dir=artifacts/kaggle/code_bundle
mkdir -p "$bundle_dir"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) git:$(git rev-parse --short HEAD) uncommitted:$(git status --porcelain . | wc -l | tr -d ' ')" > kaggle/code_version.txt
COPYFILE_DISABLE=1 tar --exclude='.git' --exclude='__pycache__' --exclude='kernels' -czf "$bundle_dir/code.tar.gz" -C .. \
  hw2/configs hw2/src hw2/kaggle hw2/third_party/DroPE
cat > "$bundle_dir/dataset-metadata.json" <<'JSON'
{"title": "drope-code", "id": "lild1tz/drope-code", "licenses": [{"name": "CC0-1.0"}]}
JSON
cat kaggle/code_version.txt
