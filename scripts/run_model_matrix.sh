#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "Usage: $0 MODEL OUTPUT_ROOT FAKEDDIT_ROOT IFND_ROOT MMFAKEBENCH_ROOT WEIBO_ROOT DRIFTBENCH_ROOT" >&2
  exit 2
fi

model=$1
output_root=$2
shift 2

datasets=(Fakeddit IFND MMFakeBench Weibo DriftBench)
roots=("$@")

extra_args=()
if [[ -n "${MODEL_PATH:-}" ]]; then
  extra_args+=(--model-path "$MODEL_PATH")
fi
if [[ -n "${REPO_PATH:-}" ]]; then
  extra_args+=(--repo-path "$REPO_PATH")
fi
if [[ -n "${MODEL_REVISION:-}" ]]; then
  extra_args+=(--revision "$MODEL_REVISION")
fi

for index in "${!datasets[@]}"; do
  dataset=${datasets[$index]}
  root=${roots[$index]}
  litevlm-fnd run \
    --model "$model" \
    --annotations "$root/source/${dataset}_test.json" \
    --data-root "$root" \
    --output-dir "$output_root/$model/${dataset}_test" \
    --gpu-id "${GPU_ID:-0}" \
    "${extra_args[@]}"
done
