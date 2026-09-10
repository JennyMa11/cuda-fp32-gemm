#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${project_dir}/results"

make -C "${project_dir}" gemm
"${project_dir}/build/gemm_benchmark" \
    --warmup "${WARMUP:-5}" \
    --iters "${ITERS:-20}" \
    --csv "${project_dir}/results/gemm_results.csv" \
    "$@"
