#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/skaushik/anaconda3/bin/python}"
RUNS_ROOT="${1:-/media/turtle-ssd/users/skaushik/Cryo-Data/Benchmark/SICD/results}"
DATASET_ROOT="${2:-/media/turtle-ssd/users/skaushik/Cryo-Data/Benchmark/SICD}"
OUTPUT_ROOT="${3:-/media/turtle-ssd/users/skaushik/Cryo-Data/Benchmark/SICD/predictions_by_modelusion}"
DEVICE="${DEVICE:-cuda}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

"${PYTHON_BIN}" "$(dirname "$0")/export_sicd_predictions.py" \
  --runs-root "${RUNS_ROOT}" \
  --dataset-root "${DATASET_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --device "${DEVICE}"
