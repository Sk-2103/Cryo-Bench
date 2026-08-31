#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR="${RESULTS_DIR:-/media/turtle-ssd/users/skaushik/GLB_data_results/pangaea}"
PYTHON_BIN="${PYTHON_BIN:-/home/skaushik/anaconda3/envs/pangaea-bench/bin/python}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRITHVI_CKPT="${PRITHVI_CKPT:-${RESULTS_DIR}/runs/20260508_074320_ddf794_prithvi_seg_upernet_glb_random/checkpoint_60.pth}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
TEST_NUM_WORKERS="${TEST_NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-80}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5}"
CKPT_INTERVAL="${CKPT_INTERVAL:-20}"
REMAINING_MODELS="${GLB_MODELS:-croma_sar dofa gfmswin prithvi_eo_v2_300 remoteclip satlasnet_mi satlasnet_si scalemae spectralgpt ssl4eo_data2vec ssl4eo_dino ssl4eo_mae_optical ssl4eo_mae_sar ssl4eo_moco terramind_large unet_encoder vit_scratch}"

mkdir -p "${RESULTS_DIR}/runs" "${RESULTS_DIR}/logs"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"

orchestrator_log="${RESULTS_DIR}/logs/resume_then_remaining.log"
echo "[$(date --iso-8601=seconds)] starting prithvi resume from ${PRITHVI_CKPT}" | tee -a "${orchestrator_log}"

set +e
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  pangaea/run.py \
  --config-name=train \
  dataset=glb_random \
  encoder=prithvi \
  decoder=seg_upernet \
  preprocessing=seg_default \
  criterion=cross_entropy \
  task=segmentation \
  work_dir="${RESULTS_DIR}/runs" \
  finetune=false \
  batch_size="${BATCH_SIZE}" \
  test_batch_size="${TEST_BATCH_SIZE}" \
  num_workers="${NUM_WORKERS}" \
  test_num_workers="${TEST_NUM_WORKERS}" \
  task.trainer.n_epochs="${EPOCHS}" \
  task.trainer.eval_interval="${EVAL_INTERVAL}" \
  task.trainer.ckpt_interval="${CKPT_INTERVAL}" \
  use_wandb=false \
  ckpt_dir="${PRITHVI_CKPT}" >"${RESULTS_DIR}/logs/$(date +%Y%m%d_%H%M%S)_prithvi_resume.log" 2>&1
resume_status=$?
set -e

echo "[$(date --iso-8601=seconds)] finished prithvi resume status=${resume_status}" | tee -a "${orchestrator_log}"
"${PYTHON_BIN}" scripts/collect_glb_pangaea_results.py --results-dir "${RESULTS_DIR}" | tee -a "${orchestrator_log}"

echo "[$(date --iso-8601=seconds)] starting remaining GLB encoders: ${REMAINING_MODELS}" | tee -a "${orchestrator_log}"
GLB_MODELS="${REMAINING_MODELS}" \
RESULTS_DIR="${RESULTS_DIR}" \
PYTHON_BIN="${PYTHON_BIN}" \
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
BATCH_SIZE="${BATCH_SIZE}" \
TEST_BATCH_SIZE="${TEST_BATCH_SIZE}" \
NUM_WORKERS="${NUM_WORKERS}" \
TEST_NUM_WORKERS="${TEST_NUM_WORKERS}" \
EPOCHS="${EPOCHS}" \
EVAL_INTERVAL="${EVAL_INTERVAL}" \
CKPT_INTERVAL="${CKPT_INTERVAL}" \
  scripts/launch_glb_pangaea_benchmark.sh

echo "[$(date --iso-8601=seconds)] unattended GLB sequence finished" | tee -a "${orchestrator_log}"
