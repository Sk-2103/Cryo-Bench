#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR="${RESULTS_DIR:-/media/turtle-ssd/users/skaushik/GLB_data_results/pangaea}"
PYTHON_BIN="${PYTHON_BIN:-/home/skaushik/anaconda3/envs/prithvi/bin/python}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
TEST_NUM_WORKERS="${TEST_NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-80}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5}"
CKPT_INTERVAL="${CKPT_INTERVAL:-20}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
START_AT="${START_AT:-}"

MODELS=(
  croma_joint
  croma_optical
  croma_sar
  dofa
  gfmswin
  prithvi
  prithvi_eo_v2_300
  remoteclip
  satlasnet_mi
  satlasnet_si
  scalemae
  spectralgpt
  ssl4eo_data2vec
  ssl4eo_dino
  ssl4eo_mae_optical
  ssl4eo_mae_sar
  ssl4eo_moco
  terramind_large
  unet_encoder
  vit_scratch
)
if [[ -n "${GLB_MODELS:-}" ]]; then
  read -r -a MODELS <<< "${GLB_MODELS}"
fi

mkdir -p "${RESULTS_DIR}/runs" "${RESULTS_DIR}/logs"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"

start_seen=false
if [[ -z "${START_AT}" ]]; then
  start_seen=true
fi

for encoder in "${MODELS[@]}"; do
  if [[ "${start_seen}" == false ]]; then
    if [[ "${encoder}" == "${START_AT}" ]]; then
      start_seen=true
    else
      continue
    fi
  fi

  finetune=false
  case "${encoder}" in
    unet_encoder|unet_encoder_mi|vit_scratch)
      finetune=true
      ;;
  esac
  decoder=seg_upernet
  case "${encoder}" in
    unet_encoder|unet_encoder_mi)
      decoder=seg_unet
      ;;
  esac
  stamp="$(date +%Y%m%d_%H%M%S)"
  log="${RESULTS_DIR}/logs/${stamp}_${encoder}.log"
  echo "[$(date --iso-8601=seconds)] starting ${encoder} decoder=${decoder} finetune=${finetune}" | tee -a "${RESULTS_DIR}/logs/launcher.log"
  set +e
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NPROC_PER_NODE}" \
    pangaea/run.py \
    --config-name=train \
    dataset=glb_random \
    encoder="${encoder}" \
    decoder="${decoder}" \
    preprocessing=seg_default \
    criterion=cross_entropy \
    task=segmentation \
    work_dir="${RESULTS_DIR}/runs" \
    finetune="${finetune}" \
    batch_size="${BATCH_SIZE}" \
    test_batch_size="${TEST_BATCH_SIZE}" \
    num_workers="${NUM_WORKERS}" \
    test_num_workers="${TEST_NUM_WORKERS}" \
    task.trainer.n_epochs="${EPOCHS}" \
    task.trainer.eval_interval="${EVAL_INTERVAL}" \
    task.trainer.ckpt_interval="${CKPT_INTERVAL}" \
    use_wandb=false >"${log}" 2>&1
  status=$?
  set -e
  echo "[$(date --iso-8601=seconds)] finished ${encoder} status=${status}" | tee -a "${RESULTS_DIR}/logs/launcher.log"
  "${PYTHON_BIN}" scripts/collect_glb_pangaea_results.py --results-dir "${RESULTS_DIR}" | tee -a "${RESULTS_DIR}/logs/launcher.log"
done
