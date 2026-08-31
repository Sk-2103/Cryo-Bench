#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/skaushik/RAMEN/pangaea-bench}"
RESULTS_DIR="${RESULTS_DIR:-/media/turtle-ssd/users/skaushik/GLB_data_results/pangaea}"
PYTHON_BIN="${PYTHON_BIN:-/home/skaushik/anaconda3/envs/pangaea-bench/bin/python}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
WATCH_LOG="${RESULTS_DIR}/logs/glb_resume_watchdog.log"
OLD_SESSION="${OLD_SESSION:-glb_pangaea}"
NEW_SESSION="${NEW_SESSION:-glb_pangaea_updated}"

RESUME_MODELS="${GLB_MODELS:-croma_sar dofa gfmswin prithvi_eo_v2_300 remoteclip satlasnet_mi satlasnet_si scalemae spectralgpt ssl4eo_data2vec ssl4eo_dino ssl4eo_mae_optical ssl4eo_mae_sar ssl4eo_moco terramind_large unet_encoder vit_scratch}"

mkdir -p "${RESULTS_DIR}/logs"
echo "[$(date --iso-8601=seconds)] watchdog started; waiting for current prithvi to finish" | tee -a "${WATCH_LOG}"

while true; do
  if grep -q "finished prithvi status=" "${RESULTS_DIR}/logs/launcher.log" 2>/dev/null; then
    break
  fi
  if ! nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null | grep -q "/envs/prithvi/bin/python"; then
    break
  fi
  sleep 30
done

echo "[$(date --iso-8601=seconds)] prithvi appears complete; stopping stale launcher if present" | tee -a "${WATCH_LOG}"
if tmux has-session -t "${OLD_SESSION}" 2>/dev/null; then
  tmux send-keys -t "${OLD_SESSION}" C-c || true
  sleep 10
fi

if tmux has-session -t "${NEW_SESSION}" 2>/dev/null; then
  echo "[$(date --iso-8601=seconds)] ${NEW_SESSION} already exists; not starting duplicate" | tee -a "${WATCH_LOG}"
  exit 0
fi

echo "[$(date --iso-8601=seconds)] starting updated unattended GLB run: ${RESUME_MODELS}" | tee -a "${WATCH_LOG}"
tmux new-session -d -s "${NEW_SESSION}" \
  "cd '${REPO_DIR}' && export PYTHONPATH='${REPO_DIR}':\${PYTHONPATH:-} MPLCONFIGDIR=/tmp/mpl PYTHON_BIN='${PYTHON_BIN}' RESULTS_DIR='${RESULTS_DIR}' CUDA_VISIBLE_DEVICES='${CUDA_DEVICES}' GLB_MODELS='${RESUME_MODELS}' && bash scripts/launch_glb_pangaea_benchmark.sh >> '${RESULTS_DIR}/logs/glb_pangaea_updated.log' 2>&1"

echo "[$(date --iso-8601=seconds)] launched ${NEW_SESSION}" | tee -a "${WATCH_LOG}"
