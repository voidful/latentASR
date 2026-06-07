#!/usr/bin/env bash
set -euo pipefail

# Evaluate checkpoints produced by run_paper_tbd_retrain.sh for the TBD paper
# tables. This script intentionally does not override --n-latent; eval.py reads
# the latent budget from each checkpoint, which is required for the N sweep.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
ROOT_DIR="$(latent_asr_repo_root "${SCRIPT_DIR}")"
cd "${ROOT_DIR}"

PYTHON_BIN="$(latent_asr_python_bin "${PYTHON_BIN:-}")"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-ASR-0.6B}"
CKPT_ROOT="${CKPT_ROOT:-${ROOT_DIR}/eval_runs/paper_tbd_retrain_20260518/checkpoints}"
BASELINE_DIR="${BASELINE_DIR:-${ROOT_DIR}/eval_runs/hf_asr_showcase_full_20260503_152506}"
N4_CKPT="${N4_CKPT:-}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/eval_runs/paper_tbd_eval_${TIMESTAMP}}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

MAX_SAMPLES_PER_CONFIG="${MAX_SAMPLES_PER_CONFIG:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
PRINT_SAMPLES="${PRINT_SAMPLES:-0}"
STREAMING="${STREAMING:-1}"
RESUME="${RESUME:-1}"

# Optional filters are regexes over the variant/dataset/theta tags.
VARIANT_FILTER="${VARIANT_FILTER:-}"
DATASET_FILTER="${DATASET_FILTER:-}"
THETA_FILTER="${THETA_FILTER:-}"

run_eval() {
  local variant="$1"
  local ckpt="$2"
  local dataset_tag="$3"
  local dataset_name="$4"
  local configs="$5"
  local theta_label="$6"
  local theta_value="$7"

  if [[ -n "${VARIANT_FILTER}" && ! "${variant}" =~ ${VARIANT_FILTER} ]]; then
    return 0
  fi
  if [[ -n "${DATASET_FILTER}" && ! "${dataset_tag}" =~ ${DATASET_FILTER} ]]; then
    return 0
  fi
  if [[ -n "${THETA_FILTER}" && ! "${theta_label}" =~ ${THETA_FILTER} ]]; then
    return 0
  fi

  latent_asr_require_file "${ckpt}" "checkpoint for ${variant}"

  local json_out="${OUT_DIR}/${variant}_${dataset_tag}_theta_${theta_label}.json"
  local log_out="${LOG_DIR}/${variant}_${dataset_tag}_theta_${theta_label}.log"
  if [[ "${RESUME}" == "1" && -f "${json_out}" ]]; then
    echo "[skip] existing ${json_out}"
    return 0
  fi

  local args=(
    --model-id "${MODEL_ID}"
    --dataset-name "${dataset_name}"
    --configs "${configs}"
    --split test
    --output-json "${json_out}"
    --max-samples-per-config "${MAX_SAMPLES_PER_CONFIG}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --latent-ckpt "${ckpt}"
    --num-beams 1
    --dynamic-halt-threshold "${theta_value}"
    --print-samples "${PRINT_SAMPLES}"
    --skip-base-model
    --skip-baseline-ft
    --skip-prompt-tuning
    --skip-lora-r16
  )
  if [[ "${STREAMING}" == "1" ]]; then
    args+=(--streaming)
  fi

  echo "------------------------------------------------------------"
  echo "Variant   : ${variant}"
  echo "Dataset   : ${dataset_tag} (${dataset_name}/${configs})"
  echo "Theta     : ${theta_label} (${theta_value})"
  echo "Checkpoint: ${ckpt}"
  echo "JSON      : ${json_out}"
  echo "LOG       : ${log_out}"
  "${PYTHON_BIN}" eval.py "${args[@]}" 2>&1 | tee "${log_out}"
}

ckpt_epoch10() {
  local variant="$1"
  printf '%s/%s/%s_epoch10.pth' "${CKPT_ROOT}" "${variant}" "${variant}"
}
if [[ -z "${N4_CKPT}" ]]; then
  N4_CKPT="$(ckpt_epoch10 n4)"
fi

echo "============================================================"
echo "LatentASR TBD Paper Evaluation"
echo "============================================================"
echo "Root dir       : ${ROOT_DIR}"
echo "Checkpoint dir : ${CKPT_ROOT}"
echo "Output dir     : ${OUT_DIR}"
echo "Baseline dir   : ${BASELINE_DIR}"
echo "N=4 ckpt       : ${N4_CKPT}"
echo "Python         : ${PYTHON_BIN}"
echo "Streaming      : ${STREAMING}"
echo "Max/config     : ${MAX_SAMPLES_PER_CONFIG}"
echo "Variant filter : ${VARIANT_FILTER:-(none)}"
echo "Dataset filter : ${DATASET_FILTER:-(none)}"
echo "Theta filter   : ${THETA_FILTER:-(none)}"
echo "============================================================"

# Component ablation: FLEURS only, deployed threshold.
for variant in component_no_bounded component_no_gate component_no_anchor; do
  run_eval "${variant}" "$(ckpt_epoch10 "${variant}")" \
    fleurs google/fleurs en_us zero 0.0
done

# N sweep: FLEURS and VoxPopuli, deployed threshold.
for variant in n1 n2 n8; do
  ckpt="$(ckpt_epoch10 "${variant}")"
  run_eval "${variant}" "${ckpt}" fleurs google/fleurs en_us zero 0.0
  run_eval "${variant}" "${ckpt}" voxpopuli facebook/voxpopuli en zero 0.0
done
run_eval n4 "${N4_CKPT}" fleurs google/fleurs en_us zero 0.0
run_eval n4 "${N4_CKPT}" voxpopuli facebook/voxpopuli en zero 0.0

# Forced-negative ablation: FLEURS threshold sweep.
for spec in full:-2.0 neg0p2:-0.2 zero:0.0 pos0p2:0.2 pos0p5:0.5; do
  IFS=':' read -r theta_label theta_value <<< "${spec}"
  run_eval pneg0 "$(ckpt_epoch10 pneg0)" \
    fleurs google/fleurs en_us "${theta_label}" "${theta_value}"
done

# Activation set scaling.
for variant in \
  activation_100 \
  activation_200 \
  activation_300 \
  activation_400 \
  activation_500 \
  activation_600 \
  activation_700 \
  activation_800
do
  ckpt="$(ckpt_epoch10 "${variant}")"
  run_eval "${variant}" "${ckpt}" fleurs google/fleurs en_us zero 0.0
  run_eval "${variant}" "${ckpt}" voxpopuli facebook/voxpopuli en zero 0.0
done

"${PYTHON_BIN}" scripts/summarize_paper_tbd.py "${OUT_DIR}" --baseline-dir "${BASELINE_DIR}"

echo "Outputs: ${OUT_DIR}"
