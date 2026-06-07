#!/usr/bin/env bash
set -euo pipefail

# Threshold sweep for the LatentASR value-head halting policy.
# Clean audio only, full splits by default, and HuggingFace streaming enabled
# so audio is not materialized under the workspace.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
ROOT_DIR="$(latent_asr_repo_root "${SCRIPT_DIR}")"
cd "${ROOT_DIR}"

LATENT_CKPT="${LATENT_CKPT:-./latent_qwen_asr_best.pth}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-ASR-0.6B}"
PYTHON_BIN="$(latent_asr_python_bin "${PYTHON_BIN:-}")"

MAX_SAMPLES_PER_CONFIG="${MAX_SAMPLES_PER_CONFIG:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
PRINT_SAMPLES="${PRINT_SAMPLES:-0}"
STREAMING="${STREAMING:-1}"
RUN_BASE="${RUN_BASE:-0}"
CASE_FILTER="${CASE_FILTER:-fleurs_en_us|voxpopuli_en}"
THRESHOLD_SPECS="${THRESHOLD_SPECS:-full:-2.0 neg0p2:-0.2 zero:0.0 pos0p2:0.2 pos0p5:0.5}"
RESUME="${RESUME:-1}"

latent_asr_require_file "${LATENT_CKPT}" "latent checkpoint"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/eval_runs/threshold_sweep_${TIMESTAMP}}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Format: tag|dataset|configs|split
CASES=(
  "fleurs_en_us|google/fleurs|en_us|test"
  "voxpopuli_en|facebook/voxpopuli|en|test"
)

EXTRA_ARGS=("$@")

run_case() {
  local theta_label="$1"
  local theta_value="$2"
  local tag="$3"
  local dataset="$4"
  local configs="$5"
  local split="$6"

  local json_out="${OUT_DIR}/${tag}_theta_${theta_label}.json"
  local log_out="${LOG_DIR}/${tag}_theta_${theta_label}.log"

  if [[ "${RESUME}" == "1" && -f "${json_out}" ]]; then
    echo "[skip] existing ${json_out}"
    return 0
  fi

  local args=(
    --model-id "${MODEL_ID}"
    --dataset-name "${dataset}"
    --configs "${configs}"
    --split "${split}"
    --output-json "${json_out}"
    --max-samples-per-config "${MAX_SAMPLES_PER_CONFIG}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --latent-ckpt "${LATENT_CKPT}"
    --n-latent 4
    --num-beams 1
    --dynamic-halt-threshold "${theta_value}"
    --print-samples "${PRINT_SAMPLES}"
    --skip-baseline-ft
    --skip-prompt-tuning
    --skip-lora-r16
  )

  if [[ "${RUN_BASE}" != "1" ]]; then
    args+=(--skip-base-model)
  fi
  if [[ "${STREAMING}" == "1" ]]; then
    args+=(--streaming)
  fi

  echo "------------------------------------------------------------"
  echo "Case      : ${tag}"
  echo "Dataset   : ${dataset}"
  echo "Configs   : ${configs}"
  echo "Split     : ${split}"
  echo "Theta     : ${theta_label} (${theta_value})"
  echo "Streaming : ${STREAMING}"
  echo "Run base  : ${RUN_BASE}"
  echo "JSON      : ${json_out}"
  echo "LOG       : ${log_out}"

  "${PYTHON_BIN}" eval.py "${args[@]}" "${EXTRA_ARGS[@]}" 2>&1 | tee "${log_out}"
}

echo "============================================================"
echo "LatentASR Threshold Sweep"
echo "============================================================"
echo "Root dir      : ${ROOT_DIR}"
echo "Output dir    : ${OUT_DIR}"
echo "Python        : ${PYTHON_BIN}"
echo "Model         : ${MODEL_ID}"
echo "Latent ckpt   : ${LATENT_CKPT}"
echo "Max/config    : ${MAX_SAMPLES_PER_CONFIG} (0 = full split)"
echo "Streaming     : ${STREAMING}"
echo "Run base      : ${RUN_BASE}"
echo "Case filter   : ${CASE_FILTER:-(none)}"
echo "Thresholds    : ${THRESHOLD_SPECS}"
echo "Extra args    : ${EXTRA_ARGS[*]:-(none)}"
echo "============================================================"

SUCCESS=()
FAILED=()

for spec in ${THRESHOLD_SPECS}; do
  IFS=':' read -r THETA_LABEL THETA_VALUE <<< "${spec}"
  for entry in "${CASES[@]}"; do
    IFS='|' read -r TAG DATASET CONFIGS SPLIT <<< "${entry}"
    if [[ -n "${CASE_FILTER}" && ! "${TAG}" =~ ${CASE_FILTER} ]]; then
      continue
    fi
    if run_case "${THETA_LABEL}" "${THETA_VALUE}" "${TAG}" "${DATASET}" "${CONFIGS}" "${SPLIT}"; then
      SUCCESS+=("${TAG}:${THETA_LABEL}")
    else
      FAILED+=("${TAG}:${THETA_LABEL}")
    fi
  done
done

"${PYTHON_BIN}" scripts/summarize_threshold_sweep.py "${OUT_DIR}"

echo ""
echo "============================================================"
echo "Threshold Sweep Summary"
echo "============================================================"
echo "Succeeded (${#SUCCESS[@]}):"
for item in "${SUCCESS[@]}"; do
  echo "  - ${item}"
done
echo "Failed (${#FAILED[@]}):"
for item in "${FAILED[@]}"; do
  echo "  - ${item}"
done
echo "Outputs: ${OUT_DIR}"
echo "============================================================"
