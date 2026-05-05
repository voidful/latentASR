#!/usr/bin/env bash
set -euo pipefail

# Run a small-to-medium HuggingFace ASR showcase for latent reasoning.
#
# Goal:
#   1. Start with small, recognizable HF ASR datasets.
#   2. Compare raw base_model vs latent_reasoning only.
#   3. Include optional noisy SNR stress cases where LR benefits are easier to see.
#
# Fast discovery:
#   ./run_lr_hf_asr_showcase.sh
#
# Full confirmation after finding promising cases:
#   MAX_SAMPLES_PER_CONFIG=0 SNR_DB_LEVELS="5 0" ./run_lr_hf_asr_showcase.sh
#
# Filter cases:
#   CASE_FILTER='fleurs|tedlium' ./run_lr_hf_asr_showcase.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
ROOT_DIR="$(latent_asr_repo_root "${SCRIPT_DIR}")"
cd "${ROOT_DIR}"

LATENT_CKPT="${LATENT_CKPT:-./latent_qwen_asr_best.pth}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-ASR-0.6B}"
PYTHON_BIN="$(latent_asr_python_bin "${PYTHON_BIN:-}")"
DYNAMIC_HALT_THRESHOLD="${DYNAMIC_HALT_THRESHOLD:-0.0}"
MAX_SAMPLES_PER_CONFIG="${MAX_SAMPLES_PER_CONFIG:-500}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
PRINT_SAMPLES="${PRINT_SAMPLES:-0}"
RUN_CLEAN="${RUN_CLEAN:-1}"
SNR_DB_LEVELS="${SNR_DB_LEVELS-10 5 0}"
CASE_FILTER="${CASE_FILTER:-}"
RESUME="${RESUME:-1}"
SNR_LEVEL_ARRAY=()
if [[ -n "${SNR_DB_LEVELS}" ]]; then
  read -r -a SNR_LEVEL_ARRAY <<< "${SNR_DB_LEVELS}"
fi

latent_asr_require_file "${LATENT_CKPT}" "latent checkpoint"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/eval_runs/hf_asr_showcase_${TIMESTAMP}}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Format: tag|dataset|configs|split
# Ordered roughly small/quick first, then broader canonical benchmarks.
HF_ASR_CASES=(
  "fleurs_en_us|google/fleurs|en_us|test"
  "minds14_en_us|PolyAI/minds14|en-US|train"
  "tedlium_release1|TwinkStart/tedlium|release1|test"
  "voxpopuli_en|facebook/voxpopuli|en|test"
  "librispeech_clean_other|openslr/librispeech_asr|clean,other|test"
)

EXTRA_ARGS=("$@")

run_case() {
  local tag="$1"
  local dataset="$2"
  local configs="$3"
  local split="$4"
  local condition="$5"
  local snr="$6"

  local json_out="${OUT_DIR}/${tag}_${condition}.json"
  local log_out="${LOG_DIR}/${tag}_${condition}.log"

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
    --dynamic-halt-threshold "${DYNAMIC_HALT_THRESHOLD}"
    --print-samples "${PRINT_SAMPLES}"
    --skip-baseline-ft
    --skip-prompt-tuning
    --skip-lora-r16
  )

  if [[ -n "${snr}" ]]; then
    args+=(--snr-db "${snr}")
  fi

  echo "------------------------------------------------------------"
  echo "Case      : ${tag}"
  echo "Dataset   : ${dataset}"
  echo "Configs   : ${configs}"
  echo "Split     : ${split}"
  echo "Condition : ${condition}"
  echo "JSON      : ${json_out}"
  echo "LOG       : ${log_out}"

  "${PYTHON_BIN}" eval.py "${args[@]}" "${EXTRA_ARGS[@]}" 2>&1 | tee "${log_out}"
}

echo "============================================================"
echo "Latent Reasoning HuggingFace ASR Showcase"
echo "============================================================"
echo "Root dir       : ${ROOT_DIR}"
echo "Output dir     : ${OUT_DIR}"
echo "Python         : ${PYTHON_BIN}"
echo "Model          : ${MODEL_ID}"
echo "Latent ckpt    : ${LATENT_CKPT}"
echo "Halt threshold : ${DYNAMIC_HALT_THRESHOLD}"
echo "Max/config     : ${MAX_SAMPLES_PER_CONFIG}"
echo "Clean pass     : ${RUN_CLEAN}"
echo "SNR levels     : ${SNR_DB_LEVELS:-(none)}"
echo "Case filter    : ${CASE_FILTER:-(none)}"
echo "Extra args     : ${EXTRA_ARGS[*]:-(none)}"
echo "============================================================"

SUCCESS=()
FAILED=()

for entry in "${HF_ASR_CASES[@]}"; do
  IFS='|' read -r TAG DATASET CONFIGS SPLIT <<< "${entry}"

  if [[ -n "${CASE_FILTER}" && ! "${TAG}" =~ ${CASE_FILTER} ]]; then
    continue
  fi

  if [[ "${RUN_CLEAN}" == "1" ]]; then
    if run_case "${TAG}" "${DATASET}" "${CONFIGS}" "${SPLIT}" "clean" ""; then
      SUCCESS+=("${TAG}:clean")
    else
      FAILED+=("${TAG}:clean")
    fi
  fi

  for SNR in "${SNR_LEVEL_ARRAY[@]}"; do
    CONDITION="snr${SNR}db"
    if run_case "${TAG}" "${DATASET}" "${CONFIGS}" "${SPLIT}" "${CONDITION}" "${SNR}"; then
      SUCCESS+=("${TAG}:${CONDITION}")
    else
      FAILED+=("${TAG}:${CONDITION}")
    fi
  done
done

echo ""
echo "============================================================"
echo "Showcase Summary"
echo "============================================================"
echo "Succeeded (${#SUCCESS[@]}):"
for item in "${SUCCESS[@]}"; do
  echo "  - ${item}"
done
echo "Failed (${#FAILED[@]}):"
for item in "${FAILED[@]}"; do
  echo "  - ${item}"
done

"${PYTHON_BIN}" scripts/summarize_lr_showcase.py "${OUT_DIR}"
echo "Outputs: ${OUT_DIR}"
echo "============================================================"
