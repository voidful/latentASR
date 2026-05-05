#!/usr/bin/env bash
set -euo pipefail

# Multilingual ASR evaluation aligned with Qwen3-ASR public benchmark families.
#
# This runner always passes --streaming so HuggingFace datasets are read lazily
# instead of materializing full audio splits under the local cache.
#
# Default run:
#   ./run_lr_multilingual_asr_streaming.sh
#
# Run every configured case:
#   CASE_FILTER='' ./run_lr_multilingual_asr_streaming.sh
#
# Quick smoke test:
#   MAX_SAMPLES_PER_CONFIG=20 CASE_FILTER='fleurs_core12' ./run_lr_multilingual_asr_streaming.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
ROOT_DIR="$(latent_asr_repo_root "${SCRIPT_DIR}")"
cd "${ROOT_DIR}"

LATENT_CKPT="${LATENT_CKPT:-./latent_qwen_asr_best.pth}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-ASR-0.6B}"
PYTHON_BIN="$(latent_asr_python_bin "${PYTHON_BIN:-}")"

DYNAMIC_HALT_THRESHOLD="${DYNAMIC_HALT_THRESHOLD:-0.0}"
MAX_SAMPLES_PER_CONFIG="${MAX_SAMPLES_PER_CONFIG:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
PRINT_SAMPLES="${PRINT_SAMPLES:-0}"
TEXT_NORMALIZER="${TEXT_NORMALIZER:-basic}"
RESUME="${RESUME:-1}"

# Default to the 30-language FLEURS grouping from the Qwen3-ASR card.
# Set CASE_FILTER='' to also run the public MLS case below.
CASE_FILTER="${CASE_FILTER:-fleurs_core12|fleurs_extra8|fleurs_extra10}"

latent_asr_require_file "${LATENT_CKPT}" "latent checkpoint"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/eval_runs/multilingual_asr_streaming_${TIMESTAMP}}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Format: tag|dataset|configs|split
# Qwen3-ASR public multilingual ASR benchmark families include FLEURS,
# CommonVoice, MLS, and MLC-SLM. FLEURS is the broadest open HF target here.
# HuggingFace facebook/multilingual_librispeech currently exposes seven
# non-English MLS configs; English is covered separately by LibriSpeech runs.
MULTILINGUAL_CASES=(
  "fleurs_core12|google/fleurs|en,zh,yue,ar,de,es,fr,it,ja,ko,pt,ru|test"
  "fleurs_extra8|google/fleurs|hi,id,ms,nl,pl,th,tr,vi|test"
  "fleurs_extra10|google/fleurs|cs,da,el,fa,fi,fil,hu,mk,ro,sv|test"
  "mls_public7|facebook/multilingual_librispeech|de,nl,es,fr,it,pl,pt|test"
)

EXTRA_ARGS=("$@")

run_case() {
  local tag="$1"
  local dataset="$2"
  local configs="$3"
  local split="$4"

  local json_out="${OUT_DIR}/${tag}_streaming_clean.json"
  local log_out="${LOG_DIR}/${tag}_streaming_clean.log"

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
    --text-normalizer "${TEXT_NORMALIZER}"
    --streaming
    --skip-baseline-ft
    --skip-prompt-tuning
    --skip-lora-r16
  )

  echo "------------------------------------------------------------"
  echo "Case      : ${tag}"
  echo "Dataset   : ${dataset}"
  echo "Configs   : ${configs}"
  echo "Split     : ${split}"
  echo "Streaming : yes"
  echo "JSON      : ${json_out}"
  echo "LOG       : ${log_out}"

  "${PYTHON_BIN}" eval.py "${args[@]}" "${EXTRA_ARGS[@]}" 2>&1 | tee "${log_out}"
}

echo "============================================================"
echo "Latent Reasoning Multilingual ASR Streaming"
echo "============================================================"
echo "Root dir        : ${ROOT_DIR}"
echo "Output dir      : ${OUT_DIR}"
echo "Python          : ${PYTHON_BIN}"
echo "Model           : ${MODEL_ID}"
echo "Latent ckpt     : ${LATENT_CKPT}"
echo "Halt threshold  : ${DYNAMIC_HALT_THRESHOLD}"
echo "Max/config      : ${MAX_SAMPLES_PER_CONFIG} (0 = full split)"
echo "Text normalizer : ${TEXT_NORMALIZER}"
echo "Case filter     : ${CASE_FILTER:-(none)}"
echo "Extra args      : ${EXTRA_ARGS[*]:-(none)}"
echo "============================================================"

SUCCESS=()
FAILED=()

for entry in "${MULTILINGUAL_CASES[@]}"; do
  IFS='|' read -r TAG DATASET CONFIGS SPLIT <<< "${entry}"

  if [[ -n "${CASE_FILTER}" && ! "${TAG}" =~ ${CASE_FILTER} ]]; then
    continue
  fi

  if run_case "${TAG}" "${DATASET}" "${CONFIGS}" "${SPLIT}"; then
    SUCCESS+=("${TAG}:streaming_clean")
  else
    FAILED+=("${TAG}:streaming_clean")
  fi
done

echo ""
echo "============================================================"
echo "Multilingual Streaming Summary"
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
