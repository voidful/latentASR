#!/usr/bin/env bash
# ==============================================================================
# run_all_modes.sh — 依序跑 LoRA、Prompt Tuning、Baseline Fine-tune、Latent Reasoning
#
# 每個 mode 獨立訓練，checkpoint 存在各自的目錄。
# 用法：
#   bash run_all_modes.sh              # 用預設設定
#   NUM_EPOCHS=5 bash run_all_modes.sh  # 自訂 epoch 數
# ==============================================================================
set -euo pipefail

# ---------- 共用設定（可透過環境變數覆蓋）----------
export MODEL_ID="${MODEL_ID:-Qwen/Qwen3-ASR-0.6B}"
export DATASET_NAME="${DATASET_NAME:-SpeechTest/extreme_asr_pony}"
export DATASET_CONFIG="${DATASET_CONFIG:-default}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export NUM_EPOCHS="${NUM_EPOCHS:-10}"
export EVAL_SAMPLES="${EVAL_SAMPLES:-1000}"
export PRETRAIN_EVAL_SAMPLES="${PRETRAIN_EVAL_SAMPLES:-10}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
ROOT_DIR="$(latent_asr_repo_root "${SCRIPT_DIR}")"
cd "${ROOT_DIR}"
PYTHON_BIN="$(latent_asr_python_bin "${PYTHON_BIN:-}")"
TRAIN_PY="${ROOT_DIR}/train.py"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT_DIR}/logs/${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

echo "=============================================="
echo " Sequential Training — ${TIMESTAMP}"
echo "  Model:   ${MODEL_ID}"
echo "  Dataset:  ${DATASET_NAME} / ${DATASET_CONFIG}"
echo "  Epochs:  ${NUM_EPOCHS}"
echo "  Logs:    ${LOG_DIR}"
echo "=============================================="

# ---------- 1. LoRA (rank=16) ----------
echo ""
echo ">>> [1/4] Training: lora_r16"
echo "----------------------------------------------"
export TRAIN_MODE=lora_r16
LOG_FILE="${LOG_DIR}/lora_r16.log"

"${PYTHON_BIN}" "${TRAIN_PY}" 2>&1 | tee "${LOG_FILE}"
LORA_EXIT=$?

if [ ${LORA_EXIT} -ne 0 ]; then
    echo "[FAIL] lora_r16 exited with code ${LORA_EXIT}"
else
    echo "[DONE] lora_r16 completed successfully"
fi

# ---------- 2. Prompt Tuning ----------
echo ""
echo ">>> [2/4] Training: prompt_tuning"
echo "----------------------------------------------"
export TRAIN_MODE=prompt_tuning
LOG_FILE="${LOG_DIR}/prompt_tuning.log"

"${PYTHON_BIN}" "${TRAIN_PY}" 2>&1 | tee "${LOG_FILE}"
PT_EXIT=$?

if [ ${PT_EXIT} -ne 0 ]; then
    echo "[FAIL] prompt_tuning exited with code ${PT_EXIT}"
else
    echo "[DONE] prompt_tuning completed successfully"
fi

# ---------- 3. Baseline Fine-tune ----------
echo ""
echo ">>> [3/4] Training: baseline (full fine-tune)"
echo "----------------------------------------------"
export TRAIN_MODE=baseline
LOG_FILE="${LOG_DIR}/baseline.log"

"${PYTHON_BIN}" "${TRAIN_PY}" 2>&1 | tee "${LOG_FILE}"
BL_EXIT=$?

if [ ${BL_EXIT} -ne 0 ]; then
    echo "[FAIL] baseline exited with code ${BL_EXIT}"
else
    echo "[DONE] baseline completed successfully"
fi

# ---------- 4. Latent Reasoning ----------
echo ""
echo ">>> [4/4] Training: latent (latent reasoning)"
echo "----------------------------------------------"
export TRAIN_MODE=latent
LOG_FILE="${LOG_DIR}/latent.log"

"${PYTHON_BIN}" "${TRAIN_PY}" 2>&1 | tee "${LOG_FILE}"
LR_EXIT=$?

if [ ${LR_EXIT} -ne 0 ]; then
    echo "[FAIL] latent exited with code ${LR_EXIT}"
else
    echo "[DONE] latent completed successfully"
fi

# ---------- Summary ----------
echo ""
echo "=============================================="
echo " Training Summary"
echo "=============================================="
echo "  lora_r16:      exit=${LORA_EXIT:-skipped}"
echo "  prompt_tuning: exit=${PT_EXIT:-skipped}"
echo "  baseline:      exit=${BL_EXIT}"
echo "  latent:        exit=${LR_EXIT}"
echo "  Logs: ${LOG_DIR}/"
echo "=============================================="

# Print best WER from each log if available
for mode in lora_r16 prompt_tuning baseline latent; do
    logf="${LOG_DIR}/${mode}.log"
    if [ -f "${logf}" ]; then
        best=$(grep -i "best.*wer" "${logf}" | tail -1 || true)
        if [ -n "${best}" ]; then
            echo "  ${mode}: ${best}"
        fi
    fi
done
