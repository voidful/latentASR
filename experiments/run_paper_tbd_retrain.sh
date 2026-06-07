#!/usr/bin/env bash
set -euo pipefail

# Retrain LatentASR variants needed by the paper TBD ablations.
# This runner intentionally saves per-epoch checkpoints. Downstream evaluation
# should use *_epoch10.pth for same-schedule comparisons unless noted otherwise.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
ROOT_DIR="$(latent_asr_repo_root "${SCRIPT_DIR}")"
cd "${ROOT_DIR}"

PYTHON_BIN="$(latent_asr_python_bin "${PYTHON_BIN:-}")"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/eval_runs/paper_tbd_retrain_${TIMESTAMP}}"
LOG_DIR="${OUT_DIR}/logs"
CKPT_DIR="${OUT_DIR}/checkpoints"
mkdir -p "${LOG_DIR}" "${CKPT_DIR}"

NUM_EPOCHS="${NUM_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
EVAL_SAMPLES="${EVAL_SAMPLES:-1}"
PRETRAIN_EVAL_SAMPLES="${PRETRAIN_EVAL_SAMPLES:-0}"
VARIANT_FILTER="${VARIANT_FILTER:-}"
DEFAULT_TRAIN_MAX="${DEFAULT_TRAIN_MAX:-0}"

# name|n_latent|train_max|p_neg|bounded|gate|anchor
VARIANTS=(
  "component_no_bounded|4|${DEFAULT_TRAIN_MAX}|0.3|0|1|1"
  "component_no_gate|4|${DEFAULT_TRAIN_MAX}|0.3|1|0|1"
  "component_no_anchor|4|${DEFAULT_TRAIN_MAX}|0.3|1|1|0"
  "n1|1|${DEFAULT_TRAIN_MAX}|0.3|1|1|1"
  "n2|2|${DEFAULT_TRAIN_MAX}|0.3|1|1|1"
  "n4|4|${DEFAULT_TRAIN_MAX}|0.3|1|1|1"
  "n8|8|${DEFAULT_TRAIN_MAX}|0.3|1|1|1"
  "pneg0|4|${DEFAULT_TRAIN_MAX}|0.0|1|1|1"
  "activation_100|4|100|0.3|1|1|1"
  "activation_200|4|200|0.3|1|1|1"
  "activation_300|4|300|0.3|1|1|1"
  "activation_400|4|400|0.3|1|1|1"
  "activation_500|4|500|0.3|1|1|1"
  "activation_600|4|600|0.3|1|1|1"
  "activation_700|4|700|0.3|1|1|1"
  "activation_800|4|800|0.3|1|1|1"
)

echo "============================================================"
echo "LatentASR paper TBD retraining"
echo "============================================================"
latent_asr_print_kv "Root" "${ROOT_DIR}"
latent_asr_print_kv "Output" "${OUT_DIR}"
latent_asr_print_kv "Python" "${PYTHON_BIN}"
latent_asr_print_kv "Epochs" "${NUM_EPOCHS}"
latent_asr_print_kv "Batch" "${BATCH_SIZE}"
latent_asr_print_kv "Grad accum" "${GRAD_ACCUM_STEPS}"
latent_asr_print_kv "Eval samples" "${EVAL_SAMPLES}"
latent_asr_print_kv "Variant filter" "${VARIANT_FILTER:-(none)}"
latent_asr_print_kv "Default train max" "${DEFAULT_TRAIN_MAX} (0 = full 811-sample activation set)"
echo "============================================================"

for spec in "${VARIANTS[@]}"; do
  IFS='|' read -r name n_latent train_max p_neg bounded gate anchor <<< "${spec}"
  if [[ -n "${VARIANT_FILTER}" && ! "${name}" =~ ${VARIANT_FILTER} ]]; then
    continue
  fi

  variant_dir="${CKPT_DIR}/${name}"
  mkdir -p "${variant_dir}"
  prefix="${variant_dir}/${name}"
  final_ckpt="${prefix}_epoch${NUM_EPOCHS}.pth"
  log_path="${LOG_DIR}/${name}.log"

  if [[ -f "${final_ckpt}" ]]; then
    echo "[skip] ${name}: found ${final_ckpt}"
    continue
  fi

  echo ""
  echo "------------------------------------------------------------"
  echo "Variant   : ${name}"
  echo "N         : ${n_latent}"
  echo "train_max : ${train_max:-0} (0 = full 811-sample activation set)"
  echo "p_neg     : ${p_neg}"
  echo "bounded   : ${bounded}"
  echo "gate      : ${gate}"
  echo "anchor    : ${anchor}"
  echo "log       : ${log_path}"
  echo "------------------------------------------------------------"

  env \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    TRAIN_MODE=latent \
    N_LATENT="${n_latent}" \
    TRAIN_MAX_SAMPLES="${train_max}" \
    VALUE_FORCED_NEG_PROB="${p_neg}" \
    LATENT_USE_BOUNDED_DELTA="${bounded}" \
    LATENT_USE_INJECTION_GATE="${gate}" \
    LATENT_USE_EMBEDDING_ANCHOR="${anchor}" \
    NUM_EPOCHS="${NUM_EPOCHS}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS}" \
    EVAL_SAMPLES="${EVAL_SAMPLES}" \
    PRETRAIN_EVAL_SAMPLES="${PRETRAIN_EVAL_SAMPLES}" \
    LOG_EVERY=25 \
    GRAD_LOG_EVERY=100 \
    CHECKPOINT_PREFIX="${prefix}" \
    "${PYTHON_BIN}" train.py 2>&1 | tee "${log_path}"
done

echo "Outputs: ${OUT_DIR}"
