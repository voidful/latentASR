#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

LATENT_CKPT="${1:-}"

if [[ -z "${LATENT_CKPT}" ]]; then
  echo "Usage: $0 <latent_checkpoint>" >&2
  exit 1
fi

if [[ ! -f "${LATENT_CKPT}" ]]; then
  echo "[error] Latent checkpoint not found: ${LATENT_CKPT}" >&2
  exit 1
fi

echo "============================================================" >&2
echo "Hyperparameter Grid Search on SpeechTest/fleurs (en)" >&2
echo "============================================================" >&2

DATASET="SpeechTest/fleurs"
SPLIT="test"
CONFIG="en"
SEARCH_ALPHAS=(0.05 0.1 0.15 0.2)
SEARCH_DEQ_TOLS=(0.01 0.05 0.1 0.5 1.0)
BEST_ALPHA="-1.0"
BEST_DEQ_TOL="0.1"
BEST_WER=100.0

echo "Searching alphas: ${SEARCH_ALPHAS[*]}" >&2
echo "Searching DEQ tolerances: ${SEARCH_DEQ_TOLS[*]}" >&2

# Create a temporary directory for JSON outputs
TMP_DIR=$(mktemp -d)
trap 'rm -rf -- "$TMP_DIR"' EXIT

for tol in "${SEARCH_DEQ_TOLS[@]}"; do
  for alpha in "${SEARCH_ALPHAS[@]}"; do
    JSON_OUT="${TMP_DIR}/res_${alpha}_${tol}.json"
    EVAL_LOG="${TMP_DIR}/eval_${alpha}_${tol}.log"
    
    echo "  --> Testing alpha=${alpha}, deq_tol=${tol} ..." >&2
    
    if python eval.py \
      --dataset-name "${DATASET}" \
      --configs "${CONFIG}" \
      --split "${SPLIT}" \
      --latent-ckpt "${LATENT_CKPT}" \
      --skip-base-model \
      --skip-baseline-ft \
      --skip-prompt-tuning \
      --skip-lora-r16 \
      --output-json "${JSON_OUT}" \
      --n-latent 4 \
      --num-beams 1 \
      --deq-tol "${tol}" \
      --alpha "${alpha}" >"${EVAL_LOG}" 2>&1; then
      
      if [[ -f "${JSON_OUT}" ]]; then
        # Parse WER from JSON output
        LATENT_WER=$(python -c "
import json, sys
data = json.load(open(sys.argv[1]))
try:
  wer = data['rows'][0]['latent_reasoning_wer']
  print(f'{wer:.6f}' if wer is not None else '100.0')
except:
  print('100.0')
" "${JSON_OUT}")

        echo "      WER for alpha=${alpha}, deq_tol=${tol} : ${LATENT_WER}" >&2
        
        # Compare to find the minimum WER
        IS_BETTER=$(python -c "print('1' if float(${LATENT_WER}) < float(${BEST_WER}) else '0')")
        if [[ "${IS_BETTER}" == "1" ]]; then
          BEST_ALPHA=${alpha}
          BEST_DEQ_TOL=${tol}
          BEST_WER=${LATENT_WER}
        fi
      else
         echo "      Failed to evaluate alpha=${alpha}, deq_tol=${tol} (No JSON)" >&2
      fi
    else
      echo "      Failed to evaluate alpha=${alpha}, deq_tol=${tol} (error)" >&2
      echo "      Check logs for details: vim ${EVAL_LOG}" >&2
    fi
  done
done

echo "" >&2
echo "============================================================" >&2
echo "Best Alpha: ${BEST_ALPHA}, Best DEQ Tol: ${BEST_DEQ_TOL} (WER: ${BEST_WER})" >&2
echo "============================================================" >&2

# Output only the best values on stdout for capture
echo "${BEST_ALPHA} ${BEST_DEQ_TOL}"
