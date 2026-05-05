#!/usr/bin/env bash
set -euo pipefail

# Run latent-generalization evaluation across all supported datasets/subsets.
#
# Defaults match the user request:
#   --baseline-ckpt ./baseline_qwen_asr_best.pth
#   --prompt-tuning-ckpt ./prompt_tuning_qwen_asr_best.pth
#   --lora-r16-ckpt ./lora_r16_qwen_asr_best
#   --latent-ckpt   ./latent_qwen_asr_best.pth
#
# Dataset policy:
#   - Common Voice / FLEURS / VoxPopuli: run fixed language sets only.
#   - Other datasets: run all available subsets/configs.
#   - Split is always test.
#
# Usage:
#   ./run_eval_all_datasets.sh
#   BASELINE_CKPT=/path/a.pth PROMPT_TUNING_CKPT=/path/prompt LORA_R16_CKPT=/path/lora LATENT_CKPT=/path/b.pth ./run_eval_all_datasets.sh
#   PREV_OUT_DIR=/path/to/old_run PROMPT_TUNING_CKPT=/path/prompt LORA_R16_CKPT=/path/lora ./run_eval_all_datasets.sh
#   ./run_eval_all_datasets.sh --print-samples 1 --max-new-tokens 96

BASELINE_CKPT="${BASELINE_CKPT:-./baseline_qwen_asr_best.pth}"
PROMPT_TUNING_CKPT="${PROMPT_TUNING_CKPT:-./prompt_tuning_qwen_asr_best.pth}"
LORA_R16_CKPT="${LORA_R16_CKPT:-./lora_r16_qwen_asr_best}"
LATENT_CKPT="${LATENT_CKPT:-./latent_qwen_asr_best.pth}"
BEST_DYNAMIC_HALT_THRESHOLD="${BEST_DYNAMIC_HALT_THRESHOLD:-0.}"
SPLIT="test"
# Per-config sample cap. 0 means run all samples.
MAX_SAMPLES_PER_CONFIG="${MAX_SAMPLES_PER_CONFIG:-0}"
# Noise levels (SNR in dB). Space-separated. Empty = clean only.
# Example: SNR_DB_LEVELS="20 10 5 0" ./run_eval_all_datasets.sh
SNR_DB_LEVELS="${SNR_DB_LEVELS:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
ROOT_DIR="$(latent_asr_repo_root "${SCRIPT_DIR}")"
PYTHON_BIN="$(latent_asr_python_bin "${PYTHON_BIN:-}")"
cd "$ROOT_DIR"

assert_ckpt_exists() {
  local label="$1"
  local path="$2"
  if [[ ! -f "$path" && ! -d "$path" ]]; then
    echo "[error] ${label} checkpoint not found: $path" >&2
    exit 1
  fi
}

ensure_ckpt_if_running() {
  local mode="$1"
  local enabled="$2"
  local path="$3"
  if [[ "${enabled}" != "1" ]]; then
    return 0
  fi
  if [[ -z "${path}" ]]; then
    echo "[error] ${mode} is required for this run but checkpoint path is empty." >&2
    exit 1
  fi
  assert_ckpt_exists "${mode}" "${path}"
}

inspect_existing_json_methods() {
  local json_path="$1"
  "${PYTHON_BIN}" - "$json_path" <<'PY'
import json
import sys
from pathlib import Path


def _has_metric_all(rows, keys):
    valid_rows = [r for r in rows if isinstance(r, dict)]
    if not valid_rows:
        return 0
    for row in valid_rows:
        ok = False
        for k in keys:
            if row.get(k) is not None:
                ok = True
                break
        if not ok:
            return 0
    return 1


out = {
    "OK": 0,
    "HAS_BASE_MODEL": 0,
    "HAS_BASELINE_FT": 0,
    "HAS_PROMPT_TUNING": 0,
    "HAS_LORA_R16": 0,
    "HAS_LATENT_REASONING": 0,
}
try:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    rows = payload.get("rows") or []
    out["OK"] = 1
    out["HAS_BASE_MODEL"] = _has_metric_all(rows, ["base_model_wer", "raw_wer", "base_wer"])
    out["HAS_BASELINE_FT"] = _has_metric_all(rows, ["baseline_ft_wer", "ft_wer", "baseline_wer"])
    out["HAS_PROMPT_TUNING"] = _has_metric_all(rows, ["prompt_tuning_wer", "prompt_wer"])
    out["HAS_LORA_R16"] = _has_metric_all(rows, ["lora_r16_wer", "lora_wer"])
    out["HAS_LATENT_REASONING"] = _has_metric_all(rows, ["latent_reasoning_wer", "latent_wer"])
except Exception:
    pass

for k, v in out.items():
    print(f"{k}={int(v)}")
PY
}

merge_eval_json() {
  local old_json="$1"
  local delta_json="$2"
  local out_json="$3"
  "${PYTHON_BIN}" - "$old_json" "$delta_json" "$out_json" <<'PY'
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _to_int(v: Any) -> int:
    try:
        return int(v or 0)
    except Exception:
        return 0


def _pick_nonempty(*vals: Any) -> Any:
    for v in vals:
        if isinstance(v, str):
            if v.strip():
                return v
        elif v is not None:
            return v
    return ""


def _diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(a - b)


def _weighted_avg(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    num = 0.0
    den = 0
    for row in rows:
        v = _to_float(row.get(key))
        n = _to_int(row.get("samples_used"))
        if v is None or n <= 0:
            continue
        num += v * n
        den += n
    if den <= 0:
        return None
    return float(num / den)


def _metric(row: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for k in keys:
        v = _to_float(row.get(k))
        if v is not None:
            return v
    return None


def _rows_by_config(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in payload.get("rows") or []:
        if not isinstance(row, dict):
            continue
        cfg = str(row.get("config", "")).strip()
        if cfg:
            out[cfg] = row
    return out


old_path = Path(sys.argv[1])
delta_path = Path(sys.argv[2])
out_path = Path(sys.argv[3])

old = json.loads(old_path.read_text(encoding="utf-8"))
delta = json.loads(delta_path.read_text(encoding="utf-8"))

old_rows = _rows_by_config(old)
delta_rows = _rows_by_config(delta)

seen = set()
configs: List[str] = []


def _push(cfg: Any) -> None:
    cfg_str = str(cfg).strip()
    if not cfg_str or cfg_str in seen:
        return
    seen.add(cfg_str)
    configs.append(cfg_str)


for cfg in old.get("configs") or []:
    _push(cfg)
for cfg in delta.get("configs") or []:
    _push(cfg)
for row in old.get("rows") or []:
    if isinstance(row, dict):
        _push(row.get("config"))
for row in delta.get("rows") or []:
    if isinstance(row, dict):
        _push(row.get("config"))

if not configs:
    for cfg in sorted(set(old_rows.keys()) | set(delta_rows.keys())):
        _push(cfg)

wer_keys = {
    "base_model_wer": ["base_model_wer", "raw_wer", "base_wer"],
    "baseline_ft_wer": ["baseline_ft_wer", "ft_wer", "baseline_wer"],
    "prompt_tuning_wer": ["prompt_tuning_wer", "prompt_wer"],
    "lora_r16_wer": ["lora_r16_wer", "lora_wer"],
    "latent_reasoning_wer": ["latent_reasoning_wer", "latent_wer"],
}
cer_keys = {
    "base_model_cer": ["base_model_cer", "raw_cer", "base_cer"],
    "baseline_ft_cer": ["baseline_ft_cer", "ft_cer", "baseline_cer"],
    "prompt_tuning_cer": ["prompt_tuning_cer", "prompt_cer"],
    "lora_r16_cer": ["lora_r16_cer", "lora_cer"],
    "latent_reasoning_cer": ["latent_reasoning_cer", "latent_cer"],
}

merged_rows: List[Dict[str, Any]] = []
for cfg in configs:
    old_row = old_rows.get(cfg, {})
    delta_row = delta_rows.get(cfg, {})

    samples_total = max(_to_int(old_row.get("samples_total")), _to_int(delta_row.get("samples_total")))
    samples_used = max(_to_int(old_row.get("samples_used")), _to_int(delta_row.get("samples_used")))
    text_column = _pick_nonempty(delta_row.get("text_column"), old_row.get("text_column"), "")

    row: Dict[str, Any] = {
        "config": cfg,
        "samples_total": samples_total,
        "samples_used": samples_used,
        "text_column": text_column,
    }

    for out_key, keys in wer_keys.items():
        row[out_key] = _metric(delta_row, keys)
        if row[out_key] is None:
            row[out_key] = _metric(old_row, keys)
    for out_key, keys in cer_keys.items():
        row[out_key] = _metric(delta_row, keys)
        if row[out_key] is None:
            row[out_key] = _metric(old_row, keys)

    row["baseline_ft_vs_base_model_wer"] = _diff(row["base_model_wer"], row["baseline_ft_wer"])
    row["baseline_ft_vs_base_model_cer"] = _diff(row["base_model_cer"], row["baseline_ft_cer"])
    row["prompt_tuning_vs_base_model_wer"] = _diff(row["base_model_wer"], row["prompt_tuning_wer"])
    row["prompt_tuning_vs_base_model_cer"] = _diff(row["base_model_cer"], row["prompt_tuning_cer"])
    row["lora_r16_vs_base_model_wer"] = _diff(row["base_model_wer"], row["lora_r16_wer"])
    row["lora_r16_vs_base_model_cer"] = _diff(row["base_model_cer"], row["lora_r16_cer"])
    row["latent_reasoning_vs_base_model_wer"] = _diff(row["base_model_wer"], row["latent_reasoning_wer"])
    row["latent_reasoning_vs_base_model_cer"] = _diff(row["base_model_cer"], row["latent_reasoning_cer"])
    row["latent_reasoning_vs_baseline_ft_wer"] = _diff(row["baseline_ft_wer"], row["latent_reasoning_wer"])
    row["latent_reasoning_vs_baseline_ft_cer"] = _diff(row["baseline_ft_cer"], row["latent_reasoning_cer"])
    row["latent_reasoning_vs_prompt_tuning_wer"] = _diff(row["prompt_tuning_wer"], row["latent_reasoning_wer"])
    row["latent_reasoning_vs_prompt_tuning_cer"] = _diff(row["prompt_tuning_cer"], row["latent_reasoning_cer"])
    row["latent_reasoning_vs_lora_r16_wer"] = _diff(row["lora_r16_wer"], row["latent_reasoning_wer"])
    row["latent_reasoning_vs_lora_r16_cer"] = _diff(row["lora_r16_cer"], row["latent_reasoning_cer"])

    merged_rows.append(row)

summary = {
    "base_model_weighted_wer": _weighted_avg(merged_rows, "base_model_wer"),
    "base_model_weighted_cer": _weighted_avg(merged_rows, "base_model_cer"),
    "baseline_ft_weighted_wer": _weighted_avg(merged_rows, "baseline_ft_wer"),
    "baseline_ft_weighted_cer": _weighted_avg(merged_rows, "baseline_ft_cer"),
    "prompt_tuning_weighted_wer": _weighted_avg(merged_rows, "prompt_tuning_wer"),
    "prompt_tuning_weighted_cer": _weighted_avg(merged_rows, "prompt_tuning_cer"),
    "lora_r16_weighted_wer": _weighted_avg(merged_rows, "lora_r16_wer"),
    "lora_r16_weighted_cer": _weighted_avg(merged_rows, "lora_r16_cer"),
    "latent_reasoning_weighted_wer": _weighted_avg(merged_rows, "latent_reasoning_wer"),
    "latent_reasoning_weighted_cer": _weighted_avg(merged_rows, "latent_reasoning_cer"),
}

summary["baseline_ft_vs_base_model_weighted_wer"] = _diff(
    summary["base_model_weighted_wer"], summary["baseline_ft_weighted_wer"]
)
summary["baseline_ft_vs_base_model_weighted_cer"] = _diff(
    summary["base_model_weighted_cer"], summary["baseline_ft_weighted_cer"]
)
summary["latent_reasoning_vs_base_model_weighted_wer"] = _diff(
    summary["base_model_weighted_wer"], summary["latent_reasoning_weighted_wer"]
)
summary["latent_reasoning_vs_base_model_weighted_cer"] = _diff(
    summary["base_model_weighted_cer"], summary["latent_reasoning_weighted_cer"]
)
summary["latent_reasoning_vs_baseline_ft_weighted_wer"] = _diff(
    summary["baseline_ft_weighted_wer"], summary["latent_reasoning_weighted_wer"]
)
summary["latent_reasoning_vs_baseline_ft_weighted_cer"] = _diff(
    summary["baseline_ft_weighted_cer"], summary["latent_reasoning_weighted_cer"]
)
summary["prompt_tuning_vs_base_model_weighted_wer"] = _diff(
    summary["base_model_weighted_wer"], summary["prompt_tuning_weighted_wer"]
)
summary["prompt_tuning_vs_base_model_weighted_cer"] = _diff(
    summary["base_model_weighted_cer"], summary["prompt_tuning_weighted_cer"]
)
summary["lora_r16_vs_base_model_weighted_wer"] = _diff(
    summary["base_model_weighted_wer"], summary["lora_r16_weighted_wer"]
)
summary["lora_r16_vs_base_model_weighted_cer"] = _diff(
    summary["base_model_weighted_cer"], summary["lora_r16_weighted_cer"]
)
summary["latent_reasoning_vs_prompt_tuning_weighted_wer"] = _diff(
    summary["prompt_tuning_weighted_wer"], summary["latent_reasoning_weighted_wer"]
)
summary["latent_reasoning_vs_prompt_tuning_weighted_cer"] = _diff(
    summary["prompt_tuning_weighted_cer"], summary["latent_reasoning_weighted_cer"]
)
summary["latent_reasoning_vs_lora_r16_weighted_wer"] = _diff(
    summary["lora_r16_weighted_wer"], summary["latent_reasoning_weighted_wer"]
)
summary["latent_reasoning_vs_lora_r16_weighted_cer"] = _diff(
    summary["lora_r16_weighted_cer"], summary["latent_reasoning_weighted_cer"]
)

latent_ckpt = _pick_nonempty(
    delta.get("latent_reasoning_checkpoint"),
    delta.get("latent_checkpoint"),
    old.get("latent_reasoning_checkpoint"),
    old.get("latent_checkpoint"),
    "",
)

payload = {
    "model_id": _pick_nonempty(delta.get("model_id"), old.get("model_id"), ""),
    "dataset_name": _pick_nonempty(delta.get("dataset_name"), old.get("dataset_name"), ""),
    "dataset_name_input": _pick_nonempty(
        delta.get("dataset_name_input"), old.get("dataset_name_input"), ""
    ),
    "split": _pick_nonempty(delta.get("split"), old.get("split"), "test"),
    "configs": configs,
    "base_model_enabled": bool(delta.get("base_model_enabled")) or bool(old.get("base_model_enabled")),
    "baseline_checkpoint": _pick_nonempty(
        delta.get("baseline_checkpoint"), old.get("baseline_checkpoint"), ""
    ),
    "prompt_tuning_checkpoint": _pick_nonempty(
        delta.get("prompt_tuning_checkpoint"), old.get("prompt_tuning_checkpoint"), ""
    ),
    "lora_r16_checkpoint": _pick_nonempty(
        delta.get("lora_r16_checkpoint"), old.get("lora_r16_checkpoint"), ""
    ),
    "latent_reasoning_checkpoint": latent_ckpt,
    "latent_checkpoint": latent_ckpt,
    "rows": merged_rows,
    "summary": summary,
}

out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[merge] merged JSON written: {out_path}")
PY
}

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-./eval_runs/generalize_${TIMESTAMP}}"
PREV_OUT_DIR="${PREV_OUT_DIR:-}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "$LOG_DIR"

# Set SKIP_GATE_CHECK=0 to enable gate checking (skip dataset if latent WER > base WER).
SKIP_GATE_CHECK="${SKIP_GATE_CHECK:-1}"

# All datasets, ordered smallest → largest (en test set).
# Each dataset acts as a gate: base+latent evaluated first,
# stops if latent WER > base WER on any dataset.
DATASETS=(
  "SpeechTest/fleurs"            # ~400
  "SpeechTest/voxpopuli"         # ~1.8K
  "SpeechTest/ASCEND"            # ~2.5K
  "SpeechTest/librispeech_asr"   # ~5.5K
  "SpeechTest/peoples_speech"    # ~10K
  "SpeechTest/common_voice_16_0" # ~16K
  "SpeechTest/gigaspeech"        # ~40K+
)

# Fixed language sets for multilingual datasets.
COMMON_VOICE_LANGS="en"
FLEURS_LANGS="en"
VOXPOPULI_LANGS="en"

EXTRA_ARGS=("$@")

# Gate check: compare latent_reasoning_wer vs base_model_wer in a JSON file.
# Returns 0 (success) if latent <= base, 1 if latent > base or data missing.
gate_check_latent_vs_base() {
  local json_path="$1"
  "${PYTHON_BIN}" - "$json_path" <<'GATE_PY'
import json, sys, pathlib
try:
    data = json.loads(pathlib.Path(sys.argv[1]).read_text())
    rows = data.get("rows") or []
    total_base_n, total_base_wer = 0, 0.0
    total_lat_n, total_lat_wer = 0, 0.0
    for r in rows:
        n = int(r.get("samples_used", 0) or 0)
        bw = r.get("base_model_wer")
        lw = r.get("latent_reasoning_wer")
        if n <= 0 or bw is None or lw is None:
            continue
        total_base_n += n; total_base_wer += float(bw) * n
        total_lat_n += n; total_lat_wer += float(lw) * n
    if total_base_n == 0 or total_lat_n == 0:
        print("GATE_RESULT=MISSING")
        sys.exit(0)
    avg_base = total_base_wer / total_base_n
    avg_lat = total_lat_wer / total_lat_n
    print(f"BASE_WER={avg_base:.6f}")
    print(f"LATENT_WER={avg_lat:.6f}")
    all_skipped = abs(avg_lat - avg_base) < 1e-9
    print(f"ALL_SKIPPED={'1' if all_skipped else '0'}")
    if avg_lat <= avg_base:
        print("GATE_RESULT=PASS")
    else:
        print(f"GATE_RESULT=FAIL")
except Exception as e:
    print(f"GATE_RESULT=ERROR:{e}")
GATE_PY
}

echo "============================================================"
echo "Latent Reasoning Generalization Evaluation"
echo "============================================================"
echo "Root dir       : $ROOT_DIR"
echo "Output dir     : $OUT_DIR"
echo "Prev output dir: ${PREV_OUT_DIR:-(none)}"
echo "Baseline ckpt  : $BASELINE_CKPT"
echo "Prompt ckpt    : $PROMPT_TUNING_CKPT"
echo "LoRA r16 ckpt  : $LORA_R16_CKPT"
echo "Latent ckpt    : $LATENT_CKPT"
echo "Halt Threshold : $BEST_DYNAMIC_HALT_THRESHOLD"
echo "Split          : $SPLIT"
echo "Max/config     : $MAX_SAMPLES_PER_CONFIG"
echo "SNR levels     : ${SNR_DB_LEVELS:-(clean only)}"
echo "Gate check     : $(if [[ $SKIP_GATE_CHECK == 1 ]]; then echo 'DISABLED'; else echo 'every dataset'; fi)"
echo "Datasets       : ${#DATASETS[@]}"
echo "Extra args     : ${EXTRA_ARGS[*]:-(none)}"
echo "============================================================"


SUCCESS=()
FAILED=()

# Skip clean evaluation when noise-only mode is requested
if [[ -n "${SNR_DB_LEVELS}" ]]; then
  echo ""
  echo ">>> Skipping clean evaluation (SNR_DB_LEVELS is set)."
  echo ""
else

for DATASET in "${DATASETS[@]}"; do
  SLUG="${DATASET//\//_}"
  JSON_OUT="${OUT_DIR}/${SLUG}.json"
  LOG_OUT="${LOG_DIR}/${SLUG}.log"
  DATASET_SCOPE="all-configs"
  DATASET_CONFIGS=""
  PREV_JSON=""
  USE_PREV_JSON=0
  CURRENT_SPLIT="${SPLIT}"

  # Default: run all methods (may be overridden by gate check or prev-JSON reuse)
  RUN_BASE_MODEL=1
  RUN_BASELINE_FT=1
  RUN_PROMPT_TUNING=1
  RUN_LORA_R16=1
  RUN_LATENT_REASONING=1

  PREV_OK=0
  PREV_HAS_BASE_MODEL=0
  PREV_HAS_BASELINE_FT=0
  PREV_HAS_PROMPT_TUNING=0
  PREV_HAS_LORA_R16=0

  # ── Resolve dataset-specific configs BEFORE gate check ────────
  case "${DATASET}" in
    "SpeechTest/common_voice_16_0")
      DATASET_SCOPE="fixed-languages"
      DATASET_CONFIGS="${COMMON_VOICE_LANGS}"
      ;;
    "SpeechTest/fleurs")
      DATASET_SCOPE="fixed-languages"
      DATASET_CONFIGS="${FLEURS_LANGS}"
      ;;
    "SpeechTest/voxpopuli")
      DATASET_SCOPE="fixed-languages"
      DATASET_CONFIGS="${VOXPOPULI_LANGS}"
      ;;
    "SpeechTest/ASCEND")
      DATASET_CONFIGS="main"
      ;;
    "SpeechTest/extreme_asr_pony")
      CURRENT_SPLIT="train"
      ;;
  esac

  # ── Phase 1: Gate Check (Base + Latent) ──────────────────────
  if [[ "${SKIP_GATE_CHECK}" != "1" ]]; then
    echo ">>> Starting Phase 1 (Gate Check): Base + Latent"
    
    GATE_JSON="${OUT_DIR}/${SLUG}.gate.json"
    
    # Ensure checkpoints for Phase 1
    ensure_ckpt_if_running "latent_reasoning" "1" "${LATENT_CKPT}"

    GATE_ARGS=(
      --dataset-name "${DATASET}"
      --split "${CURRENT_SPLIT}"
      --output-json "${GATE_JSON}"
      --max-samples-per-config "${MAX_SAMPLES_PER_CONFIG}"
      --baseline-ckpt "${BASELINE_CKPT}" # Ignored if skip-baseline-ft but good to have
      --latent-ckpt "${LATENT_CKPT}"
      --n-latent 4
      --num-beams 1
      --dynamic-halt-threshold "${BEST_DYNAMIC_HALT_THRESHOLD}"
    )
    if [[ -n "${DATASET_CONFIGS}" ]]; then
      GATE_ARGS+=(--configs "${DATASET_CONFIGS}")
    else
      GATE_ARGS+=(--all-configs)
    fi

    # Explicitly enable base/latent, disable others for gate run
    if "${PYTHON_BIN}" eval.py \
      "${GATE_ARGS[@]}" \
      --skip-baseline-ft --skip-prompt-tuning --skip-lora-r16 \
      "${EXTRA_ARGS[@]}" 2>&1 | tee "${LOG_DIR}/${SLUG}.gate.log"; then
      
      # Check Gate Result
      GATE_BASE_WER=""
      GATE_LATENT_WER=""
      GATE_RESULT=""
      GATE_ALL_SKIPPED="0"
      while IFS='=' read -r gk gv; do
        case "${gk}" in
          BASE_WER) GATE_BASE_WER="${gv}" ;;
          LATENT_WER) GATE_LATENT_WER="${gv}" ;;
          ALL_SKIPPED) GATE_ALL_SKIPPED="${gv}" ;;
          GATE_RESULT) GATE_RESULT="${gv}" ;;
        esac
      done < <(gate_check_latent_vs_base "${GATE_JSON}")

      echo "    [Gate] Base WER  : ${GATE_BASE_WER:-(unavailable)}"
      echo "    [Gate] Latent WER: ${GATE_LATENT_WER:-(unavailable)}"

      if [[ "${GATE_ALL_SKIPPED}" == "1" ]]; then
        echo "    ⏭️  LR skipped ALL samples. Skipping Phase 2 for this dataset."
        continue
      elif [[ "${GATE_RESULT}" == "PASS" ]]; then
        echo "    ✅ Gate PASSED. Continuing to Phase 2 (FT/Prompt/LoRA)."
        
        # Setup for Phase 2: Reuse Gate JSON as previous result
        PREV_JSON="${GATE_JSON}"
        USE_PREV_JSON=1
        
        # Disable Base/Latent for Phase 2 (already ran)
        RUN_BASE_MODEL=0
        RUN_LATENT_REASONING=0
        
        # Enable others for Phase 2
        RUN_BASELINE_FT=1
        RUN_PROMPT_TUNING=1
        RUN_LORA_R16=1
      else
        echo "    ❌ Gate FAILED. Skipping rest of this dataset."
        FAILED+=("${DATASET} (Gate Failed)")
        continue
      fi
    else
      echo "    [Gate] Eval failed to run."
      FAILED+=("${DATASET} (Gate Error)")
      continue
    fi
  else
    echo ">>> Gate check disabled. Running full eval."
  fi
  echo "    JSON: ${JSON_OUT}"
  echo "    LOG : ${LOG_OUT}"

  echo "    Scope: ${DATASET_SCOPE}"
  if [[ -n "${DATASET_CONFIGS}" ]]; then
    echo "    Configs: ${DATASET_CONFIGS}"
  fi

  if [[ -n "${PREV_OUT_DIR}" ]]; then
    PREV_JSON="${PREV_OUT_DIR}/${SLUG}.json"
    if [[ -f "${PREV_JSON}" ]]; then
      USE_PREV_JSON=1
      while IFS='=' read -r key value; do
        case "${key}" in
          OK) PREV_OK="${value}" ;;
          HAS_BASE_MODEL) PREV_HAS_BASE_MODEL="${value}" ;;
          HAS_BASELINE_FT) PREV_HAS_BASELINE_FT="${value}" ;;
          HAS_PROMPT_TUNING) PREV_HAS_PROMPT_TUNING="${value}" ;;
          HAS_LORA_R16) PREV_HAS_LORA_R16="${value}" ;;
          HAS_LATENT_REASONING) PREV_HAS_LATENT_REASONING="${value}" ;;
        esac
      done < <(inspect_existing_json_methods "${PREV_JSON}")

      if [[ "${PREV_OK}" != "1" ]]; then
        USE_PREV_JSON=0
        echo "    Reuse previous: parse failed, fallback to full eval"
      else
        RUN_BASE_MODEL=$((1 - PREV_HAS_BASE_MODEL))
        RUN_BASELINE_FT=$((1 - PREV_HAS_BASELINE_FT))
        RUN_PROMPT_TUNING=$((1 - PREV_HAS_PROMPT_TUNING))
        RUN_LORA_R16=$((1 - PREV_HAS_LORA_R16))
        RUN_LATENT_REASONING=$((1 - PREV_HAS_LATENT_REASONING))
        echo "    Reuse previous: ${PREV_JSON}"
        echo "    Prev coverage : base=${PREV_HAS_BASE_MODEL} baseline_ft=${PREV_HAS_BASELINE_FT} prompt=${PREV_HAS_PROMPT_TUNING} lora=${PREV_HAS_LORA_R16} latent=${PREV_HAS_LATENT_REASONING}"
      fi
    fi
  fi

  echo "    Run flags     : base=${RUN_BASE_MODEL} baseline_ft=${RUN_BASELINE_FT} prompt=${RUN_PROMPT_TUNING} lora=${RUN_LORA_R16} latent=${RUN_LATENT_REASONING}"

  ensure_ckpt_if_running "baseline_ft" "${RUN_BASELINE_FT}" "${BASELINE_CKPT}"
  ensure_ckpt_if_running "prompt_tuning" "${RUN_PROMPT_TUNING}" "${PROMPT_TUNING_CKPT}"
  ensure_ckpt_if_running "lora_r16" "${RUN_LORA_R16}" "${LORA_R16_CKPT}"
  ensure_ckpt_if_running "latent_reasoning" "${RUN_LATENT_REASONING}" "${LATENT_CKPT}"

  if [[ "${USE_PREV_JSON}" == "1" && "${RUN_BASE_MODEL}" == "0" && "${RUN_BASELINE_FT}" == "0" && "${RUN_PROMPT_TUNING}" == "0" && "${RUN_LORA_R16}" == "0" && "${RUN_LATENT_REASONING}" == "0" ]]; then
    if [[ "${PREV_JSON}" != "${JSON_OUT}" ]]; then
      cp "${PREV_JSON}" "${JSON_OUT}"
    fi
    {
      echo "[reuse] no missing method for ${DATASET}"
      echo "[reuse] copied previous JSON to ${JSON_OUT}"
    } | tee "${LOG_OUT}"
    SUCCESS+=("${DATASET}")
    continue
  fi

  EVAL_JSON_OUT="${JSON_OUT}"
  TMP_JSON=""
  if [[ "${USE_PREV_JSON}" == "1" ]]; then
    TMP_JSON="${OUT_DIR}/${SLUG}.delta.tmp"
    EVAL_JSON_OUT="${TMP_JSON}"
  fi

  CMD_ARGS=(
    --dataset-name "${DATASET}"
    --split "${CURRENT_SPLIT}"
    --output-json "${EVAL_JSON_OUT}"
    --max-samples-per-config "${MAX_SAMPLES_PER_CONFIG}"
    --n-latent 4
    --num-beams 1
    --dynamic-halt-threshold "${BEST_DYNAMIC_HALT_THRESHOLD}"
  )
  if [[ "${RUN_BASE_MODEL}" != "1" ]]; then
    CMD_ARGS+=(--skip-base-model)
  fi
  if [[ "${RUN_BASELINE_FT}" == "1" ]]; then
    CMD_ARGS+=(--baseline-ckpt "${BASELINE_CKPT}")
  fi
  if [[ "${RUN_PROMPT_TUNING}" == "1" ]]; then
    CMD_ARGS+=(--prompt-tuning-ckpt "${PROMPT_TUNING_CKPT}")
  fi
  if [[ "${RUN_LORA_R16}" == "1" ]]; then
    CMD_ARGS+=(--lora-r16-ckpt "${LORA_R16_CKPT}")
  fi
  if [[ "${RUN_LATENT_REASONING}" == "1" ]]; then
    CMD_ARGS+=(--latent-ckpt "${LATENT_CKPT}")
  fi

  if [[ -n "${DATASET_CONFIGS}" ]]; then
    CMD_ARGS+=(--configs "${DATASET_CONFIGS}")
  else
    CMD_ARGS+=(--all-configs)
  fi

  if "${PYTHON_BIN}" eval.py \
    "${CMD_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" 2>&1 | tee "${LOG_OUT}"; then
    if [[ "${USE_PREV_JSON}" == "1" ]]; then
      if merge_eval_json "${PREV_JSON}" "${TMP_JSON}" "${JSON_OUT}" 2>&1 | tee -a "${LOG_OUT}"; then
        rm -f "${TMP_JSON}"
        SUCCESS+=("${DATASET}")
      else
        FAILED+=("${DATASET}")
        echo "[warn] merge failed: ${DATASET}" | tee -a "${LOG_OUT}"
      fi
    else
      SUCCESS+=("${DATASET}")
    fi
  else
    FAILED+=("${DATASET}")
    echo "[warn] dataset failed: ${DATASET}" | tee -a "${LOG_OUT}"
  fi

  # (Gate check moved to start of loop)
done

fi  # end of clean evaluation skip

# ==============================================================
# Noise evaluation: repeat base + latent for each SNR level
# ==============================================================
if [[ -n "${SNR_DB_LEVELS}" ]]; then
  echo ""
  echo "============================================================"
  echo "Noise Robustness Evaluation"
  echo "============================================================"
  echo "SNR levels: ${SNR_DB_LEVELS}"
  echo ""

  for SNR in ${SNR_DB_LEVELS}; do
    SNR_TAG="snr${SNR}db"
    echo ">>> Noise pass: SNR=${SNR} dB"

    for DATASET in "${DATASETS[@]}"; do
      SLUG="${DATASET//\//_}"
      NOISE_JSON="${OUT_DIR}/${SLUG}_${SNR_TAG}.json"
      NOISE_LOG="${LOG_DIR}/${SLUG}_${SNR_TAG}.log"
      CURRENT_SPLIT="${SPLIT}"
      DATASET_CONFIGS=""

      case "${DATASET}" in
        "SpeechTest/common_voice_16_0")
          DATASET_CONFIGS="${COMMON_VOICE_LANGS}" ;;
        "SpeechTest/fleurs")
          DATASET_CONFIGS="${FLEURS_LANGS}" ;;
        "SpeechTest/voxpopuli")
          DATASET_CONFIGS="${VOXPOPULI_LANGS}" ;;
        "SpeechTest/ASCEND")
          DATASET_CONFIGS="main" ;;
        "SpeechTest/extreme_asr_pony")
          CURRENT_SPLIT="train" ;;
      esac

      NOISE_ARGS=(
        --dataset-name "${DATASET}"
        --split "${CURRENT_SPLIT}"
        --output-json "${NOISE_JSON}"
        --max-samples-per-config "${MAX_SAMPLES_PER_CONFIG}"
        --latent-ckpt "${LATENT_CKPT}"
        --n-latent 4
        --num-beams 1
        --dynamic-halt-threshold "${BEST_DYNAMIC_HALT_THRESHOLD}"
        --snr-db "${SNR}"
        --skip-baseline-ft --skip-prompt-tuning --skip-lora-r16
      )
      if [[ -n "${DATASET_CONFIGS}" ]]; then
        NOISE_ARGS+=(--configs "${DATASET_CONFIGS}")
      else
        NOISE_ARGS+=(--all-configs)
      fi

      echo "    [${SNR_TAG}] ${DATASET} -> ${NOISE_JSON}"
      if "${PYTHON_BIN}" eval.py \
        "${NOISE_ARGS[@]}" \
        "${EXTRA_ARGS[@]}" 2>&1 | tee "${NOISE_LOG}"; then
        SUCCESS+=("${DATASET} (${SNR_TAG})")
      else
        FAILED+=("${DATASET} (${SNR_TAG})")
        echo "    [warn] noise eval failed: ${DATASET} ${SNR_TAG}" | tee -a "${NOISE_LOG}"
      fi
    done
    echo ""
  done
fi

echo ""
echo "============================================================"
echo "Run Summary"
echo "============================================================"
echo "Succeeded (${#SUCCESS[@]}):"
for D in "${SUCCESS[@]}"; do
  echo "  - ${D}"
done
echo "Failed (${#FAILED[@]}):"
for D in "${FAILED[@]}"; do
  echo "  - ${D}"
done
echo "Outputs: ${OUT_DIR}"
echo "============================================================"

REPORT_MD="${OUT_DIR}/evaluation_report.md"
SUCCESS_CSV="$(IFS=,; echo "${SUCCESS[*]}")"
FAILED_CSV="$(IFS=,; echo "${FAILED[*]}")"

REPORT_OUT_DIR="${OUT_DIR}" \
REPORT_PATH="${REPORT_MD}" \
REPORT_BASELINE_CKPT="${BASELINE_CKPT}" \
REPORT_PROMPT_TUNING_CKPT="${PROMPT_TUNING_CKPT}" \
REPORT_LORA_R16_CKPT="${LORA_R16_CKPT}" \
REPORT_LATENT_CKPT="${LATENT_CKPT}" \
REPORT_SPLIT="${SPLIT}" \
REPORT_MAX_SAMPLES="${MAX_SAMPLES_PER_CONFIG}" \
REPORT_SUCCESS="${SUCCESS_CSV}" \
REPORT_FAILED="${FAILED_CSV}" \
"${PYTHON_BIN}" - <<'PY'
import glob
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _fmt_pct(v: Any) -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v) * 100.0:.2f}%"
    except Exception:
        return "-"


def _fmt_num(v: Any) -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v):.6f}"
    except Exception:
        return "-"


out_dir = os.environ["REPORT_OUT_DIR"]
report_path = os.environ["REPORT_PATH"]
baseline_ckpt = os.environ.get("REPORT_BASELINE_CKPT", "")
prompt_tuning_ckpt = os.environ.get("REPORT_PROMPT_TUNING_CKPT", "")
lora_r16_ckpt = os.environ.get("REPORT_LORA_R16_CKPT", "")
latent_ckpt = os.environ.get("REPORT_LATENT_CKPT", "")
split = os.environ.get("REPORT_SPLIT", "test")
max_samples = os.environ.get("REPORT_MAX_SAMPLES", "0")
success = [x for x in os.environ.get("REPORT_SUCCESS", "").split(",") if x]
failed = [x for x in os.environ.get("REPORT_FAILED", "").split(",") if x]

json_paths = sorted(glob.glob(os.path.join(out_dir, "*.json")))
records: List[Dict[str, Any]] = []
load_errors: List[str] = []

for p in json_paths:
    try:
        with open(p, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        load_errors.append(f"{Path(p).name}: {e}")
        continue

    rows = payload.get("rows") or []
    summary = payload.get("summary") or {}
    dataset_name = payload.get("dataset_name") or Path(p).stem
    sample_used = 0
    for r in rows:
        try:
            sample_used += int(r.get("samples_used", 0) or 0)
        except Exception:
            pass

    records.append(
        {
            "dataset_name": dataset_name,
            "json_name": Path(p).name,
            "rows": rows,
            "summary": summary,
            "config_count": len(rows),
            "samples_used_total": sample_used,
        }
    )

records.sort(key=lambda x: str(x["dataset_name"]).lower())

lines: List[str] = []
lines.append("# Latent Reasoning ASR Evaluation Report")
lines.append("")
lines.append(f"- Generated at (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
lines.append(f"- Output directory: `{out_dir}`")
lines.append(f"- Split: `{split}`")
lines.append(f"- Max samples per config: `{max_samples}` (`0` means full dataset)")
lines.append(f"- Baseline checkpoint: `{baseline_ckpt}`")
lines.append(f"- Prompt tuning checkpoint: `{prompt_tuning_ckpt}`")
lines.append(f"- LoRA r16 checkpoint: `{lora_r16_ckpt}`")
lines.append(f"- Latent checkpoint: `{latent_ckpt}`")
lines.append(f"- Successful datasets: `{len(success)}`")
lines.append(f"- Failed datasets: `{len(failed)}`")
lines.append("")

if success:
    lines.append("## Successful Datasets")
    lines.append("")
    for ds in success:
        lines.append(f"- `{ds}`")
    lines.append("")

if failed:
    lines.append("## Failed Datasets")
    lines.append("")
    for ds in failed:
        lines.append(f"- `{ds}`")
    lines.append("")

lines.append("## Weighted Summary By Dataset")
lines.append("")
lines.append("| Dataset | Configs | Samples Used | Raw WER | FT WER | Prompt WER | LoRA16 WER | Latent WER | Latent vs FT | Latent vs Prompt | Latent vs LoRA16 |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for rec in records:
    summary = rec["summary"]
    lines.append(
        "| {dataset} | {cfgs} | {samples} | {raw} | {ft} | {prompt} | {lora} | {lat} | {lat_vs_ft} | {lat_vs_prompt} | {lat_vs_lora} |".format(
            dataset=rec["dataset_name"],
            cfgs=rec["config_count"],
            samples=rec["samples_used_total"],
            raw=_fmt_pct(summary.get("base_model_weighted_wer")),
            ft=_fmt_pct(summary.get("baseline_ft_weighted_wer")),
            prompt=_fmt_pct(summary.get("prompt_tuning_weighted_wer")),
            lora=_fmt_pct(summary.get("lora_r16_weighted_wer")),
            lat=_fmt_pct(summary.get("latent_reasoning_weighted_wer")),
            lat_vs_ft=_fmt_pct(summary.get("latent_reasoning_vs_baseline_ft_weighted_wer")),
            lat_vs_prompt=_fmt_pct(summary.get("latent_reasoning_vs_prompt_tuning_weighted_wer")),
            lat_vs_lora=_fmt_pct(summary.get("latent_reasoning_vs_lora_r16_weighted_wer")),
        )
    )
lines.append("")

for rec in records:
    dataset_name = rec["dataset_name"]
    summary = rec["summary"]
    rows = rec["rows"]

    lines.append(f"## {dataset_name}")
    lines.append("")
    lines.append(f"- JSON: `{rec['json_name']}`")
    lines.append(f"- Config count: `{rec['config_count']}`")
    lines.append(f"- Total samples used: `{rec['samples_used_total']}`")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Base weighted WER | {_fmt_pct(summary.get('base_model_weighted_wer'))} |")
    lines.append(f"| Baseline FT weighted WER | {_fmt_pct(summary.get('baseline_ft_weighted_wer'))} |")
    lines.append(f"| Prompt tuning weighted WER | {_fmt_pct(summary.get('prompt_tuning_weighted_wer'))} |")
    lines.append(f"| LoRA r16 weighted WER | {_fmt_pct(summary.get('lora_r16_weighted_wer'))} |")
    lines.append(f"| Latent weighted WER | {_fmt_pct(summary.get('latent_reasoning_weighted_wer'))} |")
    lines.append(f"| Latent vs Baseline FT (WER) | {_fmt_pct(summary.get('latent_reasoning_vs_baseline_ft_weighted_wer'))} |")
    lines.append(f"| Latent vs Prompt tuning (WER) | {_fmt_pct(summary.get('latent_reasoning_vs_prompt_tuning_weighted_wer'))} |")
    lines.append(f"| Latent vs LoRA r16 (WER) | {_fmt_pct(summary.get('latent_reasoning_vs_lora_r16_weighted_wer'))} |")
    lines.append(f"| Base weighted CER | {_fmt_pct(summary.get('base_model_weighted_cer'))} |")
    lines.append(f"| Baseline FT weighted CER | {_fmt_pct(summary.get('baseline_ft_weighted_cer'))} |")
    lines.append(f"| Prompt tuning weighted CER | {_fmt_pct(summary.get('prompt_tuning_weighted_cer'))} |")
    lines.append(f"| LoRA r16 weighted CER | {_fmt_pct(summary.get('lora_r16_weighted_cer'))} |")
    lines.append(f"| Latent weighted CER | {_fmt_pct(summary.get('latent_reasoning_weighted_cer'))} |")
    lines.append(f"| Latent vs Baseline FT (CER) | {_fmt_pct(summary.get('latent_reasoning_vs_baseline_ft_weighted_cer'))} |")
    lines.append(f"| Latent vs Prompt tuning (CER) | {_fmt_pct(summary.get('latent_reasoning_vs_prompt_tuning_weighted_cer'))} |")
    lines.append(f"| Latent vs LoRA r16 (CER) | {_fmt_pct(summary.get('latent_reasoning_vs_lora_r16_weighted_cer'))} |")
    lines.append("")

    lines.append("| Config | N | Raw WER | FT WER | Prompt WER | LoRA16 WER | Latent WER | Latent-FT WER | Latent-Prompt WER | Latent-LoRA16 WER | Raw CER | FT CER | Prompt CER | LoRA16 CER | Latent CER | Latent-FT CER | Latent-Prompt CER | Latent-LoRA16 CER |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {cfg} | {n} | {raw_w} | {ft_w} | {pt_w} | {lora_w} | {lat_w} | {d_w} | {d_pt_w} | {d_lora_w} | {raw_c} | {ft_c} | {pt_c} | {lora_c} | {lat_c} | {d_c} | {d_pt_c} | {d_lora_c} |".format(
                cfg=row.get("config", "-"),
                n=int(row.get("samples_used", 0) or 0),
                raw_w=_fmt_pct(row.get("base_model_wer")),
                ft_w=_fmt_pct(row.get("baseline_ft_wer")),
                pt_w=_fmt_pct(row.get("prompt_tuning_wer")),
                lora_w=_fmt_pct(row.get("lora_r16_wer")),
                lat_w=_fmt_pct(row.get("latent_reasoning_wer")),
                d_w=_fmt_pct(row.get("latent_reasoning_vs_baseline_ft_wer")),
                d_pt_w=_fmt_pct(row.get("latent_reasoning_vs_prompt_tuning_wer")),
                d_lora_w=_fmt_pct(row.get("latent_reasoning_vs_lora_r16_wer")),
                raw_c=_fmt_pct(row.get("base_model_cer")),
                ft_c=_fmt_pct(row.get("baseline_ft_cer")),
                pt_c=_fmt_pct(row.get("prompt_tuning_cer")),
                lora_c=_fmt_pct(row.get("lora_r16_cer")),
                lat_c=_fmt_pct(row.get("latent_reasoning_cer")),
                d_c=_fmt_pct(row.get("latent_reasoning_vs_baseline_ft_cer")),
                d_pt_c=_fmt_pct(row.get("latent_reasoning_vs_prompt_tuning_cer")),
                d_lora_c=_fmt_pct(row.get("latent_reasoning_vs_lora_r16_cer")),
            )
        )
    lines.append("")

if load_errors:
    lines.append("## JSON Load Errors")
    lines.append("")
    for err in load_errors:
        lines.append(f"- `{err}`")
    lines.append("")

if not records:
    lines.append("## No Dataset JSON Found")
    lines.append("")
    lines.append("No per-dataset JSON files were generated in this run.")
    lines.append("")

Path(report_path).write_text("\n".join(lines), encoding="utf-8")
print(f"[report] markdown written: {report_path}")
PY

echo "Markdown report: ${REPORT_MD}"

if [[ ${#FAILED[@]} -gt 0 ]]; then
  exit 1
fi
