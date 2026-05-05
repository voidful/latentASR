#!/usr/bin/env python3
"""Write a paper-style report from clean ASR evaluation JSON files."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


QWEN_CARD_URL = "https://huggingface.co/Qwen/Qwen3-ASR-1.7B/blob/main/README.md"
MLS_CARD_URL = "https://huggingface.co/datasets/facebook/multilingual_librispeech"

FLEURS_FILES = [
    ("FLEURS core12", "fleurs_core12_streaming_clean.json"),
    ("FLEURS +8", "fleurs_extra8_streaming_clean.json"),
    ("FLEURS +10", "fleurs_extra10_streaming_clean.json"),
]

MLS_FILES = [
    ("MLS public7", "mls_public7_streaming_clean.json"),
]

ENGLISH_FILES = [
    ("FLEURS en-US", "fleurs_en_us_clean.json"),
    ("MInDS-14 en-US", "minds14_en_us_clean.json"),
    ("TED-LIUM release1", "tedlium_release1_clean.json"),
    ("VoxPopuli en", "voxpopuli_en_clean.json"),
    ("LibriSpeech clean+other", "librispeech_clean_other_clean.json"),
]

LANGUAGE_NAMES = {
    "en_us": "English",
    "cmn_hans_cn": "Mandarin Chinese",
    "yue_hant_hk": "Cantonese",
    "ar_eg": "Arabic",
    "de_de": "German",
    "es_419": "Spanish",
    "fr_fr": "French",
    "it_it": "Italian",
    "ja_jp": "Japanese",
    "ko_kr": "Korean",
    "pt_br": "Portuguese",
    "ru_ru": "Russian",
    "hi_in": "Hindi",
    "id_id": "Indonesian",
    "ms_my": "Malay",
    "nl_nl": "Dutch",
    "pl_pl": "Polish",
    "th_th": "Thai",
    "tr_tr": "Turkish",
    "vi_vn": "Vietnamese",
    "cs_cz": "Czech",
    "da_dk": "Danish",
    "el_gr": "Greek",
    "fa_ir": "Persian",
    "fi_fi": "Finnish",
    "fil_ph": "Filipino",
    "hu_hu": "Hungarian",
    "mk_mk": "Macedonian",
    "ro_ro": "Romanian",
    "sv_se": "Swedish",
    "german": "German",
    "dutch": "Dutch",
    "spanish": "Spanish",
    "french": "French",
    "italian": "Italian",
    "polish": "Polish",
    "portuguese": "Portuguese",
    "clean": "LibriSpeech test-clean",
    "other": "LibriSpeech test-other",
}


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(payload.get("rows") or [])


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _sample_count(rows: Iterable[Dict[str, Any]]) -> int:
    return sum(_as_int(row.get("samples_used")) for row in rows)


def _weighted(rows: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
    numer = 0.0
    denom = 0
    for row in rows:
        value = _as_float(row.get(key))
        count = _as_int(row.get("samples_used"))
        if value is None or count <= 0:
            continue
        numer += value * count
        denom += count
    if denom <= 0:
        return None
    return numer / denom


def _summary_value(payload: Dict[str, Any], key: str) -> Optional[float]:
    summary = payload.get("summary") or {}
    value = _as_float(summary.get(key))
    if value is not None:
        return value
    return _weighted(_rows(payload), key.replace("_weighted_", "_"))


def _pct(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value * 100.0:.3f}"


def _pp(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value * 100.0:+.3f}"


def _rel(delta: Optional[float], base: Optional[float]) -> str:
    if delta is None or base in (None, 0.0):
        return "-"
    return f"{delta / base * 100.0:+.2f}%"


def _configs(payload: Dict[str, Any]) -> str:
    values = [str(x) for x in payload.get("configs") or []]
    return ",".join(values) if values else "-"


def _split(payload: Dict[str, Any]) -> str:
    return str(payload.get("split") or "-")


def _streaming(payload: Dict[str, Any]) -> str:
    value = payload.get("streaming")
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "not recorded"


def _dataset(payload: Dict[str, Any]) -> str:
    return str(payload.get("dataset_name") or payload.get("dataset_name_input") or "-")


def _record(label: str, payload: Dict[str, Any], json_name: str) -> Dict[str, Any]:
    rows = _rows(payload)
    base_wer = _summary_value(payload, "base_model_weighted_wer")
    lr_wer = _summary_value(payload, "latent_reasoning_weighted_wer")
    base_cer = _summary_value(payload, "base_model_weighted_cer")
    lr_cer = _summary_value(payload, "latent_reasoning_weighted_cer")
    dwer = None if base_wer is None or lr_wer is None else base_wer - lr_wer
    dcer = None if base_cer is None or lr_cer is None else base_cer - lr_cer
    return {
        "label": label,
        "dataset": _dataset(payload),
        "split": _split(payload),
        "configs": _configs(payload),
        "streaming": _streaming(payload),
        "n": _sample_count(rows),
        "base_wer": base_wer,
        "lr_wer": lr_wer,
        "dwer": dwer,
        "base_cer": base_cer,
        "lr_cer": lr_cer,
        "dcer": dcer,
        "json": json_name,
    }


def _aggregate_record(
    label: str,
    dataset: str,
    split: str,
    streaming: str,
    rows: Sequence[Dict[str, Any]],
    json_name: str,
) -> Dict[str, Any]:
    base_wer = _weighted(rows, "base_model_wer")
    lr_wer = _weighted(rows, "latent_reasoning_wer")
    base_cer = _weighted(rows, "base_model_cer")
    lr_cer = _weighted(rows, "latent_reasoning_cer")
    dwer = None if base_wer is None or lr_wer is None else base_wer - lr_wer
    dcer = None if base_cer is None or lr_cer is None else base_cer - lr_cer
    return {
        "label": label,
        "dataset": dataset,
        "split": split,
        "configs": "weighted aggregate",
        "streaming": streaming,
        "n": _sample_count(rows),
        "base_wer": base_wer,
        "lr_wer": lr_wer,
        "dwer": dwer,
        "base_cer": base_cer,
        "lr_cer": lr_cer,
        "dcer": dcer,
        "json": json_name,
    }


def _metric_table(records: Sequence[Dict[str, Any]]) -> List[str]:
    lines = [
        "| Suite | Dataset | Split | Configs | Streaming | N | Base WER | LR WER | dWER | Rel. WER | Base CER | LR CER | dCER | JSON |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rec in records:
        lines.append(
            "| {label} | `{dataset}` | {split} | `{configs}` | {streaming} | {n} | {base_wer} | {lr_wer} | {dwer} | {rel} | {base_cer} | {lr_cer} | {dcer} | `{json}` |".format(
                label=rec["label"],
                dataset=rec["dataset"],
                split=rec["split"],
                configs=rec["configs"],
                streaming=rec["streaming"],
                n=rec["n"],
                base_wer=_pct(rec["base_wer"]),
                lr_wer=_pct(rec["lr_wer"]),
                dwer=_pp(rec["dwer"]),
                rel=_rel(rec["dwer"], rec["base_wer"]),
                base_cer=_pct(rec["base_cer"]),
                lr_cer=_pct(rec["lr_cer"]),
                dcer=_pp(rec["dcer"]),
                json=rec["json"],
            )
        )
    return lines


def _row_table(rows: Sequence[Tuple[str, Dict[str, Any]]]) -> List[str]:
    lines = [
        "| Group | Config | Language | N | Base WER | LR WER | dWER | Base CER | LR CER | dCER |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group, row in rows:
        cfg = str(row.get("config") or "-")
        base_wer = _as_float(row.get("base_model_wer"))
        lr_wer = _as_float(row.get("latent_reasoning_wer"))
        base_cer = _as_float(row.get("base_model_cer"))
        lr_cer = _as_float(row.get("latent_reasoning_cer"))
        dwer = None if base_wer is None or lr_wer is None else base_wer - lr_wer
        dcer = None if base_cer is None or lr_cer is None else base_cer - lr_cer
        lines.append(
            "| {group} | `{cfg}` | {lang} | {n} | {base_wer} | {lr_wer} | {dwer} | {base_cer} | {lr_cer} | {dcer} |".format(
                group=group,
                cfg=cfg,
                lang=LANGUAGE_NAMES.get(cfg, cfg.replace("_", " ").title()),
                n=_as_int(row.get("samples_used")),
                base_wer=_pct(base_wer),
                lr_wer=_pct(lr_wer),
                dwer=_pp(dwer),
                base_cer=_pct(base_cer),
                lr_cer=_pct(lr_cer),
                dcer=_pp(dcer),
            )
        )
    return lines


def _win_stats(rows: Sequence[Dict[str, Any]], metric: str) -> Tuple[int, int, int]:
    wins = ties = losses = 0
    base_key = f"base_model_{metric}"
    lr_key = f"latent_reasoning_{metric}"
    for row in rows:
        base = _as_float(row.get(base_key))
        lr = _as_float(row.get(lr_key))
        if base is None or lr is None:
            continue
        delta = base - lr
        if delta > 1e-12:
            wins += 1
        elif delta < -1e-12:
            losses += 1
        else:
            ties += 1
    return wins, ties, losses


def _paper_text(records: Sequence[Dict[str, Any]], rows: Sequence[Dict[str, Any]]) -> str:
    primary = next((r for r in records if r["label"].startswith("Primary streaming")), None)
    if primary is None:
        return (
            "The clean ASR suite evaluates the base Qwen3-ASR-0.6B model and the "
            "latent-reasoning checkpoint on full available splits. The primary "
            "multilingual aggregate was not available because at least one JSON "
            "file was missing."
        )
    wer_w, wer_t, wer_l = _win_stats(rows, "wer")
    cer_w, cer_t, cer_l = _win_stats(rows, "cer")
    return (
        "On the primary streaming multilingual clean evaluation, latent reasoning "
        f"changes weighted WER from {_pct(primary['base_wer'])}% to {_pct(primary['lr_wer'])}% "
        f"({ _pp(primary['dwer']) } pp) and weighted CER from {_pct(primary['base_cer'])}% "
        f"to {_pct(primary['lr_cer'])}% ({ _pp(primary['dcer']) } pp) over {primary['n']} "
        "utterances. Per config, LR improves/ties/regresses WER on "
        f"{wer_w}/{wer_t}/{wer_l} configs and CER on {cer_w}/{cer_t}/{cer_l} configs. "
        "The effect is therefore best described as a small, broad multilingual gain "
        "rather than a large uniform improvement."
    )


def _load_records(
    multi_dir: Path, english_dir: Path
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, Dict[str, Any]]], List[Tuple[str, Dict[str, Any]]], List[str]]:
    records: List[Dict[str, Any]] = []
    fleurs_rows: List[Tuple[str, Dict[str, Any]]] = []
    mls_rows: List[Tuple[str, Dict[str, Any]]] = []
    missing: List[str] = []

    primary_rows: List[Dict[str, Any]] = []
    fleurs_all_rows: List[Dict[str, Any]] = []

    for label, name in FLEURS_FILES:
        payload = _read_json(multi_dir / name)
        if payload is None:
            missing.append(str(multi_dir / name))
            continue
        records.append(_record(label, payload, name))
        rows = _rows(payload)
        primary_rows.extend(rows)
        fleurs_all_rows.extend(rows)
        fleurs_rows.extend((label, row) for row in rows)

    if fleurs_all_rows:
        records.append(
            _aggregate_record(
                "FLEURS 30 overall",
                "google/fleurs",
                "test",
                "yes",
                fleurs_all_rows,
                "fleurs_*_streaming_clean.json",
            )
        )

    for label, name in MLS_FILES:
        payload = _read_json(multi_dir / name)
        if payload is None:
            missing.append(str(multi_dir / name))
            continue
        records.append(_record(label, payload, name))
        rows = _rows(payload)
        primary_rows.extend(rows)
        mls_rows.extend((label, row) for row in rows)

    if primary_rows:
        records.append(
            _aggregate_record(
                "Primary streaming multilingual overall",
                "google/fleurs + facebook/multilingual_librispeech",
                "test",
                "yes",
                primary_rows,
                "fleurs_* + mls_public7",
            )
        )

    for label, name in ENGLISH_FILES:
        payload = _read_json(english_dir / name)
        if payload is None:
            missing.append(str(english_dir / name))
            continue
        records.append(_record(label, payload, name))

    return records, fleurs_rows, mls_rows, missing


def write_report(multi_dir: Path, english_dir: Path, output: Path) -> None:
    records, fleurs_rows, mls_rows, missing = _load_records(multi_dir, english_dir)
    primary_row_dicts = [row for _, row in fleurs_rows + mls_rows]
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")

    lines: List[str] = []
    lines.append("# Full ASR Experiment Report")
    lines.append("")
    lines.append(f"Generated UTC: {generated}")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Model: `Qwen/Qwen3-ASR-0.6B`.")
    lines.append("- Compared systems: base model vs. latent reasoning checkpoint `./latent_qwen_asr_best.pth`.")
    lines.append("- Acoustic condition: clean/native dataset audio only; no synthetic noise and no SNR sweep.")
    lines.append("- Delta definition: `dWER = Base WER - LR WER`, `dCER = Base CER - LR CER`; positive values mean latent reasoning is better.")
    lines.append("- Primary multilingual suite: FLEURS 30 languages plus the public HuggingFace MLS configs, all loaded with `--streaming` and full splits.")
    lines.append("- Secondary English clean suite: existing full clean runs on FLEURS-en, MInDS-14, TED-LIUM, VoxPopuli, and LibriSpeech.")
    lines.append(f"- Qwen3-ASR benchmark family reference: {QWEN_CARD_URL}")
    lines.append(f"- MLS dataset reference: {MLS_CARD_URL}")
    lines.append("")
    lines.append("The Qwen3-ASR model card reports multilingual public benchmark families including MLS, CommonVoice, MLC-SLM, and FLEURS, and lists FLEURS/FLEURS+/FLEURS++ language groupings. In this repository, `facebook/multilingual_librispeech` was checked on 2026-05-04 and exposes seven non-English public configs: `dutch`, `french`, `german`, `italian`, `polish`, `portuguese`, and `spanish`. The HF MLS card reports test-set sizes of 3,394 German, 3,075 Dutch, 2,426 French, 2,385 Spanish, 1,262 Italian, 871 Portuguese, and 520 Polish utterances, for 13,933 MLS public test utterances total. English is therefore reported separately through the LibriSpeech clean/other suite rather than being merged into the MLS JSON.")
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append("env MAX_SAMPLES_PER_CONFIG=0 CASE_FILTER='fleurs_core12|fleurs_extra8|fleurs_extra10' ./run_lr_multilingual_asr_streaming.sh")
    lines.append("env MAX_SAMPLES_PER_CONFIG=0 CASE_FILTER=mls_public7 OUT_DIR=/user_data/lr_whisper/eval_runs/multilingual_asr_streaming_20260503_182239 ./run_lr_multilingual_asr_streaming.sh")
    lines.append("RUN_CLEAN=1 SNR_DB_LEVELS= MAX_SAMPLES_PER_CONFIG=0 OUT_DIR=/user_data/lr_whisper/eval_runs/hf_asr_showcase_full_20260503_152506 ./run_lr_hf_asr_showcase.sh")
    lines.append("```")
    lines.append("")
    lines.append("## Aggregate Results")
    lines.append("")
    lines.append("All WER/CER values are percentages; deltas are percentage points.")
    lines.append("")
    lines.extend(_metric_table(records))
    lines.append("")
    lines.append("## Paper-Ready Summary")
    lines.append("")
    lines.append(_paper_text(records, primary_row_dicts))
    lines.append("")
    lines.append("## FLEURS Per-Language Results")
    lines.append("")
    lines.extend(_row_table(fleurs_rows))
    lines.append("")
    if mls_rows:
        lines.append("## MLS Public Config Results")
        lines.append("")
        lines.extend(_row_table(mls_rows))
        lines.append("")
    lines.append("## LibriSpeech Subsets")
    lines.append("")
    librispeech = _read_json(english_dir / "librispeech_clean_other_clean.json")
    if librispeech is not None:
        lines.extend(_row_table([("LibriSpeech", row) for row in _rows(librispeech)]))
    else:
        lines.append("`librispeech_clean_other_clean.json` was not found.")
    lines.append("")
    lines.append("## Reporting Notes")
    lines.append("")
    lines.append("- For languages without reliable whitespace word segmentation, especially Mandarin Chinese, Cantonese, Japanese, and Thai, CER should be interpreted as the primary metric; WER is tokenization-sensitive.")
    lines.append("- The secondary English clean suite was run before the streaming multilingual run and its JSON files do not record `streaming`; it is still clean full-split evaluation with synthetic noise disabled.")
    lines.append("- CommonVoice and MLC-SLM are part of the Qwen3-ASR benchmark family but are not included in these completed JSON artifacts. CommonVoice often requires dataset license/access handling, and MLC-SLM needs a confirmed public HF source before running in streaming mode.")
    lines.append("- The aborted `mls_public8` attempt is excluded; it used a config set that did not exist in the checked HF dataset.")
    if missing:
        lines.append("")
        lines.append("## Missing Expected JSON Files")
        lines.append("")
        for item in missing:
            lines.append(f"- `{item}`")
    lines.append("")

    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--multi-dir",
        type=Path,
        default=Path("eval_runs/multilingual_asr_streaming_20260503_182239"),
    )
    parser.add_argument(
        "--english-dir",
        type=Path,
        default=Path("eval_runs/hf_asr_showcase_full_20260503_152506"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval_runs/multilingual_asr_streaming_20260503_182239/full_experiment_report.md"),
    )
    args = parser.parse_args()
    write_report(args.multi_dir, args.english_dir, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
