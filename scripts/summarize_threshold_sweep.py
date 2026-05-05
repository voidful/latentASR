#!/usr/bin/env python3
"""Summarize LatentASR threshold-sweep JSON/log files."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_BASELINE_DIR = Path("eval_runs/hf_asr_showcase_full_20260503_152506")

BASELINE_JSON = {
    "fleurs_en_us": "fleurs_en_us_clean.json",
    "voxpopuli_en": "voxpopuli_en_clean.json",
}

DATASET_LABEL = {
    "fleurs_en_us": "FLEURS en-US",
    "voxpopuli_en": "VoxPopuli en",
}

THETA_VALUE = {
    "full": "-2.0",
    "neg0p2": "-0.2",
    "zero": "0.0",
    "pos0p2": "0.2",
    "pos0p5": "0.5",
}


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


def _pct(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value * 100.0:.3f}"


def _pp(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value * 100.0:+.3f}"


def _fmt_float(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sample_count(rows: Iterable[Dict[str, Any]]) -> int:
    return sum(_as_int(row.get("samples_used")) for row in rows)


def _weighted(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
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


def _baseline_metrics(baseline_dir: Path, tag: str) -> Tuple[Optional[float], Optional[float]]:
    name = BASELINE_JSON.get(tag)
    if not name:
        return None, None
    path = baseline_dir / name
    if not path.exists():
        return None, None
    payload = _read_json(path)
    summary = payload.get("summary") or {}
    base_wer = _as_float(summary.get("base_model_weighted_wer"))
    base_cer = _as_float(summary.get("base_model_weighted_cer"))
    if base_wer is None:
        base_wer = _weighted(payload.get("rows") or [], "base_model_wer")
    if base_cer is None:
        base_cer = _weighted(payload.get("rows") or [], "base_model_cer")
    return base_wer, base_cer


def _parse_step_distribution(log_path: Path) -> Dict[int, int]:
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="ignore").replace("\r", "\n")
    dist: Dict[int, int] = {}
    for step, count in re.findall(r"N=(\d+):\s+(\d+)\s+\(", text):
        dist[int(step)] = int(count)
    return dist


def _step_stats(dist: Dict[int, int]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    total = sum(dist.values())
    if total <= 0:
        return None, None, None
    avg_steps = sum(step * count for step, count in dist.items()) / total
    skip_rate = dist.get(0, 0) / total
    full_rate = dist.get(4, 0) / total
    return avg_steps, skip_rate, full_rate


def _tag_theta(path: Path) -> Tuple[str, str]:
    stem = path.stem
    marker = "_theta_"
    if marker not in stem:
        return stem, "unknown"
    tag, theta = stem.split(marker, 1)
    return tag, theta


def load_records(out_dir: Path, baseline_dir: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for json_path in sorted(out_dir.glob("*_theta_*.json")):
        tag, theta = _tag_theta(json_path)
        payload = _read_json(json_path)
        rows = payload.get("rows") or []
        summary = payload.get("summary") or {}
        base_wer, base_cer = _baseline_metrics(baseline_dir, tag)
        lr_wer = _as_float(summary.get("latent_reasoning_weighted_wer"))
        lr_cer = _as_float(summary.get("latent_reasoning_weighted_cer"))
        if lr_wer is None:
            lr_wer = _weighted(rows, "latent_reasoning_wer")
        if lr_cer is None:
            lr_cer = _weighted(rows, "latent_reasoning_cer")
        dist = _parse_step_distribution(out_dir / "logs" / f"{tag}_theta_{theta}.log")
        avg_steps, skip_rate, full_rate = _step_stats(dist)
        records.append(
            {
                "tag": tag,
                "dataset": DATASET_LABEL.get(tag, tag),
                "theta": theta,
                "theta_value": THETA_VALUE.get(theta, theta),
                "n": _sample_count(rows),
                "base_wer": base_wer,
                "lr_wer": lr_wer,
                "dwer": None if base_wer is None or lr_wer is None else base_wer - lr_wer,
                "base_cer": base_cer,
                "lr_cer": lr_cer,
                "dcer": None if base_cer is None or lr_cer is None else base_cer - lr_cer,
                "avg_steps": avg_steps,
                "skip_rate": skip_rate,
                "full_rate": full_rate,
                "step_dist": dist,
                "json": json_path.name,
            }
        )
    return records


def _sort_key(record: Dict[str, Any]) -> Tuple[str, int]:
    order = {"full": 0, "neg0p2": 1, "zero": 2, "pos0p2": 3, "pos0p5": 4}
    return record["tag"], order.get(record["theta"], 99)


def _table(records: List[Dict[str, Any]]) -> List[str]:
    lines = [
        "| Dataset | Theta | N | Avg steps | Skip N=0 | Full N=4 | Base WER | LR WER | dWER | Base CER | LR CER | dCER |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rec in sorted(records, key=_sort_key):
        lines.append(
            "| {dataset} | {theta} | {n} | {avg_steps} | {skip} | {full} | {base_wer} | {lr_wer} | {dwer} | {base_cer} | {lr_cer} | {dcer} |".format(
                dataset=rec["dataset"],
                theta=rec["theta_value"],
                n=rec["n"],
                avg_steps=_fmt_float(rec["avg_steps"]),
                skip=_pct(rec["skip_rate"]),
                full=_pct(rec["full_rate"]),
                base_wer=_pct(rec["base_wer"]),
                lr_wer=_pct(rec["lr_wer"]),
                dwer=_pp(rec["dwer"]),
                base_cer=_pct(rec["base_cer"]),
                lr_cer=_pct(rec["lr_cer"]),
                dcer=_pp(rec["dcer"]),
            )
        )
    return lines


def write_report(out_dir: Path, records: List[Dict[str, Any]]) -> Path:
    report_path = out_dir / "threshold_sweep_report.md"
    lines: List[str] = []
    lines.append("# LatentASR Threshold Sweep")
    lines.append("")
    lines.append(f"Generated UTC: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("`Theta=-2.0` is the no-halting control: value-head scores are bounded in [-1, 1], so no early halt is triggered.")
    lines.append("`dWER = Base WER - LR WER`; positive values mean LR is better. WER/CER and rates are percentages.")
    lines.append("")
    lines.extend(_table(records))
    lines.append("")
    lines.append("## Step Distributions")
    lines.append("")
    lines.append("| Dataset | Theta | N=0 | N=1 | N=2 | N=3 | N=4 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for rec in sorted(records, key=_sort_key):
        dist = rec["step_dist"]
        lines.append(
            "| {dataset} | {theta} | {n0} | {n1} | {n2} | {n3} | {n4} |".format(
                dataset=rec["dataset"],
                theta=rec["theta_value"],
                n0=dist.get(0, 0),
                n1=dist.get(1, 0),
                n2=dist.get(2, 0),
                n3=dist.get(3, 0),
                n4=dist.get(4, 0),
            )
        )
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    args = parser.parse_args()
    records = load_records(args.out_dir, args.baseline_dir)
    report = write_report(args.out_dir, records)
    print(report)
    print("\n".join(_table(records)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
