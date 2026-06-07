#!/usr/bin/env python3
"""Summarize paper TBD ablation evaluations into Markdown/LaTeX-ready tables."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


BASELINE_JSON = {
    "fleurs": "fleurs_en_us_clean.json",
    "voxpopuli": "voxpopuli_en_clean.json",
}


def as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def pct(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value * 100.0:.{digits}f}"


def pp(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value * 100.0:+.{digits}f}"


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def weighted(rows: Iterable[Dict[str, Any]], key: str) -> Optional[float]:
    numer = 0.0
    denom = 0
    for row in rows:
        value = as_float(row.get(key))
        count = int(row.get("samples_used") or 0)
        if value is None or count <= 0:
            continue
        numer += value * count
        denom += count
    if denom <= 0:
        return None
    return numer / denom


def baseline_metric(baseline_dir: Path, dataset_tag: str, metric: str) -> Optional[float]:
    path = baseline_dir / BASELINE_JSON[dataset_tag]
    payload = read_json(path)
    summary = payload.get("summary") or {}
    value = as_float(summary.get(f"base_model_weighted_{metric}"))
    if value is not None:
        return value
    return weighted(payload.get("rows") or [], f"base_model_{metric}")


def parse_step_distribution(log_path: Path) -> Dict[int, int]:
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="ignore").replace("\r", "\n")
    dist: Dict[int, int] = {}
    for step, count in re.findall(r"N=(\d+):\s+(\d+)\s+\(", text):
        dist[int(step)] = int(count)
    return dist


def step_stats(dist: Dict[int, int]) -> Tuple[Optional[float], Optional[float]]:
    total = sum(dist.values())
    if total <= 0:
        return None, None
    avg_steps = sum(step * count for step, count in dist.items()) / total
    skip = dist.get(0, 0) / total
    return avg_steps, skip


def parse_eval_name(path: Path) -> Optional[Tuple[str, str, str]]:
    marker = "_theta_"
    stem = path.stem
    if marker not in stem:
        return None
    prefix, theta = stem.split(marker, 1)
    for dataset_tag in ("voxpopuli", "fleurs"):
        suffix = f"_{dataset_tag}"
        if prefix.endswith(suffix):
            return prefix[: -len(suffix)], dataset_tag, theta
    return None


def load_records(out_dir: Path, baseline_dir: Path) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    records: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for path in sorted(out_dir.glob("*_theta_*.json")):
        parsed = parse_eval_name(path)
        if not parsed:
            continue
        variant, dataset_tag, theta = parsed
        payload = read_json(path)
        rows = payload.get("rows") or []
        summary = payload.get("summary") or {}
        wer = as_float(summary.get("latent_reasoning_weighted_wer"))
        cer = as_float(summary.get("latent_reasoning_weighted_cer"))
        if wer is None:
            wer = weighted(rows, "latent_reasoning_wer")
        if cer is None:
            cer = weighted(rows, "latent_reasoning_cer")
        base_wer = baseline_metric(baseline_dir, dataset_tag, "wer")
        base_cer = baseline_metric(baseline_dir, dataset_tag, "cer")
        dist = parse_step_distribution(out_dir / "logs" / f"{path.stem}.log")
        avg_steps, skip = step_stats(dist)
        key = (variant, dataset_tag, theta)
        records[key] = {
            "variant": variant,
            "dataset": dataset_tag,
            "theta": theta,
            "wer": wer,
            "cer": cer,
            "base_wer": base_wer,
            "base_cer": base_cer,
            "dwer_pp": None if wer is None or base_wer is None else wer - base_wer,
            "dcer_pp": None if cer is None or base_cer is None else cer - base_cer,
            "avg_steps": avg_steps,
            "skip": skip,
            "dist": dist,
            "json": path.name,
        }
    return records


def rec(
    records: Dict[Tuple[str, str, str], Dict[str, Any]],
    variant: str,
    dataset: str = "fleurs",
    theta: str = "zero",
) -> Optional[Dict[str, Any]]:
    return records.get((variant, dataset, theta))


def table_component(records: Dict[Tuple[str, str, str], Dict[str, Any]]) -> List[str]:
    labels = [
        ("n4", "Full \\method{} ($N{=}4$, $\\theta{=}0.0$)"),
        ("component_no_bounded", "\\quad $-$ bounded delta ($L_2$ + scale $s_k$)"),
        ("component_no_gate", "\\quad $-$ sigmoid gate ($g_k$ fixed at $1$)"),
        ("component_no_anchor", "\\quad $-$ fixed-embedding anchor ($\\mathbf{e}_{\\texttt{LT}}$ removed)"),
    ]
    lines = ["### Component Ablation", "", "| Variant | WER (%) | ΔWER (pp) |", "|---|---:|---:|"]
    for variant, label in labels:
        r = rec(records, variant)
        lines.append(f"| {label} | {pct(r['wer']) if r else '-'} | {pp(r['dwer_pp']) if r else '-'} |")
    return lines


def table_n_sweep(records: Dict[Tuple[str, str, str], Dict[str, Any]]) -> List[str]:
    variants = [("n1", "1"), ("n2", "2"), ("n4", "\\textbf{4}"), ("n8", "8")]
    lines = [
        "### N Sweep",
        "",
        "| N | FLEURS WER (%) | ΔWER (pp) | VoxPopuli WER (%) | ΔWER (pp) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for variant, label in variants:
        f = rec(records, variant, "fleurs")
        v = rec(records, variant, "voxpopuli")
        lines.append(
            f"| {label} | {pct(f['wer']) if f else '-'} | {pp(f['dwer_pp']) if f else '-'} | "
            f"{pct(v['wer']) if v else '-'} | {pp(v['dwer_pp']) if v else '-'} |"
        )
    return lines


def table_pneg(records: Dict[Tuple[str, str, str], Dict[str, Any]]) -> List[str]:
    full = rec(records, "n4")
    p0 = rec(records, "pneg0")
    skips = []
    for theta in ("full", "neg0p2", "zero", "pos0p2", "pos0p5"):
        row = rec(records, "pneg0", "fleurs", theta)
        if row and row["skip"] is not None:
            skips.append(row["skip"])
    pos = rec(records, "pneg0", "fleurs", "pos0p2")
    skip_at_pos = pos["skip"] if pos else None
    skip_range = "-" if not skips else f"[{min(skips) * 100.0:.1f}, {max(skips) * 100.0:.1f}]"
    lines = [
        "### Forced-Negative Sampling",
        "",
        "| Setting | WER (%) | ΔWER (pp) | Skip @ θ=+0.2 | Skip range (%) |",
        "|---|---:|---:|---:|---:|",
        f"| Full ($p_{{\\text{{neg}}}}{{=}}0.3$) | {pct(full['wer']) if full else '-'} | {pp(full['dwer_pp']) if full else '-'} | 100.0% | [0, 100] |",
        f"| $-$ Forced-neg ($p_{{\\text{{neg}}}}{{=}}0.0$) | {pct(p0['wer']) if p0 else '-'} | {pp(p0['dwer_pp']) if p0 else '-'} | {('-' if skip_at_pos is None else f'{skip_at_pos * 100.0:.1f}%')} | {skip_range} |",
    ]
    return lines


def table_activation(records: Dict[Tuple[str, str, str], Dict[str, Any]]) -> List[str]:
    variants = [(f"activation_{n}", str(n)) for n in range(100, 801, 100)]
    lines = [
        "### Activation Set Scaling",
        "",
        "| #utts | FLEURS WER (%) | ΔWER (pp) | VoxPopuli WER (%) | ΔWER (pp) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for variant, label in variants:
        f = rec(records, variant, "fleurs")
        v = rec(records, variant, "voxpopuli")
        lines.append(
            f"| {label} | {pct(f['wer']) if f else '-'} | {pp(f['dwer_pp']) if f else '-'} | "
            f"{pct(v['wer']) if v else '-'} | {pp(v['dwer_pp']) if v else '-'} |"
        )
    return lines


def table_pneg_sweep(records: Dict[Tuple[str, str, str], Dict[str, Any]]) -> List[str]:
    theta_values = {
        "full": "-2.0",
        "neg0p2": "-0.2",
        "zero": "0.0",
        "pos0p2": "+0.2",
        "pos0p5": "+0.5",
    }
    lines = [
        "### p_neg=0.0 FLEURS Threshold Details",
        "",
        "| θ | Avg steps | Skip (%) | WER (%) | ΔWER (pp) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for theta in ("full", "neg0p2", "zero", "pos0p2", "pos0p5"):
        r = rec(records, "pneg0", "fleurs", theta)
        if not r:
            lines.append(f"| {theta_values[theta]} | - | - | - | - |")
            continue
        avg = "-" if r["avg_steps"] is None else f"{r['avg_steps']:.2f}"
        skip = "-" if r["skip"] is None else f"{r['skip'] * 100.0:.1f}"
        lines.append(f"| {theta_values[theta]} | {avg} | {skip} | {pct(r['wer'])} | {pp(r['dwer_pp'])} |")
    return lines


def write_report(out_dir: Path, records: Dict[Tuple[str, str, str], Dict[str, Any]]) -> Path:
    lines: List[str] = ["# Paper TBD Results", ""]
    for section in (
        table_component(records),
        table_n_sweep(records),
        table_pneg(records),
        table_pneg_sweep(records),
        table_activation(records),
    ):
        lines.extend(section)
        lines.append("")
    path = out_dir / "paper_tbd_results.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    args = parser.parse_args()
    records = load_records(args.out_dir, args.baseline_dir)
    report = write_report(args.out_dir, records)
    print(report)
    print(report.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
