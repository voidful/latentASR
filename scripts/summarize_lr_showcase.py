#!/usr/bin/env python3
"""Summarize base-vs-latent ASR showcase JSON files."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _as_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _as_int(v: Any) -> int:
    try:
        return int(v or 0)
    except Exception:
        return 0


def _fmt_metric(v: Optional[float]) -> str:
    if v is None:
        return "-"
    return f"{v:.6f}"


def _fmt_rel(v: Optional[float]) -> str:
    if v is None:
        return "-"
    return f"{v:+.2f}%"


def _condition_from_name(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_clean"):
        return "clean"
    for part in stem.split("_"):
        if part.startswith("snr") and part.endswith("db"):
            return part
    return "unknown"


def _sample_count(rows: Iterable[Dict[str, Any]]) -> int:
    return sum(_as_int(row.get("samples_used")) for row in rows)


def load_records(out_dir: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in sorted(out_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            records.append(
                {
                    "json": path.name,
                    "dataset": path.stem,
                    "condition": _condition_from_name(path),
                    "error": str(exc),
                }
            )
            continue

        summary = payload.get("summary") or {}
        rows = payload.get("rows") or []
        base_wer = _as_float(summary.get("base_model_weighted_wer"))
        latent_wer = _as_float(summary.get("latent_reasoning_weighted_wer"))
        base_cer = _as_float(summary.get("base_model_weighted_cer"))
        latent_cer = _as_float(summary.get("latent_reasoning_weighted_cer"))
        delta_wer = None if base_wer is None or latent_wer is None else base_wer - latent_wer
        delta_cer = None if base_cer is None or latent_cer is None else base_cer - latent_cer
        rel_wer = None
        if delta_wer is not None and base_wer not in (None, 0.0):
            rel_wer = delta_wer / base_wer * 100.0

        records.append(
            {
                "json": path.name,
                "dataset": payload.get("dataset_name") or path.stem,
                "configs": ",".join(str(c) for c in payload.get("configs") or []),
                "condition": _condition_from_name(path),
                "samples": _sample_count(rows),
                "base_wer": base_wer,
                "latent_wer": latent_wer,
                "delta_wer": delta_wer,
                "rel_wer": rel_wer,
                "base_cer": base_cer,
                "latent_cer": latent_cer,
                "delta_cer": delta_cer,
                "error": None,
            }
        )
    return records


def write_report(out_dir: Path, records: List[Dict[str, Any]]) -> Path:
    report_path = out_dir / "showcase_report.md"
    valid = [r for r in records if not r.get("error")]
    wins = sorted(
        [r for r in valid if (r.get("delta_wer") or 0.0) > 0.0],
        key=lambda r: r.get("delta_wer") or 0.0,
        reverse=True,
    )
    regressions = sorted(
        [r for r in valid if (r.get("delta_wer") or 0.0) < 0.0],
        key=lambda r: r.get("delta_wer") or 0.0,
    )

    lines: List[str] = []
    lines.append("# LR HuggingFace ASR Showcase Report")
    lines.append("")
    lines.append(f"- Generated UTC: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    lines.append(f"- Output directory: `{out_dir}`")
    lines.append("- Delta WER is `base_model_wer - latent_reasoning_wer`; positive means LR is better.")
    lines.append("")

    lines.append("## Best LR Wins")
    lines.append("")
    lines.append("| Rank | Dataset | Configs | Condition | N | Base WER | LR WER | Delta WER | Relative | Delta CER | JSON |")
    lines.append("|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|")
    for rank, rec in enumerate(wins[:20], start=1):
        lines.append(
            "| {rank} | {dataset} | {configs} | {condition} | {samples} | {base} | {lat} | {delta} | {rel} | {dcer} | `{json}` |".format(
                rank=rank,
                dataset=rec["dataset"],
                configs=rec["configs"] or "-",
                condition=rec["condition"],
                samples=rec["samples"],
                base=_fmt_metric(rec["base_wer"]),
                lat=_fmt_metric(rec["latent_wer"]),
                delta=_fmt_metric(rec["delta_wer"]),
                rel=_fmt_rel(rec["rel_wer"]),
                dcer=_fmt_metric(rec["delta_cer"]),
                json=rec["json"],
            )
        )
    if not wins:
        lines.append("| - | - | - | - | - | - | - | - | - | - | - |")
    lines.append("")

    lines.append("## All Cases")
    lines.append("")
    lines.append("| Dataset | Configs | Condition | N | Base WER | LR WER | Delta WER | Relative | Base CER | LR CER | Delta CER | JSON |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for rec in sorted(valid, key=lambda r: (r["condition"], r["dataset"], r["configs"])):
        lines.append(
            "| {dataset} | {configs} | {condition} | {samples} | {base} | {lat} | {delta} | {rel} | {bcer} | {lcer} | {dcer} | `{json}` |".format(
                dataset=rec["dataset"],
                configs=rec["configs"] or "-",
                condition=rec["condition"],
                samples=rec["samples"],
                base=_fmt_metric(rec["base_wer"]),
                lat=_fmt_metric(rec["latent_wer"]),
                delta=_fmt_metric(rec["delta_wer"]),
                rel=_fmt_rel(rec["rel_wer"]),
                bcer=_fmt_metric(rec["base_cer"]),
                lcer=_fmt_metric(rec["latent_cer"]),
                dcer=_fmt_metric(rec["delta_cer"]),
                json=rec["json"],
            )
        )
    lines.append("")

    if regressions:
        lines.append("## Regressions To Check")
        lines.append("")
        lines.append("| Dataset | Configs | Condition | N | Delta WER | Relative | JSON |")
        lines.append("|---|---|---|---:|---:|---:|---|")
        for rec in regressions[:20]:
            lines.append(
                "| {dataset} | {configs} | {condition} | {samples} | {delta} | {rel} | `{json}` |".format(
                    dataset=rec["dataset"],
                    configs=rec["configs"] or "-",
                    condition=rec["condition"],
                    samples=rec["samples"],
                    delta=_fmt_metric(rec["delta_wer"]),
                    rel=_fmt_rel(rec["rel_wer"]),
                    json=rec["json"],
                )
            )
        lines.append("")

    errors = [r for r in records if r.get("error")]
    if errors:
        lines.append("## JSON Load Errors")
        lines.append("")
        for rec in errors:
            lines.append(f"- `{rec['json']}`: {rec['error']}")
        lines.append("")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: summarize_lr_showcase.py OUT_DIR", file=sys.stderr)
        return 2
    out_dir = Path(sys.argv[1]).expanduser().resolve()
    records = load_records(out_dir)
    report = write_report(out_dir, records)
    wins = sum(1 for r in records if not r.get("error") and (r.get("delta_wer") or 0.0) > 0.0)
    losses = sum(1 for r in records if not r.get("error") and (r.get("delta_wer") or 0.0) < 0.0)
    print(f"records={len(records)} wins={wins} regressions={losses}")
    print(f"report={report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
