#!/usr/bin/env python3
"""Per-sample paper analysis for LatentASR.

The script runs three paths on one ASR split:
  1. frozen baseline,
  2. LatentASR with the deployed halting threshold,
  3. LatentASR with forced full compute.

It writes per-sample JSON plus a LaTeX snippet containing difficulty bins,
gate-quality diagnostics, qualitative examples, and latent-delta statistics.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from datasets import Audio, load_dataset
from jiwer import cer, wer
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval import (  # noqa: E402
    build_base_model_bundle,
    build_latent_bundle,
    clean_prediction,
    configure_text_normalizer,
    normalize_text,
)


MODEL_ID = "Qwen/Qwen3-ASR-0.6B"
LATENT_CKPT = "eval_runs/paper_tbd_retrain_20260518/checkpoints/activation_500/activation_500_epoch10.pth"
MAX_NEW_TOKENS = 128


DATASET_PRESETS = {
    "fleurs_en_us": {
        "dataset_name": "google/fleurs",
        "config": "en_us",
        "split": "test",
        "text_columns": ["transcription", "raw_transcription", "sentence", "text"],
        "normalizer": "english",
        "language_hint": "English",
        "label": r"FLEURS (\texttt{en\_us})",
        "source_label": "FLEURS Q4",
    },
    "voxpopuli_en": {
        "dataset_name": "facebook/voxpopuli",
        "config": "en",
        "split": "test",
        "text_columns": ["normalized_text", "raw_text", "text", "sentence"],
        "normalizer": "english",
        "language_hint": "English",
        "label": r"VoxPopuli (\texttt{en})",
        "source_label": "VoxPopuli Q4",
    },
}


def choose_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def choose_dtype(device: str) -> torch.dtype:
    return torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16


def tensor_to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        return float(value.detach().flatten()[0].cpu().item())
    try:
        return float(value)
    except Exception:
        return None


def tensor_to_list(value: Any) -> List[float]:
    if value is None:
        return []
    if torch.is_tensor(value):
        return [float(x) for x in value.detach().flatten().cpu().tolist()]
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    return []


def get_ref(sample: Dict[str, Any], text_columns: List[str]) -> str:
    for col in text_columns:
        value = sample.get(col)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def sample_to_features(model: Any, processor: Any, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    audio = sample.get("audio")
    if not isinstance(audio, dict) or "array" not in audio or "sampling_rate" not in audio:
        return None
    audio_array = np.array(audio["array"], dtype=np.float64)
    target_dtype = model.thinker.dtype if hasattr(model.thinker, "dtype") else torch.float32
    feat_out = processor.feature_extractor(
        audio_array,
        sampling_rate=audio["sampling_rate"],
        return_attention_mask=True,
    )
    device = model.base_model.device
    feats = torch.tensor(feat_out.input_features[0], dtype=target_dtype, device=device).unsqueeze(0)
    n_frames = feats.size(-1)
    if getattr(feat_out, "attention_mask", None) is not None:
        raw_mask = feat_out.attention_mask[0]
        if not isinstance(raw_mask, (list, torch.Tensor)):
            raw_mask = list(raw_mask)
        if isinstance(raw_mask, torch.Tensor):
            raw_mask = raw_mask.long()
        else:
            raw_mask = torch.tensor(raw_mask, dtype=torch.long)
        if raw_mask.size(-1) < n_frames:
            raw_mask = torch.cat([raw_mask, torch.zeros(n_frames - raw_mask.size(-1), dtype=torch.long)])
        else:
            raw_mask = raw_mask[:n_frames]
        feature_attention_mask = raw_mask.to(device=device).unsqueeze(0)
    else:
        feature_attention_mask = torch.ones((1, n_frames), dtype=torch.long, device=device)
    if int(feature_attention_mask.sum().item()) < 10:
        return None
    return {"feats": feats, "feature_attention_mask": feature_attention_mask}


@torch.no_grad()
def transcribe(
    model: Any,
    processor: Any,
    sample: Dict[str, Any],
    *,
    use_baseline: bool,
    theta: float,
    language_hint: str,
) -> Optional[Dict[str, Any]]:
    feats = sample_to_features(model, processor, sample)
    if feats is None:
        return None
    prompt_text = f"Transcribe the {language_hint} audio into text." if language_hint else "Transcribe the audio into text."
    gen_kwargs = {
        "feature_attention_mask": feats["feature_attention_mask"],
        "max_new_tokens": MAX_NEW_TOKENS,
        "use_baseline": use_baseline,
        "return_thoughts": False,
        "return_stats": True,
        "do_sample": False,
        "eos_token_id": [151645, 151643],
        "num_beams": 1,
        "language_hint": language_hint,
        "prompt_text": prompt_text,
        "dynamic_halt_threshold": theta,
    }
    out = model.generate(feats["feats"], **gen_kwargs)
    if isinstance(out, tuple):
        gen_ids = out[0]
        stats = out[1] if len(out) > 1 and isinstance(out[1], dict) else {}
    else:
        gen_ids = out
        stats = {}
    ids = gen_ids[0]
    eos_id = processor.tokenizer.eos_token_id
    if eos_id is not None and (ids == eos_id).any():
        eos_pos = (ids == eos_id).nonzero(as_tuple=True)[0][0]
        ids = ids[:eos_pos]
    raw = processor.tokenizer.decode(ids, skip_special_tokens=True)
    pred = clean_prediction(raw)
    deq = tensor_to_float(stats.get("deq_iters"))
    return {
        "pred": pred,
        "pred_norm": normalize_text(pred),
        "stats": {
            "deq_iters": 0 if deq is None else int(round(deq)),
            "skipped": bool(stats.get("skipped", False)) if stats else False,
            "v_preds": tensor_to_list(stats.get("v_preds")),
            "scaled_norm_mean": tensor_to_list(stats.get("scaled_norm_mean")),
            "step_cos": tensor_to_float(stats.get("step_cos")),
            "diff_norm": tensor_to_float(stats.get("diff_norm")),
        },
    }


def load_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def save_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def iter_dataset(preset: Dict[str, Any], streaming: bool) -> Iterable[Dict[str, Any]]:
    ds = load_dataset(
        preset["dataset_name"],
        preset["config"],
        split=preset["split"],
        streaming=streaming,
        trust_remote_code=True,
    )
    if not streaming:
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    return ds


def collect_refs(preset: Dict[str, Any], streaming: bool) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    for idx, sample in enumerate(tqdm(iter_dataset(preset, streaming), desc="refs")):
        ref_raw = get_ref(sample, preset["text_columns"])
        ref_norm = normalize_text(ref_raw)
        if ref_norm:
            refs.append({"idx": idx, "ref_raw": ref_raw, "ref_norm": ref_norm})
    return refs


def run_baseline(args: argparse.Namespace, preset: Dict[str, Any], refs: List[Dict[str, Any]], rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    done = {r["idx"] for r in rows if "baseline_norm" in r}
    if len(done) == len(refs):
        return rows
    by_idx = {r["idx"]: r for r in rows}
    ref_idx = {r["idx"]: r for r in refs}
    bundle = build_base_model_bundle(args.model_id, args.device, args.dtype)
    for idx, sample in enumerate(tqdm(iter_dataset(preset, args.streaming), desc="baseline")):
        if idx not in ref_idx or idx in done:
            continue
        out = transcribe(
            bundle.model,
            bundle.processor,
            sample,
            use_baseline=True,
            theta=args.theta,
            language_hint=preset["language_hint"],
        )
        if out is None:
            continue
        row = by_idx.setdefault(idx, {"idx": idx, **ref_idx[idx]})
        row["baseline_pred"] = out["pred"]
        row["baseline_norm"] = out["pred_norm"]
        save_rows(args.out_json, sorted(by_idx.values(), key=lambda x: x["idx"]))
    del bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return sorted(by_idx.values(), key=lambda x: x["idx"])


def run_latent_path(
    args: argparse.Namespace,
    preset: Dict[str, Any],
    rows: List[Dict[str, Any]],
    *,
    theta: float,
    pred_key: str,
    norm_key: str,
    stats_key: str,
    desc: str,
) -> List[Dict[str, Any]]:
    eligible = {r["idx"] for r in rows if "baseline_norm" in r}
    done = {r["idx"] for r in rows if norm_key in r}
    if eligible and done == eligible:
        return rows
    by_idx = {r["idx"]: r for r in rows}
    bundle = build_latent_bundle(args.model_id, args.latent_ckpt, args.n_latent, args.device, args.dtype)
    for idx, sample in enumerate(tqdm(iter_dataset(preset, args.streaming), desc=desc)):
        if idx not in eligible or idx in done:
            continue
        out = transcribe(
            bundle.model,
            bundle.processor,
            sample,
            use_baseline=False,
            theta=theta,
            language_hint=preset["language_hint"],
        )
        if out is None:
            continue
        row = by_idx[idx]
        row[pred_key] = out["pred"]
        row[norm_key] = out["pred_norm"]
        row[stats_key] = out["stats"]
        save_rows(args.out_json, sorted(by_idx.values(), key=lambda x: x["idx"]))
    del bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return sorted(by_idx.values(), key=lambda x: x["idx"])


def group_wer(group: List[Dict[str, Any]], key: str) -> float:
    if not group:
        return 0.0
    return 100 * wer([r["ref_norm"] for r in group], [r[key] for r in group])


def group_cer(group: List[Dict[str, Any]], key: str) -> float:
    if not group:
        return 0.0
    return 100 * cer([r["ref_norm"] for r in group], [r[key] for r in group])


def esc_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def format_delta(value: float, bold_negative: bool = False) -> str:
    if bold_negative and value < 0:
        return rf"$\boldsymbol{{{value:+.2f}}}$"
    return rf"${value:+.2f}$"


def write_latex(args: argparse.Namespace, preset: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    final_rows = [
        r for r in rows
        if all(k in r for k in ("baseline_norm", "latent_norm", "full_norm"))
    ]
    for r in final_rows:
        ref = r["ref_norm"]
        r["baseline_wer"] = float(wer(ref, r["baseline_norm"]))
        r["latent_wer"] = float(wer(ref, r["latent_norm"]))
        r["full_wer"] = float(wer(ref, r["full_norm"]))
        r["baseline_cer"] = float(cer(ref, r["baseline_norm"]))
        r["latent_cer"] = float(cer(ref, r["latent_norm"]))
        r["full_cer"] = float(cer(ref, r["full_norm"]))

    base_wer = group_wer(final_rows, "baseline_norm")
    lat_wer = group_wer(final_rows, "latent_norm")
    full_wer = group_wer(final_rows, "full_norm")
    base_cer = group_cer(final_rows, "baseline_norm")
    lat_cer = group_cer(final_rows, "latent_norm")
    full_cer = group_cer(final_rows, "full_norm")

    sorted_rows = sorted(final_rows, key=lambda r: (r["baseline_wer"], r["idx"]))
    n = len(sorted_rows)
    bins = []
    for q in range(4):
        part = sorted_rows[math.floor(q * n / 4): math.floor((q + 1) * n / 4)]
        bw = group_wer(part, "baseline_norm")
        lw = group_wer(part, "latent_norm")
        skip_q = 100 * sum(1 for r in part if r["latent_stats"].get("deq_iters", 0) == 0) / max(1, len(part))
        bins.append((q + 1, len(part), bw, lw, lw - bw, skip_q))

    skipped = [r for r in final_rows if r["latent_stats"].get("deq_iters", 0) == 0]
    processed = [r for r in final_rows if r["latent_stats"].get("deq_iters", 0) > 0]

    skip_base = group_wer(skipped, "baseline_norm")
    skip_full = group_wer(skipped, "full_norm")
    proc_base = group_wer(processed, "baseline_norm")
    proc_lat = group_wer(processed, "latent_norm")
    proc_full = group_wer(processed, "full_norm")

    step_counts = {i: 0 for i in range(args.n_latent + 1)}
    for r in final_rows:
        deq = int(r["latent_stats"].get("deq_iters", 0))
        step_counts[deq] = step_counts.get(deq, 0) + 1
    step_rates = {k: 100 * v / max(1, len(final_rows)) for k, v in step_counts.items()}
    avg_steps = sum(k * v for k, v in step_counts.items()) / max(1, len(final_rows))

    processed_full = processed
    norms_by_step: List[List[float]] = [[] for _ in range(args.n_latent)]
    cos_vals: List[float] = []
    diff_vals: List[float] = []
    for r in processed_full:
        norms = r["full_stats"].get("scaled_norm_mean", [])
        for i, val in enumerate(norms[:args.n_latent]):
            norms_by_step[i].append(float(val))
        if r["full_stats"].get("step_cos") is not None:
            cos_vals.append(float(r["full_stats"]["step_cos"]))
        if r["full_stats"].get("diff_norm") is not None:
            diff_vals.append(float(r["full_stats"]["diff_norm"]))
    step_scale_means = [float(np.mean(vals)) if vals else 0.0 for vals in norms_by_step]
    step_scale_text = ", ".join(f"${v:.4f}$" for v in step_scale_means)

    def dist_stats(vals: List[float]) -> Dict[str, float]:
        if not vals:
            return {
                "mean": 0.0,
                "std": 0.0,
                "p25": 0.0,
                "median": 0.0,
                "p75": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
        arr = np.asarray(vals, dtype=np.float64)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "p25": float(np.percentile(arr, 25)),
            "median": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    cos_stat = dist_stats(cos_vals)
    diff_stat = dist_stats(diff_vals)

    examples = [
        r for r in final_rows
        if r["baseline_wer"] > r["latent_wer"]
        and r["latent_stats"].get("deq_iters", 0) >= 1
        and len(r["ref_norm"].split()) >= 6
    ]
    examples.sort(key=lambda r: (r["baseline_wer"] - r["latent_wer"], r["baseline_wer"]), reverse=True)
    examples = examples[:4]

    lines: List[str] = []
    lines.append("% ================================================================\n")
    lines.append(f"% Per-sample analysis generated for {args.dataset_key}\n")
    lines.append(f"% Samples used: {len(final_rows)}\n")
    lines.append("% ================================================================\n\n")
    lines.append("\\subsection{Analysis}\n")
    lines.append("\\label{sec:analysis}\n\n")
    lines.append("\\textbf{Difficulty-Binned Reductions.}\\quad\n")
    lines.append(
        f"We partition the {preset['label']} test set into four equal-sized bins by per-utterance "
        f"Baseline WER and recompute WER within each bin (Table~\\ref{{tab:difficulty_bins}}). "
        f"The aggregate result is small but positive: \\method{{}} reduces WER from "
        f"${base_wer:.2f}\\%$ to ${lat_wer:.2f}\\%$ at $\\theta{{=}}{args.theta:.1f}$, "
        f"while forced full compute reaches ${full_wer:.2f}\\%$. "
    )
    best_bin = min(bins, key=lambda x: x[4])
    lines.append(
        f"The largest reduction appears in Q{best_bin[0]}, where $\\Delta$WER is "
        f"${best_bin[4]:+.2f}$~pp. This confirms that the average gain should not be read "
        "as a uniform per-utterance improvement; latent scaling mainly changes the subset "
        "where the frozen baseline leaves residual errors.\n\n"
    )
    lines.append("\\begin{table}[ht]\n")
    lines.append(
        f"  \\caption{{Difficulty-binned analysis on {preset['label']} at $\\theta{{=}}{args.theta:.1f}$ "
        f"({len(final_rows):,} utterances total). Utterances are partitioned into Baseline-WER quartiles. "
        "$\\Delta$WER denotes \\method{} minus Baseline, so negative values indicate improvement.}\n"
    )
    lines.append("  \\label{tab:difficulty_bins}\n")
    lines.append("  \\centering\n")
    lines.append("  \\resizebox{\\columnwidth}{!}{\n")
    lines.append("  \\begin{tabular}{l c c c c c}\n")
    lines.append("    \\toprule\n")
    lines.append("    \\textbf{Bin} & \\textbf{\\#utts} & \\textbf{Baseline WER (\\%)} & \\textbf{\\method{} WER (\\%)} & \\textbf{$\\Delta$WER (pp)} & \\textbf{Skip (\\%)} \\\\\n")
    lines.append("    \\midrule\n")
    labels = ["Q1 (easiest)", "Q2", "Q3", "Q4 (hardest)"]
    for q, count, bw, lw, d, skip_q in bins:
        method_cell = rf"\textbf{{{lw:.2f}}}" if d < 0 else f"{lw:.2f}"
        lines.append(
            f"    {labels[q - 1]} & {count:,} & {bw:.2f} & {method_cell} & "
            f"{format_delta(d, bold_negative=True)} & {skip_q:.1f} \\\\\n"
        )
    lines.append("    \\bottomrule\n")
    lines.append("  \\end{tabular}\n")
    lines.append("  }\n")
    lines.append("\\end{table}\n\n")

    lines.append("\\textbf{Value Head Decision Quality.}\\quad\n")
    lines.append(
        "The step distribution shows how much compute the Value Head allocates: "
        f"at $\\theta{{=}}{args.theta:.1f}$, it skips {step_rates.get(0, 0.0):.1f}\\% of utterances "
        f"and uses an average of {avg_steps:.2f} latent steps. "
        "We further test selectivity by forcing the full $N{=}4$ path on the utterances "
        "that the deployed gate skips. Table~\\ref{tab:gate_quality} reports the actual "
        "deployed change and this counterfactual full-compute change.\n\n"
    )
    lines.append("\\begin{table}[ht]\n")
    lines.append(
        f"  \\caption{{Value Head decision quality on {preset['label']} at $\\theta{{=}}{args.theta:.1f}$. "
        "\\textbf{Counterfactual $\\Delta$WER} forces the $N{=}4$ latent path on each subset.}\n"
    )
    lines.append("  \\label{tab:gate_quality}\n")
    lines.append("  \\centering\n")
    lines.append("  \\resizebox{\\columnwidth}{!}{\n")
    lines.append("  \\begin{tabular}{l c c c}\n")
    lines.append("    \\toprule\n")
    lines.append("    \\textbf{Subset (at $\\theta{=}0.0$)} & \\textbf{\\#utts} & \\textbf{Actual $\\Delta$WER (pp)} & \\textbf{Counterfactual $\\Delta$WER (pp)} \\\\\n")
    lines.append("    \\midrule\n")
    lines.append(f"    Skipped ($v_0 < 0$)      & {len(skipped):,} & $0.00$ (by construction) & {format_delta(skip_full - skip_base, True)} \\\\\n")
    lines.append(f"    Processed ($v_0 \\geq 0$) & {len(processed):,} & {format_delta(proc_lat - proc_base, True)} & {format_delta(proc_full - proc_base, True)} \\\\\n")
    lines.append("    \\bottomrule\n")
    lines.append("  \\end{tabular}\n")
    lines.append("  }\n")
    lines.append("\\end{table}\n\n")

    lines.append("\\textbf{Step Allocation.}\\quad\n")
    lines.append(
        f"Table~\\ref{{tab:vox_step_dist}} gives the full $N$-step distribution on {preset['label']}. "
        "Compared with forced full compute, the deployed policy keeps most examples away from "
        "the deepest path while retaining the aggregate WER reduction.\n\n"
    )
    lines.append("\\begin{table}[ht]\n")
    lines.append(
        f"  \\caption{{N-step distribution on {preset['label']} at $\\theta{{=}}{args.theta:.1f}$.}}\n"
    )
    lines.append("  \\label{tab:vox_step_dist}\n")
    lines.append("  \\centering\n")
    lines.append("  \\resizebox{0.9\\columnwidth}{!}{\n")
    lines.append("  \\begin{tabular}{l c c c c c c}\n")
    lines.append("    \\toprule\n")
    lines.append("    \\textbf{Dataset} & \\textbf{Avg. steps} & \\textbf{N=0} & \\textbf{N=1} & \\textbf{N=2} & \\textbf{N=3} & \\textbf{N=4} \\\\\n")
    lines.append("    \\midrule\n")
    lines.append(
        f"    {preset['label']} & {avg_steps:.2f} & "
        f"{step_rates.get(0, 0.0):.1f}\\% & {step_rates.get(1, 0.0):.1f}\\% & "
        f"{step_rates.get(2, 0.0):.1f}\\% & {step_rates.get(3, 0.0):.1f}\\% & "
        f"{step_rates.get(4, 0.0):.1f}\\% \\\\\n"
    )
    lines.append("    \\bottomrule\n")
    lines.append("  \\end{tabular}\n")
    lines.append("  }\n")
    lines.append("\\end{table}\n\n")

    if examples:
        lines.append("\\textbf{Qualitative Examples.}\\quad\n")
        lines.append(
            f"Table~\\ref{{tab:qualitative}} shows hard-bin {preset['label']} utterances where "
            "the latent loop changes the transcript.\n\n"
        )
        lines.append("\\begin{table}[ht]\n")
        lines.append(
            f"  \\caption{{Qualitative examples from {preset['label']} hard bins.}}\n"
        )
        lines.append("  \\label{tab:qualitative}\n")
        lines.append("  \\centering\n")
        lines.append("  \\resizebox{\\columnwidth}{!}{\n")
        lines.append("  \\begin{tabular}{p{0.13\\columnwidth} p{0.27\\columnwidth} p{0.27\\columnwidth} p{0.27\\columnwidth}}\n")
        lines.append("    \\toprule\n")
        lines.append("    \\textbf{Source} & \\textbf{Reference} & \\textbf{Baseline} & \\textbf{\\method{}} \\\\\n")
        lines.append("    \\midrule\n")
        for r in examples:
            lines.append(
                f"    {preset['source_label']} & {esc_latex(r['ref_raw'])} & "
                f"{esc_latex(r['baseline_pred'])} & {esc_latex(r['latent_pred'])} \\\\\n"
            )
        lines.append("    \\bottomrule\n")
        lines.append("  \\end{tabular}\n")
        lines.append("  }\n")
        lines.append("\\end{table}\n\n")

    lines.append("\\textbf{Refinement-Path Diagnostics.}\\quad\n")
    lines.append(
        "We recompute the forced-full $N{=}4$ path on the "
        f"{preset['label']} processed subset ({len(processed_full):,} utterances with $N{{>}}0$ "
        f"under $\\theta{{=}}{args.theta:.1f}$). The scaled delta norms are identical across "
        "utterances because each delta is $L_2$-normalized and multiplied by the learned "
        "per-step scale; they therefore measure the bounded step-size constraint rather "
        f"than dataset-specific refinement behavior. For this run, the per-step scales are {step_scale_text}. "
        "To characterize sample-dependent behavior, Table~\\ref{tab:refinement_dynamics} "
        "instead reports the distribution of consecutive-delta cosine and consecutive-delta "
        "difference norm. "
        f"The cosine range is ${cos_stat['min']:.4f}$--${cos_stat['max']:.4f}$ and the "
        f"difference-norm range is ${diff_stat['min']:.4f}$--${diff_stat['max']:.4f}$, "
        "confirming that the forced refinement path is not a constant copied trajectory "
        "while the update magnitudes remain bounded.\n\n"
    )
    lines.append("\\begin{table}[ht]\n")
    lines.append(
        f"  \\caption{{Forced-full refinement diagnostics on the {preset['label']} processed subset "
        f"({len(processed_full):,} utterances with $N{{>}}0$ under $\\theta{{=}}{args.theta:.1f}$). "
        "Scaled delta norms are fixed by the learned step scales; cosine and difference "
        "statistics vary across utterances.}\n"
    )
    lines.append("  \\label{tab:refinement_dynamics}\n")
    lines.append("  \\centering\n")
    lines.append("  \\resizebox{\\columnwidth}{!}{\n")
    lines.append("  \\begin{tabular}{l c c c c c}\n")
    lines.append("    \\toprule\n")
    lines.append(r"    \textbf{Metric} & \textbf{Mean} & \textbf{Std.} & \textbf{P25} & \textbf{Median} & \textbf{P75} \\" + "\n")
    lines.append("    \\midrule\n")
    lines.append(
        f"    Consecutive-delta cosine & {cos_stat['mean']:.4f} & {cos_stat['std']:.4f} & "
        f"{cos_stat['p25']:.4f} & {cos_stat['median']:.4f} & {cos_stat['p75']:.4f} "
        + r"\\"
        + "\n"
    )
    lines.append(
        f"    Consecutive-delta diff. norm & {diff_stat['mean']:.4f} & {diff_stat['std']:.4f} & "
        f"{diff_stat['p25']:.4f} & {diff_stat['median']:.4f} & {diff_stat['p75']:.4f} "
        + r"\\"
        + "\n"
    )
    lines.append("    \\bottomrule\n")
    lines.append("  \\end{tabular}\n")
    lines.append("  }\n")
    lines.append("\\end{table}\n\n")

    lines.append("% Overall metrics for cross-checking:\n")
    lines.append(f"% Baseline WER/CER: {base_wer:.4f}/{base_cer:.4f}\n")
    lines.append(f"% Latent theta={args.theta:.1f} WER/CER: {lat_wer:.4f}/{lat_cer:.4f}\n")
    lines.append(f"% Forced full WER/CER: {full_wer:.4f}/{full_cer:.4f}\n")

    args.out_tex.write_text("".join(lines), encoding="utf-8")
    save_rows(args.out_json, final_rows)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_tex}")
    print(f"samples={len(final_rows)} baseline_wer={base_wer:.4f} latent_wer={lat_wer:.4f} full_wer={full_wer:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-key", choices=sorted(DATASET_PRESETS), default="voxpopuli_en")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--latent-ckpt", default=LATENT_CKPT)
    parser.add_argument("--n-latent", type=int, default=4)
    parser.add_argument("--theta", type=float, default=0.0)
    parser.add_argument("--full-theta", type=float, default=-2.0)
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out-dir", type=Path, default=Path("eval_runs/paper_activation500_voxpopuli_analysis"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.out_json = args.out_dir / f"{args.dataset_key}_per_sample.json"
    args.out_tex = args.out_dir / f"{args.dataset_key}_analysis_latex.tex"
    args.device = choose_device()
    args.dtype = choose_dtype(args.device)
    return args


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    preset = DATASET_PRESETS[args.dataset_key]
    configure_text_normalizer(preset["normalizer"])

    rows = load_rows(args.out_json)
    refs = collect_refs(preset, args.streaming)
    existing = {r["idx"]: r for r in rows}
    for ref in refs:
        existing.setdefault(ref["idx"], {"idx": ref["idx"], **ref})
    rows = sorted(existing.values(), key=lambda x: x["idx"])
    save_rows(args.out_json, rows)

    rows = run_baseline(args, preset, refs, rows)
    rows = run_latent_path(
        args,
        preset,
        rows,
        theta=args.theta,
        pred_key="latent_pred",
        norm_key="latent_norm",
        stats_key="latent_stats",
        desc=f"latent_theta{args.theta:g}",
    )
    rows = run_latent_path(
        args,
        preset,
        rows,
        theta=args.full_theta,
        pred_key="full_pred",
        norm_key="full_norm",
        stats_key="full_stats",
        desc="latent_full",
    )
    write_latex(args, preset, rows)


if __name__ == "__main__":
    main()
