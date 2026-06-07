#!/usr/bin/env python3
"""Per-sample analysis for the paper TODO tables.

This script evaluates the 500-utterance LatentASR checkpoint on FLEURS en_us
with baseline, deployed halting (theta=0), and forced full compute
(theta=-2). It writes per-sample metrics plus a compact Markdown report that
can be pasted into the paper.
"""

from __future__ import annotations

import gc
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import Audio, load_dataset
from jiwer import cer, wer
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval import (
    build_base_model_bundle,
    build_latent_bundle,
    clean_prediction,
    configure_text_normalizer,
    normalize_text,
)


MODEL_ID = "Qwen/Qwen3-ASR-0.6B"
LATENT_CKPT = "eval_runs/paper_tbd_retrain_20260518/checkpoints/activation_500/activation_500_epoch10.pth"
OUT_DIR = Path("eval_runs/paper_activation500_todo_analysis")
OUT_JSON = OUT_DIR / "fleurs_en_us_per_sample.json"
OUT_REPORT = OUT_DIR / "paper_todo_replacements.md"
MAX_NEW_TOKENS = 128
BASELINE_WER = 4.8999


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
) -> Optional[Dict[str, Any]]:
    feats = sample_to_features(model, processor, sample)
    if feats is None:
        return None
    gen_kwargs = {
        "feature_attention_mask": feats["feature_attention_mask"],
        "max_new_tokens": MAX_NEW_TOKENS,
        "use_baseline": use_baseline,
        "return_thoughts": False,
        "return_stats": True,
        "do_sample": False,
        "eos_token_id": [151645, 151643],
        "num_beams": 1,
        "language_hint": "English",
        "prompt_text": "Transcribe the English audio into text.",
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
    skipped = bool(stats.get("skipped", False)) if stats else False
    v_preds = tensor_to_list(stats.get("v_preds"))
    scaled_norm = tensor_to_list(stats.get("scaled_norm_mean"))
    return {
        "pred": pred,
        "pred_norm": normalize_text(pred),
        "stats": {
            "deq_iters": 0 if deq is None else int(round(deq)),
            "skipped": skipped,
            "v_preds": v_preds,
            "scaled_norm_mean": scaled_norm,
            "step_cos": tensor_to_float(stats.get("step_cos")),
            "diff_norm": tensor_to_float(stats.get("diff_norm")),
        },
    }


def pct(x: float) -> str:
    return f"{x:.2f}"


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_text_normalizer("english")
    device = choose_device()
    dtype = choose_dtype(device)

    ds = load_dataset("google/fleurs", "en_us", split="test", trust_remote_code=True)
    refs = []
    for sample in ds:
        ref_raw = sample.get("transcription") or sample.get("raw_transcription") or sample.get("sentence") or ""
        refs.append({"raw": ref_raw, "norm": normalize_text(ref_raw)})
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    rows: List[Dict[str, Any]] = []

    base_bundle = build_base_model_bundle(MODEL_ID, device, dtype)
    for idx, sample in enumerate(tqdm(ds, desc="baseline")):
        ref = refs[idx]
        out = transcribe(base_bundle.model, base_bundle.processor, sample, use_baseline=True, theta=0.0)
        if out is None or not ref["norm"]:
            continue
        rows.append(
            {
                "idx": idx,
                "ref_raw": ref["raw"],
                "ref_norm": ref["norm"],
                "baseline_pred": out["pred"],
                "baseline_norm": out["pred_norm"],
            }
        )
    del base_bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    latent_bundle = build_latent_bundle(MODEL_ID, LATENT_CKPT, 4, device, dtype)
    by_idx = {r["idx"]: r for r in rows}
    for idx, sample in enumerate(tqdm(ds, desc="latent_theta0")):
        if idx not in by_idx:
            continue
        out = transcribe(latent_bundle.model, latent_bundle.processor, sample, use_baseline=False, theta=0.0)
        if out is None:
            continue
        by_idx[idx]["latent_pred"] = out["pred"]
        by_idx[idx]["latent_norm"] = out["pred_norm"]
        by_idx[idx]["latent_stats"] = out["stats"]

    for idx, sample in enumerate(tqdm(ds, desc="latent_full")):
        if idx not in by_idx:
            continue
        out = transcribe(latent_bundle.model, latent_bundle.processor, sample, use_baseline=False, theta=-2.0)
        if out is None:
            continue
        by_idx[idx]["full_pred"] = out["pred"]
        by_idx[idx]["full_norm"] = out["pred_norm"]
        by_idx[idx]["full_stats"] = out["stats"]

    final_rows = []
    for r in rows:
        if "latent_norm" not in r or "full_norm" not in r:
            continue
        ref = r["ref_norm"]
        b = r["baseline_norm"]
        l = r["latent_norm"]
        f = r["full_norm"]
        r["baseline_wer"] = float(wer(ref, b))
        r["latent_wer"] = float(wer(ref, l))
        r["full_wer"] = float(wer(ref, f))
        r["baseline_cer"] = float(cer(ref, b))
        r["latent_cer"] = float(cer(ref, l))
        r["full_cer"] = float(cer(ref, f))
        final_rows.append(r)

    OUT_JSON.write_text(json.dumps(final_rows, indent=2, ensure_ascii=False))

    refs_all = [r["ref_norm"] for r in final_rows]
    base_all = [r["baseline_norm"] for r in final_rows]
    lat_all = [r["latent_norm"] for r in final_rows]
    full_all = [r["full_norm"] for r in final_rows]

    base_wer = 100 * wer(refs_all, base_all)
    lat_wer = 100 * wer(refs_all, lat_all)
    full_wer = 100 * wer(refs_all, full_all)

    sorted_rows = sorted(final_rows, key=lambda r: (r["baseline_wer"], r["idx"]))
    n = len(sorted_rows)
    bins = []
    for q in range(4):
        part = sorted_rows[math.floor(q * n / 4): math.floor((q + 1) * n / 4)]
        refs_q = [r["ref_norm"] for r in part]
        b_q = [r["baseline_norm"] for r in part]
        l_q = [r["latent_norm"] for r in part]
        skip_q = 100 * sum(1 for r in part if r["latent_stats"].get("deq_iters", 0) == 0) / max(1, len(part))
        bw = 100 * wer(refs_q, b_q)
        lw = 100 * wer(refs_q, l_q)
        bins.append((q + 1, len(part), bw, lw, lw - bw, skip_q))

    skipped = [r for r in final_rows if r["latent_stats"].get("deq_iters", 0) == 0]
    processed = [r for r in final_rows if r["latent_stats"].get("deq_iters", 0) > 0]

    def group_wer(group: List[Dict[str, Any]], key: str) -> float:
        if not group:
            return 0.0
        return 100 * wer([r["ref_norm"] for r in group], [r[key] for r in group])

    skip_base = group_wer(skipped, "baseline_norm")
    skip_full = group_wer(skipped, "full_norm")
    proc_base = group_wer(processed, "baseline_norm")
    proc_lat = group_wer(processed, "latent_norm")
    proc_full = group_wer(processed, "full_norm")

    improvements = [
        r for r in final_rows
        if r["baseline_wer"] > r["latent_wer"]
        and r["latent_stats"].get("deq_iters", 0) >= 1
        and len(r["ref_norm"].split()) >= 6
    ]
    improvements.sort(key=lambda r: (r["baseline_wer"] - r["latent_wer"], r["baseline_wer"]), reverse=True)
    examples = improvements[:4]

    # Forced-full latent statistics on the processed subset.
    full_processed = [r for r in final_rows if r["latent_stats"].get("deq_iters", 0) > 0]
    norms_by_step: List[List[float]] = [[] for _ in range(4)]
    cos_vals: List[float] = []
    diff_vals: List[float] = []
    for r in full_processed:
        norms = r["full_stats"].get("scaled_norm_mean", [])
        for i, val in enumerate(norms[:4]):
            norms_by_step[i].append(float(val))
        if r["full_stats"].get("step_cos") is not None:
            cos_vals.append(float(r["full_stats"]["step_cos"]))
        if r["full_stats"].get("diff_norm") is not None:
            diff_vals.append(float(r["full_stats"]["diff_norm"]))

    report = []
    report.append("# Paper TODO replacements\n")
    report.append(f"Samples used: {len(final_rows)}. FLEURS baseline WER {base_wer:.4f}, theta=0 LatentASR WER {lat_wer:.4f}, forced-full WER {full_wer:.4f}.\n")
    report.append("## Difficulty-Binned Reductions\n")
    report.append("| Bin | #utts | Baseline WER (%) | LatentASR WER (%) | ΔWER (pp) | Skip (%) |\n")
    report.append("|---|---:|---:|---:|---:|---:|\n")
    for q, count, bw, lw, d, skip_q in bins:
        label = ["Q1 (easiest)", "Q2", "Q3", "Q4 (hardest)"][q - 1]
        report.append(f"| {label} | {count} | {bw:.2f} | {lw:.2f} | {d:+.2f} | {skip_q:.1f} |\n")
    total_delta = sum((r["latent_wer"] - r["baseline_wer"]) for r in final_rows)
    q4_delta = sum((r["latent_wer"] - r["baseline_wer"]) for r in bins and sorted_rows[math.floor(3*n/4):])
    report.append(f"\nQ1 skip rate: {bins[0][5]:.1f}%. Q4 utterance-level ΔWER contribution over total utterance-level ΔWER: {100*q4_delta/total_delta if total_delta else 0:.1f}%.\n")

    report.append("\n## Value Head Decision Quality\n")
    report.append("| Subset (theta=0) | #utts | Actual ΔWER (pp) | Counterfactual ΔWER (pp) |\n")
    report.append("|---|---:|---:|---:|\n")
    report.append(f"| Skipped (v0 < 0) | {len(skipped)} | 0.00 | {skip_full - skip_base:+.2f} |\n")
    report.append(f"| Processed (v0 >= 0) | {len(processed)} | {proc_lat - proc_base:+.2f} | {proc_full - proc_base:+.2f} |\n")

    report.append("\n## Qualitative Examples\n")
    report.append("| Source | Reference | Baseline | LatentASR |\n")
    report.append("|---|---|---|---|\n")
    for r in examples:
        report.append(f"| FLEURS Q4 | {r['ref_raw']} | {r['baseline_pred']} | {r['latent_pred']} |\n")

    report.append("\n## Forced-Full Refinement Stats on Processed Subset\n")
    report.append("| Step k | Mean scaled delta norm |\n")
    report.append("|---:|---:|\n")
    for i, vals in enumerate(norms_by_step, start=1):
        report.append(f"| {i} | {np.mean(vals) if vals else 0.0:.4f} |\n")
    report.append(f"\nMean consecutive-delta cosine under forced full path: {np.mean(cos_vals) if cos_vals else 0.0:.4f}. Mean delta-difference norm: {np.mean(diff_vals) if diff_vals else 0.0:.4f}.\n")

    OUT_REPORT.write_text("".join(report))
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
