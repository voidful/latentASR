"""
Per-sample analysis: When does latent reasoning help most?

Runs both baseline and latent_reasoning on FLEURS (en_us) and VoxPopuli (en),
collects per-sample WER, and bins results by:
  1. Utterance length (word count)
  2. Baseline difficulty (baseline WER per sample)
"""
import os, sys, json, re, torch, numpy as np
from pathlib import Path
from collections import defaultdict
from jiwer import wer as compute_wer, cer as compute_cer
from datasets import load_dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from eval import (
    build_base_model_bundle, build_latent_bundle, resolve_language_hint,
    normalize_text, clean_prediction, resolve_im_start_id, ensure_front_prompt_token,
    resolve_text_column, choose_device, choose_dtype, sanitize_generation_config,
    FALLBACK_TEXT_COLUMNS,
)

MODEL_ID = "Qwen/Qwen3-ASR-0.6B"
LATENT_CKPT = "./latent_qwen_asr_best.pth"

DATASETS = [
    ("SpeechTest/fleurs", "en_us", "test"),
    ("SpeechTest/voxpopuli", "en", "test"),
]


def per_sample_wer(ref: str, pred: str) -> float:
    if not ref.strip():
        return 0.0
    try:
        return float(compute_wer(ref, pred))
    except Exception:
        return 1.0


def per_sample_cer(ref: str, pred: str) -> float:
    if not ref.strip():
        return 0.0
    try:
        return float(compute_cer(ref, pred))
    except Exception:
        return 1.0


def run_single_sample(model, processor, audio_array, sr, language_hint, prompt_text, use_baseline, dynamic_halt_threshold=0.0):
    target_dtype = model.thinker.dtype if hasattr(model.thinker, "dtype") else torch.float32
    feat_out = processor.feature_extractor(audio_array, sampling_rate=sr, return_attention_mask=True)
    feats = torch.tensor(feat_out.input_features[0], dtype=target_dtype, device=model.base_model.device).unsqueeze(0)
    n_frames = feats.size(-1)

    if getattr(feat_out, "attention_mask", None) is not None:
        raw_mask = feat_out.attention_mask[0]
        if not isinstance(raw_mask, torch.Tensor):
            raw_mask = torch.tensor(list(raw_mask), dtype=torch.long)
        else:
            raw_mask = raw_mask.long()
        if raw_mask.size(-1) < n_frames:
            raw_mask = torch.cat([raw_mask, torch.zeros(n_frames - raw_mask.size(-1), dtype=torch.long)])
        else:
            raw_mask = raw_mask[:n_frames]
        feature_attention_mask = raw_mask.to(device=model.base_model.device).unsqueeze(0)
    else:
        feature_attention_mask = torch.ones((1, n_frames), dtype=torch.long, device=model.base_model.device)

    effective_len = feature_attention_mask.sum().item()
    if effective_len < 10:
        return None, {}

    gen_kwargs = {
        "feature_attention_mask": feature_attention_mask,
        "max_new_tokens": 128,
        "use_baseline": use_baseline,
        "return_thoughts": False,
        "return_stats": True,
        "do_sample": False,
        "eos_token_id": [151645, 151643],
        "num_beams": 1,
        "language_hint": language_hint,
        "prompt_text": prompt_text,
        "dynamic_halt_threshold": dynamic_halt_threshold,
    }

    gen_output = model.generate(feats, **gen_kwargs)
    if isinstance(gen_output, tuple):
        gen_ids = gen_output[0]
        stats = gen_output[1] if isinstance(gen_output[1], dict) else (gen_output[2] if len(gen_output) > 2 and isinstance(gen_output[2], dict) else {})
    else:
        gen_ids = gen_output
        stats = {}

    ids = gen_ids[0]
    eos_id = processor.tokenizer.eos_token_id
    if eos_id is not None and (ids == eos_id).any():
        eos_pos = (ids == eos_id).nonzero(as_tuple=True)[0][0]
        ids = ids[:eos_pos]

    pred_raw = processor.tokenizer.decode(ids, skip_special_tokens=True)
    pred = clean_prediction(pred_raw)
    return pred, stats


@torch.no_grad()
def main():
    device = choose_device("auto")
    dtype = choose_dtype("auto", device)

    all_records = []

    for dataset_name, config_name, split in DATASETS:
        print(f"\n{'='*60}")
        print(f"Processing {dataset_name} / {config_name}")
        print(f"{'='*60}")

        ds = load_dataset(dataset_name, config_name, split=split, trust_remote_code=True)
        column_names = list(getattr(ds, "column_names", []))
        text_column = resolve_text_column(dataset_name, column_names)
        language_hint = resolve_language_hint(dataset_name, config_name)
        if language_hint:
            prompt_text = f"Transcribe the {language_hint} audio into text."
        else:
            prompt_text = "Transcribe the audio into text."

        # --- Build base model ---
        print("[1/2] Building base model...")
        base_bundle = build_base_model_bundle(MODEL_ID, device, dtype)

        base_preds = []
        base_refs = []
        audio_durations = []

        for i in tqdm(range(len(ds)), desc="Baseline"):
            sample = ds[i]
            ref = sample.get(text_column, "")
            if not isinstance(ref, str) or not ref.strip():
                base_preds.append(None)
                base_refs.append(None)
                audio_durations.append(0)
                continue

            audio = sample.get("audio")
            if not isinstance(audio, dict):
                base_preds.append(None)
                base_refs.append(None)
                audio_durations.append(0)
                continue

            audio_array = np.array(audio["array"], dtype=np.float64)
            sr = audio["sampling_rate"]
            duration = len(audio_array) / sr

            pred, _ = run_single_sample(
                base_bundle.model, base_bundle.processor,
                audio_array, sr, language_hint, prompt_text, use_baseline=True,
            )
            if pred is None:
                base_preds.append(None)
                base_refs.append(None)
                audio_durations.append(0)
                continue

            ref_norm = normalize_text(ref)
            pred_norm = normalize_text(pred)
            if not ref_norm:
                base_preds.append(None)
                base_refs.append(None)
                audio_durations.append(0)
                continue

            base_preds.append(pred_norm)
            base_refs.append(ref_norm)
            audio_durations.append(duration)

        # Release base model
        del base_bundle
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- Build latent model ---
        print("[2/2] Building latent reasoning model...")
        latent_bundle = build_latent_bundle(MODEL_ID, LATENT_CKPT, 4, device, dtype)

        latent_preds = []
        latent_v_preds = []
        latent_deq_iters = []

        for i in tqdm(range(len(ds)), desc="Latent Reasoning"):
            if base_refs[i] is None:
                latent_preds.append(None)
                latent_v_preds.append(None)
                latent_deq_iters.append(0)
                continue

            sample = ds[i]
            audio = sample.get("audio")
            audio_array = np.array(audio["array"], dtype=np.float64)
            sr = audio["sampling_rate"]

            pred, stats = run_single_sample(
                latent_bundle.model, latent_bundle.processor,
                audio_array, sr, language_hint, prompt_text, use_baseline=False,
            )
            if pred is None:
                latent_preds.append(None)
                latent_v_preds.append(None)
                latent_deq_iters.append(0)
                continue

            pred_norm = normalize_text(pred)
            latent_preds.append(pred_norm)

            if "v_preds" in stats and stats["v_preds"] is not None:
                latent_v_preds.append(stats["v_preds"][0].tolist())
            else:
                latent_v_preds.append(None)

            iters_t = stats.get("deq_iters", None)
            latent_deq_iters.append(int(iters_t.item()) if iters_t is not None and hasattr(iters_t, "item") else 0)

        # Release latent model
        del latent_bundle
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- Collect per-sample records ---
        for i in range(len(ds)):
            if base_refs[i] is None or latent_preds[i] is None:
                continue

            ref = base_refs[i]
            word_count = len(ref.split())
            base_wer_i = per_sample_wer(ref, base_preds[i])
            latent_wer_i = per_sample_wer(ref, latent_preds[i])
            base_cer_i = per_sample_cer(ref, base_preds[i])
            latent_cer_i = per_sample_cer(ref, latent_preds[i])

            all_records.append({
                "dataset": f"{dataset_name}/{config_name}",
                "idx": i,
                "ref": ref,
                "base_pred": base_preds[i],
                "latent_pred": latent_preds[i],
                "word_count": word_count,
                "audio_duration": audio_durations[i],
                "base_wer": base_wer_i,
                "latent_wer": latent_wer_i,
                "delta_wer": base_wer_i - latent_wer_i,  # positive = LR helped
                "base_cer": base_cer_i,
                "latent_cer": latent_cer_i,
                "delta_cer": base_cer_i - latent_cer_i,
                "v_preds": latent_v_preds[i],
                "deq_iters": latent_deq_iters[i],
            })

    # ========== ANALYSIS ==========
    print(f"\n{'='*70}")
    print(f"ANALYSIS: When does latent reasoning help most?")
    print(f"Total samples with valid comparisons: {len(all_records)}")
    print(f"{'='*70}")

    # Overall stats
    helped = [r for r in all_records if r["delta_wer"] > 0]
    hurt = [r for r in all_records if r["delta_wer"] < 0]
    same = [r for r in all_records if r["delta_wer"] == 0]
    print(f"\nOverall: Helped={len(helped)}  Hurt={len(hurt)}  Same={len(same)}")

    avg_delta = np.mean([r["delta_wer"] for r in all_records])
    print(f"Average delta WER (positive=LR better): {avg_delta:.4f}")

    # --- Bin by utterance length (word count) ---
    print(f"\n--- By Utterance Length (word count) ---")
    bins_len = [(1, 5, "1-5 words"), (6, 10, "6-10 words"), (11, 20, "11-20 words"), (21, 999, "21+ words")]
    print(f"{'Bin':<15} {'N':>6} {'Helped':>8} {'Hurt':>8} {'Same':>8} {'Avg ΔWER':>10} {'Avg ΔCER':>10}")
    for lo, hi, label in bins_len:
        subset = [r for r in all_records if lo <= r["word_count"] <= hi]
        if not subset:
            continue
        n_helped = sum(1 for r in subset if r["delta_wer"] > 0)
        n_hurt = sum(1 for r in subset if r["delta_wer"] < 0)
        n_same = sum(1 for r in subset if r["delta_wer"] == 0)
        avg_d_wer = np.mean([r["delta_wer"] for r in subset])
        avg_d_cer = np.mean([r["delta_cer"] for r in subset])
        print(f"{label:<15} {len(subset):>6} {n_helped:>8} {n_hurt:>8} {n_same:>8} {avg_d_wer:>+10.4f} {avg_d_cer:>+10.4f}")

    # --- Bin by baseline difficulty ---
    print(f"\n--- By Baseline Difficulty (baseline WER) ---")
    bins_diff = [
        (0.0, 0.001, "Perfect (WER=0)"),
        (0.001, 0.1, "Easy (0<WER<10%)"),
        (0.1, 0.3, "Medium (10-30%)"),
        (0.3, 0.5, "Hard (30-50%)"),
        (0.5, 999, "Very Hard (50%+)"),
    ]
    print(f"{'Bin':<22} {'N':>6} {'Helped':>8} {'Hurt':>8} {'Same':>8} {'Avg ΔWER':>10} {'Avg ΔCER':>10}")
    for lo, hi, label in bins_diff:
        subset = [r for r in all_records if lo <= r["base_wer"] < hi]
        if not subset:
            continue
        n_helped = sum(1 for r in subset if r["delta_wer"] > 0)
        n_hurt = sum(1 for r in subset if r["delta_wer"] < 0)
        n_same = sum(1 for r in subset if r["delta_wer"] == 0)
        avg_d_wer = np.mean([r["delta_wer"] for r in subset])
        avg_d_cer = np.mean([r["delta_cer"] for r in subset])
        print(f"{label:<22} {len(subset):>6} {n_helped:>8} {n_hurt:>8} {n_same:>8} {avg_d_wer:>+10.4f} {avg_d_cer:>+10.4f}")

    # --- Bin by audio duration ---
    print(f"\n--- By Audio Duration ---")
    bins_dur = [(0, 3, "<3s"), (3, 6, "3-6s"), (6, 10, "6-10s"), (10, 999, "10s+")]
    print(f"{'Bin':<15} {'N':>6} {'Helped':>8} {'Hurt':>8} {'Same':>8} {'Avg ΔWER':>10} {'Avg ΔCER':>10}")
    for lo, hi, label in bins_dur:
        subset = [r for r in all_records if lo <= r["audio_duration"] < hi]
        if not subset:
            continue
        n_helped = sum(1 for r in subset if r["delta_wer"] > 0)
        n_hurt = sum(1 for r in subset if r["delta_wer"] < 0)
        n_same = sum(1 for r in subset if r["delta_wer"] == 0)
        avg_d_wer = np.mean([r["delta_wer"] for r in subset])
        avg_d_cer = np.mean([r["delta_cer"] for r in subset])
        print(f"{label:<15} {len(subset):>6} {n_helped:>8} {n_hurt:>8} {n_same:>8} {avg_d_wer:>+10.4f} {avg_d_cer:>+10.4f}")

    # --- Value Head analysis ---
    v_records = [r for r in all_records if r["v_preds"] is not None]
    if v_records:
        print(f"\n--- Value Head Score vs Actual Improvement ---")
        bins_v = [(-1.0, -0.5, "v<-0.5"), (-0.5, 0.0, "-0.5≤v<0"), (0.0, 0.5, "0≤v<0.5"), (0.5, 1.01, "v≥0.5")]
        print(f"{'v_pred bin':<15} {'N':>6} {'Helped':>8} {'Hurt':>8} {'Avg ΔWER':>10}")
        for lo, hi, label in bins_v:
            subset = [r for r in v_records if lo <= r["v_preds"][0] < hi]
            if not subset:
                continue
            n_helped = sum(1 for r in subset if r["delta_wer"] > 0)
            n_hurt = sum(1 for r in subset if r["delta_wer"] < 0)
            avg_d = np.mean([r["delta_wer"] for r in subset])
            print(f"{label:<15} {len(subset):>6} {n_helped:>8} {n_hurt:>8} {avg_d:>+10.4f}")

    # Save raw records
    out_path = "/Users/voidful/PycharmProjects/lr_whisper/lr_help_analysis.json"
    with open(out_path, "w") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)
    print(f"\nRaw records saved to: {out_path}")


if __name__ == "__main__":
    main()
