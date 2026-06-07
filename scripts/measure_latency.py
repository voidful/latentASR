#!/usr/bin/env python3
"""Measure batch-1 inference latency for baseline and LatentASR settings."""

from __future__ import annotations

import argparse
import json
import sys
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from datasets import Audio, load_dataset

from eval import (
    build_base_model_bundle,
    build_latent_bundle,
    choose_device,
    choose_dtype,
    clean_prediction,
    resolve_language_hint,
    release_model,
)
from utils import set_seed


DATASETS = {
    "fleurs": ("google/fleurs", "en_us", "test"),
    "voxpopuli": ("facebook/voxpopuli", "en", "test"),
}


def collect_samples(
    dataset_name: str,
    config: str,
    split: str,
    n: int,
    streaming: bool,
    sampling_rate: int,
) -> List[Dict[str, Any]]:
    ds = load_dataset(
        dataset_name,
        config,
        split=split,
        trust_remote_code=True,
        streaming=streaming,
    )
    if not streaming:
        ds = ds.cast_column("audio", Audio(sampling_rate=sampling_rate))
    samples: List[Dict[str, Any]] = []
    for sample in ds:
        audio = sample.get("audio")
        if isinstance(audio, dict) and "array" in audio and "sampling_rate" in audio:
            samples.append(audio)
        if len(samples) >= n:
            break
    return samples


def run_one(
    model: Any,
    processor: Any,
    audio: Dict[str, Any],
    use_baseline: bool,
    threshold: float,
    language_hint: Optional[str],
    max_new_tokens: int,
) -> Tuple[str, Dict[str, Any]]:
    prompt_text = (
        f"Transcribe the {language_hint} audio into text."
        if language_hint
        else "Transcribe the audio into text."
    )
    target_dtype = model.thinker.dtype if hasattr(model.thinker, "dtype") else torch.float32
    feat_out = processor.feature_extractor(
        np.asarray(audio["array"], dtype=np.float64),
        sampling_rate=audio["sampling_rate"],
        return_attention_mask=True,
    )
    feats = torch.tensor(
        feat_out.input_features[0],
        dtype=target_dtype,
        device=model.base_model.device,
    ).unsqueeze(0)
    n_frames = feats.size(-1)
    raw_mask = getattr(feat_out, "attention_mask", None)
    if raw_mask is not None:
        mask = torch.tensor(raw_mask[0], dtype=torch.long)
        if mask.size(-1) < n_frames:
            mask = torch.cat([mask, torch.zeros(n_frames - mask.size(-1), dtype=torch.long)])
        else:
            mask = mask[:n_frames]
        feature_attention_mask = mask.to(model.base_model.device).unsqueeze(0)
    else:
        feature_attention_mask = torch.ones((1, n_frames), dtype=torch.long, device=model.base_model.device)

    out = model.generate(
        feats,
        feature_attention_mask=feature_attention_mask,
        max_new_tokens=max_new_tokens,
        use_baseline=use_baseline,
        return_thoughts=False,
        return_stats=True,
        do_sample=False,
        eos_token_id=[151645, 151643],
        num_beams=1,
        language_hint=language_hint,
        prompt_text=prompt_text,
        dynamic_halt_threshold=threshold,
    )
    if isinstance(out, tuple):
        gen_ids = out[0]
        stats = out[1] if isinstance(out[1], dict) else {}
    else:
        gen_ids = out
        stats = {}
    text = clean_prediction(processor.tokenizer.decode(gen_ids[0], skip_special_tokens=True))
    return text, stats


def measure_setting(
    model: Any,
    processor: Any,
    samples: List[Dict[str, Any]],
    use_baseline: bool,
    threshold: float,
    language_hint: Optional[str],
    warmup: int,
    measure: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    n_total = min(len(samples), warmup + measure)
    if n_total <= warmup:
        raise ValueError(f"Need > warmup samples, got {len(samples)}")
    times: List[float] = []
    steps: List[int] = []
    for idx, audio in enumerate(samples[:n_total]):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _, stats = run_one(
            model=model,
            processor=processor,
            audio=audio,
            use_baseline=use_baseline,
            threshold=threshold,
            language_hint=language_hint,
            max_new_tokens=max_new_tokens,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if idx >= warmup:
            times.append(dt_ms)
            iters = stats.get("deq_iters")
            if hasattr(iters, "item"):
                steps.append(int(iters.item()))
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "std_ms": statistics.pstdev(times) if len(times) > 1 else 0.0,
        "n": len(times),
        "avg_steps": (sum(steps) / len(steps)) if steps else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--latent-ckpt", default="latent_qwen_asr_best.pth")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    args = parser.parse_args()

    set_seed(42)
    device = choose_device(args.device)
    dtype = choose_dtype(args.dtype, device)
    output: Dict[str, Any] = {
        "model_id": args.model_id,
        "latent_ckpt": args.latent_ckpt,
        "device": device,
        "dtype": str(dtype),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "warmup": args.warmup,
        "samples": args.samples,
        "results": {},
    }

    base_bundle = build_base_model_bundle(args.model_id, device=device, dtype=dtype)
    latent_bundle = build_latent_bundle(
        args.model_id,
        checkpoint_path=args.latent_ckpt,
        n_latent_override=-1,
        device=device,
        dtype=dtype,
    )
    try:
        sampling_rate = int(getattr(base_bundle.processor.feature_extractor, "sampling_rate", 16000) or 16000)
        for tag, (dataset_name, config, split) in DATASETS.items():
            print(f"[latency] collect {tag}: {dataset_name}/{config}/{split}")
            samples = collect_samples(
                dataset_name=dataset_name,
                config=config,
                split=split,
                n=args.warmup + args.samples,
                streaming=args.streaming,
                sampling_rate=sampling_rate,
            )
            language_hint = resolve_language_hint(dataset_name, config)
            output["results"][tag] = {}
            settings = [
                ("baseline", base_bundle.model, base_bundle.processor, True, 0.0),
                ("theta_full", latent_bundle.model, latent_bundle.processor, False, -2.0),
                ("theta_zero", latent_bundle.model, latent_bundle.processor, False, 0.0),
                ("theta_skip", latent_bundle.model, latent_bundle.processor, False, 0.5),
            ]
            for name, model, processor, use_baseline, threshold in settings:
                print(f"[latency] {tag} {name}")
                output["results"][tag][name] = measure_setting(
                    model=model,
                    processor=processor,
                    samples=samples,
                    use_baseline=use_baseline,
                    threshold=threshold,
                    language_hint=language_hint,
                    warmup=args.warmup,
                    measure=args.samples,
                    max_new_tokens=args.max_new_tokens,
                )
                print(output["results"][tag][name])
    finally:
        release_model(latent_bundle)
        release_model(base_bundle)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
