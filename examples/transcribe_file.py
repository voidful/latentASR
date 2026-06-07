"""Transcribe one audio file with the released LatentASR adapter."""

from __future__ import annotations

import argparse

import torch
from datasets import Audio

from eval import build_latent_bundle, choose_device, choose_dtype, clean_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe one audio file with LatentASR.")
    parser.add_argument("audio", help="Path to an audio file.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--latent-ckpt", default="checkpoints/latentASR_adapter.pth")
    parser.add_argument("--theta", type=float, default=0.0, help="Dynamic halting threshold.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--language", default="English")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    dtype = choose_dtype(args.dtype, device)
    bundle = build_latent_bundle(
        model_id=args.model_id,
        checkpoint_path=args.latent_ckpt,
        n_latent_override=-1,
        device=device,
        dtype=dtype,
    )
    model = bundle.model
    processor = bundle.processor

    target_sr = int(getattr(processor.feature_extractor, "sampling_rate", 16000) or 16000)
    audio = Audio(sampling_rate=target_sr).decode_example({"path": args.audio})
    feat_out = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_attention_mask=True,
    )
    target_dtype = model.thinker.dtype if hasattr(model.thinker, "dtype") else torch.float32
    feats = torch.tensor(
        feat_out.input_features[0],
        dtype=target_dtype,
        device=model.base_model.device,
    ).unsqueeze(0)
    n_frames = feats.size(-1)
    if getattr(feat_out, "attention_mask", None) is not None:
        mask = torch.tensor(feat_out.attention_mask[0], dtype=torch.long)
        if mask.size(-1) < n_frames:
            mask = torch.cat([mask, torch.zeros(n_frames - mask.size(-1), dtype=torch.long)])
        else:
            mask = mask[:n_frames]
        feature_attention_mask = mask.to(device=model.base_model.device).unsqueeze(0)
    else:
        feature_attention_mask = torch.ones((1, n_frames), dtype=torch.long, device=model.base_model.device)

    gen_ids, stats = model.generate(
        feats,
        feature_attention_mask=feature_attention_mask,
        max_new_tokens=args.max_new_tokens,
        use_baseline=False,
        return_stats=True,
        do_sample=False,
        eos_token_id=model.stop_ids,
        num_beams=1,
        language_hint=args.language,
        dynamic_halt_threshold=args.theta,
    )
    text = clean_prediction(processor.tokenizer.decode(gen_ids[0], skip_special_tokens=True))
    steps = int(stats.get("deq_iters", torch.tensor(0)).item()) if isinstance(stats, dict) else -1
    print(text)
    print(f"[latent_steps={steps} theta={args.theta}]")


if __name__ == "__main__":
    main()

