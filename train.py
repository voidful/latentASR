import gc

import re
import string
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from datasets import load_dataset
from tqdm import tqdm
from jiwer import wer

# Tee stdout/stderr to log file immediately on import.
from utils import Logger, set_seed, env_flag, mode_label
from config import get_config, TrainingConfig
from losses import _fmt, trajectory_regularization_loss
from data import prepare_dataset, DataCollatorQwenASR
from peft_utils import attach_peft_adapter, save_peft_adapter_checkpoint
from model import LatentQwenASR

sys.stdout = Logger()
sys.stderr = sys.stdout

from qwen_asr import Qwen3ASRModel  # type: ignore
from transformers import GenerationConfig


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(
    model: LatentQwenASR,
    processor: Any,
    eval_dataset: Any,
    num_samples: int,
    use_baseline: bool,
) -> Tuple[float, int]:
    """Compute the word error rate (WER) on a subset of evaluation data.

    Returns:
        wer_value: WER over valid samples
        processed: number of samples processed
    """
    model.eval()
    preds: List[str] = []
    refs: List[str] = []
    it = iter(eval_dataset)
    prompt_active = bool(getattr(model, "use_soft_prompt", False))
    effective_baseline = use_baseline or (not model.use_latent and not prompt_active)
    if effective_baseline:
        desc = "Baseline"
    elif model.use_latent:
        desc = f"Latent (N={model.n_latent})"
    else:
        desc = f"Prompt (N={model.n_latent})"
    print(f"\n--- {desc} ---")

    processed = 0
    total = num_samples
    try:
        ds_len = len(eval_dataset)
        total = min(num_samples, ds_len)
    except Exception:
        total = num_samples

    for _ in tqdm(range(total), desc="Evaluating"):
        try:
            sample = next(it)
        except StopIteration:
            if processed == 0:
                print("Warning: eval dataset exhausted before any samples were read.")
            else:
                print(f"Warning: eval dataset exhausted early at {processed} samples.")
            break
        processed += 1
        target_dtype = model.thinker.dtype if hasattr(model.thinker, "dtype") else torch.float32

        feats = torch.tensor(sample["input_features"], dtype=target_dtype).unsqueeze(0).to(model.base_model.device)
        B, F, T = feats.shape
        if "feature_attention_mask" in sample:
            fam = sample["feature_attention_mask"]
            if not isinstance(fam, torch.Tensor):
                fam = torch.tensor(fam, dtype=torch.long)
            if fam.size(-1) < T:
                fam = torch.cat([fam, torch.zeros(T - fam.size(-1), dtype=torch.long)])
            elif fam.size(-1) > T:
                fam = fam[:T]
            feature_attention_mask = fam.unsqueeze(0).to(feats.device)
        else:
            feature_attention_mask = torch.ones((B, T), dtype=torch.long, device=feats.device)

        if effective_baseline:
            gen_ids = model.generate(
                feats,
                feature_attention_mask=feature_attention_mask,
                max_new_tokens=128,
                use_baseline=True,
                return_thoughts=False,
                do_sample=False,
                eos_token_id=model.stop_ids,
                num_beams=1,
            )
            thoughts = None
        else:
            if model.use_latent:
                gen_ids, thoughts = model.generate(
                    feats,
                    feature_attention_mask=feature_attention_mask,
                    max_new_tokens=128,
                    use_baseline=False,
                    return_thoughts=True,
                    do_sample=False,
                    eos_token_id=model.stop_ids,
                    num_beams=1,
                )
            else:
                gen_ids = model.generate(
                    feats,
                    feature_attention_mask=feature_attention_mask,
                    max_new_tokens=128,
                    use_baseline=False,
                    return_thoughts=False,
                    do_sample=False,
                    eos_token_id=model.stop_ids,
                    num_beams=1,
                )
                thoughts = None

        stop_ids = getattr(model, "stop_ids", [processor.tokenizer.eos_token_id])
        ids = gen_ids[0]
        earliest_stop = ids.numel()
        for sid in stop_ids:
            if sid is None:
                continue
            matches = (ids == sid).nonzero(as_tuple=True)[0]
            if matches.numel() > 0:
                pos = int(matches[0].item())
                if pos < earliest_stop:
                    earliest_stop = pos
        if earliest_stop < ids.numel():
            ids = ids[:earliest_stop]

        pred_text_raw = processor.tokenizer.decode(ids, skip_special_tokens=True)
        pred_text = re.sub(r"language\s+\w+<asr_text>", "", pred_text_raw, flags=re.IGNORECASE)
        if "<asr_text>" in pred_text:
            pred_text = pred_text.split("<asr_text>")[1]
        pred_text = pred_text.strip()

        def _normalize(t: str) -> str:
            t = t.lower()
            t = t.translate(str.maketrans("", "", string.punctuation))
            return " ".join(t.split())

        ref_text = sample["reference_text"]
        preds.append(_normalize(pred_text))
        refs.append(_normalize(ref_text))

        if len(preds) <= 5:
            print(f"\n[Sample {len(preds)}]")
            print(f"  Ref:  {ref_text.strip()}")
            print(f"  Pred: {pred_text.strip()}")
            print(f"  Norm Pred: {_normalize(pred_text)}")
            print(f"  Ids:  {gen_ids[0].tolist()[:20]}...")
            if thoughts is not None:
                thought_text = ""
                try:
                    t_vecs = thoughts[0].float()
                    t_norm = t_vecs / (t_vecs.norm(dim=-1, keepdim=True) + 1e-8)
                    emb_weight = model.embed_tokens.weight.float()
                    vocab_size = emb_weight.size(0)
                    if vocab_size > 10000:
                        idx = torch.randperm(vocab_size, device=emb_weight.device)[:10000]
                        emb_sub = emb_weight[idx]
                        emb_norm = emb_sub / (emb_sub.norm(dim=-1, keepdim=True) + 1e-8)
                        sims = torch.matmul(t_norm, emb_norm.t())
                        top_vals, top_ids = sims.topk(3, dim=-1)
                        thought_lines = []
                        for i in range(len(top_ids)):
                            sub_ids = idx[top_ids[i]].tolist()
                            toks = processor.tokenizer.convert_ids_to_tokens(sub_ids)
                            thought_lines.append(f"T{i}:{toks}")
                        thought_text = " | ".join(thought_lines)
                    else:
                        emb_norm = emb_weight / (emb_weight.norm(dim=-1, keepdim=True) + 1e-8)
                        sims = torch.matmul(t_norm, emb_norm.t())
                        top_vals, top_ids = sims.topk(3, dim=-1)
                        thought_lines = []
                        for i in range(len(top_ids)):
                            toks = processor.tokenizer.convert_ids_to_tokens(top_ids[i].tolist())
                            thought_lines.append(f"T{i}:{toks}")
                        thought_text = " | ".join(thought_lines)
                except Exception as e:
                    thought_text = f"Error decoding: {e}"
                print(f"  Thoughts: {thought_text}")

    valid = [(p, r) for p, r in zip(preds, refs) if r]
    if not valid:
        return 1.0, processed
    vp, vr = zip(*valid)
    return wer(list(vr), list(vp)), processed


def _run_eval_pair(
    model: LatentQwenASR,
    processor: Any,
    eval_ds_clean: Any,
    eval_ds_other: Any,
    cfg: TrainingConfig,
    num_samples: int,
    label: str,
    primary_eval_name: str,
    primary_use_baseline: bool,
    train_mode_tag: str,
) -> Tuple[float, int, float, int, float, int, float, int]:
    """Run baseline + latent evaluation on both test-clean and test-other.

    Returns:
        (wer_base_clean, n_base_clean,
         wer_lat_clean,  n_lat_clean,
         wer_base_other, n_base_other,
         wer_lat_other,  n_lat_other)
    """
    print("\n[test-clean]")
    wer_base_clean, n_base_clean = evaluate_model(
        model, processor, eval_ds_clean,
        num_samples=num_samples, use_baseline=primary_use_baseline,
    )
    print(
        f">>> {primary_eval_name} WER ({label}, test-clean): "
        f"{wer_base_clean:.4f} ({wer_base_clean * 100:.2f}%) | n={n_base_clean}"
    )
    if model.use_latent:
        wer_lat_clean, n_lat_clean = evaluate_model(
            model, processor, eval_ds_clean,
            num_samples=num_samples, use_baseline=False,
        )
        print(
            f">>> Latent WER ({label}, test-clean):   "
            f"{wer_lat_clean:.4f} ({wer_lat_clean * 100:.2f}%) | n={n_lat_clean}"
        )
    else:
        wer_lat_clean, n_lat_clean = wer_base_clean, n_base_clean
        print(f">>> Latent WER ({label}, test-clean):   skipped ({train_mode_tag} mode)")

    n_other_total = len(list(eval_ds_other)) if hasattr(eval_ds_other, '__iter__') else len(eval_ds_other)
    if n_other_total > 0:
        print("\n[test-other]")
        wer_base_other, n_base_other = evaluate_model(
            model, processor, eval_ds_other,
            num_samples=num_samples, use_baseline=primary_use_baseline,
        )
        print(
            f">>> {primary_eval_name} WER ({label}, test-other): "
            f"{wer_base_other:.4f} ({wer_base_other * 100:.2f}%) | n={n_base_other}"
        )
        if model.use_latent:
            wer_lat_other, n_lat_other = evaluate_model(
                model, processor, eval_ds_other,
                num_samples=num_samples, use_baseline=False,
            )
            print(
                f">>> Latent WER ({label}, test-other):   "
                f"{wer_lat_other:.4f} ({wer_lat_other * 100:.2f}%) | n={n_lat_other}"
            )
        else:
            wer_lat_other, n_lat_other = wer_base_other, n_base_other
            print(f">>> Latent WER ({label}, test-other):   skipped ({train_mode_tag} mode)")
    else:
        wer_base_other, n_base_other = 1.0, 0
        wer_lat_other, n_lat_other = 1.0, 0

    return (
        wer_base_clean, n_base_clean,
        wer_lat_clean, n_lat_clean,
        wer_base_other, n_base_other,
        wer_lat_other, n_lat_other,
    )


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = get_config()
    set_seed()

    print(f"CUDA Available: {torch.cuda.is_available()}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        cfg.model_id,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map=device if device == "cuda" else None,
    )
    asr_model = asr_wrapper.model
    processor = asr_wrapper.processor

    print(f"Using device: {device}")
    if cfg.use_latent_reasoning:
        active_n_latent = cfg.n_latent
    elif cfg.use_prompt_tuning:
        active_n_latent = cfg.prompt_tuning_num_virtual_tokens
    else:
        active_n_latent = 0
    if (cfg.use_latent_reasoning or cfg.use_prompt_tuning) and active_n_latent <= 0:
        raise ValueError(
            f"{mode_label(cfg.train_mode)} requires a positive front-token count, "
            f"got {active_n_latent}."
        )

    freeze_base_default = cfg.use_latent_reasoning or cfg.use_prompt_tuning
    if cfg.use_peft_mode:
        freeze_base_default = False
    freeze_base = env_flag("FREEZE_BASE", default=freeze_base_default)
    if cfg.use_peft_mode and freeze_base:
        print("[warn] FREEZE_BASE=1 is incompatible with PEFT adapters. Forcing FREEZE_BASE=0.")
        freeze_base = False
    freeze_audio_stack_default = cfg.use_latent_reasoning or cfg.use_prompt_tuning or cfg.use_peft_mode
    freeze_audio_stack = env_flag("FREEZE_AUDIO_STACK", default=freeze_audio_stack_default)
    print(
        f"Training mode: {cfg.train_mode} ({mode_label(cfg.train_mode)}) | "
        f"use_latent={cfg.use_latent_reasoning} | n_latent={active_n_latent} | "
        f"freeze_base={freeze_base} | freeze_audio_stack={freeze_audio_stack}"
    )
    peft_metadata: Dict[str, Any] = {}

    bos_id = processor.tokenizer.bos_token_id
    eos_id = processor.tokenizer.eos_token_id
    pad_id = processor.tokenizer.pad_token_id

    print("=== Tokenizer Special Tokens ===")
    print(f"  bos_token_id: {bos_id} ({processor.tokenizer.bos_token!r})")
    print(f"  eos_token_id: {eos_id} ({processor.tokenizer.eos_token!r})")
    _im_end_check = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    print(f"  im_end_id (by literal): {_im_end_check}")
    if _im_end_check != eos_id:
        print(f"  [WARN] im_end_id ({_im_end_check}) != eos_token_id ({eos_id}) — using literal im_end_id")
    print(f"  pad_token_id: {pad_id}")

    added_tokens = list(processor.tokenizer.added_tokens_encoder.keys())[:20]
    print(f"  First 20 added tokens: {added_tokens}")

    im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    if im_start_id is None or im_start_id == processor.tokenizer.unk_token_id:
        im_start_id = eos_id
    print(f"  im_start_id: {im_start_id}")

    start_id = im_start_id if im_start_id is not None else (eos_id if eos_id is not None else 0)
    lang_id = start_id
    transcribe_id = start_id

    if active_n_latent > 0:
        print("  Adding special <|latent|> token...")
        special_tokens_dict = {"additional_special_tokens": ["<|latent|>"]}
        num_added_toks = processor.tokenizer.add_special_tokens(special_tokens_dict)
        if num_added_toks > 0:
            print(f"  Resizing model embeddings to {len(processor.tokenizer)}...")
            asr_model.thinker.resize_token_embeddings(len(processor.tokenizer))
        nt_id = processor.tokenizer.convert_tokens_to_ids("<|latent|>")
        print(f"  nt_id: {nt_id} ({processor.tokenizer.convert_ids_to_tokens(nt_id)!r})")
    else:
        nt_id = -1
        print(f"  Front prompt token disabled in {mode_label(cfg.train_mode)} mode.")

    print(f"LANG_ID={lang_id}, TRANSCRIBE_ID={transcribe_id}, NT_ID={nt_id}")

    if cfg.use_peft_mode:
        print(f"Attaching PEFT adapter for mode={mode_label(cfg.train_mode)}...")
        peft_metadata = attach_peft_adapter(asr_model=asr_model, cfg=cfg)
        print(f"PEFT metadata: {peft_metadata}")

    # Load datasets
    thought_mode = cfg.thought_mode
    thought_group_size = cfg.thought_group_size
    print(f"Thought mode: {thought_mode} (group_size={thought_group_size})")

    def _make_dataset_fn(split_nt_id: int, split_n_latent: int) -> Any:
        """Return a prepare_dataset lambda bound to the current thought config."""
        return lambda batch: prepare_dataset(
            processor, lang_id, transcribe_id,
            split_nt_id, split_n_latent, batch,
            thought_mode=thought_mode,
            thought_group_size=thought_group_size,
        )

    print("Loading train dataset...")
    train_split = "train" if "extreme_asr_pony" in cfg.dataset_name else "train.100"
    train_ds = load_dataset(cfg.dataset_name, cfg.dataset_config, split=train_split)
    if cfg.train_max_samples > 0:
        max_train = min(int(cfg.train_max_samples), len(train_ds))
        train_ds = train_ds.shuffle(seed=42).select(range(max_train))
        print(
            f"Train dataset subsampled: {max_train} samples "
            f"(TRAIN_MAX_SAMPLES={cfg.train_max_samples})"
        )
    
    # Dynamically remove columns that exist in the loaded dataset
    cols_to_remove = ["audio", "file", "id", "chapter_id", "speaker_id"]
    train_remove = [c for c in cols_to_remove if c in train_ds.column_names]
    train_ds = train_ds.map(
        _make_dataset_fn(nt_id, active_n_latent),
        remove_columns=train_remove,
    )
    print(f"Train dataset size: {len(train_ds)} samples")

    try:
        if "extreme_asr_pony" in cfg.dataset_name:
            # Pony doesn't have test splits, so we just take a small validation slice from train
            # To avoid dropping train samples, we reload a separate train slice just for eval
            eval_ds_clean = load_dataset(cfg.dataset_name, cfg.dataset_config, split="train[:5%]")
        else:
            eval_ds_clean = load_dataset(cfg.dataset_name, "clean", split="test")

        clean_remove = [c for c in cols_to_remove if c in eval_ds_clean.column_names]
        eval_ds_clean = eval_ds_clean.map(
            _make_dataset_fn(nt_id, active_n_latent),
            remove_columns=clean_remove,
        )
    except Exception as e:
        print(f"Warning: Could not load test-clean split: {e}")
        eval_ds_clean = []

    try:
        if "extreme_asr_pony" in cfg.dataset_name:
            # No test-other for pony
            eval_ds_other = []
        else:
            eval_ds_other = load_dataset(cfg.dataset_name, "other", split="test")
            other_remove = [c for c in cols_to_remove if c in eval_ds_other.column_names]
            eval_ds_other = eval_ds_other.map(
                _make_dataset_fn(nt_id, active_n_latent),
                remove_columns=other_remove,
            )
    except Exception as e:
        print(f"Warning: Could not load test-other split: {e}")
        eval_ds_other = []
    
    # Safely get lengths
    n_clean = len(list(eval_ds_clean)) if hasattr(eval_ds_clean, '__iter__') else len(eval_ds_clean)
    n_other = len(list(eval_ds_other)) if hasattr(eval_ds_other, '__iter__') else len(eval_ds_other)
    print(f"Eval dataset sizes: test-clean={n_clean}, test-other={n_other}")

    collator = DataCollatorQwenASR(processor)
    dl_kwargs: Dict[str, Any] = {"batch_size": cfg.batch_size, "shuffle": True, "collate_fn": collator}
    if device == "cuda":
        dl_kwargs["num_workers"] = 2
        dl_kwargs["pin_memory"] = True
    train_loader = DataLoader(train_ds, **dl_kwargs)

    model = LatentQwenASR(
        asr_model,
        processor,
        n_latent=active_n_latent,
        nt_token_id=nt_id,
        lang_token_id=lang_id,
        transcribe_token_id=transcribe_id,
        freeze_base=freeze_base,
        use_latent=cfg.use_latent_reasoning,
        use_soft_prompt=cfg.use_prompt_tuning,
        soft_prompt_init_mode=cfg.prompt_tuning_init_mode,
        soft_prompt_init_text=cfg.prompt_tuning_init_text,
        user_prompt_text=cfg.user_prompt_text,
        delta_tanh_c=cfg.delta_tanh_c,
        scale_max=cfg.scale_max,
        scale_init=cfg.scale_init,
        thought_mode=cfg.thought_mode,
        thought_group_size=cfg.thought_group_size,
        halt_threshold=cfg.halt_threshold,
        latent_drop_prob=cfg.latent_drop_prob,
        latent_input_noise_std=cfg.latent_input_noise_std,
        latent_use_bounded_delta=cfg.latent_use_bounded_delta,
        latent_use_injection_gate=cfg.latent_use_injection_gate,
        latent_use_embedding_anchor=cfg.latent_use_embedding_anchor,
        freeze_audio_stack=freeze_audio_stack,
    ).to(device)

    train_mode_tag = mode_label(cfg.train_mode)

    # Optimizer
    if model.use_latent:
        optim_params = [
            {
                "params": (
                    list(model.init_proj.parameters())
                    + list(model.delta_proj.parameters())
                    + list(model.step_proj.parameters())
                    + [model.step_embed]
                ),
                "lr": cfg.lr_adapter,
            },
            {"params": [model.log_scale], "lr": cfg.lr_scale},
            {
                "params": (
                    list(model.value_head.parameters())
                    + list(model.thought_ln.parameters())
                    + list(model.injection_gate.parameters())
                ),
                "lr": cfg.lr_adapter,
            },
        ]
        optimizer = torch.optim.AdamW(optim_params, weight_decay=0.01)
    elif model.use_soft_prompt:
        if not model.soft_prompt_embed.requires_grad:
            raise RuntimeError("Prompt-tuning active but soft_prompt_embed is frozen.")
        optimizer = torch.optim.AdamW(
            [model.soft_prompt_embed],
            lr=cfg.lr_prompt_tuning,
            weight_decay=0.01,
        )
        print(
            f"Optimizer ({train_mode_tag}): AdamW lr={cfg.lr_prompt_tuning} "
            f"trainable_params={model.soft_prompt_embed.numel()}"
        )
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError(f"No trainable parameters found in {train_mode_tag} mode.")
        non_latent_lr = cfg.lr_lora_r16 if cfg.use_lora_r16 else cfg.lr_baseline_ft
        optimizer = torch.optim.AdamW(trainable_params, lr=non_latent_lr, weight_decay=0.01)
        print(
            f"Optimizer ({train_mode_tag}): AdamW lr={non_latent_lr} "
            f"trainable_params={sum(p.numel() for p in trainable_params)}"
        )
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    grad_accum_steps = max(1, int(cfg.grad_accum_steps))
    effective_batch = cfg.batch_size * grad_accum_steps
    print(
        f"Gradient accumulation: micro_batch={cfg.batch_size} × accum={grad_accum_steps} "
        f"= effective_batch={effective_batch}"
    )

    batches_per_epoch = len(train_loader)
    optim_steps_per_epoch = max(1, batches_per_epoch // grad_accum_steps)
    total_optim_steps = optim_steps_per_epoch * cfg.num_epochs

    use_lr_schedule = cfg.use_prompt_tuning or cfg.use_lora_r16
    scheduler = None
    if use_lr_schedule:
        warmup_steps = min(100, max(1, total_optim_steps // 10))
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_optim_steps - warmup_steps))
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
        print(
            f"LR schedule: warmup={warmup_steps} optim-steps -> cosine decay "
            f"(total={total_optim_steps} optim-steps)"
        )

    primary_eval_name = "Prompt" if model.use_soft_prompt else "Baseline"
    primary_use_baseline = not model.use_soft_prompt

    # Pre-training evaluation
    print("\n" + "=" * 50)
    print("Pre-Training Evaluation")
    print("=" * 50)
    print(f"Pre-training eval samples: {cfg.pretrain_eval_samples}")
    (
        wer_base_pre_clean, n_base_pre_clean,
        wer_lat_pre_clean, n_lat_pre_clean,
        wer_base_pre_other, n_base_pre_other,
        wer_lat_pre_other, n_lat_pre_other,
    ) = _run_eval_pair(
        model, processor, eval_ds_clean, eval_ds_other, cfg,
        num_samples=cfg.pretrain_eval_samples,
        label="pre",
        primary_eval_name=primary_eval_name,
        primary_use_baseline=primary_use_baseline,
        train_mode_tag=train_mode_tag,
    )

    print("\n" + "=" * 50)
    print("Starting Training...")
    print("=" * 50)

    global_step = 0
    best_wer = float("inf")
    if model.use_latent:
        ckpt_prefix = cfg.checkpoint_prefix or "latent_qwen_asr"
        best_metric_name = "latent"
    elif cfg.use_prompt_tuning:
        ckpt_prefix = cfg.checkpoint_prefix or "prompt_tuning_qwen_asr"
        best_metric_name = "prompt_tuning"
    elif cfg.use_lora_r16:
        ckpt_prefix = cfg.checkpoint_prefix or "lora_r16_qwen_asr"
        best_metric_name = "lora_r16"
    else:
        ckpt_prefix = cfg.checkpoint_prefix or "baseline_qwen_asr"
        best_metric_name = "baseline"


    for epoch in range(1, cfg.num_epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{cfg.num_epochs}")
        print(f"{'='*50}")

        model.train()
        running = 0.0
        epoch_loss = 0.0
        num_batches = 0
        accum_count = 0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            target_dtype = model.thinker.dtype if hasattr(model.thinker, "dtype") else torch.float32
            input_features = batch["input_features"].to(device, dtype=target_dtype)
            feature_attention_mask = batch["feature_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits, stats, deltas, states, labels_for_loss, initial_state = model(
                input_features,
                labels,
                feature_attention_mask=feature_attention_mask,
                global_step=global_step,
            )

            # Bug fix: guard against all-masked batches that would produce NaN CE loss.
            if (labels_for_loss != -100).sum() == 0:
                optimizer.zero_grad(set_to_none=True)
                continue

            eos_counts = (labels_for_loss == int(model.im_end_id)).sum(dim=1)
            if not torch.all(eos_counts >= 1):
                raise AssertionError(
                    f"Found sample(s) without <|im_end|> supervision in batch: {eos_counts.tolist()}"
                )
            if global_step == 0:
                print(
                    "[sanity] <|im_end|> targets per sample (first batch): "
                    f"min={int(eos_counts.min().item())} max={int(eos_counts.max().item())}"
                )



            # Global average CE Loss for Text Decoder
            ce = loss_fct(logits.reshape(-1, logits.size(-1)), labels_for_loss.reshape(-1))

            if model.use_latent:
                deltas_f = deltas.float()
                states_f = states.float()

                # ---- Value Head: Tanh Delta CE Loss (Continuous Impact) ----
                # Delta_CE = cl_baseline - cl_lr
                # if CE_lr < CE_baseline (LR helps), delta is positive
                # if CE_lr > CE_baseline (LR hurts), delta is negative
                baseline_ce = stats.get("baseline_ce", None)  # (B,)
                
                if baseline_ce is not None and "predicted_value" in stats and "lr_ce" in stats:
                    predicted_value = stats["predicted_value"].float()
                    predicted_value_flat = predicted_value.reshape(-1)
                    lr_ce_per_sample = stats["lr_ce"]  # (B,)
                    lr_fixes = stats.get("lr_fixes", baseline_ce.new_zeros(baseline_ce.size(0)))
                    lr_breaks = stats.get("lr_breaks", baseline_ce.new_zeros(baseline_ce.size(0)))
                    baseline_acc = stats.get("baseline_acc", baseline_ce.new_zeros(baseline_ce.size(0)))
                    lr_acc = stats.get("lr_acc", baseline_ce.new_zeros(baseline_ce.size(0)))
                    
                    with torch.no_grad():
                        batch_bl_ce = baseline_ce.mean()
                        batch_lr_ce = lr_ce_per_sample.mean()
                        
                        # ---- Negative Sampling: force "LR is harmful" examples ----
                        # During training, LR CE is directly optimized so lr_acc >= bl_acc
                        # and lr_ce < bl_ce almost always → target always positive → overconfidence.
                        # Permuting lr metrics across batch doesn't help on OOD data
                        # because ALL samples have lr_ce < bl_ce.
                        # Fix: with probability p_neg, compute target normally then negate it:
                        #   target = -|target|   (forced negative)
                        # This tells the Value Head "this delta is wrong" ~30% of the time.
                        p_neg = float(cfg.value_forced_neg_prob)
                        is_neg_sample = torch.rand(1).item() < p_neg
                        
                        acc_diff = lr_acc - baseline_acc  # (B,), range ~[-0.3, 0.3]
                        
                        # Fallback: when both accuracies are 0 (extreme OOD), acc_diff is
                        # uninformative for that utterance. Use CE difference as the
                        # per-utterance surrogate signal instead of waiting for the
                        # whole minibatch to be degenerate.
                        both_zero = (baseline_acc.abs() < 1e-6) & (lr_acc.abs() < 1e-6)  # (B,)
                        # Scale factor 3 (reduced from 10 to avoid saturation):
                        # a ±0.1 accuracy diff maps to tanh(±0.3) ≈ ±0.29
                        # a ±0.3 accuracy diff maps to tanh(±0.9) ≈ ±0.72
                        acc_target = torch.tanh(acc_diff * 3.0).view(-1)
                        ce_diff = (baseline_ce - lr_ce_per_sample).clamp(-2.0, 2.0)  # (B,)
                        ce_target = torch.tanh(ce_diff * 0.5).view(-1)
                        target_value = torch.where(both_zero.view(-1), ce_target, acc_target)
                        
                        # Label smoothing: shrink toward 0 to prevent target saturation
                        target_value = target_value * 0.9
                        
                        # Force negative: flip target to -|target| for negative samples
                        if is_neg_sample:
                            target_value = -target_value.abs()

                    if target_value.numel() != predicted_value_flat.numel():
                        if predicted_value_flat.numel() % target_value.numel() != 0:
                            raise RuntimeError(
                                "Value-head prediction/target size mismatch: "
                                f"pred={predicted_value_flat.numel()} target={target_value.numel()}"
                            )
                        steps_per_sample = predicted_value_flat.numel() // target_value.numel()
                        target_value = (
                            target_value.unsqueeze(1)
                            .expand(-1, steps_per_sample)
                            .reshape(-1)
                        )

                    l_value = F.mse_loss(predicted_value_flat, target_value)

                    if global_step % 5 == 0:
                        pos_ratio = (target_value > 0).float().mean().item()
                        neg_ratio = (target_value < 0).float().mean().item()
                        fixes_mean = lr_fixes.mean().item()
                        breaks_mean = lr_breaks.mean().item()
                        bl_ce_mean = batch_bl_ce.item()
                        lr_ce_mean = batch_lr_ce.item()
                        bl_acc_mean = baseline_acc.mean().item() if baseline_acc is not None else 0.0
                        lr_acc_mean = lr_acc.mean().item() if lr_acc is not None else 0.0
                        acc_diff_mean = acc_diff.mean().item()
                        print(f"\n[Value-DEBUG] step={global_step} "
                              f"bl_acc={bl_acc_mean:.4f} lr_acc={lr_acc_mean:.4f} "
                              f"acc_diff={acc_diff_mean:.4f} "
                              f"fixes={fixes_mean:.1f} breaks={breaks_mean:.1f} "
                              f"bl_ce={bl_ce_mean:.4f} lr_ce={lr_ce_mean:.4f} "
                              f"pos_rate={pos_ratio:.2f} neg_rate={neg_ratio:.2f} "
                              f"neg_sample={'Y' if is_neg_sample else 'N'} "
                              f"tgt={target_value.mean().item():.4f} "
                              f"pred={predicted_value_flat.mean().item():.4f} l_value={l_value.item():.4f}")
                else:
                    l_value = ce.new_tensor(0.0).float()

                # Decay w_cycle linearly over training to encourage deviation from baseline
                progress = global_step / max(1, float(total_optim_steps))
                current_w_cycle = cfg.w_cycle * max(0.0, 1.0 - progress)

                # Cycle consistency loss
                traj = torch.cat([initial_state.float().unsqueeze(1), states_f], dim=1)
                l_cycle = trajectory_regularization_loss(traj, alpha=0.3) if current_w_cycle > 0 else states_f.new_tensor(0.0)

                loss = (
                    ce
                    + current_w_cycle * l_cycle
                    + cfg.w_value * l_value  # Value Head MSE Loss
                )

                if torch.isnan(loss):
                    print(f"\n[FATAL] NaN Loss Detected at step {global_step}!")
                    print(f"  ce: {ce.item()} | l_cycle: {l_cycle.item()} | l_value: {l_value.item() if isinstance(l_value, torch.Tensor) else l_value}")
                    if isinstance(l_value, torch.Tensor) and torch.isnan(l_value):
                        print(f"  --- Value Head Breakdown ---")
                        print(f"  predicted_value: {predicted_value.item():.4f} | target_value: {target_value.item():.4f}")
            else:
                loss = ce
                if torch.isnan(loss):
                    print(f"\n[FATAL] NaN Baseline Loss Detected at step {global_step}!")
                    print(f"  ce: {ce.item()}")

            scaled_loss = loss / grad_accum_steps
            if scaled_loss.requires_grad:
                scaled_loss.backward()
            accum_count += 1

            if accum_count % grad_accum_steps == 0:
                if global_step % cfg.grad_log_every == 0:
                    with torch.no_grad():
                        if model.use_latent:
                            scale_grad_norm = (
                                0.0 if model.log_scale.grad is None
                                else model.log_scale.grad.norm().item()
                            )
                            delta_proj_grad_norm = (
                                0.0 if model.delta_proj.weight.grad is None
                                else model.delta_proj.weight.grad.norm().item()
                            )
                            step_embed_grad_norm = (
                                0.0 if model.step_embed.grad is None
                                else model.step_embed.grad.norm().item()
                            )
                            if delta_proj_grad_norm == 0.0:
                                print(f"[WARNING] step={global_step}: delta_proj gradients are ZERO!")
                            init_grad_norm = (
                                0.0 if model.init_proj.weight.grad is None
                                else model.init_proj.weight.grad.norm().item()
                            )
                            delta_grad_norm = (
                                0.0 if model.delta_proj.weight.grad is None
                                else model.delta_proj.weight.grad.norm().item()
                            )
                            scales = model.step_scales().detach().cpu()
                            v_loss_val = l_value.item() if isinstance(l_value, torch.Tensor) else 0.0
                            print(
                                f"[grad] scale_g_norm={scale_grad_norm:.6f} | "
                                f"init_g_norm={init_grad_norm:.4f} | delta_g_norm={delta_grad_norm:.4f} | "
                                f"step_embed_g={step_embed_grad_norm:.4f} | "
                                f"ce={ce.item():.4f} | v_mse={v_loss_val:.4f} | v_pred={stats.get('predicted_value', torch.tensor(0.0)).mean().item():.4f} | "
                                f"scales={_fmt(scales)}"
                            )
                        else:
                            grad_norms = [
                                p.grad.norm().item()
                                for p in model.parameters()
                                if p.requires_grad and p.grad is not None
                            ]
                            mean_grad = float(np.mean(grad_norms)) if grad_norms else 0.0
                            print(f"[grad] {train_mode_tag}_grad_norm_mean={mean_grad:.6f}")

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            running += loss.item()
            epoch_loss += loss.item()
            num_batches += 1

            if num_batches % 10 == 0:
                avg = running / 10.0
                running = 0.0
                with torch.no_grad():
                    if model.use_latent:
                        scale_now = model.step_scales().mean().item()
                        val_info = ""
                        try:
                            if "predicted_value" in stats:
                                v_pred = stats["predicted_value"].mean().item() if hasattr(stats["predicted_value"], 'mean') else float(stats["predicted_value"])
                                val_info = f" | v_pred:{v_pred:.3f}"
                        except: pass
                        pbar.set_description(
                            f"Epoch {epoch} | loss {avg:.4f} | scale {scale_now:.4f}{val_info}"
                        )
                    else:
                        pbar.set_description(f"Epoch {epoch} | loss {avg:.4f} | {train_mode_tag}")

            if num_batches % cfg.log_every == 0:
                if model.use_latent:
                    with torch.no_grad():
                        rn_m = stats["raw_norm_mean"].cpu()
                        rn_s = stats["raw_norm_std"].cpu()
                        sn_m = stats["scaled_norm_mean"].cpu()
                        sn_s = stats["scaled_norm_std"].cpu()
                        cm = stats["cos_mean"].cpu()
                        cs = stats["cos_std"].cpu()
                        sc = stats["scales"].cpu()
                        dn = stats["diff_norm"].item()
                        st_c = stats["step_cos"].item()
                        deq_i = stats.get("deq_iters", torch.tensor(0.0)).item()
                        v_loss = l_value.item() if "l_value" in locals() else 0.0
                    thought_text = ""
                    try:
                        t_vecs = states[0]
                        t_norm = t_vecs / (t_vecs.norm(dim=-1, keepdim=True) + 1e-8)
                        emb_weight = model.embed_tokens.weight
                        emb_norm = emb_weight / (emb_weight.norm(dim=-1, keepdim=True) + 1e-8)
                        sims = torch.matmul(t_norm, emb_norm.t())
                        _, top_ids = sims.topk(3, dim=-1)
                        thought_lines = []
                        for i in range(len(top_ids)):
                            toks = processor.tokenizer.convert_ids_to_tokens(top_ids[i].tolist())
                            thought_lines.append(f"T{i}:{toks}")
                        thought_text = " | ".join(thought_lines)
                    except Exception as e:
                        thought_text = f"Error decoding: {e}"

                    print(
                        f"[loss] ce={ce.item():.4f} | "
                        f"traj={l_cycle.item():.4f} | v_loss={v_loss:.4f}\n"
                        f"[thoughts] {thought_text}\n"
                        f"[latent-metrics]\n"
                        f"  deq_iters:        {deq_i:.1f}\n"
                        f"  raw_norm_mean:    {_fmt(rn_m)}\n"
                        f"  raw_norm_std:     {_fmt(rn_s)}\n"
                        f"  scaled_norm_mean: {_fmt(sn_m)}\n"
                        f"  scaled_norm_std:  {_fmt(sn_s)}\n"
                        f"  cos_mean:         {_fmt(cm)}\n"
                        f"  cos_std:          {_fmt(cs)}\n"
                        f"  scales:           {_fmt(sc)}\n"
                        f"  smoothness:       diff_norm={dn:.4f} | step_cos={st_c:.4f}"
                    )
                else:
                    print(f"[loss] ce={ce.item():.4f} | {train_mode_tag}")

        # Flush remaining accumulated gradients at end of epoch.
        if accum_count % grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        print(f"\n[Epoch {epoch}] Average Loss: {avg_epoch_loss:.4f} | Total Optim Steps: {global_step}")

        # Epoch evaluation
        print(f"\n--- Epoch {epoch} Evaluation ---")
        (
            wer_clean_base, n_clean_base,
            wer_clean_lat, n_clean_lat,
            wer_other_base, n_other_base,
            wer_other_lat, n_other_lat,
        ) = _run_eval_pair(
            model, processor, eval_ds_clean, eval_ds_other, cfg,
            num_samples=cfg.eval_samples,
            label=f"epoch {epoch}",
            primary_eval_name=primary_eval_name,
            primary_use_baseline=primary_use_baseline,
            train_mode_tag=train_mode_tag,
        )

        current_metric = wer_clean_lat if model.use_latent else wer_clean_base

        # Save checkpoint per epoch
        ckpt_payload: Dict[str, Any] = {
            "epoch": epoch,
            "global_step": global_step,
            "model_id": cfg.model_id,
            "train_mode": cfg.train_mode,
            "n_latent": active_n_latent,
            "freeze_base": freeze_base,
            "freeze_audio_stack": freeze_audio_stack,
            "wer_clean_baseline": wer_clean_base,
            "wer_other_baseline": wer_other_base,
        }
        if peft_metadata:
            ckpt_payload["peft"] = peft_metadata
        if model.use_latent:
            ckpt_payload.update(
                {
                    "delta_tanh_c": cfg.delta_tanh_c,
                    "value_forced_neg_prob": cfg.value_forced_neg_prob,
                    "latent_use_bounded_delta": cfg.latent_use_bounded_delta,
                    "latent_use_injection_gate": cfg.latent_use_injection_gate,
                    "latent_use_embedding_anchor": cfg.latent_use_embedding_anchor,
                    "train_max_samples": cfg.train_max_samples,
                    "init_proj": model.init_proj.state_dict(),
                    "delta_proj": model.delta_proj.state_dict(),
                    "step_proj": model.step_proj.state_dict(),
                    "step_embed": model.step_embed.detach().cpu(),
                    "log_scale": model.log_scale.detach().cpu(),
                    "value_head": model.value_head.state_dict(),
                    "injection_gate": model.injection_gate.state_dict(),
                    "wer_clean_latent": wer_clean_lat,
                    "wer_other_latent": wer_other_lat,
                }
            )
            ckpt_path = f"{ckpt_prefix}_epoch{epoch}.pth"
            torch.save(ckpt_payload, ckpt_path)
        elif model.use_soft_prompt:
            ckpt_payload.update(
                {
                    "soft_prompt_embed": model.soft_prompt_embed.detach().cpu(),
                    "prompt_tuning_init_mode": cfg.prompt_tuning_init_mode,
                    "prompt_tuning_init_text": cfg.prompt_tuning_init_text,
                }
            )
            ckpt_path = f"{ckpt_prefix}_epoch{epoch}.pth"
            torch.save(ckpt_payload, ckpt_path)
        elif cfg.use_peft_mode:
            ckpt_path = f"{ckpt_prefix}_epoch{epoch}"
            save_peft_adapter_checkpoint(model=model, ckpt_dir=ckpt_path, metadata=ckpt_payload)
        else:
            ckpt_payload.update({"model_state_dict": model.state_dict()})
            ckpt_path = f"{ckpt_prefix}_epoch{epoch}.pth"
            torch.save(ckpt_payload, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

        if current_metric < best_wer:
            best_wer = current_metric
            if cfg.use_peft_mode and (not model.use_latent):
                best_ckpt_path = f"{ckpt_prefix}_best"
                save_peft_adapter_checkpoint(model=model, ckpt_dir=best_ckpt_path, metadata=ckpt_payload)
            else:
                best_ckpt_path = f"{ckpt_prefix}_best.pth"
                torch.save(ckpt_payload, best_ckpt_path)
            print(f"New best model! WER={best_wer:.4f} -> Saved to {best_ckpt_path}")

        model.train()

    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"Total epochs: {cfg.num_epochs}")
    print(f"Total steps: {global_step}")
    print(f"Best WER (test-clean, {best_metric_name}): {best_wer:.4f}")
    print(f"\nCheckpoints saved:")
    epoch_suffix = "" if (cfg.use_peft_mode and (not model.use_latent)) else ".pth"
    best_suffix = "" if (cfg.use_peft_mode and (not model.use_latent)) else ".pth"
    for e in range(1, cfg.num_epochs + 1):
        print(f"  - {ckpt_prefix}_epoch{e}{epoch_suffix}")
    print(f"  - {ckpt_prefix}_best{best_suffix} (best model)")
    print("\nDone.")


if __name__ == "__main__":
    main()
