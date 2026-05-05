"""PEFT adapter utilities for lr_whisper."""

import os
import re
from typing import Any, Dict

import torch
import torch.nn as nn

from config import TrainingConfig


def attach_peft_adapter(
    asr_model: nn.Module,
    cfg: TrainingConfig,
) -> Dict[str, Any]:
    """Attach a LoRA PEFT adapter to *asr_model.thinker*.

    Args:
        asr_model: The base ASR model (has a ``.thinker`` attribute).
        cfg: Training configuration providing LoRA hyperparameters.

    Returns:
        A metadata dict describing the adapter configuration.

    Raises:
        RuntimeError: If the ``peft`` package is not installed.
        ValueError: If ``cfg.train_mode`` is not ``"lora_r16"``.
    """
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except Exception as e:
        raise RuntimeError(
            "PEFT mode requested but `peft` package is unavailable."
        ) from e

    if cfg.train_mode == "lora_r16":
        rank = int(cfg.lora_rank)
        alpha = int(cfg.lora_alpha)
        target_modules = list(cfg.lora_target_modules)

        print(
            f"[lora] rank={rank} alpha={alpha} dropout={cfg.lora_dropout} "
            f"target_modules={target_modules}"
        )

        if any("audio" in m for m in target_modules):
            # User explicitly listed audio modules — respect that.
            pass
        else:
            # Convert plain module names into a regex that excludes audio_tower.
            escaped = [re.escape(m) for m in target_modules]
            target_modules = r"^(?!.*audio_tower).*\b(" + "|".join(escaped) + r")$"
            print(f"[lora] Using regex target_modules to exclude audio_tower: {target_modules}")

        peft_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        asr_model.thinker = get_peft_model(asr_model.thinker, peft_cfg)
        asr_model.thinker.print_trainable_parameters()
        return {
            "peft_mode": "lora_r16",
            "rank": rank,
            "alpha": alpha,
            "dropout": cfg.lora_dropout,
            "target_modules": target_modules,
        }

    raise ValueError(
        f"attach_peft_adapter called with unsupported train_mode={cfg.train_mode!r}. "
        "Only lora_r16 uses PEFT in this trainer."
    )


def save_peft_adapter_checkpoint(
    model: nn.Module,
    ckpt_dir: str,
    metadata: Dict[str, Any],
) -> None:
    """Save a PEFT adapter checkpoint to *ckpt_dir*.

    Args:
        model: The ``LatentQwenASR`` wrapper (has ``.base_model.thinker``).
        ckpt_dir: Directory path for the checkpoint.
        metadata: Arbitrary metadata dict saved alongside the adapter.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    model.base_model.thinker.save_pretrained(ckpt_dir, safe_serialization=False)
    torch.save(metadata, os.path.join(ckpt_dir, "training_state.pth"))
