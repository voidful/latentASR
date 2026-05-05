"""Shared utilities for lr_whisper training and evaluation."""

import os
import random
import sys
from typing import Optional


class Logger(object):
    """Tee stdout/stderr to a log file."""

    def __init__(self, filename: str = "train_latent.log") -> None:
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()


def set_seed(seed: int = 42) -> None:
    """Set seeds for deterministic behaviour across multiple libraries."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable."""
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def normalize_train_mode(mode: str) -> str:
    """Normalise training mode aliases to a canonical string."""
    m = (mode or "").strip().lower()
    aliases = {
        "prompt": "prompt_tuning",
        "prompt-tuning": "prompt_tuning",
        "prompt_tune": "prompt_tuning",
        "lora": "lora_r16",
        "lora16": "lora_r16",
        "lora-16": "lora_r16",
        "lora_r16": "lora_r16",
    }
    return aliases.get(m, m)


def mode_label(train_mode: str) -> str:
    """Return a human-readable label for a training mode."""
    if train_mode == "prompt_tuning":
        return "prompt-tuning"
    if train_mode == "lora_r16":
        return "lora-r16"
    return train_mode
