"""Training configuration dataclass for lr_whisper.

All hyperparameters are read from environment variables via ``TrainingConfig.from_env()``.
Use ``get_config()`` to obtain a process-wide singleton (lazy, never called at import time).
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

from utils import env_flag, normalize_train_mode


@dataclass
class TrainingConfig:
    # Model / dataset
    model_id: str = "Qwen/Qwen3-ASR-0.6B"
    dataset_name: str = "openslr/librispeech_asr"
    dataset_config: str = "clean"

    # Training batch / epoch
    batch_size: int = 4
    grad_accum_steps: int = 4
    num_epochs: int = 10
    eval_samples: int = 1000
    pretrain_eval_samples: int = 10

    # Latent tokens
    n_latent: int = 4
    is_deq_train: bool = False
    deq_train_iters: int = 15
    halt_threshold: float = 0.0
    latent_drop_prob: float = 0.15
    latent_input_noise_std: float = 0.5

    # Thought layout
    thought_mode: str = "prefix"       # "prefix" | "interleaved"
    thought_group_size: int = 1        # words per NT token in interleaved mode

    # Training mode
    train_mode: str = "latent"

    # Learning rates
    lr_adapter: float = 1e-4

    lr_scale: float = 5e-5
    lr_baseline_ft: float = 1e-5
    lr_prompt_tuning: float = 5e-4
    lr_lora_r16: float = 1e-4

    # Prompt text
    user_prompt_text: str = "Transcribe the audio into text."

    # Prompt-tuning hyperparameters
    prompt_tuning_init_mode: str = "text"
    prompt_tuning_num_virtual_tokens: int = 20
    prompt_tuning_init_text: str = "Transcribe the audio into text."

    # LoRA hyperparameters
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )


    # Delta stabilisation / scale
    delta_tanh_c: float = 5.0
    scale_max: float = 3.0
    scale_init: float = 0.2

    # Regularisation weights
    w_smooth: float = 0.0  # Disabled — replaced by w_cycle
    w_cycle: float = 0.1  # Cycle consistency loss (vocabulary-independent) - main regularizer
    w_value: float = 3.0

    # Logging intervals
    log_every: int = 50
    grad_log_every: int = 50

    # Derived flags (set in __post_init__)
    use_latent_reasoning: bool = False
    use_prompt_tuning: bool = False
    use_lora_r16: bool = False
    use_peft_mode: bool = False

    def __post_init__(self) -> None:
        valid_modes = {"latent", "baseline", "prompt_tuning", "lora_r16"}
        if self.train_mode not in valid_modes:
            raise ValueError(
                f"Unsupported train_mode={self.train_mode!r}. "
                f"Use one of: {sorted(valid_modes)}"
            )
        valid_thought_modes = {"prefix", "interleaved"}
        if self.thought_mode not in valid_thought_modes:
            raise ValueError(
                f"Unsupported thought_mode={self.thought_mode!r}. "
                f"Use one of: {sorted(valid_thought_modes)}"
            )
        if self.thought_group_size < 1:
            raise ValueError(f"thought_group_size must be >= 1, got {self.thought_group_size}")
        self.use_latent_reasoning = self.train_mode == "latent"
        self.use_prompt_tuning = self.train_mode == "prompt_tuning"
        self.use_lora_r16 = self.train_mode == "lora_r16"
        self.use_peft_mode = self.use_lora_r16

    @classmethod
    def from_env(cls) -> "TrainingConfig":
        """Construct a ``TrainingConfig`` by reading environment variables."""
        lora_modules_raw = os.getenv(
            "LORA_TARGET_MODULES",
            "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        )
        lora_target_modules = [x.strip() for x in lora_modules_raw.split(",") if x.strip()]

        train_mode = normalize_train_mode(os.getenv("TRAIN_MODE", "latent"))
        default_dataset = "openslr/librispeech_asr"
        default_config = "clean"
        if train_mode == "latent":
            default_dataset = "SpeechTest/extreme_asr_pony"
            default_config = "default"

        return cls(
            model_id=os.getenv("MODEL_ID", "Qwen/Qwen3-ASR-0.6B"),
            dataset_name=os.getenv("DATASET_NAME", default_dataset),
            dataset_config=os.getenv("DATASET_CONFIG", default_config),
            batch_size=int(os.getenv("BATCH_SIZE", "8")),
            grad_accum_steps=int(os.getenv("GRAD_ACCUM_STEPS", "4")),
            num_epochs=int(os.getenv("NUM_EPOCHS", "10")),
            eval_samples=int(os.getenv("EVAL_SAMPLES", "1000")),
            pretrain_eval_samples=int(os.getenv("PRETRAIN_EVAL_SAMPLES", "10")),
            n_latent=int(os.getenv("N_LATENT", "4")),
            is_deq_train=os.getenv("IS_DEQ_TRAIN", "true").strip().lower() == "true",
            deq_train_iters=int(os.getenv("DEQ_TRAIN_ITERS", "15")),
            halt_threshold=float(os.getenv("HALT_THRESHOLD", "0.0")),
            latent_drop_prob=float(os.getenv("LATENT_DROP_PROB", "0.15")),
            latent_input_noise_std=float(os.getenv("LATENT_INPUT_NOISE_STD", "0.5")),
            thought_mode=os.getenv("THOUGHT_MODE", "prefix").strip().lower(),
            thought_group_size=int(os.getenv("THOUGHT_GROUP_SIZE", "1")),
            train_mode=train_mode,
            lr_adapter=float(os.getenv("LR_ADAPTER", "1e-4")),
            lr_scale=float(os.getenv("LR_SCALE", "5e-5")),
            lr_baseline_ft=float(os.getenv("LR_BASELINE_FT", "1e-5")),
            lr_prompt_tuning=float(os.getenv("LR_PROMPT_TUNING", "5e-4")),
            lr_lora_r16=float(os.getenv("LR_LORA_R16", "1e-4")),
            user_prompt_text=os.getenv(
                "USER_PROMPT_TEXT", "Transcribe the audio into text."
            ).strip(),
            prompt_tuning_init_mode=os.getenv("PROMPT_TUNING_INIT_MODE", "text").strip().lower(),
            prompt_tuning_num_virtual_tokens=int(
                os.getenv("PROMPT_TUNING_NUM_VIRTUAL_TOKENS", "20")
            ),
            prompt_tuning_init_text=os.getenv(
                "PROMPT_TUNING_INIT_TEXT", "Transcribe the audio into text."
            ).strip(),
            lora_rank=int(os.getenv("LORA_RANK", "16")),
            lora_alpha=int(os.getenv("LORA_ALPHA", "32")),
            lora_dropout=float(os.getenv("LORA_DROPOUT", "0.05")),
            lora_target_modules=lora_target_modules,
            delta_tanh_c=float(os.getenv("DELTA_TANH_C", "5.0")),
            scale_max=float(os.getenv("SCALE_MAX", "3.0")),
            scale_init=float(os.getenv("SCALE_INIT", "0.2")),
            w_smooth=float(os.getenv("W_SMOOTH", "0.0")),
            w_cycle=float(os.getenv("W_CYCLE", "0.1")),
            w_value=float(os.getenv("W_VALUE", "3.0")),
            log_every=int(os.getenv("LOG_EVERY", "50")),
            grad_log_every=int(os.getenv("GRAD_LOG_EVERY", "50")),
        )


_config: Optional[TrainingConfig] = None


def get_config() -> TrainingConfig:
    """Return the process-wide ``TrainingConfig`` singleton (lazy init from env)."""
    global _config
    if _config is None:
        _config = TrainingConfig.from_env()
    return _config
