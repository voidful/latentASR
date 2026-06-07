"""
Evaluate latent-reasoning and baseline/PEFT ASR models.

Supported datasets:
  - SpeechTest/common_voice_16_0
  - SpeechTest/peoples_speech
  - SpeechTest/librispeech_asr
  - SpeechTest/fleurs
  - SpeechTest/voxpopuli
  - SpeechTest/gigaspeech
  - SpeechTest/ASCEND
  - SpeechTest/extreme_asr_pony
  - openslr/librispeech_asr
  - google/fleurs
  - facebook/voxpopuli
  - PolyAI/minds14
  - LIUM/tedlium
  - edinburghcstr/ami

Unknown HuggingFace datasets are also accepted with generic ASR defaults:
an `audio` column plus one of the fallback text columns.

Example:
    python eval.py \
      --baseline-ckpt baseline_qwen_asr_best.pth \
      --prompt-tuning-ckpt prompt_tuning_qwen_asr_best \
      --lora-r16-ckpt lora_r16_qwen_asr_best \
      --latent-ckpt latent_qwen_asr_best.pth \
      --max-samples-per-config 500 \
      --output-json cv16_eval.json
"""

import argparse
import gc
import itertools
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Audio, get_dataset_config_names, load_dataset
from jiwer import cer, wer
from tqdm import tqdm

from qwen_asr import Qwen3ASRModel  # type: ignore
from model import LatentQwenASR
from utils import set_seed


COMMONVOICE_12_LANGS = [
    "en",
]

FLEURS_12_LANGS = [
    "en",
]

VOXPOPULI_30_LANGS = [
    "en",
]

QWEN_FLEURS_CORE12_LANGS = [
    "en",
    "zh",
    "yue",
    "ar",
    "de",
    "es",
    "fr",
    "it",
    "ja",
    "ko",
    "pt",
    "ru",
]

QWEN_FLEURS_EXTRA8_LANGS = [
    "hi",
    "id",
    "ms",
    "nl",
    "pl",
    "th",
    "tr",
    "vi",
]

QWEN_FLEURS_EXTRA10_LANGS = [
    "cs",
    "da",
    "el",
    "fa",
    "fi",
    "fil",
    "hu",
    "mk",
    "ro",
    "sv",
]

QWEN_FLEURS_30_LANGS = (
    QWEN_FLEURS_CORE12_LANGS
    + QWEN_FLEURS_EXTRA8_LANGS
    + QWEN_FLEURS_EXTRA10_LANGS
)

MLS_PUBLIC7_LANGS = ["de", "nl", "es", "fr", "it", "pl", "pt"]

COMMONVOICE_CONFIG_ALIASES = {
    "zh": "zh-CN",
    "zh-cn": "zh-CN",
    "zh_cn": "zh-CN",
    "zh_tw": "zh-TW",
    "zh-tw": "zh-TW",
}

COMMONVOICE_LANGUAGE_HINTS = {
    "en": "English",
}

GENERAL_LANGUAGE_HINTS = {
    "ar": "Arabic",
    "ar_eg": "Arabic",
    "ar-eg": "Arabic",
    "ara": "Arabic",
    "arb": "Arabic",
    "cs": "Czech",
    "cs_cz": "Czech",
    "cs-cz": "Czech",
    "czech": "Czech",
    "da": "Danish",
    "da_dk": "Danish",
    "da-dk": "Danish",
    "danish": "Danish",
    "en": "English",
    "en_us": "English",
    "en-us": "English",
    "en_gb": "English",
    "en-gb": "English",
    "en_au": "English",
    "en-au": "English",
    "english": "English",
    "fr": "French",
    "fr_fr": "French",
    "fr-fr": "French",
    "french": "French",
    "de": "German",
    "de_de": "German",
    "de-de": "German",
    "german": "German",
    "es": "Spanish",
    "es_es": "Spanish",
    "es-es": "Spanish",
    "es_419": "Spanish",
    "spanish": "Spanish",
    "it": "Italian",
    "it_it": "Italian",
    "it-it": "Italian",
    "italian": "Italian",
    "pt": "Portuguese",
    "pt_pt": "Portuguese",
    "pt-pt": "Portuguese",
    "pt_br": "Portuguese",
    "pt-br": "Portuguese",
    "portuguese": "Portuguese",
    "nl": "Dutch",
    "nl_nl": "Dutch",
    "nl-nl": "Dutch",
    "dutch": "Dutch",
    "pl": "Polish",
    "pl_pl": "Polish",
    "pl-pl": "Polish",
    "polish": "Polish",
    "zh": "Chinese",
    "zh_cn": "Chinese",
    "zh-cn": "Chinese",
    "cmn": "Chinese",
    "cmn_hans_cn": "Chinese",
    "cmn-hans-cn": "Chinese",
    "yue": "Cantonese",
    "yue_hant_hk": "Cantonese",
    "yue-hant-hk": "Cantonese",
    "hi": "Hindi",
    "hi_in": "Hindi",
    "hi-in": "Hindi",
    "id": "Indonesian",
    "id_id": "Indonesian",
    "id-id": "Indonesian",
    "ja": "Japanese",
    "ja_jp": "Japanese",
    "ja-jp": "Japanese",
    "ko": "Korean",
    "ko_kr": "Korean",
    "ko-kr": "Korean",
    "ms": "Malay",
    "ms_my": "Malay",
    "ms-my": "Malay",
    "ru": "Russian",
    "ru_ru": "Russian",
    "ru-ru": "Russian",
    "th": "Thai",
    "th_th": "Thai",
    "th-th": "Thai",
    "tr": "Turkish",
    "tr_tr": "Turkish",
    "tr-tr": "Turkish",
    "vi": "Vietnamese",
    "vi_vn": "Vietnamese",
    "vi-vn": "Vietnamese",
    "el": "Greek",
    "el_gr": "Greek",
    "el-gr": "Greek",
    "fa": "Persian",
    "fa_ir": "Persian",
    "fa-ir": "Persian",
    "fi": "Finnish",
    "fi_fi": "Finnish",
    "fi-fi": "Finnish",
    "fil": "Filipino",
    "fil_ph": "Filipino",
    "fil-ph": "Filipino",
    "hu": "Hungarian",
    "hu_hu": "Hungarian",
    "hu-hu": "Hungarian",
    "mk": "Macedonian",
    "mk_mk": "Macedonian",
    "mk-mk": "Macedonian",
    "ro": "Romanian",
    "ro_ro": "Romanian",
    "ro-ro": "Romanian",
    "sv": "Swedish",
    "sv_se": "Swedish",
    "sv-se": "Swedish",
}

FALLBACK_TEXT_COLUMNS = [
    "sentence",
    "text",
    "normalized_text",
    "raw_text",
    "transcript",
    "transcription",
    "english_transcription",
]

# Language-code family aliases used when dataset config IDs differ from
# requested language IDs (e.g., zh -> cmn_hans_cn, fil -> tl, ms -> msa).
LANGUAGE_CODE_ALIASES = {
    "en": ["eng"],
    "zh": ["zho", "cmn", "cmn_hans_cn", "cmn_hans"],
    "zh_tw": ["cmn_hant_tw", "cmn_hant", "zho_hant", "zh_hant_tw"],
    "yue": ["yue_hant_hk", "zh_hk", "zh_yue"],
    "ar": ["ara", "arb"],
    "de": ["deu", "ger"],
    "es": ["spa", "es_419"],
    "fr": ["fra", "fre"],
    "it": ["ita"],
    "ja": ["jpn"],
    "ko": ["kor"],
    "pt": ["por"],
    "ru": ["rus"],
    "id": ["ind", "in"],
    "ms": ["msa", "may", "zsm"],
    "fil": ["tgl", "tl"],
    "th": ["tha"],
    "vi": ["vie"],
    "tr": ["tur"],
    "hi": ["hin"],
    "nl": ["nld", "dut"],
    "sv": ["swe"],
    "da": ["dan"],
    "fi": ["fin"],
    "pl": ["pol"],
    "cs": ["ces", "cze"],
    "fa": ["fas", "per", "pes"],
    "el": ["ell", "gre"],
    "hu": ["hun"],
    "mk": ["mkd", "mac"],
    "ro": ["ron", "rum"],
}

DATASET_PRESETS = {
    "SpeechTest/common_voice_16_0": {
        "default_strategy": "fixed_languages",
        "default_languages": COMMONVOICE_12_LANGS,
        "config_aliases": COMMONVOICE_CONFIG_ALIASES,
        "language_hints": COMMONVOICE_LANGUAGE_HINTS,
        "preferred_text_columns": ["sentence"],
    },
    "SpeechTest/peoples_speech": {
        "default_strategy": "all_available",
        "config_aliases": {},
        "language_hints": {"test": "English", "en": "English"},
        "preferred_text_columns": ["text"],
    },
    "SpeechTest/librispeech_asr": {
        "default_strategy": "all_available",
        "config_aliases": {},
        "language_hints": {
            "clean": "English",
            "other": "English",
            "en": "English",
        },
        "preferred_text_columns": ["text"],
    },
    "SpeechTest/fleurs": {
        "default_strategy": "fixed_languages",
        "default_languages": FLEURS_12_LANGS,
        "config_aliases": {},
        "language_hints": COMMONVOICE_LANGUAGE_HINTS,
        "preferred_text_columns": ["transcription", "raw_transcription", "text"],
    },
    "SpeechTest/voxpopuli": {
        "default_strategy": "fixed_languages",
        "default_languages": VOXPOPULI_30_LANGS,
        "config_aliases": {},
        "language_hints": {
            "en": "English",
        },
        "preferred_text_columns": ["normalized_text", "raw_text", "text"],
    },
    "SpeechTest/gigaspeech": {
        "default_strategy": "all_available",
        "config_aliases": {},
        "language_hints": {"test": "English", "en": "English"},
        "preferred_text_columns": ["text"],
    },
    "SpeechTest/ASCEND": {
        "default_strategy": "all_available",
        "config_aliases": {},
        "language_hints": {},
        "preferred_text_columns": ["transcription", "text"],
    },
    "SpeechTest/extreme_asr_pony": {
        "default_strategy": "all_available",
        "config_aliases": {},
        "language_hints": {},
        "preferred_text_columns": ["text"],
    },
    "openslr/librispeech_asr": {
        "default_strategy": "default_configs",
        "default_configs": ["clean", "other"],
        "config_aliases": {},
        "language_hints": GENERAL_LANGUAGE_HINTS,
        "preferred_text_columns": ["text"],
    },
    "google/fleurs": {
        "default_strategy": "fixed_languages",
        "default_languages": FLEURS_12_LANGS,
        "config_aliases": {},
        "language_hints": GENERAL_LANGUAGE_HINTS,
        "preferred_text_columns": ["transcription", "raw_transcription", "text"],
    },
    "facebook/voxpopuli": {
        "default_strategy": "fixed_languages",
        "default_languages": VOXPOPULI_30_LANGS,
        "config_aliases": {},
        "language_hints": GENERAL_LANGUAGE_HINTS,
        "preferred_text_columns": ["normalized_text", "raw_text", "text"],
    },
    "PolyAI/minds14": {
        "default_strategy": "default_configs",
        "default_configs": ["en-US"],
        "config_aliases": {
            "en": "en-US",
            "en_us": "en-US",
            "en-us": "en-US",
        },
        "language_hints": GENERAL_LANGUAGE_HINTS,
        "preferred_text_columns": ["transcription", "english_transcription", "text"],
    },
    "facebook/multilingual_librispeech": {
        "default_strategy": "fixed_languages",
        "default_languages": MLS_PUBLIC7_LANGS,
        "config_aliases": {
            "de": "german",
            "nl": "dutch",
            "es": "spanish",
            "fr": "french",
            "it": "italian",
            "pl": "polish",
            "pt": "portuguese",
        },
        "language_hints": GENERAL_LANGUAGE_HINTS,
        "preferred_text_columns": ["transcript", "text"],
    },
    "LIUM/tedlium": {
        "default_strategy": "default_configs",
        "default_configs": ["release1"],
        "config_aliases": {
            "r1": "release1",
            "r2": "release2",
            "r3": "release3",
        },
        "language_hints": GENERAL_LANGUAGE_HINTS,
        "preferred_text_columns": ["text"],
    },
    "TwinkStart/tedlium": {
        "default_strategy": "default_configs",
        "default_configs": ["release1"],
        "config_aliases": {
            "r1": "release1",
            "r2": "release2",
            "r3": "release3",
        },
        "language_hints": GENERAL_LANGUAGE_HINTS,
        "preferred_text_columns": ["text"],
    },
    "edinburghcstr/ami": {
        "default_strategy": "default_configs",
        "default_configs": ["ihm"],
        "config_aliases": {},
        "language_hints": GENERAL_LANGUAGE_HINTS,
        "preferred_text_columns": ["text", "transcript", "sentence"],
    },
}


@dataclass
class ModelBundle:
    model: LatentQwenASR
    processor: any
    checkpoint: Dict[str, any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate five ASR settings on SpeechTest or HuggingFace ASR datasets: "
            "base_model, baseline_ft, prompt_tuning, lora_r16, latent_reasoning."
        )
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=os.getenv("MODEL_ID", "Qwen/Qwen3-ASR-0.6B"),
        help="Base Qwen3-ASR model ID.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="SpeechTest/common_voice_16_0",
        help="HuggingFace dataset name.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--configs",
        type=str,
        default="",
        help=(
            "Comma-separated config names (dataset-specific). "
            "Default: dataset preset "
            "(fixed-language datasets use preset languages; "
            "others use preset configs or all subsets). "
            "Ignored when --all-configs is set."
        ),
    )
    parser.add_argument(
        "--all-configs",
        action="store_true",
        help="Evaluate every config in the dataset.",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=0,
        help="Limit number of configs (0 = no limit).",
    )
    parser.add_argument(
        "--max-samples-per-config",
        type=int,
        default=0,
        help="Maximum test samples per config (0 = all).",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help=(
            "Load datasets with HuggingFace streaming=True and iterate samples "
            "sequentially without materializing the split on disk."
        ),
    )
    parser.add_argument(
        "--text-normalizer",
        type=str,
        default="english",
        choices=["english", "basic", "simple"],
        help=(
            "Text normalization before WER/CER. Use 'basic' or 'simple' for "
            "multilingual evaluations to avoid English-specific rewriting."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Max new tokens for generation.",
    )
    parser.add_argument(
        "--baseline-ckpt",
        type=str,
        default="",
        help="Path to baseline fine-tuned checkpoint (.pth).",
    )
    parser.add_argument(
        "--prompt-tuning-ckpt",
        type=str,
        default="",
        help=(
            "Path to prompt-tuning checkpoint. "
            "Supports .pth (wrapper state_dict) or a PEFT adapter directory."
        ),
    )
    parser.add_argument(
        "--lora-r16-ckpt",
        type=str,
        default="",
        help=(
            "Path to LoRA(rank=16) checkpoint. "
            "Supports .pth (wrapper state_dict) or a PEFT adapter directory."
        ),
    )
    parser.add_argument(
        "--latent-ckpt",
        type=str,
        default="",
        help="Path to latent reasoning checkpoint (.pth).",
    )
    parser.add_argument(
        "--skip-base-model",
        action="store_true",
        help="Skip raw pretrained model inference baseline.",
    )
    parser.add_argument(
        "--skip-baseline-ft",
        action="store_true",
        help="Skip baseline fine-tuned model evaluation.",
    )
    parser.add_argument(
        "--skip-prompt-tuning",
        action="store_true",
        help="Skip prompt tuning model evaluation.",
    )
    parser.add_argument(
        "--skip-lora-r16",
        action="store_true",
        help="Skip LoRA r16 model evaluation.",
    )
    parser.add_argument(
        "--n-latent",
        type=int,
        default=-1,
        help="Override latent token count. If < 0, use checkpoint n_latent.",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=1,
        help="Number of beams for beam search.",
    )
    parser.add_argument(
        "--dynamic-halt-threshold",
        type=float,
        default=0.0,
        help="Value head threshold for early halting of latent thoughts.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run inference.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Inference dtype.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--print-samples",
        type=int,
        default=2,
        help="Number of qualitative samples to print per config.",
    )
    parser.add_argument(
        "--print-gate-skips",
        action="store_true",
        help=(
            "Also print latent_reasoning samples skipped by the value-head gate. "
            "Default is off to keep large sweep logs readable."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="If set, write detailed results to this JSON file.",
    )
    parser.add_argument(
        "--no-language-hint",
        action="store_true",
        help="Disable language-specific prompting and use generic ASR prompt.",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="Transcribe the {language} audio into text.",
        help=(
            "User prompt template. Use {language} placeholder. "
            "Ignored when --no-language-hint is set."
        ),
    )
    parser.add_argument(
        "--snr-db",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Add Gaussian noise to audio at specified SNR levels (dB). "
            "Can specify multiple values, e.g. --snr-db 20 10 5. "
            "Each SNR level will be evaluated as a separate row. "
            "If not set, no noise is added."
        ),
    )
    return parser.parse_args()


def choose_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but CUDA is not available.")
    return device_arg


def choose_dtype(dtype_arg: str, device: str) -> torch.dtype:
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16 if device == "cuda" else torch.float32
    if dtype_arg == "bfloat16":
        return torch.bfloat16 if device == "cuda" else torch.float32

    # auto
    if device != "cuda":
        return torch.float32
    major = torch.cuda.get_device_capability(0)[0]
    if major >= 8:
        return torch.bfloat16
    return torch.float16


def get_dataset_preset(dataset_name: str) -> Dict[str, any]:
    return DATASET_PRESETS.get(dataset_name, {})


def resolve_dataset_name(dataset_name: str) -> str:
    key = (dataset_name or "").strip()
    if not key:
        raise ValueError("dataset name is empty.")

    if key in DATASET_PRESETS:
        return key

    lowered = key.lower()
    for canonical in DATASET_PRESETS:
        if canonical.lower() == lowered:
            return canonical

    print(
        f"[warn] dataset '{dataset_name}' has no local preset; "
        "using generic HuggingFace ASR defaults."
    )
    return key


def _map_config_alias(name: str, alias_map: Dict[str, str]) -> str:
    n = name.strip()
    if not n:
        return n
    lowered = n.lower()
    if lowered in alias_map:
        return alias_map[lowered]
    return n


def resolve_language_hint(dataset_name: str, config_name: str) -> Optional[str]:
    preset = get_dataset_preset(dataset_name)
    hints_raw = preset.get("language_hints", {})
    hints = {k.lower(): v for k, v in hints_raw.items()}

    lowered = config_name.lower()
    if lowered in hints:
        return hints[lowered]

    # Generic base-language fallback (e.g. "fr_ca" -> "fr").
    base = lowered.split("-")[0].split("_")[0]
    for key, val in hints.items():
        if key.split("-")[0].split("_")[0] == base:
            return val
    return None


def _norm_cfg_key(v: str) -> str:
    return (v or "").strip().lower().replace("-", "_")


def _base_lang(v: str) -> str:
    return _norm_cfg_key(v).split("_")[0]


def _language_candidate_forms(lang_code: str) -> List[str]:
    code = _norm_cfg_key(lang_code)
    out = {
        code,
        code.replace("_", "-"),
    }
    base = _base_lang(code)
    out.add(base)

    # Common alias expansions for multilingual dataset configs.
    if code == "zh":
        out.update({"zh_cn", "zh-cn", "zh_hans", "zh-hans"})
    if code in {"zh_tw", "zhtw"}:
        out.update({"zh_tw", "zh-tw", "zh_hant", "zh-hant", "zh_hk", "zh-hk"})
    if code == "yue":
        out.update({"yue_hk", "yue-hk", "yue_hant_hk", "yue-hant-hk", "zh_yue", "zh-yue", "zh_hk", "zh-hk"})
    if code == "fil":
        out.update({"fil_ph", "fil-ph", "tl", "tl_ph", "tl-ph", "tgl", "tgl_ph", "tgl-ph"})

    # Expand across ISO variants / dataset-specific aliases.
    base_aliases = LANGUAGE_CODE_ALIASES.get(base, [])
    for alias in base_aliases:
        alias_norm = _norm_cfg_key(alias)
        out.update(
            {
                alias_norm,
                alias_norm.replace("_", "-"),
                _base_lang(alias_norm),
            }
        )

    return list(out)


def _pick_config_for_language(lang_code: str, available_configs: List[str]) -> Optional[str]:
    if not available_configs:
        return None
    infos = []
    for cfg in available_configs:
        norm = _norm_cfg_key(cfg)
        infos.append((cfg, norm, _base_lang(norm)))

    candidates = set(_language_candidate_forms(lang_code))
    code_norm = _norm_cfg_key(lang_code)
    code_base = _base_lang(code_norm)

    # 1) exact-ish match against candidate forms
    exact_hits = [t for t in infos if t[1] in candidates]
    if exact_hits:
        # Prefer exact code, then prefix match (e.g. en_us for en), then lexical.
        exact_hits.sort(
            key=lambda t: (
                0 if t[1] == code_norm else 1,
                0 if t[1].startswith(code_base + "_") else 1,
                t[1],
            )
        )
        return exact_hits[0][0]

    # 2) base-language fallback
    base_hits = [t for t in infos if t[2] == code_base]
    if base_hits:
        # For Chinese variants, bias toward region when specified.
        def _rank(tup: Tuple[str, str, str]) -> Tuple[int, int, str]:
            norm = tup[1]
            if code_norm in {"zh_tw", "zhtw"}:
                zh_tw_bias = 0 if ("_tw" in norm or "hant" in norm) else 1
                return (zh_tw_bias, 0 if norm.startswith(code_base + "_") else 1, norm)
            if code_norm == "zh":
                zh_cn_bias = 0 if ("_cn" in norm or "hans" in norm) else 1
                return (zh_cn_bias, 0 if norm == "zh" else 1, norm)
            return (0, 0 if norm.startswith(code_base + "_") else 1, norm)

        base_hits.sort(key=_rank)
        return base_hits[0][0]

    return None


def resolve_language_configs(
    language_codes: List[str],
    available_configs: List[str],
) -> Tuple[List[str], List[str], List[Tuple[str, str]]]:
    resolved: List[str] = []
    missing: List[str] = []
    mapping: List[Tuple[str, str]] = []
    used = set()
    for code in language_codes:
        cfg = _pick_config_for_language(code, available_configs)
        if cfg is None:
            missing.append(code)
            continue
        mapping.append((code, cfg))
        if cfg in used:
            continue
        resolved.append(cfg)
        used.add(cfg)
    return resolved, missing, mapping


def resolve_configs(dataset_name: str, config_arg: str, all_configs: bool, max_configs: int) -> List[str]:
    available_configs = get_dataset_config_names(dataset_name, trust_remote_code=True)
    available_set = set(available_configs)
    preset = get_dataset_preset(dataset_name)
    alias_raw = preset.get("config_aliases", {})
    alias_map = {k.lower(): v for k, v in alias_raw.items()}

    if all_configs:
        configs = available_configs
    else:
        raw_configs = [x.strip() for x in config_arg.split(",") if x.strip()]
        default_strategy = str(preset.get("default_strategy", "all_available"))

        # Fixed language datasets: map language codes -> actual config names.
        use_language_mapping = default_strategy == "fixed_languages"
        if use_language_mapping:
            if not raw_configs:
                raw_configs = list(preset.get("default_languages", []))
            mapped_langs = [_map_config_alias(x, alias_map) for x in raw_configs]
            configs, missing_langs, mapping = resolve_language_configs(mapped_langs, available_configs)
            if mapping:
                mapping_str = ", ".join([f"{src}->{dst}" for src, dst in mapping])
                print(f"[info] {dataset_name}: language->config mapping: {mapping_str}")
            if missing_langs:
                print(
                    f"[warn] {dataset_name}: skip missing language configs: {missing_langs}"
                )
        else:
            if not raw_configs:
                if default_strategy == "all_available":
                    raw_configs = available_configs.copy()
                else:
                    raw_configs = list(preset.get("default_configs", []))

            mapped = [_map_config_alias(x, alias_map) for x in raw_configs]
            seen = set()
            configs = []
            for cfg in mapped:
                if cfg and cfg not in seen:
                    configs.append(cfg)
                    seen.add(cfg)

            missing = [cfg for cfg in configs if cfg not in available_set]
            if missing:
                raise ValueError(
                    "Unknown configs after alias mapping: "
                    f"{missing}. Available configs include: {sorted(list(available_set))[:20]} ..."
                )

    if not configs:
        raise ValueError("No configs resolved. Provide --configs or use --all-configs.")
    if max_configs > 0:
        configs = configs[:max_configs]
    return configs


def dataset_column_names(raw_ds: any) -> List[str]:
    column_names = getattr(raw_ds, "column_names", None)
    if column_names:
        return list(column_names)
    features = getattr(raw_ds, "features", None)
    if features:
        try:
            return list(features.keys())
        except Exception:
            pass
    return []


def resolve_text_column(dataset_name: str, column_names: List[str]) -> Optional[str]:
    preset = get_dataset_preset(dataset_name)
    preferred = list(preset.get("preferred_text_columns", []))
    candidates = preferred + FALLBACK_TEXT_COLUMNS
    seen = set()
    deduped = []
    for c in candidates:
        if c not in seen:
            deduped.append(c)
            seen.add(c)
    for c in deduped:
        if c in column_names:
            return c
    return None


def resolve_im_start_id(tokenizer: any) -> int:
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    if im_start_id is None or im_start_id == tokenizer.unk_token_id:
        if tokenizer.bos_token_id is not None:
            return int(tokenizer.bos_token_id)
        if tokenizer.eos_token_id is not None:
            return int(tokenizer.eos_token_id)
        return 0
    return int(im_start_id)


def ensure_front_prompt_token(
    asr_model: any,
    tokenizer: any,
    token_name: str = "<|latent|>",
) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token_name)
    if token_id is None or token_id == tokenizer.unk_token_id:
        added = tokenizer.add_special_tokens({"additional_special_tokens": [token_name]})
        if added > 0:
            asr_model.thinker.resize_token_embeddings(len(tokenizer))
        token_id = tokenizer.convert_tokens_to_ids(token_name)
    if token_id is None or token_id < 0:
        raise RuntimeError(f"Failed to resolve front prompt token id for {token_name!r}.")
    return int(token_id)


def extract_soft_prompt_tensor(ckpt: Dict[str, any]) -> Optional[torch.Tensor]:
    tensor = ckpt.get("soft_prompt_embed")
    if torch.is_tensor(tensor):
        return tensor
    state = ckpt.get("model_state_dict")
    if isinstance(state, dict):
        tensor = state.get("soft_prompt_embed")
        if torch.is_tensor(tensor):
            return tensor
    return None


def load_checkpoint(path: str) -> Dict[str, any]:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        return ckpt
    return {"model_state_dict": ckpt}


def is_path_file_or_dir(path: str) -> bool:
    return os.path.isfile(path) or os.path.isdir(path)


def build_plain_bundle(
    model_id: str,
    device: str,
    dtype: torch.dtype,
) -> Tuple[LatentQwenASR, any]:
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=device if device == "cuda" else None,
    )
    asr_model = asr_wrapper.model
    processor = asr_wrapper.processor

    start_id = resolve_im_start_id(processor.tokenizer)
    model = LatentQwenASR(
        asr_model=asr_model,
        processor=processor,
        n_latent=0,
        nt_token_id=-1,
        lang_token_id=start_id,
        transcribe_token_id=start_id,
        freeze_base=False,
        use_latent=False,
    ).to(device)
    return model, processor


def maybe_attach_peft_adapter(
    model: LatentQwenASR,
    adapter_path: str,
    mode_name: str,
    expected_lora_rank: Optional[int] = None,
) -> None:
    try:
        from peft import PeftConfig, PeftModel
    except Exception as e:
        raise RuntimeError(
            f"[{mode_name}] Loading PEFT adapter requires `peft` package: {e}"
        ) from e

    peft_cfg = PeftConfig.from_pretrained(adapter_path)
    peft_type = str(getattr(peft_cfg, "peft_type", "unknown"))
    print(f"[{mode_name}] adapter detected: peft_type={peft_type} path={adapter_path}")

    if expected_lora_rank is not None:
        rank = getattr(peft_cfg, "r", None)
        if rank is None:
            rank_pattern = getattr(peft_cfg, "rank_pattern", None)
            if isinstance(rank_pattern, dict) and rank_pattern:
                unique_ranks = sorted({int(v) for v in rank_pattern.values()})
                if len(unique_ranks) == 1:
                    rank = unique_ranks[0]
        if rank is None:
            print(
                f"[warn][{mode_name}] unable to verify LoRA rank from adapter config."
            )
        elif int(rank) != int(expected_lora_rank):
            print(
                f"[warn][{mode_name}] expected LoRA rank={expected_lora_rank}, got rank={rank}."
            )

    adapted = PeftModel.from_pretrained(
        model.thinker,
        adapter_path,
        is_trainable=False,
    )
    model.base_model.thinker = adapted
    model.thinker = model.base_model.thinker


def sanitize_generation_config(model: LatentQwenASR) -> None:
    gen_cfg = getattr(model.thinker, "generation_config", None)
    if gen_cfg is None:
        return
    # Keep decoding deterministic and avoid warnings from stale sampling flags.
    if hasattr(gen_cfg, "do_sample"):
        gen_cfg.do_sample = False
    if hasattr(gen_cfg, "temperature"):
        gen_cfg.temperature = 1.0
    if hasattr(gen_cfg, "top_p"):
        gen_cfg.top_p = 1.0
    if hasattr(gen_cfg, "typical_p"):
        gen_cfg.typical_p = 1.0


def build_base_model_bundle(
    model_id: str,
    device: str,
    dtype: torch.dtype,
) -> ModelBundle:
    print(f"\n[Load base_model] model={model_id}")
    ckpt: Dict[str, any] = {}
    model, processor = build_plain_bundle(
        model_id=model_id,
        device=device,
        dtype=dtype,
    )

    sanitize_generation_config(model)
    model.eval()
    return ModelBundle(model=model, processor=processor, checkpoint=ckpt)


def build_baseline_ft_bundle(
    model_id: str,
    checkpoint_path: str,
    device: str,
    dtype: torch.dtype,
) -> ModelBundle:
    print(f"\n[Load baseline_ft] model={model_id} ckpt={checkpoint_path}")
    ckpt = load_checkpoint(checkpoint_path)
    model, processor = build_plain_bundle(
        model_id=model_id,
        device=device,
        dtype=dtype,
    )

    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[baseline_ft] missing keys: {len(missing)}")
    if unexpected:
        print(f"[baseline_ft] unexpected keys: {len(unexpected)}")

    sanitize_generation_config(model)
    model.eval()
    return ModelBundle(model=model, processor=processor, checkpoint=ckpt)


def build_prompt_tuning_bundle(
    model_id: str,
    checkpoint_path: str,
    device: str,
    dtype: torch.dtype,
) -> ModelBundle:
    print(f"\n[Load prompt_tuning] model={model_id} ckpt={checkpoint_path}")
    if os.path.isdir(checkpoint_path):
        # Backward compatibility: old prompt-tuning PEFT adapters.
        model, processor = build_plain_bundle(
            model_id=model_id,
            device=device,
            dtype=dtype,
        )
        maybe_attach_peft_adapter(
            model=model,
            adapter_path=checkpoint_path,
            mode_name="prompt_tuning",
        )
        ckpt = {
            "adapter_path": checkpoint_path,
            "checkpoint_type": "peft_adapter",
        }
    else:
        ckpt = load_checkpoint(checkpoint_path)
        soft_prompt = extract_soft_prompt_tensor(ckpt)

        if soft_prompt is not None:
            asr_wrapper = Qwen3ASRModel.from_pretrained(
                model_id,
                dtype=dtype,
                device_map=device if device == "cuda" else None,
            )
            asr_model = asr_wrapper.model
            processor = asr_wrapper.processor

            nt_id = ensure_front_prompt_token(asr_model, processor.tokenizer)
            start_id = resolve_im_start_id(processor.tokenizer)
            n_latent = int(ckpt.get("n_latent", int(soft_prompt.size(0))))
            if n_latent <= 0:
                n_latent = int(soft_prompt.size(0))
            if n_latent <= 0:
                raise ValueError("prompt_tuning checkpoint has empty soft prompt tensor.")

            model = LatentQwenASR(
                asr_model=asr_model,
                processor=processor,
                n_latent=n_latent,
                nt_token_id=nt_id,
                lang_token_id=start_id,
                transcribe_token_id=start_id,
                freeze_base=False,
                use_latent=False,
                use_soft_prompt=True,
                soft_prompt_init_mode="random",
            ).to(device)

            prompt = soft_prompt.detach()
            if prompt.dim() != 2:
                raise ValueError(
                    f"soft_prompt_embed must be 2D [n_latent, hidden], got shape={tuple(prompt.shape)}"
                )
            if prompt.size(0) < n_latent:
                reps = (n_latent + prompt.size(0) - 1) // prompt.size(0)
                prompt = prompt.repeat((reps, 1))
            prompt = prompt[:n_latent]
            model.soft_prompt_embed.data.copy_(
                prompt.to(
                    device=model.soft_prompt_embed.device,
                    dtype=model.soft_prompt_embed.dtype,
                )
            )
            print(f"[prompt_tuning] loaded front soft prompt: n_latent={n_latent}")
        else:
            # Fallback for legacy `.pth` checkpoints that saved full wrapper state.
            model, processor = build_plain_bundle(
                model_id=model_id,
                device=device,
                dtype=dtype,
            )
            state = ckpt.get("model_state_dict", ckpt)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                print(f"[prompt_tuning] missing keys: {len(missing)}")
            if unexpected:
                print(f"[prompt_tuning] unexpected keys: {len(unexpected)}")

    sanitize_generation_config(model)
    model.eval()
    return ModelBundle(model=model, processor=processor, checkpoint=ckpt)


def build_lora_r16_bundle(
    model_id: str,
    checkpoint_path: str,
    device: str,
    dtype: torch.dtype,
) -> ModelBundle:
    print(f"\n[Load lora_r16] model={model_id} ckpt={checkpoint_path}")
    model, processor = build_plain_bundle(
        model_id=model_id,
        device=device,
        dtype=dtype,
    )

    if os.path.isdir(checkpoint_path):
        maybe_attach_peft_adapter(
            model=model,
            adapter_path=checkpoint_path,
            mode_name="lora_r16",
            expected_lora_rank=None,
        )
        ckpt = {
            "adapter_path": checkpoint_path,
            "checkpoint_type": "peft_adapter",
        }
    else:
        ckpt = load_checkpoint(checkpoint_path)
        state = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[lora_r16] missing keys: {len(missing)}")
        if unexpected:
            print(f"[lora_r16] unexpected keys: {len(unexpected)}")

    sanitize_generation_config(model)
    model.eval()
    return ModelBundle(model=model, processor=processor, checkpoint=ckpt)


def build_latent_bundle(
    model_id: str,
    checkpoint_path: str,
    n_latent_override: int,
    device: str,
    dtype: torch.dtype,
) -> ModelBundle:
    print(f"\n[Load latent_reasoning] model={model_id} ckpt={checkpoint_path}")
    ckpt = load_checkpoint(checkpoint_path)

    n_latent = int(ckpt.get("n_latent", 4))
    if n_latent_override >= 0:
        n_latent = int(n_latent_override)
    if n_latent <= 0:
        raise ValueError(f"Invalid latent token count: {n_latent}")

    asr_wrapper = Qwen3ASRModel.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=device if device == "cuda" else None,
    )
    asr_model = asr_wrapper.model
    processor = asr_wrapper.processor

    token_id = ensure_front_prompt_token(asr_model, processor.tokenizer)

    start_id = resolve_im_start_id(processor.tokenizer)
    model = LatentQwenASR(
        asr_model=asr_model,
        processor=processor,
        n_latent=n_latent,
        nt_token_id=int(token_id),
        lang_token_id=start_id,
        transcribe_token_id=start_id,
        freeze_base=False,
        use_latent=True,
        latent_use_bounded_delta=bool(ckpt.get("latent_use_bounded_delta", True)),
        latent_use_injection_gate=bool(ckpt.get("latent_use_injection_gate", True)),
        latent_use_embedding_anchor=bool(ckpt.get("latent_use_embedding_anchor", True)),
    ).to(device)

    # Prefer latent-adapter payload format from train.py.
    if "init_proj" in ckpt:
        model.init_proj.load_state_dict(ckpt["init_proj"], strict=True)
        model.delta_proj.load_state_dict(ckpt["delta_proj"], strict=True)
        model.step_proj.load_state_dict(ckpt["step_proj"], strict=True)
        if "step_embed" in ckpt:
            model.step_embed.data.copy_(
                ckpt["step_embed"].to(
                    device=model.step_embed.device,
                    dtype=model.step_embed.dtype,
                )
            )
        if "log_scale" in ckpt:
            model.log_scale.data.copy_(
                ckpt["log_scale"].to(
                    device=model.log_scale.device,
                    dtype=model.log_scale.dtype,
                )
            )
        if "value_head" in ckpt:
            model.value_head.load_state_dict(ckpt["value_head"], strict=True)
        if "injection_gate" in ckpt:
            model.injection_gate.load_state_dict(ckpt["injection_gate"], strict=True)
    else:
        # Fallback: checkpoint may contain full wrapper state_dict.
        state = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[latent_reasoning] missing keys: {len(missing)}")
        if unexpected:
            print(f"[latent_reasoning] unexpected keys: {len(unexpected)}")

    sanitize_generation_config(model)
    model.eval()
    return ModelBundle(model=model, processor=processor, checkpoint=ckpt)


_global_normalizer = None
_global_normalizer_kind = "english"


def configure_text_normalizer(kind: str) -> None:
    global _global_normalizer, _global_normalizer_kind
    kind = (kind or "english").strip().lower()
    if kind not in {"english", "basic", "simple"}:
        raise ValueError(f"Unknown text normalizer: {kind}")
    if kind != _global_normalizer_kind:
        _global_normalizer = None
    _global_normalizer_kind = kind


def simple_text_normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.lower()
    text = "".join(
        " " if unicodedata.category(ch).startswith("P") else ch
        for ch in text
    )
    return re.sub(r"\s+", " ", text).strip()


class _SimpleTextNormalizer:
    def __call__(self, text: str) -> str:
        return simple_text_normalize(text)


def build_text_normalizer(kind: str):
    if kind == "simple":
        return _SimpleTextNormalizer()

    if kind == "basic":
        try:
            from transformers.models.whisper.english_normalizer import BasicTextNormalizer
            return BasicTextNormalizer()
        except Exception:
            return _SimpleTextNormalizer()

    try:
        # We use the standard Whisper English normalizer for English-heavy legacy runs.
        from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
        import json, urllib.request, os

        mapping_path = os.path.join(os.path.dirname(__file__), "english_normalizer.json")
        if not os.path.exists(mapping_path):
            url = "https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/whisper/english_normalizer.json"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode("utf-8"))
            with open(mapping_path, "w") as f:
                json.dump(data, f)
        else:
            with open(mapping_path, "r") as f:
                data = json.load(f)
        return EnglishTextNormalizer(data)
    except Exception:
        try:
            from transformers.models.whisper.english_normalizer import BasicTextNormalizer
            return BasicTextNormalizer()
        except Exception:
            return _SimpleTextNormalizer()

def normalize_text(text: str) -> str:
    global _global_normalizer
    if _global_normalizer is None:
        _global_normalizer = build_text_normalizer(_global_normalizer_kind)
    normalized = _global_normalizer(text)
    
    if isinstance(normalized, str):
        if _global_normalizer_kind == "english":
            # LibriSpeech specific normalizations: expand common contractions
            # that Whisper leaves as "m" or "re".
            contractions = {
                r"\bi m\b": "i am",
                r"\byou re\b": "you are",
                r"\bwe re\b": "we are",
                r"\bthey re\b": "they are",
                r"\bhe s\b": "he is",
                r"\bshe s\b": "she is",
                r"\bit s\b": "it is",
                r"\bthat s\b": "that is",
                r"\bwho s\b": "who is",
                r"\bwhat s\b": "what is",
                r"\bthere s\b": "there is",
                r"\bhere s\b": "here is",
                r"\bwhere s\b": "where is",
                r"\blet s\b": "let us",
                r"\bhow s\b": "how is",
                # Standard possessive stripping (boy's -> boy s -> boys)
                r"\b s\b": "s",
            }
            for pat, repl in contractions.items():
                normalized = re.sub(pat, repl, normalized)
            
        return re.sub(r"\s+", " ", normalized).strip()
    return ""

def clean_prediction(text: str) -> str:
    text = text or ""
    text = re.sub(r"language\s+\w+<asr_text>", "", text, flags=re.IGNORECASE)
    if "<asr_text>" in text:
        text = text.split("<asr_text>", 1)[-1]
    return text.strip()


@torch.no_grad()
def add_noise_at_snr(waveform: np.ndarray, snr_db: float) -> np.ndarray:
    """Add Gaussian noise to a waveform at a target Signal-to-Noise Ratio (dB).
    
    SNR_dB = 10 * log10(P_signal / P_noise)
    => P_noise = P_signal / 10^(SNR_dB / 10)
    => std_noise = sqrt(P_noise)
    """
    signal_power = np.mean(waveform ** 2)
    if signal_power < 1e-10:  # silence guard
        return waveform
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = np.random.normal(0, np.sqrt(noise_power), waveform.shape)
    return (waveform + noise).astype(waveform.dtype)


def evaluate_config(
    model: LatentQwenASR,
    processor: any,
    dataset_name: str,
    config_name: str,
    split: str,
    use_baseline: bool,
    mode_name: str,
    max_samples: int,
    max_new_tokens: int,
    print_samples: int,
    language_hint: Optional[str],
    prompt_template: str,
    disable_language_hint: bool,
    num_beams: int = 1,
    dynamic_halt_threshold: float = 0.0,
    snr_db: Optional[float] = None,
    print_gate_skips: bool = False,
    streaming: bool = False,
) -> Dict[str, any]:
    raw_ds = load_dataset(
        dataset_name,
        config_name,
        split=split,
        trust_remote_code=True,
        streaming=streaming,
    )
    known_total: Optional[int]
    try:
        known_total = len(raw_ds)
    except TypeError:
        known_total = None
    total = known_total if known_total is not None else 0
    if max_samples > 0:
        total = min(total, max_samples) if known_total is not None else max_samples
    column_names = dataset_column_names(raw_ds)
    text_column = resolve_text_column(dataset_name, column_names)
    audio_column = "audio" if "audio" in column_names else None

    if audio_column is None and not streaming:
        raise ValueError(
            f"Dataset {dataset_name}/{config_name} does not have an 'audio' column. "
            f"Columns: {column_names}"
        )
    target_sampling_rate = int(getattr(processor.feature_extractor, "sampling_rate", 0) or 0)
    if target_sampling_rate > 0 and audio_column is not None:
        raw_ds = raw_ds.cast_column(audio_column, Audio(sampling_rate=target_sampling_rate))

    preds: List[str] = []
    refs: List[str] = []
    shown = 0
    skipped = 0
    reasoned = 0

    # Value Head aggregate statistics collectors
    all_v_preds: List[List[float]] = []   # per-sample list of v_pred values
    all_deq_iters: List[int] = []          # per-sample thinking steps taken
    all_gates: List[float] = []            # per-sample gate values
    all_scales: List[List[float]] = []     # per-sample scale values
    n_skipped_by_gate = 0                  # count of N=0 (gate skip)

    effective_language_hint = None if disable_language_hint else language_hint
    if effective_language_hint:
        try:
            prompt_text = prompt_template.format(language=effective_language_hint)
        except Exception:
            prompt_text = f"Transcribe the {effective_language_hint} audio into text."
    else:
        prompt_text = "Transcribe the audio into text."

    sample_iter = iter(raw_ds)
    if max_samples > 0:
        sample_iter = itertools.islice(sample_iter, max_samples)
    pbar = tqdm(sample_iter, total=total if total > 0 else None, desc=f"{config_name}:{mode_name}", leave=False)
    seen = 0
    for i, sample in enumerate(pbar):
        seen += 1

        if (text_column is None or audio_column is None) and isinstance(sample, dict):
            sample_columns = list(sample.keys())
            if text_column is None:
                text_column = resolve_text_column(dataset_name, sample_columns)
            if audio_column is None and "audio" in sample_columns:
                audio_column = "audio"

        if audio_column is None:
            skipped += 1
            continue

        if text_column is not None:
            ref = sample.get(text_column, "")
        else:
            ref = ""
            for cand in FALLBACK_TEXT_COLUMNS:
                v = sample.get(cand, "")
                if isinstance(v, str) and v.strip():
                    ref = v
                    break
        if not isinstance(ref, str) or not ref.strip():
            skipped += 1
            continue

        audio = sample.get(audio_column)
        if not isinstance(audio, dict) or "array" not in audio or "sampling_rate" not in audio:
            skipped += 1
            continue

        audio_array = np.array(audio["array"], dtype=np.float64)
        if snr_db is not None:
            audio_array = add_noise_at_snr(audio_array, snr_db)

        target_dtype = model.thinker.dtype if hasattr(model.thinker, "dtype") else torch.float32
        feat_out = processor.feature_extractor(
            audio_array,
            sampling_rate=audio["sampling_rate"],
            return_attention_mask=True,
        )
        feats = torch.tensor(feat_out.input_features[0], dtype=target_dtype, device=model.base_model.device).unsqueeze(0)
        n_frames = feats.size(-1)
        
        if getattr(feat_out, "attention_mask", None) is not None:
            # Coerce mask to match the Mel frame length (same logic as data.py)
            raw_mask = feat_out.attention_mask[0]
            if not isinstance(raw_mask, (list, torch.Tensor)):
                raw_mask = list(raw_mask)
            if isinstance(raw_mask, torch.Tensor):
                raw_mask = raw_mask.long()
            else:
                raw_mask = torch.tensor(raw_mask, dtype=torch.long)
            # Pad or truncate to n_frames
            if raw_mask.size(-1) < n_frames:
                raw_mask = torch.cat([raw_mask, torch.zeros(n_frames - raw_mask.size(-1), dtype=torch.long)])
            else:
                raw_mask = raw_mask[:n_frames]
            feature_attention_mask = raw_mask.to(device=model.base_model.device).unsqueeze(0)
        else:
            feature_attention_mask = torch.ones((1, n_frames), dtype=torch.long, device=model.base_model.device)

        # Guard: if effective feature length is too short for audio tower, skip sample
        effective_len = feature_attention_mask.sum().item()
        if effective_len < 10:
            skipped += 1
            continue

        gen_kwargs = {
            "feature_attention_mask": feature_attention_mask,
            "max_new_tokens": max_new_tokens,
            "use_baseline": use_baseline,
            "return_thoughts": False,
            "return_stats": True,
            "do_sample": False,
            "eos_token_id": [151645, 151643],
            "num_beams": num_beams,
            "language_hint": effective_language_hint,
            "prompt_text": prompt_text,
            "dynamic_halt_threshold": dynamic_halt_threshold,
        }

        gen_output = model.generate(feats, **gen_kwargs)
        if isinstance(gen_output, tuple):
            gen_ids = gen_output[0]
            stats = gen_output[1] if isinstance(gen_output[1], dict) else (gen_output[2] if len(gen_output)>2 and isinstance(gen_output[2], dict) else {})
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

        pred_norm = normalize_text(pred)
        ref_norm = normalize_text(ref)
        if not ref_norm:
            skipped += 1
            continue

        preds.append(pred_norm)
        refs.append(ref_norm)

        # Always print if it was skipped (N=0 from Gate) OR if under the print limit
        is_skipped_lr = stats.get("skipped", False)
        
        if mode_name == "latent_reasoning":
            if not is_skipped_lr:
                reasoned += 1

        if shown < print_samples or (print_gate_skips and is_skipped_lr):
            if not is_skipped_lr:
                shown += 1
            print(f"\n[{config_name} | {mode_name} | sample {shown if not is_skipped_lr else 'SKIPPED'}]")
            print(f"  Ref : {ref.strip()}")
            print(f"  Pred: {pred}")
            if "v_preds" in stats and stats["v_preds"] is not None:
                v_vals = stats["v_preds"][0].tolist()
                v_str = " | ".join([f"{v:.4f}" for v in v_vals])
                iters_t = stats.get("deq_iters", None)
                deq_iters = int(iters_t.item()) if iters_t is not None and hasattr(iters_t, "item") else -1
                if is_skipped_lr:
                    print(f"  v_preds: [{v_str}] -> SKIPPED (N=0)")
                else:
                    print(f"  v_preds: [{v_str}] -> {deq_iters} step(s)")

        # --- Collect Value Head stats for aggregate summary ---
        if mode_name == "latent_reasoning":
            iters_t = stats.get("deq_iters", None)
            deq_n = int(iters_t.item()) if iters_t is not None and hasattr(iters_t, "item") else 0
            all_deq_iters.append(deq_n)
            if is_skipped_lr:
                n_skipped_by_gate += 1
            if "v_preds" in stats and stats["v_preds"] is not None:
                all_v_preds.append(stats["v_preds"][0].tolist())
            if "gate" in stats and stats["gate"] is not None:
                g = stats["gate"]
                all_gates.append(float(g[0].item()) if hasattr(g, "item") or g.dim() > 0 else float(g))
            if "scales" in stats and stats["scales"] is not None:
                s = stats["scales"]
                all_scales.append(s.tolist() if hasattr(s, "tolist") else list(s))

    # --- Print Value Head aggregate summary ---
    if mode_name == "latent_reasoning" and all_deq_iters:
        import collections
        n_total_eval = len(all_deq_iters)
        n_dist = collections.Counter(all_deq_iters)
        print(f"\n{'='*60}")
        print(f"[Value-Head Summary] {config_name} | {mode_name}")
        print(f"{'='*60}")
        print(f"  Total samples evaluated : {n_total_eval}")
        print(f"  Skipped by gate (N=0)   : {n_skipped_by_gate} ({100*n_skipped_by_gate/max(1,n_total_eval):.1f}%)")
        print(f"  Reasoned (N>0)          : {n_total_eval - n_skipped_by_gate} ({100*(n_total_eval - n_skipped_by_gate)/max(1,n_total_eval):.1f}%)")
        print(f"  --- N-step distribution ---")
        for k in sorted(n_dist.keys()):
            pct = 100 * n_dist[k] / n_total_eval
            bar = '█' * int(pct / 2)
            print(f"    N={k}: {n_dist[k]:5d} ({pct:5.1f}%) {bar}")
        if all_v_preds:
            flat_v = [v for vs in all_v_preds for v in vs]
            if flat_v:
                import numpy as _np
                v_arr = _np.array(flat_v)
                print(f"  --- v_pred statistics (all steps, all samples) ---")
                print(f"    count : {len(v_arr)}")
                print(f"    mean  : {v_arr.mean():.4f}")
                print(f"    std   : {v_arr.std():.4f}")
                print(f"    min   : {v_arr.min():.4f}")
                print(f"    max   : {v_arr.max():.4f}")
                print(f"    median: {float(_np.median(v_arr)):.4f}")
                # Histogram buckets
                edges = [-1.0, -0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8, 1.0]
                print(f"  --- v_pred histogram ---")
                for j in range(len(edges) - 1):
                    lo, hi = edges[j], edges[j + 1]
                    cnt = int(((v_arr >= lo) & (v_arr < hi)).sum())
                    pct = 100 * cnt / len(v_arr)
                    bar = '▓' * int(pct / 2)
                    print(f"    [{lo:+.1f}, {hi:+.1f}): {cnt:5d} ({pct:5.1f}%) {bar}")
                # Also count == 1.0
                cnt_one = int((v_arr >= 0.8).sum())
                # First-step v_pred (initial gate signal)
                first_v = [vs[0] for vs in all_v_preds if vs]
                if first_v:
                    fv = _np.array(first_v)
                    print(f"  --- v_pred[0] (initial gate signal) ---")
                    print(f"    mean  : {fv.mean():.4f}")
                    print(f"    std   : {fv.std():.4f}")
                    print(f"    < 0   : {int((fv < 0).sum()):5d} ({100*(fv < 0).mean():.1f}%)")
                    print(f"    >= 0  : {int((fv >= 0).sum()):5d} ({100*(fv >= 0).mean():.1f}%)")
        if all_gates:
            import numpy as _np
            g_arr = _np.array(all_gates)
            print(f"  --- Gate confidence (mapped to [0,1]) ---")
            print(f"    mean  : {g_arr.mean():.4f}")
            print(f"    std   : {g_arr.std():.4f}")
            print(f"    min   : {g_arr.min():.4f}")
            print(f"    max   : {g_arr.max():.4f}")
        if all_scales:
            import numpy as _np
            s_arr = _np.array(all_scales)
            if s_arr.ndim == 2:
                print(f"  --- Delta scales per step ---")
                for si in range(s_arr.shape[1]):
                    col = s_arr[:, si]
                    print(f"    step {si}: mean={col.mean():.4f} std={col.std():.4f}")
            elif s_arr.ndim == 1:
                print(f"  --- Delta scales ---")
                print(f"    mean={s_arr.mean():.4f} std={s_arr.std():.4f}")
        print(f"{'='*60}\n")

    if not refs:
        return {
            "config": config_name,
            "mode": mode_name,
            "language_hint": effective_language_hint,
            "text_column": text_column,
            "samples_total": total if total > 0 else seen,
            "samples_used": 0,
            "samples_skipped": skipped,
            "samples_reasoned": reasoned,
            "wer": 1.0,
            "cer": 1.0,
        }

    return {
        "config": config_name,
        "mode": mode_name,
        "language_hint": effective_language_hint,
        "text_column": text_column,
        "samples_total": total if total > 0 else seen,
        "samples_used": len(refs),
        "samples_skipped": skipped,
        "samples_reasoned": reasoned,
        "wer": float(wer(refs, preds)),
        "cer": float(cer(refs, preds)),
    }


def weighted_avg(rows: List[Dict[str, any]], metric_key: str) -> Optional[float]:
    numer = 0.0
    denom = 0
    for row in rows:
        n = int(row.get("samples_used", 0))
        if n <= 0:
            continue
        m = row.get(metric_key)
        if m is None:
            continue
        numer += float(m) * n
        denom += n
    if denom == 0:
        return None
    return numer / float(denom)


def print_summary_table(rows: List[Dict[str, any]]) -> None:
    header = (
        f"{'Config':<12} {'N':>7} "
        f"{'Raw-WER':>10} {'FT-WER':>10} {'PT-WER':>10} {'LoRA-WER':>10} {'Lat-WER':>10} "
        f"{'Raw-CER':>10} {'FT-CER':>10} {'PT-CER':>10} {'LoRA-CER':>10} {'Lat-CER':>10}"
    )
    print("\n" + header)
    print("-" * len(header))

    for row in rows:
        cfg = row["config"]
        n = row.get("samples_used", 0)
        raw_wer = row.get("base_model_wer")
        ft_wer = row.get("baseline_ft_wer")
        pt_wer = row.get("prompt_tuning_wer")
        lora_wer = row.get("lora_r16_wer")
        lat_wer = row.get("latent_reasoning_wer")
        raw_cer = row.get("base_model_cer")
        ft_cer = row.get("baseline_ft_cer")
        pt_cer = row.get("prompt_tuning_cer")
        lora_cer = row.get("lora_r16_cer")
        lat_cer = row.get("latent_reasoning_cer")

        print(
            f"{cfg:<12} {n:>7d} "
            f"{(f'{raw_wer:.4f}' if raw_wer is not None else '-'):>10} "
            f"{(f'{ft_wer:.4f}' if ft_wer is not None else '-'):>10} "
            f"{(f'{pt_wer:.4f}' if pt_wer is not None else '-'):>10} "
            f"{(f'{lora_wer:.4f}' if lora_wer is not None else '-'):>10} "
            f"{(f'{lat_wer:.4f}' if lat_wer is not None else '-'):>10} "
            f"{(f'{raw_cer:.4f}' if raw_cer is not None else '-'):>10} "
            f"{(f'{ft_cer:.4f}' if ft_cer is not None else '-'):>10} "
            f"{(f'{pt_cer:.4f}' if pt_cer is not None else '-'):>10} "
            f"{(f'{lora_cer:.4f}' if lora_cer is not None else '-'):>10} "
            f"{(f'{lat_cer:.4f}' if lat_cer is not None else '-'):>10}"
        )


def release_model(bundle: Optional[ModelBundle]) -> None:
    if bundle is None:
        return
    del bundle.model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    configure_text_normalizer(args.text_normalizer)

    args.baseline_ckpt = (args.baseline_ckpt or "").strip()
    args.prompt_tuning_ckpt = (args.prompt_tuning_ckpt or "").strip()
    args.lora_r16_ckpt = (args.lora_r16_ckpt or "").strip()
    args.latent_ckpt = (args.latent_ckpt or "").strip()
    run_base_model = not bool(args.skip_base_model)
    run_baseline_ft = bool(args.baseline_ckpt) and not bool(args.skip_baseline_ft)
    run_prompt_tuning = bool(args.prompt_tuning_ckpt) and not bool(args.skip_prompt_tuning)
    run_lora_r16 = bool(args.lora_r16_ckpt) and not bool(args.skip_lora_r16)
    run_latent_reasoning = bool(args.latent_ckpt)
    if not (
        run_base_model
        or run_baseline_ft
        or run_prompt_tuning
        or run_lora_r16
        or run_latent_reasoning
    ):
        raise ValueError(
            "No evaluation mode enabled. Provide at least one of --baseline-ckpt, "
            "--prompt-tuning-ckpt, --lora-r16-ckpt, --latent-ckpt, "
            "or remove --skip-base-model."
        )
    if run_baseline_ft and not os.path.isfile(args.baseline_ckpt):
        raise FileNotFoundError(
            f"--baseline-ckpt expects a .pth file, got: {args.baseline_ckpt}"
        )
    if run_latent_reasoning and not os.path.isfile(args.latent_ckpt):
        raise FileNotFoundError(
            f"--latent-ckpt expects a .pth file, got: {args.latent_ckpt}"
        )
    if run_prompt_tuning and not is_path_file_or_dir(args.prompt_tuning_ckpt):
        raise FileNotFoundError(
            f"--prompt-tuning-ckpt path not found: {args.prompt_tuning_ckpt}"
        )
    if run_lora_r16 and not is_path_file_or_dir(args.lora_r16_ckpt):
        raise FileNotFoundError(
            f"--lora-r16-ckpt path not found: {args.lora_r16_ckpt}"
        )

    dataset_name = resolve_dataset_name(args.dataset_name)
    device = choose_device(args.device)
    dtype = choose_dtype(args.dtype, device)
    configs = resolve_configs(dataset_name, args.configs, args.all_configs, args.max_configs)

    print("=== Evaluation Setup ===")
    print(f"Device: {device}")
    print(f"Dtype : {dtype}")
    print(f"Model : {args.model_id}")
    print(f"Data  : {dataset_name} ({args.split})")
    print(f"Configs ({len(configs)}): {configs}")
    print(f"Max samples/config: {'all' if args.max_samples_per_config <= 0 else args.max_samples_per_config}")
    print(f"Streaming:            {'yes' if args.streaming else 'no'}")
    print(f"Text normalizer:      {args.text_normalizer}")
    print(f"Run base_model:       {'yes' if run_base_model else 'no'}")
    print(f"Run baseline_ft:      {'yes' if run_baseline_ft else 'no'}")
    print(f"Run prompt_tuning:    {'yes' if run_prompt_tuning else 'no'}")
    print(f"Run lora_r16:         {'yes' if run_lora_r16 else 'no'}")
    print(f"Run latent_reasoning: {'yes' if run_latent_reasoning else 'no'}")
    print(f"Language hint: {'off' if args.no_language_hint else 'on'}")
    if not args.no_language_hint:
        print(f"Prompt template: {args.prompt_template}")
    print(f"Num beams:            {args.num_beams}")
    print(f"Print gate skips:     {'yes' if args.print_gate_skips else 'no'}")

    base_model_rows: Dict[str, Dict[str, any]] = {}
    baseline_ft_rows: Dict[str, Dict[str, any]] = {}
    prompt_tuning_rows: Dict[str, Dict[str, any]] = {}
    lora_r16_rows: Dict[str, Dict[str, any]] = {}
    latent_reasoning_rows: Dict[str, Dict[str, any]] = {}

    base_model_bundle: Optional[ModelBundle] = None
    baseline_ft_bundle: Optional[ModelBundle] = None
    prompt_tuning_bundle: Optional[ModelBundle] = None
    lora_r16_bundle: Optional[ModelBundle] = None
    latent_reasoning_bundle: Optional[ModelBundle] = None

    if run_base_model:
        try:
            base_model_bundle = build_base_model_bundle(
                model_id=args.model_id,
                device=device,
                dtype=dtype,
            )
            for cfg in configs:
                language_hint = resolve_language_hint(dataset_name, cfg)
                out = evaluate_config(
                    model=base_model_bundle.model,
                    processor=base_model_bundle.processor,
                    dataset_name=dataset_name,
                    config_name=cfg,
                    split=args.split,
                    use_baseline=True,
                    mode_name="base_model",
                    max_samples=args.max_samples_per_config,
                    max_new_tokens=args.max_new_tokens,
                    print_samples=args.print_samples,
                    language_hint=language_hint,
                    prompt_template=args.prompt_template,
                    disable_language_hint=args.no_language_hint,
                    num_beams=args.num_beams,
                    dynamic_halt_threshold=args.dynamic_halt_threshold,
                    snr_db=args.snr_db[0] if args.snr_db else None,
                    print_gate_skips=args.print_gate_skips,
                    streaming=args.streaming,
                )
                base_model_rows[cfg] = out
                print(
                    f"[BaseModel][{cfg}] "
                    f"lang={out.get('language_hint') or 'auto'} "
                    f"text_col={out.get('text_column') or '-'} "
                    f"WER={out['wer']:.4f} CER={out['cer']:.4f} "
                    f"n={out['samples_used']} skip={out['samples_skipped']}"
                )
        finally:
            release_model(base_model_bundle)
    else:
        print("\n[Skip] base_model evaluation skipped (--skip-base-model).")

    if run_baseline_ft:
        try:
            baseline_ft_bundle = build_baseline_ft_bundle(
                model_id=args.model_id,
                checkpoint_path=args.baseline_ckpt,
                device=device,
                dtype=dtype,
            )
            for cfg in configs:
                language_hint = resolve_language_hint(dataset_name, cfg)
                out = evaluate_config(
                    model=baseline_ft_bundle.model,
                    processor=baseline_ft_bundle.processor,
                    dataset_name=dataset_name,
                    config_name=cfg,
                    split=args.split,
                    use_baseline=True,
                    mode_name="baseline_ft",
                    max_samples=args.max_samples_per_config,
                    max_new_tokens=args.max_new_tokens,
                    print_samples=args.print_samples,
                    language_hint=language_hint,
                    prompt_template=args.prompt_template,
                    disable_language_hint=args.no_language_hint,
                    num_beams=args.num_beams,
                    dynamic_halt_threshold=args.dynamic_halt_threshold,
                    snr_db=args.snr_db[0] if args.snr_db else None,
                    print_gate_skips=args.print_gate_skips,
                    streaming=args.streaming,
                )
                baseline_ft_rows[cfg] = out
                print(
                    f"[BaselineFT][{cfg}] "
                    f"lang={out.get('language_hint') or 'auto'} "
                    f"text_col={out.get('text_column') or '-'} "
                    f"WER={out['wer']:.4f} CER={out['cer']:.4f} "
                    f"n={out['samples_used']} skip={out['samples_skipped']}"
                )
        finally:
            release_model(baseline_ft_bundle)
    else:
        print("\n[Skip] baseline_ft evaluation skipped (no --baseline-ckpt).")

    if run_prompt_tuning:
        try:
            prompt_tuning_bundle = build_prompt_tuning_bundle(
                model_id=args.model_id,
                checkpoint_path=args.prompt_tuning_ckpt,
                device=device,
                dtype=dtype,
            )
            for cfg in configs:
                language_hint = resolve_language_hint(dataset_name, cfg)
                out = evaluate_config(
                    model=prompt_tuning_bundle.model,
                    processor=prompt_tuning_bundle.processor,
                    dataset_name=dataset_name,
                    config_name=cfg,
                    split=args.split,
                    use_baseline=False,
                    mode_name="prompt_tuning",
                    max_samples=args.max_samples_per_config,
                    max_new_tokens=args.max_new_tokens,
                    print_samples=args.print_samples,
                    language_hint=language_hint,
                    prompt_template=args.prompt_template,
                    disable_language_hint=args.no_language_hint,
                    num_beams=args.num_beams,
                    dynamic_halt_threshold=args.dynamic_halt_threshold,
                    snr_db=args.snr_db[0] if args.snr_db else None,
                    print_gate_skips=args.print_gate_skips,
                    streaming=args.streaming,
                )
                prompt_tuning_rows[cfg] = out
                print(
                    f"[PromptTuning][{cfg}] "
                    f"text_col={out.get('text_column') or '-'} "
                    f"WER={out['wer']:.4f} CER={out['cer']:.4f} "
                    f"n={out['samples_used']} skip={out['samples_skipped']}"
                )
        finally:
            release_model(prompt_tuning_bundle)
    else:
        print("\n[Skip] prompt_tuning evaluation skipped (no --prompt-tuning-ckpt).")

    if run_lora_r16:
        try:
            lora_r16_bundle = build_lora_r16_bundle(
                model_id=args.model_id,
                checkpoint_path=args.lora_r16_ckpt,
                device=device,
                dtype=dtype,
            )
            for cfg in configs:
                language_hint = resolve_language_hint(dataset_name, cfg)
                out = evaluate_config(
                    model=lora_r16_bundle.model,
                    processor=lora_r16_bundle.processor,
                    dataset_name=dataset_name,
                    config_name=cfg,
                    split=args.split,
                    use_baseline=True,
                    mode_name="lora_r16",
                    max_samples=args.max_samples_per_config,
                    max_new_tokens=args.max_new_tokens,
                    print_samples=args.print_samples,
                    language_hint=language_hint,
                    prompt_template=args.prompt_template,
                    disable_language_hint=args.no_language_hint,
                    num_beams=args.num_beams,
                    dynamic_halt_threshold=args.dynamic_halt_threshold,
                    snr_db=args.snr_db[0] if args.snr_db else None,
                    print_gate_skips=args.print_gate_skips,
                    streaming=args.streaming,
                )
                lora_r16_rows[cfg] = out
                print(
                    f"[LoRA-R16][{cfg}] "
                    f"text_col={out.get('text_column') or '-'} "
                    f"WER={out['wer']:.4f} CER={out['cer']:.4f} "
                    f"n={out['samples_used']} skip={out['samples_skipped']}"
                )
        finally:
            release_model(lora_r16_bundle)
    else:
        print("\n[Skip] lora_r16 evaluation skipped (no --lora-r16-ckpt).")

    if run_latent_reasoning:
        try:
            latent_reasoning_bundle = build_latent_bundle(
                model_id=args.model_id,
                checkpoint_path=args.latent_ckpt,
                n_latent_override=args.n_latent,
                device=device,
                dtype=dtype,
            )
            for cfg in configs:
                language_hint = resolve_language_hint(dataset_name, cfg)
                out = evaluate_config(
                    model=latent_reasoning_bundle.model,
                    processor=latent_reasoning_bundle.processor,
                    dataset_name=dataset_name,
                    config_name=cfg,
                    split=args.split,
                    use_baseline=False,
                    mode_name="latent_reasoning",
                    max_samples=args.max_samples_per_config,
                    max_new_tokens=args.max_new_tokens,
                    print_samples=args.print_samples,
                    language_hint=language_hint,
                    prompt_template=args.prompt_template,
                    disable_language_hint=args.no_language_hint,
                    num_beams=args.num_beams,
                    dynamic_halt_threshold=args.dynamic_halt_threshold,
                    snr_db=args.snr_db[0] if args.snr_db else None,
                    print_gate_skips=args.print_gate_skips,
                    streaming=args.streaming,
                )
                latent_reasoning_rows[cfg] = out
                n_used = out['samples_used']
                n_reasoned = out.get('samples_reasoned', 0)
                reason_pct = (n_reasoned / n_used * 100) if n_used > 0 else 0.0
                print(
                    f"[LatentReasoning][{cfg}] "
                    f"lang={out.get('language_hint') or 'auto'} "
                    f"text_col={out.get('text_column') or '-'} "
                    f"WER={out['wer']:.4f} CER={out['cer']:.4f} "
                    f"n={n_used} skip={out['samples_skipped']} "
                    f"reasoned={n_reasoned}/{n_used} ({reason_pct:.1f}%)"
                )
        finally:
            release_model(latent_reasoning_bundle)
    else:
        print("\n[Skip] latent_reasoning evaluation skipped (no --latent-ckpt).")

    merged_rows: List[Dict[str, any]] = []
    for cfg in configs:
        base = base_model_rows.get(cfg, {})
        ft = baseline_ft_rows.get(cfg, {})
        prompt = prompt_tuning_rows.get(cfg, {})
        lora = lora_r16_rows.get(cfg, {})
        lat = latent_reasoning_rows.get(cfg, {})
        text_col = (
            base.get("text_column")
            or ft.get("text_column")
            or prompt.get("text_column")
            or lora.get("text_column")
            or lat.get("text_column")
        )
        row = {
            "config": cfg,
            "samples_total": int(
                max(
                    base.get("samples_total", 0),
                    ft.get("samples_total", 0),
                    prompt.get("samples_total", 0),
                    lora.get("samples_total", 0),
                    lat.get("samples_total", 0),
                )
            ),
            "samples_used": int(
                max(
                    base.get("samples_used", 0),
                    ft.get("samples_used", 0),
                    prompt.get("samples_used", 0),
                    lora.get("samples_used", 0),
                    lat.get("samples_used", 0),
                )
            ),
            "text_column": text_col,
            "base_model_wer": base.get("wer"),
            "base_model_cer": base.get("cer"),
            "baseline_ft_wer": ft.get("wer"),
            "baseline_ft_cer": ft.get("cer"),
            "prompt_tuning_wer": prompt.get("wer"),
            "prompt_tuning_cer": prompt.get("cer"),
            "lora_r16_wer": lora.get("wer"),
            "lora_r16_cer": lora.get("cer"),
            "latent_reasoning_wer": lat.get("wer"),
            "latent_reasoning_cer": lat.get("cer"),
            "baseline_ft_vs_base_model_wer": None
            if base.get("wer") is None or ft.get("wer") is None
            else float(base["wer"] - ft["wer"]),
            "baseline_ft_vs_base_model_cer": None
            if base.get("cer") is None or ft.get("cer") is None
            else float(base["cer"] - ft["cer"]),
            "prompt_tuning_vs_base_model_wer": None
            if base.get("wer") is None or prompt.get("wer") is None
            else float(base["wer"] - prompt["wer"]),
            "prompt_tuning_vs_base_model_cer": None
            if base.get("cer") is None or prompt.get("cer") is None
            else float(base["cer"] - prompt["cer"]),
            "lora_r16_vs_base_model_wer": None
            if base.get("wer") is None or lora.get("wer") is None
            else float(base["wer"] - lora["wer"]),
            "lora_r16_vs_base_model_cer": None
            if base.get("cer") is None or lora.get("cer") is None
            else float(base["cer"] - lora["cer"]),
            "latent_reasoning_vs_base_model_wer": None
            if base.get("wer") is None or lat.get("wer") is None
            else float(base["wer"] - lat["wer"]),
            "latent_reasoning_vs_base_model_cer": None
            if base.get("cer") is None or lat.get("cer") is None
            else float(base["cer"] - lat["cer"]),
            "latent_reasoning_vs_baseline_ft_wer": None
            if ft.get("wer") is None or lat.get("wer") is None
            else float(ft["wer"] - lat["wer"]),
            "latent_reasoning_vs_baseline_ft_cer": None
            if ft.get("cer") is None or lat.get("cer") is None
            else float(ft["cer"] - lat["cer"]),
            "latent_reasoning_vs_prompt_tuning_wer": None
            if prompt.get("wer") is None or lat.get("wer") is None
            else float(prompt["wer"] - lat["wer"]),
            "latent_reasoning_vs_prompt_tuning_cer": None
            if prompt.get("cer") is None or lat.get("cer") is None
            else float(prompt["cer"] - lat["cer"]),
            "latent_reasoning_vs_lora_r16_wer": None
            if lora.get("wer") is None or lat.get("wer") is None
            else float(lora["wer"] - lat["wer"]),
            "latent_reasoning_vs_lora_r16_cer": None
            if lora.get("cer") is None or lat.get("cer") is None
            else float(lora["cer"] - lat["cer"]),
        }
        merged_rows.append(row)

    print_summary_table(merged_rows)

    base_list = [base_model_rows[c] for c in configs if c in base_model_rows]
    ft_list = [baseline_ft_rows[c] for c in configs if c in baseline_ft_rows]
    prompt_list = [prompt_tuning_rows[c] for c in configs if c in prompt_tuning_rows]
    lora_list = [lora_r16_rows[c] for c in configs if c in lora_r16_rows]
    lat_list = [latent_reasoning_rows[c] for c in configs if c in latent_reasoning_rows]
    summary = {
        "base_model_weighted_wer": weighted_avg(base_list, "wer"),
        "base_model_weighted_cer": weighted_avg(base_list, "cer"),
        "baseline_ft_weighted_wer": weighted_avg(ft_list, "wer"),
        "baseline_ft_weighted_cer": weighted_avg(ft_list, "cer"),
        "prompt_tuning_weighted_wer": weighted_avg(prompt_list, "wer"),
        "prompt_tuning_weighted_cer": weighted_avg(prompt_list, "cer"),
        "lora_r16_weighted_wer": weighted_avg(lora_list, "wer"),
        "lora_r16_weighted_cer": weighted_avg(lora_list, "cer"),
        "latent_reasoning_weighted_wer": weighted_avg(lat_list, "wer"),
        "latent_reasoning_weighted_cer": weighted_avg(lat_list, "cer"),
    }
    if summary["base_model_weighted_wer"] is not None and summary["baseline_ft_weighted_wer"] is not None:
        summary["baseline_ft_vs_base_model_weighted_wer"] = (
            summary["base_model_weighted_wer"] - summary["baseline_ft_weighted_wer"]
        )
    else:
        summary["baseline_ft_vs_base_model_weighted_wer"] = None
    if summary["base_model_weighted_cer"] is not None and summary["baseline_ft_weighted_cer"] is not None:
        summary["baseline_ft_vs_base_model_weighted_cer"] = (
            summary["base_model_weighted_cer"] - summary["baseline_ft_weighted_cer"]
        )
    else:
        summary["baseline_ft_vs_base_model_weighted_cer"] = None

    if summary["base_model_weighted_wer"] is not None and summary["latent_reasoning_weighted_wer"] is not None:
        summary["latent_reasoning_vs_base_model_weighted_wer"] = (
            summary["base_model_weighted_wer"] - summary["latent_reasoning_weighted_wer"]
        )
    else:
        summary["latent_reasoning_vs_base_model_weighted_wer"] = None
    if summary["base_model_weighted_cer"] is not None and summary["latent_reasoning_weighted_cer"] is not None:
        summary["latent_reasoning_vs_base_model_weighted_cer"] = (
            summary["base_model_weighted_cer"] - summary["latent_reasoning_weighted_cer"]
        )
    else:
        summary["latent_reasoning_vs_base_model_weighted_cer"] = None

    if summary["baseline_ft_weighted_wer"] is not None and summary["latent_reasoning_weighted_wer"] is not None:
        summary["latent_reasoning_vs_baseline_ft_weighted_wer"] = (
            summary["baseline_ft_weighted_wer"] - summary["latent_reasoning_weighted_wer"]
        )
    else:
        summary["latent_reasoning_vs_baseline_ft_weighted_wer"] = None
    if summary["baseline_ft_weighted_cer"] is not None and summary["latent_reasoning_weighted_cer"] is not None:
        summary["latent_reasoning_vs_baseline_ft_weighted_cer"] = (
            summary["baseline_ft_weighted_cer"] - summary["latent_reasoning_weighted_cer"]
        )
    else:
        summary["latent_reasoning_vs_baseline_ft_weighted_cer"] = None

    if summary["base_model_weighted_wer"] is not None and summary["prompt_tuning_weighted_wer"] is not None:
        summary["prompt_tuning_vs_base_model_weighted_wer"] = (
            summary["base_model_weighted_wer"] - summary["prompt_tuning_weighted_wer"]
        )
    else:
        summary["prompt_tuning_vs_base_model_weighted_wer"] = None
    if summary["base_model_weighted_cer"] is not None and summary["prompt_tuning_weighted_cer"] is not None:
        summary["prompt_tuning_vs_base_model_weighted_cer"] = (
            summary["base_model_weighted_cer"] - summary["prompt_tuning_weighted_cer"]
        )
    else:
        summary["prompt_tuning_vs_base_model_weighted_cer"] = None

    if summary["base_model_weighted_wer"] is not None and summary["lora_r16_weighted_wer"] is not None:
        summary["lora_r16_vs_base_model_weighted_wer"] = (
            summary["base_model_weighted_wer"] - summary["lora_r16_weighted_wer"]
        )
    else:
        summary["lora_r16_vs_base_model_weighted_wer"] = None
    if summary["base_model_weighted_cer"] is not None and summary["lora_r16_weighted_cer"] is not None:
        summary["lora_r16_vs_base_model_weighted_cer"] = (
            summary["base_model_weighted_cer"] - summary["lora_r16_weighted_cer"]
        )
    else:
        summary["lora_r16_vs_base_model_weighted_cer"] = None

    if summary["prompt_tuning_weighted_wer"] is not None and summary["latent_reasoning_weighted_wer"] is not None:
        summary["latent_reasoning_vs_prompt_tuning_weighted_wer"] = (
            summary["prompt_tuning_weighted_wer"] - summary["latent_reasoning_weighted_wer"]
        )
    else:
        summary["latent_reasoning_vs_prompt_tuning_weighted_wer"] = None
    if summary["prompt_tuning_weighted_cer"] is not None and summary["latent_reasoning_weighted_cer"] is not None:
        summary["latent_reasoning_vs_prompt_tuning_weighted_cer"] = (
            summary["prompt_tuning_weighted_cer"] - summary["latent_reasoning_weighted_cer"]
        )
    else:
        summary["latent_reasoning_vs_prompt_tuning_weighted_cer"] = None

    if summary["lora_r16_weighted_wer"] is not None and summary["latent_reasoning_weighted_wer"] is not None:
        summary["latent_reasoning_vs_lora_r16_weighted_wer"] = (
            summary["lora_r16_weighted_wer"] - summary["latent_reasoning_weighted_wer"]
        )
    else:
        summary["latent_reasoning_vs_lora_r16_weighted_wer"] = None
    if summary["lora_r16_weighted_cer"] is not None and summary["latent_reasoning_weighted_cer"] is not None:
        summary["latent_reasoning_vs_lora_r16_weighted_cer"] = (
            summary["lora_r16_weighted_cer"] - summary["latent_reasoning_weighted_cer"]
        )
    else:
        summary["latent_reasoning_vs_lora_r16_weighted_cer"] = None

    print("\n=== Weighted Summary ===")
    for k, v in summary.items():
        if v is None:
            print(f"{k}: -")
        else:
            print(f"{k}: {v:.6f}")

    if args.output_json:
        payload = {
            "model_id": args.model_id,
            "dataset_name": dataset_name,
            "dataset_name_input": args.dataset_name,
            "split": args.split,
            "configs": configs,
            "streaming": args.streaming,
            "text_normalizer": args.text_normalizer,
            "base_model_enabled": run_base_model,
            "baseline_checkpoint": args.baseline_ckpt,
            "prompt_tuning_checkpoint": args.prompt_tuning_ckpt,
            "lora_r16_checkpoint": args.lora_r16_ckpt,
            "latent_reasoning_checkpoint": args.latent_ckpt,
            "latent_checkpoint": args.latent_ckpt,
            "rows": merged_rows,
            "summary": summary,
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nSaved JSON: {args.output_json}")


if __name__ == "__main__":
    main()
