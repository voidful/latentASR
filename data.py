"""Dataset preparation and collation for lr_whisper."""

import numpy as np
from typing import Any, Dict, List, Optional

import torch


# ---------------------------------------------------------------------------
# Audio token sequence helpers
# ---------------------------------------------------------------------------

def build_audio_token_seq(
    audio_bos_id: int,
    audio_token_id: int,
    audio_eos_id: int,
    l_audio: int,
    device: torch.device,
) -> torch.Tensor:
    """Build ``[<|audio_start|>, <audio_pad>×L, <|audio_end|>]`` token tensor.

    Args:
        audio_bos_id: Token id for ``<|audio_start|>``.
        audio_token_id: Token id for the audio placeholder (repeated L times).
        audio_eos_id: Token id for ``<|audio_end|>``.
        l_audio: Number of audio placeholder tokens.
        device: Target device.

    Returns:
        1-D LongTensor of length ``l_audio + 2``.
    """
    audio_bos = torch.tensor([audio_bos_id], dtype=torch.long, device=device)
    audio_eos = torch.tensor([audio_eos_id], dtype=torch.long, device=device)
    audio_pads = torch.full((l_audio,), audio_token_id, dtype=torch.long, device=device)
    return torch.cat([audio_bos, audio_pads, audio_eos])


def _coerce_feat_attention_mask(
    mask: Any,
    target_len: int,
) -> torch.Tensor:
    """Coerce a feature-attention mask to a LongTensor of length *target_len*.

    Pads with zeros or truncates as needed.

    Args:
        mask: A list, numpy array, or Tensor representing the mask.
        target_len: Required length.

    Returns:
        1-D LongTensor of length ``target_len``.
    """
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask, dtype=torch.long)
    mask = mask.long()
    cur = mask.size(-1)
    if cur < target_len:
        mask = torch.cat([mask, torch.zeros(target_len - cur, dtype=torch.long)])
    elif cur > target_len:
        mask = mask[:target_len]
    return mask


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def prepare_dataset(
    processor: Any,
    lang_id: int,
    transcribe_id: int,
    nt_id: Optional[int],
    n_latent: int,
    batch: Dict[str, Any],
    thought_mode: str = "prefix",
    thought_group_size: int = 1,
    language: str = "English",
) -> Dict[str, Any]:
    """Prepare a single sample for training.

    This helper converts raw audio and text into model inputs and labels.  The
    audio waveform is passed through the processor's feature extractor and
    encoded into log-Mel features.  The transcript text is tokenised without
    special tokens.

    Labels use Qwen Chat Template format to be compatible with the pre-trained
    model.  Two layout modes are supported:

    Prefix mode (default)::

        <|im_start|>assistant\\n[NT*N]transcription<|im_end|>

    Interleaved mode (``thought_mode="interleaved"``)::

        <|im_start|>assistant\\n[NT][word1][NT][word2]...<|im_end|>

    In interleaved mode, one NT token is placed before each group of
    ``thought_group_size`` word tokens.  NT positions are masked in the model's
    labels so no CE loss is computed on them.

    The user turn with audio is handled as input in the forward method.

    Args:
        processor: The Qwen3-ASR processor returned by ``asr_wrapper.processor``.
        lang_id: Integer token id for ``<|im_start|>``.
        transcribe_id: Unused (kept for API compatibility).
        nt_id: Integer token id for the latent "NT" token.
        n_latent: Number of latent tokens to prepend (prefix mode only).
        batch: A dictionary with keys ``audio`` and ``text`` as provided by
            ``load_dataset``.
        thought_mode: ``"prefix"`` (default) or ``"interleaved"``.
        thought_group_size: Words per NT token in interleaved mode (default 1).

    Returns:
        A dictionary containing the input features, decoder labels and
        reference text.
    """
    audio = batch["audio"]
    feat_out = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"],
        return_attention_mask=True,
    )
    feats = feat_out.input_features[0]
    if hasattr(feat_out, "attention_mask") and feat_out.attention_mask is not None:
        feat_attention_mask = feat_out.attention_mask[0]
    else:
        feat_attention_mask = [1] * len(feats[0]) if hasattr(feats, "__len__") else None

    raw_text = batch["text"].strip()
    if raw_text:
        fmt_text = raw_text.capitalize()
        # Add a period if it doesn't have one to match base model grammar
        if not fmt_text.endswith((".", "!", "?")):
            fmt_text += "."
    else:
        fmt_text = ""

    text_ids = processor.tokenizer(fmt_text, add_special_tokens=False).input_ids

    im_start = lang_id
    _im_end_token = "<|im_end|>"
    im_end = processor.tokenizer.convert_tokens_to_ids(_im_end_token)
    if im_end is None or im_end == processor.tokenizer.unk_token_id:
        im_end = processor.tokenizer.eos_token_id

    # Original: asst_nl = processor.tokenizer.encode("assistant\n", add_special_tokens=False)
    # if not asst_nl:
    #     raise ValueError("Tokenizer returned empty ids for 'assistant\\n'.")

    # New construction for asst_nl to include language formatting
    asst = processor.tokenizer.encode("assistant", add_special_tokens=False)
    nl = processor.tokenizer.encode("\n", add_special_tokens=False)
    if not asst or not nl:
        raise ValueError("Tokenizer returned empty ids for 'assistant' or '\\n'.")
    asr_prefix_ids = processor.tokenizer.encode(f"language {language}<asr_text>", add_special_tokens=False)
    asst_nl_prefix = asst + nl + asr_prefix_ids

    # For labels, we mask the prefix part including `asst_nl`
    # Let's verify `nt_id` logic.
    interleaved = (thought_mode == "interleaved")
    if interleaved:
        # Interleave: [NT, tok1, NT, tok2, ...] with one NT per thought_group_size words.
        if nt_id is None or nt_id < 0:
            raise ValueError("interleaved thought_mode requires a valid nt_id.")
        group = max(1, int(thought_group_size))
        # Build interleaved sequence
        num_toks = len(text_ids)
        num_groups = (num_toks + group - 1) // group
        actual_nt = min(n_latent, num_groups) # Use n_latent as max NTs

        interleaved_body: List[int] = []
        for i, tok in enumerate(text_ids):
            if i % group == 0 and (i // group) < actual_nt:
                interleaved_body.append(nt_id)
            interleaved_body.append(tok)
        body = interleaved_body
    else:
        # Prefix mode: [NT*N, tok1, tok2, ...]
        latent_tokens: List[int] = []
        if n_latent > 0:
            if nt_id is None or nt_id < 0:
                raise ValueError("n_latent > 0 requires a valid nt_id.")
            latent_tokens = [nt_id] * int(n_latent)
        body = latent_tokens + text_ids

    labels: List[int] = [im_start] + asst_nl_prefix + body + [im_end]
    if labels[-1] != im_end:
        raise ValueError("Label sequence must end with <|im_end|>.")

    result: Dict[str, Any] = {
        "input_features": feats,
        "labels": labels,
        "reference_text": fmt_text, # evaluate on formatted text to allow cleaner debug
    }
    if feat_attention_mask is not None:
        result["feature_attention_mask"] = feat_attention_mask
    return result


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

class DataCollatorQwenASR:
    """Collate function for Qwen-ASR latent finetuning.

    This collator pads the list of input feature arrays into a batch tensor and
    pads the decoder label sequences.  Padding positions in the labels are
    replaced with ``-100`` so that they are ignored by the cross-entropy loss.

    Args:
        processor: The Qwen3-ASR processor used for tokenisation and feature
            extraction.
    """

    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Pad input features
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch_inputs = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt", return_attention_mask=True
        )

        # Build feature_attention_mask from per-sample masks saved in
        # prepare_dataset.  The collator's pad() creates an attention_mask
        # but it's ALL 1s when every sample is pre-padded to 3000 frames.
        if "feature_attention_mask" in features[0]:
            max_len = batch_inputs["input_features"].size(-1)
            masks = []
            for f in features:
                m = f["feature_attention_mask"]
                if hasattr(m, "tolist"):
                    m = m if isinstance(m, list) else m.tolist()
                m = list(m) + [0] * (max_len - len(m))
                masks.append(m[:max_len])
            feature_attention_mask = torch.tensor(masks, dtype=torch.long)
        else:
            feature_attention_mask = batch_inputs["attention_mask"]

        # Pad labels using the tokenizer
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        labels = labels_batch["input_ids"]
        attn = labels_batch["attention_mask"]
        labels = labels.masked_fill(attn.eq(0), -100)

        return {
            "input_features": batch_inputs["input_features"],
            "feature_attention_mask": feature_attention_mask,
            "labels": labels,
        }
