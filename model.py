"""LatentQwenASR model for lr_whisper.

The ``LatentQwenASR`` class wraps a frozen Qwen3-ASR base model and adds a
small controller network that emits per-step delta vectors (latent tokens).
"""

import math
import re
import string
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from jiwer import wer

from data import build_audio_token_seq, _coerce_feat_attention_mask

def _detach_kv(kv_tuple: Any) -> Any:
    """Detach past_key_values to prevent excessive BPTT exploding gradients."""
    if kv_tuple is None:
        return None
    # If it's a raw tuple (older transformers standard)
    if isinstance(kv_tuple, tuple):
        return tuple(tuple(t.detach() for t in layer) for layer in kv_tuple)

    # Transformers >= 4.38 uses Cache objects like DynamicCache
    if hasattr(kv_tuple, "to_legacy_cache"):
        legacy_tuple = kv_tuple.to_legacy_cache()
        detached_tuple = tuple(tuple(t.detach() for t in layer) for layer in legacy_tuple)
        return type(kv_tuple).from_legacy_cache(detached_tuple)

    # Fallback to cloning attributes if the specific class doesn't match
    import copy
    new_cache = copy.copy(kv_tuple)
    if hasattr(new_cache, "key_cache"):
        new_cache.key_cache = [k.detach() for k in getattr(kv_tuple, "key_cache", [])]
        new_cache.value_cache = [v.detach() for v in getattr(kv_tuple, "value_cache", [])]
    return new_cache


class LatentQwenASR(nn.Module):
    """Wrapper around Qwen3-ASR that supports latent prompt injection.

    The class keeps the underlying ASR model frozen and introduces a small
    controller network that emits delta vectors for each latent token.  These
    deltas are normalised and scaled, then added to the token embeddings at
    the NT-token positions.  During training, the deltas are inserted into the
    decoder embeddings at the appropriate positions; during generation, they
    are applied just before decoding.

    Design note - float32 parameters
    -----------------------------------
    ``log_scale`` is kept in **float32** even when the rest of the model is in bfloat16
    to preserve gradient precision for the per-step scales.
    """

    @staticmethod
    def _resolve_text_model(thinker: nn.Module) -> nn.Module:
        """Resolve the inner text decoder module across wrapped backends (e.g. PEFT)."""
        queue: List[object] = [thinker]
        seen: set = set()
        while queue:
            node = queue.pop(0)
            if node is None:
                continue
            node_id = id(node)
            if node_id in seen:
                continue
            seen.add(node_id)
            emb = getattr(node, "embed_tokens", None)
            if isinstance(emb, nn.Module):
                return node
            for attr in ("model", "base_model", "module"):
                child = getattr(node, attr, None)
                if child is not None and child is not node:
                    queue.append(child)
        raise AttributeError(
            "Unable to resolve text decoder with `embed_tokens` from thinker module."
        )

    @staticmethod
    def _resolve_embed_tokens(text_model: nn.Module) -> nn.Module:
        emb = getattr(text_model, "embed_tokens", None)
        if isinstance(emb, nn.Module):
            return emb
        if hasattr(text_model, "get_input_embeddings"):
            emb = text_model.get_input_embeddings()
            if isinstance(emb, nn.Module):
                return emb
        raise AttributeError("Unable to resolve input embedding module.")

    def __init__(
        self,
        asr_model: nn.Module,
        processor: Any,
        n_latent: int,
        nt_token_id: int,
        lang_token_id: int,
        transcribe_token_id: int,
        freeze_base: bool = True,
        use_latent: bool = True,
        use_soft_prompt: bool = False,
        soft_prompt_init_mode: str = "text",
        soft_prompt_init_text: str = "",
        user_prompt_text: str = "Transcribe the audio into text.",

        delta_tanh_c: float = 5.0,
        scale_max: float = 3.0,
        scale_init: float = 0.2,
        thought_mode: str = "prefix",
        thought_group_size: int = 1,
        halt_threshold: float = 0.0,
        latent_drop_prob: float = 0.0,
        latent_input_noise_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.base_model = asr_model
        self.processor = processor
        self.config = getattr(asr_model, "config", None)
        self.n_latent = int(n_latent)
        self.use_latent = bool(use_latent and self.n_latent > 0)
        self.nt_token_id = int(nt_token_id)
        self.use_soft_prompt = bool(
            use_soft_prompt
            and (not self.use_latent)
            and self.n_latent > 0
            and self.nt_token_id >= 0
        )
        self.soft_prompt_init_mode = (soft_prompt_init_mode or "text").strip().lower()
        self.soft_prompt_init_text = soft_prompt_init_text or ""
        self.lang_token_id = int(lang_token_id)
        self.transcribe_token_id = int(transcribe_token_id)

        # Hyperparams stored from constructor args (originally read from env globals).

        self.delta_tanh_c = float(delta_tanh_c)
        self.halt_threshold = float(halt_threshold)
        self.latent_drop_prob = float(latent_drop_prob)
        self.latent_input_noise_std = float(latent_input_noise_std)

        # Qwen3-ASR audio token ID
        _audio_tok = None
        if self.config and hasattr(self.config, "audio_token_id") and self.config.audio_token_id is not None:
            _audio_tok = int(self.config.audio_token_id)
        if _audio_tok is None:
            _thinker_cfg = getattr(self.base_model, "config", None) or getattr(
                getattr(self.base_model, "thinker", None), "config", None
            )
            if _thinker_cfg is not None and hasattr(_thinker_cfg, "audio_token_id") and _thinker_cfg.audio_token_id is not None:
                _audio_tok = int(_thinker_cfg.audio_token_id)
        if _audio_tok is None:
            for _name in ("<|AUDIO|>", "<|audio|>", "<|audio_content|>", "<|audio_pad|>"):
                target_tok = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor
                if hasattr(target_tok, "convert_tokens_to_ids"):
                    _try = target_tok.convert_tokens_to_ids(_name)
                    unk_id = getattr(target_tok, "unk_token_id", None)
                    if _try is not None and _try != unk_id:
                        _audio_tok = int(_try)
                        break
        if _audio_tok is None:
            _audio_tok = getattr(self.processor, "audio_token_id", None)
        if _audio_tok is None:
            _audio_tok = 151646  # last-resort fallback
        self.audio_token_id = _audio_tok
        
        target_tok = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor
        _decoded = target_tok.convert_ids_to_tokens(self.audio_token_id) if hasattr(target_tok, "convert_ids_to_tokens") else str(self.audio_token_id)
        print(f"[init] audio_token_id={self.audio_token_id} decodes_to={_decoded!r}")

        # Freeze all parameters in the underlying model for latent-adapter training.
        if freeze_base:
            for p in self.base_model.parameters():
                p.requires_grad = False

        # Always freeze audio_tower regardless of training mode.
        audio_tower = getattr(self.base_model, "thinker", self.base_model)
        audio_tower = getattr(audio_tower, "audio_tower", None)
        if audio_tower is not None:
            n_frozen = 0
            for p in audio_tower.parameters():
                if p.requires_grad:
                    p.requires_grad = False
                    n_frozen += 1
            if n_frozen > 0:
                print(f"[init] Froze {n_frozen} audio_tower parameters (always frozen).")
        # Also freeze multi_modal_projector (audio→text projection).
        projector = getattr(getattr(self.base_model, "thinker", self.base_model), "multi_modal_projector", None)
        if projector is not None:
            n_frozen_proj = 0
            for p in projector.parameters():
                if p.requires_grad:
                    p.requires_grad = False
                    n_frozen_proj += 1
            if n_frozen_proj > 0:
                print(f"[init] Froze {n_frozen_proj} multi_modal_projector parameters (always frozen).")

        self.thinker = self.base_model.thinker
        self.text_model = self._resolve_text_model(self.thinker)
        self.embed_tokens = self._resolve_embed_tokens(self.text_model)

        if self.config and hasattr(self.config, "text_config"):
            d = self.config.text_config.hidden_size
        elif self.config and hasattr(self.config, "hidden_size"):
            d = self.config.hidden_size
        else:
            if hasattr(self.embed_tokens, "embedding_dim"):
                d = int(self.embed_tokens.embedding_dim)
            else:
                d = int(self.embed_tokens.weight.size(-1))

        print(f"[LatentQwenASR] Using hidden_size={d}")

        target_dtype = self.thinker.dtype if hasattr(self.thinker, "dtype") else torch.float32

        self.init_proj = nn.Linear(d, d, bias=True).to(dtype=target_dtype)
        self.delta_proj = nn.Linear(d, d, bias=True).to(dtype=target_dtype)

        self.step_embed = nn.Parameter(torch.zeros(self.n_latent, d, dtype=target_dtype))
        nn.init.normal_(self.step_embed, mean=0.0, std=0.02)
        self.step_proj = nn.Linear(d, d, bias=False).to(dtype=target_dtype)
        self.scale_max = float(scale_max)



        scale_init_val = float(scale_init)
        if scale_init_val <= 0:
            scale_init_val = 1.0
        self.log_scale = nn.Parameter(
            torch.full((self.n_latent,), float(np.log(scale_init_val)), dtype=torch.float32)
        )

        # Stability: LayerNorm for the recurrent thought loop
        self.thought_ln = nn.LayerNorm(d, elementwise_affine=False).to(dtype=target_dtype)

        # Thought Quality Analyzer (Value Head) predicting expected CE error
        self.value_head = nn.Linear(d, 1).to(dtype=target_dtype)

        # Soft Gated Injection (Scheme 4)
        self.injection_gate = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.Sigmoid()
        ).to(dtype=target_dtype)
        # Initialize gate weights to 0 so sigmoid(0) = 0.5 (neutral start)
        nn.init.zeros_(self.injection_gate[0].weight)
        nn.init.zeros_(self.injection_gate[0].bias)
        self._hidden_size = d

        # Front-token prompt tuning
        self.soft_prompt_embed = nn.Parameter(
            torch.zeros(self.n_latent, d, dtype=target_dtype),
            requires_grad=self.use_soft_prompt,
        )
        if self.use_soft_prompt:
            init_prompt = None
            if self.soft_prompt_init_mode == "text" and self.soft_prompt_init_text.strip():
                init_ids = self.processor.tokenizer.encode(
                    self.soft_prompt_init_text.strip(),
                    add_special_tokens=False,
                )
                if init_ids:
                    with torch.no_grad():
                        emb_device = self.embed_tokens.weight.device
                        init_ids_t = torch.tensor(init_ids, dtype=torch.long, device=emb_device)
                        base = self.embed_tokens(init_ids_t).detach()
                    if base.numel() > 0:
                        if base.size(0) < self.n_latent:
                            reps = (self.n_latent + base.size(0) - 1) // base.size(0)
                            base = base.repeat((reps, 1))
                        init_prompt = base[: self.n_latent]
            if init_prompt is None:
                init_prompt = torch.empty((self.n_latent, d), dtype=target_dtype)
                nn.init.normal_(init_prompt, mean=0.0, std=0.02)
            self.soft_prompt_embed.data.copy_(
                init_prompt.to(device=self.soft_prompt_embed.device, dtype=self.soft_prompt_embed.dtype)
            )
        else:
            self.soft_prompt_embed.requires_grad = False

        # Cast trainable modules to backbone dtype
        target_dtype = self.base_model.dtype if hasattr(self.base_model, "dtype") else torch.float32
        self.init_proj.to(target_dtype)
        self.delta_proj.to(target_dtype)
        self.step_embed.data = self.step_embed.data.to(target_dtype)
        self.step_proj.to(target_dtype)
        self.soft_prompt_embed.data = self.soft_prompt_embed.data.to(target_dtype)
        # log_scale intentionally stays float32 for per-step scale gradient precision.

        # Qwen3-ASR Special Audio Tokens
        self.audio_bos_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|audio_start|>")
        self.audio_eos_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|audio_end|>")
        if self.audio_bos_token_id is None:
            self.audio_bos_token_id = self.processor.tokenizer.bos_token_id
        if self.audio_eos_token_id is None:
            self.audio_eos_token_id = self.processor.tokenizer.eos_token_id

        # Chat-template token pieces
        self.im_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        if self.im_start_id is None or self.im_start_id == self.processor.tokenizer.unk_token_id:
            self.im_start_id = self.lang_token_id
        _im_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if _im_end_id is None or _im_end_id == self.processor.tokenizer.unk_token_id:
            _im_end_id = self.processor.tokenizer.eos_token_id
        self.im_end_id = _im_end_id
        # Unified stop-token set for generation AND eval truncation.
        _stop_ids = {int(self.im_end_id)}
        _eot_id = self.processor.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        if _eot_id is not None and _eot_id != self.processor.tokenizer.unk_token_id:
            _stop_ids.add(int(_eot_id))
        if self.processor.tokenizer.eos_token_id is not None:
            _stop_ids.add(int(self.processor.tokenizer.eos_token_id))
        self.stop_ids = sorted(_stop_ids)
        self.user_nl_ids = self.processor.tokenizer.encode("user\n", add_special_tokens=False)
        self.asst_nl_ids = self.processor.tokenizer.encode("assistant\n", add_special_tokens=False)
        self.asr_prefix_ids = self.processor.tokenizer.encode("language English<asr_text>", add_special_tokens=False)
        self.nl_ids = self.processor.tokenizer.encode("\n", add_special_tokens=False)
        self.user_prompt_text = user_prompt_text
        if self.user_prompt_text:
            self.user_prompt_ids = self.processor.tokenizer.encode(
                self.user_prompt_text, add_special_tokens=False
            )
        else:
            self.user_prompt_ids = []
        self._warned_generate_fallback = False
        self.thought_mode = thought_mode.strip().lower()
        self.thought_group_size = max(1, int(thought_group_size))

        # In non-latent modes, keep latent controller out of optimization.
        if not self.use_latent:
            for p in self.init_proj.parameters():
                p.requires_grad = False
            for p in self.delta_proj.parameters():
                p.requires_grad = False
            for p in self.step_proj.parameters():
                p.requires_grad = False
            for p in self.thought_ln.parameters():
                p.requires_grad = False
            self.step_embed.requires_grad = False

            self.log_scale.requires_grad = False
        if not self.use_soft_prompt:
            self.soft_prompt_embed.requires_grad = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_chat_template_tensors(
        self,
        device: torch.device,
        language: str = "English",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build shared chat-template pieces for train/eval consistency."""
        system_nl_ids = self.processor.tokenizer.encode("system\n", add_special_tokens=False)
        system_turn = torch.tensor(
            [self.im_start_id] + system_nl_ids + [self.im_end_id] + self.nl_ids,
            dtype=torch.long,
            device=device,
        )
        user_prefix = torch.tensor(
            [self.im_start_id] + self.user_nl_ids,
            dtype=torch.long,
            device=device,
        )
        if self.user_prompt_ids:
            user_prompt = torch.tensor(self.user_prompt_ids, dtype=torch.long, device=device)
        else:
            user_prompt = torch.empty((0,), dtype=torch.long, device=device)
        user_suffix = torch.tensor(
            [self.im_end_id] + self.nl_ids,
            dtype=torch.long,
            device=device,
        )
        asr_prefix_ids = self.processor.tokenizer.encode(f"language {language}<asr_text>", add_special_tokens=False)
        assistant_prefix = torch.tensor(
            [self.im_start_id] + self.asst_nl_ids + asr_prefix_ids,
            dtype=torch.long,
            device=device,
        )
        return system_turn, user_prefix, user_prompt, user_suffix, assistant_prefix

    def _build_full_sequence(
        self,
        audio_toks: torch.Tensor,
        text_all: torch.Tensor,
        system_turn: torch.Tensor,
        user_prefix: torch.Tensor,
        user_prompt: torch.Tensor,
        user_suffix: torch.Tensor,
        label_pad: int = -100,
        n_extra_mask: int = 0,
        mask_all_nt: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build full input_ids and aligned labels for one sample.

        The sequence layout is::

            [system_turn, user_prefix, audio_toks, user_prompt,
             user_suffix, text_all]

        Labels clone input_ids, then mask everything up to (and including)
        the assistant control tokens, plus an optional ``n_extra_mask``
        additional positions (e.g. NT tokens in latent prefix mode).

        When ``mask_all_nt=True``, all NT-token positions anywhere in the
        sequence are additionally masked.  Use this for interleaved mode where
        NT tokens are scattered throughout the assistant turn.

        Args:
            audio_toks: 1-D token tensor for the audio segment.
            text_all: 1-D token tensor for the target text (assistant turn).
            system_turn: Shared system-turn token tensor.
            user_prefix: Shared user-prefix token tensor.
            user_prompt: Shared user-prompt token tensor.
            user_suffix: Shared user-suffix token tensor.
            label_pad: Padding value for ignored label positions.
            n_extra_mask: Number of additional positions after the assistant
                prefix to mask (set to ``self.n_latent`` for latent prefix
                mode, 0 for native/baseline/interleaved).
            mask_all_nt: If True, additionally mask every NT-token position
                in the full sequence (for interleaved thought mode).

        Returns:
            ``(full_ids, full_labels)`` - 1-D LongTensors.
        """
        full_ids = torch.cat(
            [system_turn, user_prefix, audio_toks, user_prompt, user_suffix, text_all]
        )
        n_prefix = (
            system_turn.size(0)
            + user_prefix.size(0)
            + audio_toks.size(0)
            + user_prompt.size(0)
            + user_suffix.size(0)
        )
        # Mask: <|im_start|> + "assistant\n" + n_extra_mask (e.g. NT tokens)
        n_asst_control = 1 + len(self.asst_nl_ids) + n_extra_mask
        n_total_mask = n_prefix + n_asst_control
        full_labels = full_ids.clone()
        full_labels[:n_total_mask] = label_pad
        if mask_all_nt and self.nt_token_id >= 0:
            full_labels[full_ids == self.nt_token_id] = label_pad
        return full_ids, full_labels

    def _inject_latent_deltas(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.LongTensor,
        deltas: torch.Tensor,
    ) -> torch.Tensor:
        """Inject latent delta embeddings at NT-token positions.

        Replaces the NT-token positions in *inputs_embeds* with::

            base_nt_embedding + delta

        Deltas are always injected at full strength.  The Value Head controls
        hard routing (N=0 skip / early halt) instead of soft scaling.
        """
        B = inputs_embeds.size(0)
        K = deltas.size(1)
        device = inputs_embeds.device
        nt_mask = (input_ids == self.nt_token_id)
        nt_counts = nt_mask.sum(dim=1)

        max_nt = int(nt_counts.max().item())
        if K == 1 and max_nt > 1:
            deltas = deltas.expand(B, max_nt, -1)
            K = max_nt

        assert (nt_counts == K).all(), (
            f"NT token count mismatch! {nt_counts} vs expected {K}"
        )
        nt_token_ids = torch.full(
            (B, K), self.nt_token_id, dtype=torch.long, device=device
        )
        base_nt = self.embed_tokens(nt_token_ids).to(dtype=inputs_embeds.dtype)
        delta_scaled = deltas.float().to(dtype=inputs_embeds.dtype)

        # Soft Gated Injection (Scheme 4)
        # 1. Concatenate base embedding and proposed delta to form context
        gate_input = torch.cat([base_nt, delta_scaled], dim=-1)
        # 2. Compute gate weights (0 to 1)
        gating_weight = self.injection_gate(gate_input)
        # 3. Apply soft gate to delta and form final injected prefix
        gated_delta = delta_scaled * gating_weight
        nt_embeds = base_nt + gated_delta
        batch_idx, seq_idx = torch.nonzero(nt_mask, as_tuple=True)
        result = inputs_embeds.clone()
        result[batch_idx, seq_idx, :] = nt_embeds.to(dtype=inputs_embeds.dtype).view(-1, self._hidden_size)
        return result

    def _compute_prefix_state(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.LongTensor,
        strict: bool = True,
        language: str = "English",
    ) -> Tuple[torch.Tensor, object, torch.LongTensor]:
        """Compute prefix state and KV cache up to the first NT token."""
        B, _, _ = inputs_embeds.shape
        device = inputs_embeds.device
        text_model = self.text_model

        nt_mask = (input_ids == self.nt_token_id)
        nt_counts = nt_mask.sum(dim=1)
        if strict:
            assert (nt_counts == self.n_latent).all(), (
                f"NT token count mismatch! {nt_counts} vs {self.n_latent}"
            )

        first_nt = nt_mask.float().argmax(dim=1)
        if strict:
            asr_prefix_ids = self.processor.tokenizer.encode(
                f"language {language}<asr_text>", add_special_tokens=False
            )
            expected = torch.tensor(
                [self.im_start_id] + self.asst_nl_ids + asr_prefix_ids, dtype=torch.long, device=device
            )
            exp_len = expected.numel()
            for i in range(B):
                idx = int(first_nt[i].item())
                start = idx - exp_len
                assert start >= 0, (
                    f"Prompt too short to contain assistant prefix (idx={idx}, exp_len={exp_len})."
                )
                actual = input_ids[i, start:idx]
                assert torch.equal(actual, expected), (
                    "Assistant prefix mismatch before NT. "
                    f"expected={expected.tolist()} actual={actual.tolist()}"
                )

        prefix_lens = first_nt
        if strict:
            assert (prefix_lens > 0).all(), f"Invalid prefix length(s): {prefix_lens}"

        max_prefix = int(prefix_lens.max().item())
        prefix_embeds = inputs_embeds[:, :max_prefix, :]
        prefix_attn = (
            torch.arange(max_prefix, device=device).unsqueeze(0) < prefix_lens.unsqueeze(1)
        ).long()
        if prefix_attn.numel() > 0:
            prefix_embeds = prefix_embeds * prefix_attn.unsqueeze(-1)

        with torch.no_grad():
            out = text_model(
                inputs_embeds=prefix_embeds,
                attention_mask=prefix_attn,
                use_cache=True,
                return_dict=True,
            )
        last_positions = torch.clamp(prefix_lens - 1, min=0)
        batch_idx = torch.arange(B, device=device)
        last = out.last_hidden_state[batch_idx, last_positions].detach()
        return last, out.past_key_values, prefix_attn

    def _embed_scale(self) -> float:
        # Qwen3-ASR does NOT scale embeddings by sqrt(d_model).
        return 1.0


    def step_scales(self) -> torch.Tensor:
        s = F.softplus(self.log_scale)
        s = torch.clamp(s, max=self.scale_max)
        return s[:self.n_latent]

    def forced_decoder_ids_latent(self) -> List[Tuple[int, int]]:
        ids: List[Tuple[int, int]] = [(0, self.im_start_id)]
        cur = 1
        for tid in self.asst_nl_ids:
            ids.append((cur, tid))
            cur += 1
        for i in range(self.n_latent):
            ids.append((cur + i, self.nt_token_id))
        return ids

    def forced_decoder_ids_baseline(self) -> List[Tuple[int, int]]:
        ids: List[Tuple[int, int]] = [(0, self.im_start_id)]
        cur = 1
        for tid in self.asst_nl_ids:
            ids.append((cur, tid))
            cur += 1
        return ids

    def _shift_right(self, labels: torch.LongTensor) -> torch.LongTensor:
        pad_id = self.processor.tokenizer.eos_token_id
        decoder_input_ids = labels.new_full(labels.shape, pad_id)
        decoder_input_ids[:, 0] = (
            self.processor.tokenizer.bos_token_id
            if self.processor.tokenizer.bos_token_id is not None
            else self.lang_token_id
        )
        shifted = labels[:, :-1].clone()
        shifted = shifted.masked_fill(shifted == -100, pad_id)
        decoder_input_ids[:, 1:] = shifted
        return decoder_input_ids

    def _encode_audio(
        self,
        input_features: torch.FloatTensor,
        feature_attention_mask: torch.LongTensor,
    ) -> Tuple[torch.Tensor, List[int]]:
        """Manually encode audio features to get their exact lengths per sample."""
        if feature_attention_mask is not None:
            feature_lens = torch.sum(feature_attention_mask, dim=1)
        else:
            feature_lens = torch.full(
                (input_features.size(0),), input_features.size(2),
                device=input_features.device, dtype=torch.long,
            )

        audio_features = []
        lengths = []
        audio_tower = self.thinker.audio_tower
        projector = getattr(self.thinker, "multi_modal_projector", None)

        for i in range(len(input_features)):
            feat = input_features[i]
            l = feature_lens[i]
            inp = feat[:, :l]
            length_tensor = l.unsqueeze(0)
            out = audio_tower(inp, feature_lens=length_tensor).last_hidden_state
            if projector is not None:
                out = projector(out)
            out = out.squeeze(0)
            audio_features.append(out)
            lengths.append(out.size(0))

        return torch.cat(audio_features, dim=0), lengths

    def _get_audio_lengths(
        self,
        input_features: torch.FloatTensor,
        feature_attention_mask: torch.LongTensor,
    ) -> List[int]:
        """Compute audio output lengths using the exact same formula as the thinker."""
        if feature_attention_mask is not None:
            feature_lens = torch.sum(feature_attention_mask, dim=1)
        else:
            feature_lens = torch.full(
                (input_features.size(0),), input_features.size(2),
                device=input_features.device, dtype=torch.long,
            )

        input_lengths_leave = feature_lens % 100
        feat_lengths = (input_lengths_leave - 1) // 2 + 1
        output_lengths = (
            ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1
            + (feature_lens // 100) * 13
        )
        return output_lengths.tolist()

    def _get_native_audio_embeds(
        self,
        input_ids: torch.LongTensor,
        input_features: torch.FloatTensor,
        feature_attention_mask: torch.LongTensor,
        attention_mask: torch.LongTensor,
    ) -> torch.Tensor:
        """Capture the thinker's native audio-fused embeddings via a forward hook."""
        captured: Dict[str, torch.Tensor] = {}

        def _hook(module: nn.Module, args: Any, kwargs: Any) -> None:
            embeds = kwargs.get("inputs_embeds", None)
            if embeds is None and len(args) > 0:
                embeds = args[0]
            if embeds is not None:
                captured["embeds"] = embeds
            return None

        hook = self.text_model.register_forward_pre_hook(_hook, with_kwargs=True)
        try:
            with torch.no_grad():
                self.thinker(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    input_features=input_features,
                    feature_attention_mask=feature_attention_mask,
                    labels=None,
                    return_dict=True,
                )
        finally:
            hook.remove()

        if "embeds" not in captured:
            raise RuntimeError(
                "_get_native_audio_embeds: forward hook did not capture embeddings. "
                "The thinker's internal forward may use a different code path."
            )
        return captured["embeds"].detach().clone()

    def _compute_continuous_thoughts(
        self,
        initial_state: torch.Tensor,
        prefix_past_key_values: Optional[object] = None,
        prefix_attention_mask: Optional[torch.LongTensor] = None,
        halt_threshold: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute recurrent latent states + per-step deltas using DEQ fixed-point iteration."""
        B = initial_state.size(0)
        D = initial_state.size(-1)
        device = initial_state.device
        text_model = self.text_model

        initial_norm = initial_state / (initial_state.norm(dim=-1, keepdim=True) + 1e-6)
        h = self.init_proj(initial_norm)

        # Remove random corruption as teacher forcing artificially lowers CE
        _corrupted = False
        
        scales = self.step_scales()
        thought_embeds: List[torch.Tensor] = []
        state_embeds: List[torch.Tensor] = []
        raw_norms: List[torch.Tensor] = []
        scaled_norms: List[torch.Tensor] = []
        v_preds: List[torch.Tensor] = []
        past_kv = prefix_past_key_values
        running_attn = prefix_attention_mask

        # ---- Dynamic Causal Thinking Loop ----
        # Instead of fixed-point iteration (which doesn't converge on frozen LLMs),
        # we generate a sequence of thought tokens.
        max_iter = self.n_latent
        if halt_threshold is None:
            halt_threshold = getattr(self, "halt_threshold", 0.0)

        current_embed = h

        # ---- N=0 Skip Check (Tanh) ----
        # v_pred ∈ [-1,1]: expected delta CE (positive means LR helps).
        # If v_pred < threshold → model is confident LR is NOT needed → skip.
        if not self.training:
            v_init = torch.tanh(self.value_head(h.unsqueeze(1)).squeeze(-1).squeeze(-1))  # (B,)
            if (v_init < halt_threshold).all():
                # Return empty deltas → generate() will remove all NT tokens
                empty_deltas = torch.zeros(B, 0, D, device=device, dtype=h.dtype)
                empty_states = torch.zeros(B, 0, D, device=device, dtype=h.dtype)
                stats = {
                    "predicted_value": v_init.detach(),
                    "deq_iters": torch.tensor(0.0, device=device),
                    "raw_norm_mean": torch.zeros(0, device=device),
                    "raw_norm_std": torch.zeros(0, device=device),
                    "scaled_norm_mean": torch.zeros(0, device=device),
                    "scaled_norm_std": torch.zeros(0, device=device),
                    "cos_mean": torch.zeros(0, device=device),
                    "cos_std": torch.zeros(0, device=device),
                    "scales": scales.detach(),
                    "diff_norm": torch.tensor(0.0, device=device),
                    "step_cos": torch.tensor(0.0, device=device),
                    "v_preds": v_init.unsqueeze(1).detach(),
                    "gate": v_init.unsqueeze(1).detach(),
                    "skipped": True,
                }
                return empty_deltas, empty_states, stats

        with torch.no_grad():
            for k in range(max_iter):
                # The input for step k is step_embed[k] projected, added to the normalized state
                step_idx = min(k, self.step_embed.size(0) - 1)
                step = self.step_proj(self.step_embed[step_idx]).view(1, 1, -1).expand(B, 1, -1)
                
                h_norm = self.thought_ln(current_embed)
                h_input = h_norm.unsqueeze(1) + step

                step_attn = None
                if running_attn is not None:
                    # Causal: attention mask GROWS by 1 at each step
                    one = torch.ones((B, 1), dtype=running_attn.dtype, device=device)
                    step_attn = torch.cat([running_attn, one], dim=1)

                out = text_model(
                    inputs_embeds=h_input,
                    past_key_values=past_kv,
                    attention_mask=step_attn,
                    use_cache=True,
                    return_dict=True,
                )

                # Next state
                h_next = out.last_hidden_state.squeeze(1)
                
                # Compute delta to inject into this token's output representation
                delta_raw = self.delta_proj(h_next)
                delta_raw_f32 = delta_raw.float()
                norm_val = delta_raw_f32.norm(dim=-1, keepdim=True)
                delta_dir = (delta_raw_f32 / (norm_val + 1e-6)).to(dtype=delta_raw.dtype)
                
                scale = scales[step_idx]
                delta = delta_dir * scale

                # Early Halt Check (Value Head, tanh) — BEFORE committing delta
                v_raw = self.value_head(h_next.unsqueeze(1)).squeeze(-1).squeeze(-1)  # (B,)
                v_pred = torch.tanh(v_raw)

                # Halt when v_pred drops below threshold (model says this step is harmful)
                if not self.training:
                    if (v_pred < halt_threshold).all():
                        break

                # Only commit the delta if we did NOT halt
                thought_embeds.append(delta)
                state_embeds.append(h_next)
                raw_norms.append(delta_raw.norm(dim=-1))
                scaled_norms.append(delta.norm(dim=-1))
                v_preds.append(v_pred)

                # Update context for the *next* iteration
                past_kv = out.past_key_values
                running_attn = step_attn
                current_embed = h_next

            # Grab the past_kv produced exactly prior to the final step
            with torch.no_grad():
                # We need to re-run up to final_step_idx-1 to get the exact past_kv?
                # Actually, no. We can just run from initial_state and re-accumulate.
                # BUT since max_iter <= n_latent is small (e.g. 4), we can just re-run 
                # the *entire* sequence with gradients enabled!
                pass
                
        # To make things simple and correct for training with such a small sequence (N=4),
        # we can just re-run the FULL causal generation with gradients enabled and overwrite.
        # This gives exact gradients through all thinking steps (BPTT).
        if self.training:
            thought_embeds_grad = []
            state_embeds_grad = []
            current_embed_grad = h  # Keep grad flow to init_proj
            running_attn_grad = prefix_attention_mask
            past_kv_grad = prefix_past_key_values
            
            # Re-run for exactly the number of steps we decided to take in the no_grad pass
            for k in range(len(thought_embeds)):
                step_idx = min(k, self.step_embed.size(0) - 1)
                step = self.step_proj(self.step_embed[step_idx]).view(1, 1, -1).expand(B, 1, -1)
                
                h_norm = self.thought_ln(current_embed_grad)
                h_input = h_norm.unsqueeze(1) + step

                step_attn = None
                if running_attn_grad is not None:
                    one = torch.ones((B, 1), dtype=running_attn_grad.dtype, device=device)
                    step_attn = torch.cat([running_attn_grad, one], dim=1)

                out = text_model(
                    inputs_embeds=h_input,
                    past_key_values=past_kv_grad,  # Evolve KV cache like inference!
                    attention_mask=step_attn,
                    use_cache=True,
                    return_dict=True,
                )

                h_next_g = out.last_hidden_state.squeeze(1)
                
                delta_raw_g = self.delta_proj(h_next_g)
                delta_raw_f32_g = delta_raw_g.float()
                norm_val_g = delta_raw_f32_g.norm(dim=-1, keepdim=True)
                delta_dir_g = (delta_raw_f32_g / (norm_val_g + 1e-6)).to(dtype=delta_raw_g.dtype)
                
                scale_g = scales[step_idx]
                delta_g = delta_dir_g * scale_g

                thought_embeds_grad.append(delta_g)
                state_embeds_grad.append(h_next_g)

                # Detach state so CE-loss gradients don't flow back to previous steps.
                # Each step is independently trained on its own delta (1-step truncated BPTT).
                past_kv_grad = out.past_key_values
                running_attn_grad = step_attn
                current_embed_grad = h_next_g.detach()
                
            # Override the no_grad lists with the gradient-tracked ones
            thought_embeds = thought_embeds_grad
            state_embeds = state_embeds_grad
            
        # Stack sequences — guard against empty list when all steps halted
        if not thought_embeds:
            # All steps rejected by Value Head → return empty (K=0)
            deltas = torch.zeros((B, 0, D), device=device, dtype=h.dtype)
            states = torch.zeros((B, 0, D), device=device, dtype=h.dtype)
            # Use initial state for value prediction
            predicted_value = torch.tanh(self.value_head(h.unsqueeze(1)))
            stats = {
                "predicted_value": predicted_value.squeeze(-1).squeeze(-1),
                "deq_iters": torch.tensor(0.0, device=device),
                "raw_norm_mean": torch.zeros(self.n_latent, device=device),
                "raw_norm_std": torch.zeros(self.n_latent, device=device),
                "scaled_norm_mean": torch.zeros(self.n_latent, device=device),
                "scaled_norm_std": torch.zeros(self.n_latent, device=device),
                "cos_mean": torch.zeros(self.n_latent, device=device),
                "cos_std": torch.zeros(self.n_latent, device=device),
                "scales": scales.detach(),
                "diff_norm": torch.tensor(0.0, device=device),
                "step_cos": torch.tensor(0.0, device=device),
                "v_preds": None,
                "gate": torch.zeros((B, 1), device=device),
                "corrupted": _corrupted,
                "skipped": True,
            }
            return deltas, states, stats

        deltas = torch.stack(thought_embeds, dim=1) # (B, K, D)
        states = torch.stack(state_embeds, dim=1)   # (B, K, D)
        
        # predicted_value: tanh [-1, 1] estimating LR's impact on CE
        predicted_value = torch.tanh(self.value_head(states[:, -1:, :])) # (B, 1, 1) -> [-1, 1]

        # Gate: average tanh v_pred across all steps, scaled to [0, 1] for gated delta injection
        v_preds_stacked = torch.stack(v_preds, dim=1) if v_preds else None  # (B, K)
        if v_preds_stacked is not None:
            gate = (v_preds_stacked.mean(dim=1, keepdim=True) + 1.0) / 2.0  # (B, 1) — mean confidence mapped to [0, 1]
        else:
            gate = (predicted_value.squeeze(-1) + 1.0) / 2.0  # fallback, scaled to [0, 1]

        deq_iters = len(thought_embeds)

        raw_tensor = torch.stack(raw_norms, dim=1)
        scaled_tensor = torch.stack(scaled_norms, dim=1)
        scales_val = scales.detach()

        with torch.no_grad():
            flat_states = states.view(-1, D)
            flat_states_norm = F.normalize(flat_states, dim=-1)

            word_emb_weight = self.embed_tokens.weight
            vocab_size = word_emb_weight.size(0)
            if vocab_size > 10000:
                idx = torch.randperm(vocab_size, device=device)[:10000]
                word_emb_sub = word_emb_weight[idx]
            else:
                word_emb_sub = word_emb_weight

            word_emb_norm = F.normalize(word_emb_sub, dim=-1)
            sim_matrix = torch.matmul(flat_states_norm, word_emb_norm.t())
            if sim_matrix.numel() > 0:
                max_sim, _ = sim_matrix.max(dim=-1)
                cos_mean_val = max_sim.mean()
                cos_std_val = max_sim.std(unbiased=False)
            else:
                cos_mean_val = flat_states_norm.new_tensor(0.0)
                cos_std_val = flat_states_norm.new_tensor(0.0)

        if deltas.size(1) > 1:
            diffs = deltas[:, 1:] - deltas[:, :-1]
            diff_norm_val = diffs.norm(dim=-1).mean()
            step_cos_val = F.cosine_similarity(deltas[:, 1:], deltas[:, :-1], dim=-1).mean()
        else:
            diff_norm_val = deltas.new_tensor(0.0)
            step_cos_val = deltas.new_tensor(0.0)

        stats = {
            "predicted_value": predicted_value.squeeze(-1).squeeze(-1),  # (B,) for per-sample MSE (delta CE)
            "deq_iters": torch.tensor(deq_iters, dtype=torch.float32, device=device),
            "raw_norm_mean": raw_tensor.mean(dim=0).detach() if raw_tensor.ndim > 1 else raw_tensor.detach(),
            "raw_norm_std": raw_tensor.std(dim=0, unbiased=False).detach() if raw_tensor.ndim > 1 else raw_tensor.new_zeros(raw_tensor.size(-1)),
            "scaled_norm_mean": scaled_tensor.mean(dim=0).detach() if scaled_tensor.ndim > 1 else scaled_tensor.detach(),
            "scaled_norm_std": scaled_tensor.std(dim=0, unbiased=False).detach() if scaled_tensor.ndim > 1 else scaled_tensor.new_zeros(scaled_tensor.size(-1)),
            "cos_mean": torch.full((self.n_latent,), cos_mean_val.item(), device=device),
            "cos_std": torch.full((self.n_latent,), cos_std_val.item(), device=device),
            "scales": scales_val,
            "diff_norm": diff_norm_val.detach(),
            "step_cos": step_cos_val.detach(),
            "v_preds": v_preds_stacked.detach() if v_preds_stacked is not None else None,
            "gate": gate.detach(),
            "corrupted": _corrupted,
        }
        return deltas, states, stats

    def _compute_causal_thoughts(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Two-pass causal thought injection for interleaved mode.

        Each NT token at position ``k`` receives a thought delta computed from
        the hidden state at position ``k-1`` (the token immediately before it).
        This makes each thought *causal*: it can attend to all previous context
        including the word generated just before it, which aligns with ASR's
        left-to-right generation structure.

        Pass 1 (``no_grad``): forward the full sequence with **base** NT
        embeddings to capture all hidden states ``h_all``.

        Pass 2 (with grad): for each NT at position k, compute a delta from
        ``h_all[:, k-1, :]`` via ``init_proj → delta_proj → normalize →
        scale``.  Inject the modified NT embedding and return the updated
        ``inputs_embeds``.

        Args:
            inputs_embeds: ``(B, SeqLen, D)`` embedding tensor (base NT
                embeddings already placed at NT positions by
                ``_get_native_audio_embeds``).
            input_ids: ``(B, SeqLen)`` token id tensor to locate NT positions.

        Returns:
            ``(modified_embeds, deltas_padded, states_padded, stats)`` where
            ``deltas_padded`` and ``states_padded`` have shape
            ``(B, max_nt, D)`` (zero-padded to the batch maximum NT count).
        """
        B, SeqLen, D = inputs_embeds.shape
        device = inputs_embeds.device

        nt_mask = (input_ids == self.nt_token_id)  # (B, SeqLen)
        batch_idx, nt_pos = torch.nonzero(nt_mask, as_tuple=True)

        if len(nt_pos) == 0:
            empty_d = inputs_embeds.new_zeros((B, 0, D))
            empty_s = inputs_embeds.new_zeros((B, 0, D))
            dummy = inputs_embeds.new_tensor(0.0)
            stats: Dict[str, torch.Tensor] = {
                "raw_norm_mean": inputs_embeds.new_zeros((0,)),
                "raw_norm_std": inputs_embeds.new_zeros((0,)),
                "scaled_norm_mean": inputs_embeds.new_zeros((0,)),
                "scaled_norm_std": inputs_embeds.new_zeros((0,)),
                "cos_mean": inputs_embeds.new_zeros((0,)),
                "cos_std": inputs_embeds.new_zeros((0,)),
                "scales": inputs_embeds.new_zeros((0,)),
                "diff_norm": dummy,
                "step_cos": dummy,
            }
            return inputs_embeds, empty_d, empty_s, stats

        # --- Pass 1: no_grad forward to capture all hidden states ----------
        with torch.no_grad():
            out1 = self.text_model(
                inputs_embeds=inputs_embeds,
                use_cache=False,
                return_dict=True,
            )
        h_all = out1.last_hidden_state  # (B, SeqLen, D) - detached in no_grad

        # Context for each NT at position k is h_all[:, k-1, :]
        ctx_pos = (nt_pos - 1).clamp(min=0)  # (N_total,)
        contexts = h_all[batch_idx, ctx_pos]  # (N_total, D)

        # --- Pass 2: generate deltas (with grad) ---------------------------
        ctx_norm = F.normalize(contexts.float(), dim=-1).to(dtype=inputs_embeds.dtype)
        h_proj = self.init_proj(ctx_norm)         # (N_total, D)
        delta_raw = self.delta_proj(h_proj)       # (N_total, D)
        # Avoid PyTorch F.normalize backward bug in float16 for near-zero norms
        # Must compute in float32 because 1 / 1e-6 > 65504 (overflows FP16 to Inf)
        delta_raw_f32 = delta_raw.float()
        norm_val = delta_raw_f32.norm(dim=-1, keepdim=True)
        delta_dir = (delta_raw_f32 / (norm_val + 1e-6)).to(dtype=delta_raw.dtype)

        scale = self.step_scales().mean()         # scalar - shared across all positions
        delta = delta_dir * scale                 # (N_total, D)

        # Build NT embeddings and inject
        # base_nt_ids and nt_embeds are moved down to use the scaled delta


        # --- Pack into (B, max_nt, D) for loss functions -------------------
        nt_counts = nt_mask.sum(dim=1)  # (B,)
        max_nt = int(nt_counts.max().item())
        deltas_padded = inputs_embeds.new_zeros((B, max_nt, D))
        states_padded = inputs_embeds.new_zeros((B, max_nt, D))
        for b in range(B):
            b_mask = (batch_idx == b)
            cnt = int(nt_counts[b].item())
            if cnt > 0:
                deltas_padded[b, :cnt] = delta[b_mask]
                states_padded[b, :cnt] = contexts[b_mask]

        # --- Stats for logging (shape matches prefix-mode stats) ------------
        with torch.no_grad():
            delta_for_stats = delta.float()
            raw_norms = delta_raw.norm(dim=-1)       # (N_total,)
            scaled_norms = delta_for_stats.norm(dim=-1)
            flat_ctx_norm = F.normalize(contexts, dim=-1)
            word_emb = self.embed_tokens.weight
            vocab_size = word_emb.size(0)
            if vocab_size > 10000:
                idx = torch.randperm(vocab_size, device=device)[:10000]
                word_emb_sub = word_emb[idx]
            else:
                word_emb_sub = word_emb
            word_emb_norm = F.normalize(word_emb_sub, dim=-1)
            sim_matrix = torch.matmul(flat_ctx_norm.to(word_emb_norm.dtype), word_emb_norm.t())
            max_sim, _ = sim_matrix.max(dim=-1)
            cos_mean_val = max_sim.mean()
            cos_std_val = max_sim.std()
            # Compute diff_norm and step_cos across padded deltas
            if max_nt > 1:
                diff = deltas_padded[:, 1:, :] - deltas_padded[:, :-1, :]
                diff_norm_val = diff.norm(dim=-1).mean()
                step_cos_val = F.cosine_similarity(
                    deltas_padded[:, 1:, :], deltas_padded[:, :-1, :], dim=-1
                ).mean()
            else:
                diff_norm_val = deltas_padded.new_tensor(0.0)
                step_cos_val = deltas_padded.new_tensor(0.0)

        # Inject deltas into embeddings (grad-tracked for training)
        base_nt_ids = torch.full(
            (len(batch_idx),), self.nt_token_id, dtype=torch.long, device=device
        )
        base_nt = self.embed_tokens(base_nt_ids).to(dtype=inputs_embeds.dtype)
        nt_embeds = base_nt + delta.to(dtype=inputs_embeds.dtype)

        result_embeds = inputs_embeds.clone()
        result_embeds[batch_idx, nt_pos] = nt_embeds

        n_out = max_nt
        stats = {
            "raw_norm_mean": raw_norms.mean().unsqueeze(0).expand(n_out).detach(),
            "raw_norm_std": raw_norms.std(unbiased=False).unsqueeze(0).expand(n_out).detach(),
            "scaled_norm_mean": scaled_norms.mean().unsqueeze(0).expand(n_out).detach(),
            "scaled_norm_std": scaled_norms.std(unbiased=False).unsqueeze(0).expand(n_out).detach(),
            "cos_mean": torch.full((n_out,), cos_mean_val.item(), device=device),
            "cos_std": torch.full((n_out,), cos_std_val.item(), device=device),
            "scales": scale.unsqueeze(0).expand(n_out).detach(),
            "diff_norm": diff_norm_val.detach(),
            "step_cos": step_cos_val.detach(),
        }
        return result_embeds, deltas_padded, states_padded, stats

    @torch.no_grad()
    def _generate_interleaved(
        self,
        base_embeds: torch.Tensor,
        attention_mask: torch.LongTensor,
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> torch.LongTensor:
        """Token-by-token generation with interleaved thought injection.

        Before each group of ``self.thought_group_size`` generated word tokens,
        a thought delta is computed from the current sequence's last hidden
        state, and an NT token (with the modified embedding) is prepended.

        NT tokens in the output are included as ``self.nt_token_id`` so that
        callers using ``skip_special_tokens=True`` in ``tokenizer.decode`` will
        strip them automatically (since ``<|latent|>`` is a registered special
        token).

        Args:
            base_embeds: ``(1, SeqLen, D)`` audio-fused embeddings of the
                prompt prefix (no NT tokens appended yet).
            attention_mask: ``(1, SeqLen)`` attention mask for *base_embeds*.
            max_new_tokens: Maximum number of word tokens to generate.
            do_sample: Whether to sample instead of greedy decode.
            temperature: Sampling temperature (ignored if ``do_sample=False``).

        Returns:
            ``(1, N_generated)`` LongTensor of token ids including interleaved
            NT tokens and ending with the first EOS/stop token encountered.
        """
        eos_ids: set = set(self.stop_ids)
        device = base_embeds.device
        safe_temp = float(temperature) if float(temperature) > 0 else 1.0

        cur_embeds = base_embeds.clone()
        cur_attn = attention_mask.clone()
        generated: List[torch.Tensor] = []
        words_generated = 0

        nt_id_t = torch.tensor([[self.nt_token_id]], dtype=torch.long, device=device)
        base_nt_emb = self.embed_tokens(nt_id_t).to(dtype=cur_embeds.dtype)  # (1, 1, D)

        def _append_thought() -> None:
            """Compute a thought from the last hidden state and inject NT."""
            nonlocal cur_embeds, cur_attn
            # Get context = last hidden state of current sequence
            ctx_out = self.text_model(
                inputs_embeds=cur_embeds,
                use_cache=False,
                return_dict=True,
            )
            h_last = ctx_out.last_hidden_state[:, -1:, :]  # (1, 1, D)
            h_norm = F.normalize(h_last.float().squeeze(1), dim=-1).to(dtype=cur_embeds.dtype)
            h_proj = self.init_proj(h_norm)
            delta_raw = self.delta_proj(h_proj)
            delta_dir = F.normalize(delta_raw, dim=-1)
            scale = self.step_scales().mean()
            delta_scaled = (delta_dir * scale).float().to(dtype=cur_embeds.dtype)
            
            # Use Tanh Value Head for Interleaved mode as well
            if hasattr(self, "value_head"):
                v_raw = self.value_head(h_norm) # (1, 1)
                v_pred = (torch.tanh(v_raw) + 1.0) / 2.0 # (1, 1) -> scaled to [0, 1]
                halt_thresh = float(getattr(self.config, "halt_threshold", 0.0))
                # Skip injecting NT if predicted value < threshold
                if v_pred.squeeze().item() < halt_thresh:
                    return

            nt_emb_mod = base_nt_emb + delta_scaled.unsqueeze(1)  # (1, 1, D)
            cur_embeds = torch.cat([cur_embeds, nt_emb_mod], dim=1)
            cur_attn = torch.cat([cur_attn, cur_attn.new_ones((1, 1))], dim=1)
            generated.append(nt_id_t.clone())

        for _ in range(max_new_tokens + len(generated)):
            # Inject thought before each new group of words
            if words_generated % self.thought_group_size == 0:
                _append_thought()

            # Generate next word token
            out = self.thinker(
                inputs_embeds=cur_embeds,
                attention_mask=cur_attn,
                input_features=None,
                use_cache=False,
                return_dict=True,
            )
            next_logits = out.logits[:, -1, :]
            if do_sample:
                probs = F.softmax(next_logits / safe_temp, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

            tok_id = int(next_token[0, 0].item())
            generated.append(next_token)
            if tok_id in eos_ids:
                break

            next_emb = self.embed_tokens(next_token)
            cur_embeds = torch.cat([cur_embeds, next_emb], dim=1)
            cur_attn = torch.cat([cur_attn, cur_attn.new_ones((1, 1))], dim=1)
            words_generated += 1

        if not generated:
            return torch.zeros((1, 0), dtype=torch.long, device=device)
        return torch.cat(generated, dim=1)

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def _forward_native(
        self,
        input_features: torch.FloatTensor,
        labels: torch.LongTensor,
        feature_attention_mask: torch.LongTensor,
        language_hint: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Native thinker forward for LoRA / baseline FT."""
        B = input_features.size(0)
        device = input_features.device

        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.processor.tokenizer.eos_token_id or 151643

        input_ids_list = []
        labels_list = []
        label_pad = -100

        audio_lengths = self._get_audio_lengths(input_features, feature_attention_mask)
        system_turn, user_prefix, user_prompt, user_suffix, _ = (
            self._build_chat_template_tensors(device, language=language_hint or "English")
        )

        for i in range(B):
            valid_len = (labels[i] != label_pad).sum().item()
            text_all = labels[i, :valid_len]
            l_audio = audio_lengths[i]

            audio_toks = build_audio_token_seq(
                self.audio_bos_token_id, self.audio_token_id, self.audio_eos_token_id,
                l_audio, device,
            )
            full_ids, full_labels = self._build_full_sequence(
                audio_toks, text_all, system_turn, user_prefix, user_prompt, user_suffix,
                label_pad=label_pad, n_extra_mask=0,
            )
            input_ids_list.append(full_ids)
            labels_list.append(full_labels)

        max_len = max(x.size(0) for x in input_ids_list)
        input_ids_batch = torch.full((B, max_len), pad_token_id, dtype=torch.long, device=device)
        labels_batch = torch.full((B, max_len), label_pad, dtype=torch.long, device=device)
        for i in range(B):
            l = input_ids_list[i].size(0)
            input_ids_batch[i, :l] = input_ids_list[i]
            labels_batch[i, :l] = labels_list[i]

        attention_mask = (input_ids_batch != pad_token_id).long()

        out = self.thinker(
            input_ids=input_ids_batch,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=None,
            return_dict=True,
        )
        logits = out.logits

        # One-time audio diagnostic
        if not getattr(self, "_native_diag_done", False):
            self._native_diag_done = True
            import sys as _s
            _d = _s.stderr.write
            _d(f"\n[native-diag] thinker type: {type(self.thinker).__name__}\n")
            _d(f"[native-diag] has base_model (PEFT): {hasattr(self.thinker, 'base_model')}\n")
            n_audio_ph = (input_ids_batch[0] == self.audio_token_id).sum().item()
            feat_valid = (
                feature_attention_mask[0].sum().item()
                if feature_attention_mask is not None
                else input_features.size(2)
            )
            _d(f"[native-diag] audio_token_id={self.audio_token_id} decoded={self.processor.tokenizer.convert_ids_to_tokens(self.audio_token_id)!r}\n")
            _d(f"[native-diag] input_features: {list(input_features.shape)}, valid_mel_frames={feat_valid}\n")
            _d(f"[native-diag] input_ids shape: {list(input_ids_batch.shape)}\n")
            _d(f"[native-diag] input_features shape: {list(input_features.shape)}\n")
            _d(f"[native-diag] audio placeholder tokens in input_ids[0]: {n_audio_ph}\n")
            _d(f"[native-diag] audio_bos/eos in ids[0]: bos={int((input_ids_batch[0]==self.audio_bos_token_id).sum())} eos={int((input_ids_batch[0]==self.audio_eos_token_id).sum())}\n")
            with torch.no_grad():
                try:
                    zero_feat = torch.zeros_like(input_features[:1])
                    out_noaud = self.thinker(
                        input_ids=input_ids_batch[:1],
                        attention_mask=attention_mask[:1],
                        input_features=zero_feat,
                        feature_attention_mask=feature_attention_mask[:1] if feature_attention_mask is not None else None,
                        labels=None,
                        return_dict=True,
                    )
                    diff = (logits[0] - out_noaud.logits[0]).abs().mean().item()
                    _d(f"[native-diag] logit diff (real vs zero audio): {diff:.6f}\n")
                    if diff < 0.01:
                        _d(f"[native-diag] ⚠️  AUDIO NOT BEING USED! Logits identical with zero audio.\n")
                    else:
                        _d(f"[native-diag] ✓ Audio IS affecting logits (diff={diff:.4f})\n")
                except Exception as e:
                    _d(f"[native-diag] zero-audio test failed: {e}\n")
            tok_str = self.processor.tokenizer.decode(input_ids_batch[0, :15].tolist())
            _d(f"[native-diag] input_ids[0][:15] decoded: {repr(tok_str)}\n")
            _s.stderr.flush()

        # Pre-shift labels for loss
        labels_for_loss = torch.full_like(labels_batch, label_pad)
        labels_for_loss[:, :-1] = labels_batch[:, 1:]

        D = self._hidden_size
        dummy = logits.new_tensor(0.0)
        stats = {
            "raw_norm_mean": logits.new_zeros((0,)),
            "raw_norm_std": logits.new_zeros((0,)),
            "scaled_norm_mean": logits.new_zeros((0,)),
            "scaled_norm_std": logits.new_zeros((0,)),
            "cos_mean": logits.new_zeros((0,)),
            "cos_std": logits.new_zeros((0,)),
            "scales": logits.new_zeros((0,)),
            "diff_norm": dummy,
            "step_cos": dummy,
        }
        deltas = logits.new_zeros((B, 0, D))
        states = logits.new_zeros((B, 0, D))
        initial_state = logits.new_zeros((B, D))

        return logits, stats, deltas, states, labels_for_loss, initial_state

    def forward(
        self,
        input_features: torch.FloatTensor,
        labels: torch.LongTensor,
        feature_attention_mask: Optional[torch.LongTensor] = None,
        global_step: int = 0,
        language_hint: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for training."""
        B = input_features.size(0)
        device = input_features.device

        if feature_attention_mask is None:
            feature_attention_mask = torch.ones(
                (B, input_features.size(2)), dtype=torch.long, device=device
            )

        # Fast path: use thinker's native forward for LoRA / baseline FT.
        if not self.use_latent and not self.use_soft_prompt:
            return self._forward_native(input_features, labels, feature_attention_mask, language_hint=language_hint)

        audio_lengths = self._get_audio_lengths(input_features, feature_attention_mask)

        new_input_ids_list = []
        new_labels_list = []

        label_pad = -100
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.processor.tokenizer.eos_token_id or 151643

        system_turn, user_prefix, user_prompt, user_suffix, _ = (
            self._build_chat_template_tensors(device, language=language_hint or "English")
        )

        interleaved_mode = self.use_latent and self.thought_mode == "interleaved"

        # +++ LATENT DROP +++
        drop_latent_this_batch = False
        if self.training and self.use_latent and not interleaved_mode:
            drop_prob = self.latent_drop_prob
            if drop_prob > 0.0 and torch.rand(1).item() < drop_prob:
                drop_latent_this_batch = True
        
        actual_n_latent = 0 if drop_latent_this_batch else self.n_latent

        for i in range(B):
            valid_len = (labels[i] != label_pad).sum().item()
            text_all = labels[i, :valid_len]
            if drop_latent_this_batch:
                text_all = text_all[text_all != self.nt_token_id]

            l_audio = audio_lengths[i]

            audio_toks = build_audio_token_seq(
                self.audio_bos_token_id, self.audio_token_id, self.audio_eos_token_id,
                l_audio, device,
            )
            if interleaved_mode:
                # NT tokens are scattered throughout text_all; mask all NT positions.
                full_ids, full_labels = self._build_full_sequence(
                    audio_toks, text_all, system_turn, user_prefix, user_prompt, user_suffix,
                    label_pad=label_pad, n_extra_mask=0, mask_all_nt=True,
                )
            else:
                # Prefix mode: all NT tokens appear contiguously after asst_nl.
                full_ids, full_labels = self._build_full_sequence(
                    audio_toks, text_all, system_turn, user_prefix, user_prompt, user_suffix,
                    label_pad=label_pad, n_extra_mask=actual_n_latent,
                )
            new_input_ids_list.append(full_ids)
            new_labels_list.append(full_labels)

        max_len = max(x.size(0) for x in new_input_ids_list)
        new_input_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long, device=device)
        new_labels_raw = torch.full((B, max_len), label_pad, dtype=torch.long, device=device)

        for i in range(B):
            l = new_input_ids_list[i].size(0)
            new_input_ids[i, :l] = new_input_ids_list[i]
            new_labels_raw[i, :l] = new_labels_list[i]

        attention_mask = (new_input_ids != pad_token_id).long()

        inputs_embeds = self._get_native_audio_embeds(
            input_ids=new_input_ids,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            attention_mask=attention_mask,
        )

        # +++ LATENT INPUT NOISE (Value Head Regularization) +++
        # Save pre-LR embeddings (with noise) for fair baseline CE comparison
        pre_lr_embeds = None
        if self.training:
            noise_std = self.latent_input_noise_std
            if noise_std > 0.0:
                noise = torch.randn_like(inputs_embeds) * noise_std
                inputs_embeds = inputs_embeds + noise
            # Save the (possibly noisy) embeddings BEFORE LR injection
            if self.use_latent:
                pre_lr_embeds = inputs_embeds.detach().clone()

        # Latent / soft-prompt injection
        if self.use_latent and interleaved_mode:
            # Interleaved: two-pass causal thought generation.
            inputs_embeds, deltas, states, stats = self._compute_causal_thoughts(
                inputs_embeds, new_input_ids
            )
            initial_state = inputs_embeds.new_zeros((B, self._hidden_size))
        elif self.use_latent:
            if actual_n_latent == 0:
                initial_state = inputs_embeds.new_zeros((B, self._hidden_size))
                deltas = inputs_embeds.new_zeros((B, 0, self._hidden_size))
                states = inputs_embeds.new_zeros((B, 0, self._hidden_size))
                stats = {
                    "raw_norm_mean": inputs_embeds.new_zeros((0,)),
                    "raw_norm_std": inputs_embeds.new_zeros((0,)),
                    "scaled_norm_mean": inputs_embeds.new_zeros((0,)),
                    "scaled_norm_std": inputs_embeds.new_zeros((0,)),
                    "cos_mean": inputs_embeds.new_zeros((0,)),
                    "cos_std": inputs_embeds.new_zeros((0,)),
                    "scales": inputs_embeds.new_zeros((0,)),
                    "diff_norm": inputs_embeds.new_tensor(0.0),
                    "step_cos": inputs_embeds.new_tensor(0.0),
                }
            else:
                # Prefix mode: recurrent thought loop from audio-prefix state.
                initial_state, prefix_past_kv, prefix_attn = self._compute_prefix_state(
                    inputs_embeds, new_input_ids, strict=True
                )
                deltas, states, stats = self._compute_continuous_thoughts(
                    initial_state,
                    prefix_past_key_values=prefix_past_kv,
                    prefix_attention_mask=prefix_attn,
                )
                inputs_embeds = self._inject_latent_deltas(
                    inputs_embeds, new_input_ids, deltas
                )
        elif self.use_soft_prompt:
            nt_mask = (new_input_ids == self.nt_token_id)
            nt_counts = nt_mask.sum(dim=1)
            assert (nt_counts == self.n_latent).all(), (
                f"NT token count mismatch! {nt_counts} vs {self.n_latent}"
            )
            prompt = self.soft_prompt_embed.to(dtype=inputs_embeds.dtype)
            prompt = prompt.unsqueeze(0).expand(B, -1, -1).contiguous()
            batch_idx, seq_idx = torch.nonzero(nt_mask, as_tuple=True)
            prompt_embeds = inputs_embeds.clone()
            prompt_embeds[batch_idx, seq_idx, :] = prompt.view(-1, self._hidden_size)
            inputs_embeds = prompt_embeds
            initial_state = inputs_embeds.new_zeros((B, self._hidden_size))
            deltas = inputs_embeds.new_zeros((B, 0, self._hidden_size))
            states = inputs_embeds.new_zeros((B, 0, self._hidden_size))
            stats = {
                "raw_norm_mean": inputs_embeds.new_zeros((0,)),
                "raw_norm_std": inputs_embeds.new_zeros((0,)),
                "scaled_norm_mean": inputs_embeds.new_zeros((0,)),
                "scaled_norm_std": inputs_embeds.new_zeros((0,)),
                "cos_mean": inputs_embeds.new_zeros((0,)),
                "cos_std": inputs_embeds.new_zeros((0,)),
                "scales": inputs_embeds.new_zeros((0,)),
                "diff_norm": inputs_embeds.new_tensor(0.0),
                "step_cos": inputs_embeds.new_tensor(0.0),
            }
        else:
            initial_state = inputs_embeds.new_zeros((B, self._hidden_size))
            deltas = inputs_embeds.new_zeros((B, 0, self._hidden_size))
            states = inputs_embeds.new_zeros((B, 0, self._hidden_size))
            stats = {
                "raw_norm_mean": inputs_embeds.new_zeros((0,)),
                "raw_norm_std": inputs_embeds.new_zeros((0,)),
                "scaled_norm_mean": inputs_embeds.new_zeros((0,)),
                "scaled_norm_std": inputs_embeds.new_zeros((0,)),
                "cos_mean": inputs_embeds.new_zeros((0,)),
                "cos_std": inputs_embeds.new_zeros((0,)),
                "scales": inputs_embeds.new_zeros((0,)),
                "diff_norm": inputs_embeds.new_tensor(0.0),
                "step_cos": inputs_embeds.new_tensor(0.0),
            }

        out = self.thinker(
            inputs_embeds=inputs_embeds,
            input_ids=new_input_ids,
            attention_mask=attention_mask,
            input_features=None,
            feature_attention_mask=None,
            labels=None,
            use_cache=False,
            return_dict=True,
        )
        logits = out.logits

        new_labels = torch.full_like(new_labels_raw, label_pad)
        new_labels[:, :-1] = new_labels_raw[:, 1:]

        if logits.size(1) != new_labels.size(1):
            if logits.size(1) > new_labels.size(1):
                shift = logits.size(1) - new_labels.size(1)
                logits = logits[:, shift:, :]
            else:
                new_labels = new_labels[:, -logits.size(1):]

        # ---- Baseline CE vs LR CE for Value Head MSE (Delta CE) ----
        # Compute per-sample CE for both baseline (no LR) and LR-enhanced
        # forward passes. The difference (baseline_ce - lr_ce) forms the
        # continuous target for the Value Head via tanh compression.
        if self.use_latent and self.training and deltas.size(1) > 0:
            with torch.no_grad():
                # Use the SAME (possibly noisy) embeddings but WITHOUT LR deltas
                # This ensures a fair comparison: noisy-no-LR vs noisy-with-LR
                if pre_lr_embeds is not None:
                    baseline_embeds = pre_lr_embeds
                else:
                    baseline_embeds = self._get_native_audio_embeds(
                        input_ids=new_input_ids,
                        input_features=input_features,
                        feature_attention_mask=feature_attention_mask,
                        attention_mask=attention_mask,
                    )
                baseline_out = self.thinker(
                    inputs_embeds=baseline_embeds,
                    input_ids=new_input_ids,
                    attention_mask=attention_mask,
                    input_features=None,
                    feature_attention_mask=None,
                    labels=None,
                    use_cache=False,
                    return_dict=True,
                )
                baseline_logits = baseline_out.logits
                if baseline_logits.size(1) != new_labels.size(1):
                    if baseline_logits.size(1) > new_labels.size(1):
                        shift = baseline_logits.size(1) - new_labels.size(1)
                        baseline_logits = baseline_logits[:, shift:, :]
                    else:
                        bl_labels = new_labels[:, -baseline_logits.size(1):]
                else:
                    bl_labels = new_labels

                # Per-SAMPLE CE: for each sample in batch, compute CE on valid tokens.
                # This is used by train.py to form the continuous delta_ce target.
                valid_mask = (bl_labels != -100)  # (B, T)
                valid_per_sample = valid_mask.float().sum(dim=1).clamp(min=1)  # (B,)

                # For value head accuracy: exclude asr_prefix tokens
                # ("language", "English", "<asr_text>") which are trivially correct
                # and dilute the real transcript accuracy signal.
                asr_prefix_set = set(self.asr_prefix_ids)
                asr_prefix_mask = torch.zeros_like(bl_labels, dtype=torch.bool)
                for tid in asr_prefix_set:
                    asr_prefix_mask |= (bl_labels == tid)
                acc_mask = valid_mask & ~asr_prefix_mask  # exclude prefix tokens
                acc_per_sample = acc_mask.float().sum(dim=1).clamp(min=1)

                if not hasattr(self, '_acc_debug_printed'):
                    self._acc_debug_printed = True
                    total_tokens = bl_labels.size(1)
                    ce_valid = valid_per_sample.mean().item()
                    acc_valid = acc_per_sample.mean().item()
                    sample_acc_toks = bl_labels[0][acc_mask[0]][:20].tolist()
                    sample_decoded = self.processor.tokenizer.decode(sample_acc_toks)
                    print(f"\n[ACC-DEBUG] total={total_tokens} ce_valid={ce_valid:.0f} acc_valid={acc_valid:.0f}")
                    print(f"[ACC-DEBUG] first 20 acc label ids: {sample_acc_toks}")
                    print(f"[ACC-DEBUG] decoded: {sample_decoded}")
                if valid_mask.any():
                    # Per-sample baseline CE (uses full valid_mask for loss)
                    loss_fct_none = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
                    bl_ce_per_token = loss_fct_none(
                        baseline_logits.reshape(-1, baseline_logits.size(-1)),
                        bl_labels.reshape(-1),
                    ).view(B, -1)  # (B, T)
                    baseline_ce_per_sample = bl_ce_per_token.sum(dim=1) / valid_per_sample  # (B,)
                    stats["baseline_ce"] = baseline_ce_per_sample  # (B,)

                    # Per-sample LR CE (for fair Delta CE comparison)
                    lr_logits_for_ce = logits
                    if lr_logits_for_ce.size(1) != bl_labels.size(1):
                        lr_logits_for_ce = lr_logits_for_ce[:, -bl_labels.size(1):, :]
                    lr_ce_per_token = loss_fct_none(
                        lr_logits_for_ce.reshape(-1, lr_logits_for_ce.size(-1)),
                        bl_labels.reshape(-1),
                    ).view(B, -1)
                    lr_ce_per_sample = lr_ce_per_token.sum(dim=1) / valid_per_sample
                    stats["lr_ce"] = lr_ce_per_sample # (B,)

                    # Per-sample accuracy for VALUE HEAD (uses acc_mask, excludes prefix)
                    baseline_preds = baseline_logits.argmax(dim=-1)  # (B, T)
                    lr_preds = logits.argmax(dim=-1)                 # (B, T)
                    if lr_preds.size(1) != bl_labels.size(1):
                        lr_preds = lr_preds[:, -bl_labels.size(1):]
                    baseline_correct = ((baseline_preds == bl_labels) & acc_mask).float()
                    lr_correct = ((lr_preds == bl_labels) & acc_mask).float()
                    baseline_acc = baseline_correct.sum(dim=1) / acc_per_sample  # (B,)
                    lr_acc = lr_correct.sum(dim=1) / acc_per_sample              # (B,)
                    stats["baseline_acc"] = baseline_acc
                    stats["lr_acc"] = lr_acc

                    # Error-conditional metrics for value head target (uses acc_mask)
                    baseline_wrong = ((baseline_preds != bl_labels) & acc_mask)  # (B, T)
                    baseline_right = ((baseline_preds == bl_labels) & acc_mask)  # (B, T)
                    lr_fixes = ((lr_preds == bl_labels) & baseline_wrong).float().sum(dim=1)   # (B,)
                    lr_breaks = ((lr_preds != bl_labels) & baseline_right).float().sum(dim=1)  # (B,)
                    baseline_errors = baseline_wrong.float().sum(dim=1).clamp(min=1)           # (B,)
                    stats["lr_fixes"] = lr_fixes
                    stats["lr_breaks"] = lr_breaks
                    stats["baseline_errors"] = baseline_errors
                else:
                    B_size = logits.size(0)
                    stats["baseline_ce"] = logits.new_zeros(B_size)
                    stats["baseline_acc"] = logits.new_ones(B_size)
                    stats["lr_acc"] = logits.new_ones(B_size)
                    stats["lr_fixes"] = logits.new_zeros(B_size)
                    stats["lr_breaks"] = logits.new_zeros(B_size)
                    stats["baseline_errors"] = logits.new_ones(B_size)

        return logits, stats, deltas, states, new_labels, initial_state

    @torch.no_grad()
    def _generate_with_forward_fallback(
        self,
        start_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        attention_mask: torch.LongTensor,
        max_new_tokens: int,
        eos_token_id: Optional[Any],
        do_sample: bool,
        temperature: float,
    ) -> torch.LongTensor:
        """Fallback decoder when ``.generate(inputs_embeds=...)`` is unsupported."""
        if eos_token_id is None:
            eos_ids: List[int] = []
        elif isinstance(eos_token_id, (list, tuple, set)):
            eos_ids = [int(x) for x in eos_token_id]
        else:
            eos_ids = [int(eos_token_id)]

        generated_tokens: List[torch.Tensor] = []
        safe_temp = float(temperature) if float(temperature) > 0 else 1.0

        cur_ids = start_ids.clone()
        cur_embeds = inputs_embeds.clone()
        cur_attn = attention_mask.clone()

        for _ in range(int(max_new_tokens)):
            out = self.thinker(
                inputs_embeds=cur_embeds,
                attention_mask=cur_attn,
                input_features=None,
                use_cache=False,
                return_dict=True,
            )
            next_logits = out.logits[:, -1, :]

            if do_sample:
                probs = F.softmax(next_logits / safe_temp, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

            generated_tokens.append(next_token)
            if eos_ids and int(next_token[0, 0].item()) in eos_ids:
                break

            next_embed = self.embed_tokens(next_token)
            cur_ids = torch.cat([cur_ids, next_token], dim=1)
            cur_embeds = torch.cat([cur_embeds, next_embed], dim=1)
            cur_attn = torch.cat(
                [
                    cur_attn,
                    torch.ones(
                        (cur_attn.size(0), 1),
                        dtype=cur_attn.dtype,
                        device=cur_attn.device,
                    ),
                ],
                dim=1,
            )

        if not generated_tokens:
            return start_ids.new_empty((start_ids.size(0), 0))
        return torch.cat(generated_tokens, dim=1)

    @torch.no_grad()
    def generate(
        self,
        input_features: torch.FloatTensor,
        feature_attention_mask: Optional[torch.LongTensor] = None,
        max_new_tokens: int = 128,
        use_baseline: bool = False,
        return_thoughts: bool = False,
        return_stats: bool = False,
        language_hint: Optional[str] = None,
        prompt_text: Optional[str] = None,
        **gen_kwargs: Any,
    ) -> Any:
        """Generate a transcription with or without latent prompting."""
        language_hint = gen_kwargs.pop("language_hint", language_hint)
        prompt_text = gen_kwargs.pop("prompt_text", prompt_text)
        dynamic_halt_threshold = gen_kwargs.pop("dynamic_halt_threshold", getattr(self, "halt_threshold", 0.0))

        B = input_features.size(0)
        assert B == 1, "Evaluation assumes batch size 1."
        device = input_features.device

        if feature_attention_mask is None:
            feature_attention_mask = torch.ones(
                (B, input_features.size(2)), dtype=torch.long, device=device
            )
        audio_lengths = self._get_audio_lengths(input_features, feature_attention_mask)

        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.processor.tokenizer.eos_token_id or 151643

        l_audio = audio_lengths[0]
        audio_toks = build_audio_token_seq(
            self.audio_bos_token_id, self.audio_token_id, self.audio_eos_token_id,
            l_audio, device,
        ).unsqueeze(0)

        system_turn, user_prefix, user_prompt, user_suffix, assistant_prefix = (
            self._build_chat_template_tensors(device, language=language_hint or "English")
        )
        prefix = torch.cat(
            [system_turn, user_prefix, audio_toks.squeeze(0), user_prompt, user_suffix, assistant_prefix],
            dim=0,
        ).unsqueeze(0)

        latent_active = (not use_baseline) and self.use_latent
        soft_prompt_active = (not use_baseline) and self.use_soft_prompt
        interleaved_active = latent_active and self.thought_mode == "interleaved"

        # For prefix mode: prepend NT tokens to the prompt.
        # For interleaved mode: no upfront NT tokens; they are injected during generation.
        front_prompt_active = latent_active or soft_prompt_active
        if front_prompt_active and not interleaved_active:
            nt_toks = torch.tensor([self.nt_token_id] * self.n_latent, device=device).unsqueeze(0)
            start_ids = torch.cat([prefix, nt_toks], dim=1)
        else:
            start_ids = prefix

        attention_mask = (start_ids != pad_token_id).long()
        inputs_embeds = self._get_native_audio_embeds(
            input_ids=start_ids,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            attention_mask=attention_mask,
        )

        states = None
        stats = {}
        if interleaved_active:
            # Interleaved generate: NT tokens injected token-by-token.
            fallback_do_sample = bool(gen_kwargs.get("do_sample", False))
            fallback_temperature = float(gen_kwargs.get("temperature", 1.0))
            gen_ids = self._generate_interleaved(
                base_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=fallback_do_sample,
                temperature=fallback_temperature,
            )
            if return_thoughts:
                return gen_ids, states
            return gen_ids

        if latent_active:
            initial_state, prefix_past_kv, prefix_attn = self._compute_prefix_state(
                inputs_embeds,
                start_ids,
                strict=True,
                language=language_hint or "English",
            )
            deltas, states, stats = self._compute_continuous_thoughts(
                initial_state,
                prefix_past_key_values=prefix_past_kv,
                prefix_attention_mask=prefix_attn,
                halt_threshold=dynamic_halt_threshold,
            )

            K = deltas.size(1)
            if K == 0:
                # N=0 skip: remove ALL NT tokens, completely reverting to base generation.
                remove_count = self.n_latent
                start_ids = start_ids[:, :-remove_count]
                inputs_embeds = inputs_embeds[:, :-remove_count, :]
                attention_mask = attention_mask[:, :-remove_count]
            elif K < self.n_latent:
                # N > 0 but early halted: Remove the unneeded NT tokens instead of zero-padding them.
                # This prevents the LLM from attending to "blank" un-updated base_nt embeddings.
                remove_count = self.n_latent - K
                start_ids = start_ids[:, :-remove_count]
                inputs_embeds = inputs_embeds[:, :-remove_count, :]
                attention_mask = attention_mask[:, :-remove_count]
                inputs_embeds = self._inject_latent_deltas(
                    inputs_embeds, start_ids, deltas
                )
            else:
                inputs_embeds = self._inject_latent_deltas(
                    inputs_embeds, start_ids, deltas
                )
        elif soft_prompt_active:
            nt_mask = (start_ids == self.nt_token_id)
            nt_counts = nt_mask.sum(dim=1)
            assert (nt_counts == self.n_latent).all(), f"Gen NT count mismatch! {nt_counts}"
            prompt = self.soft_prompt_embed.to(dtype=inputs_embeds.dtype)
            batch_idx, seq_idx = torch.nonzero(nt_mask, as_tuple=True)
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[batch_idx, seq_idx, :] = prompt.view(-1, self._hidden_size)

        if front_prompt_active:
            is_nt = (start_ids == self.nt_token_id)
            nt_counts = is_nt.sum(dim=1)
            assert (nt_counts <= self.n_latent).all(), f"Gen NT count mismatch! {nt_counts}"

        fallback_eos = gen_kwargs.get("eos_token_id", None)
        fallback_do_sample = bool(gen_kwargs.get("do_sample", False))
        fallback_temperature = float(gen_kwargs.get("temperature", 1.0))

        try:
            if latent_active and K == 0:
                # N=0 skip -> we stripped the NT tokens from start_ids.
                # Do NOT pass inputs_embeds so Qwen generates organically from text + audio ids.
                # Must provide input_features so audio encoder handles the raw audio.
                gen_out = self.thinker.generate(
                    input_ids=start_ids,
                    attention_mask=attention_mask,
                    input_features=input_features,
                    feature_attention_mask=feature_attention_mask,
                    pad_token_id=pad_token_id,
                    max_new_tokens=max_new_tokens,
                    **gen_kwargs,
                )
            else:
                gen_out = self.thinker.generate(
                    inputs_embeds=inputs_embeds,
                    input_ids=start_ids,
                    attention_mask=attention_mask,
                    input_features=None,
                    pad_token_id=pad_token_id,
                    max_new_tokens=max_new_tokens,
                    **gen_kwargs,
                )
            start_len = start_ids.size(1)
            gen_ids = gen_out[:, start_len:]
        except ValueError as e:
            msg = str(e)
            if "inputs_embeds" not in msg:
                raise
            if not self._warned_generate_fallback:
                print(
                    "[warn] thinker.generate does not support inputs_embeds; "
                    "using fallback decoding loop."
                )
                self._warned_generate_fallback = True
            gen_ids = self._generate_with_forward_fallback(
                start_ids=start_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                eos_token_id=fallback_eos,
                do_sample=fallback_do_sample,
                temperature=fallback_temperature,
            )
        if return_stats:
            if return_thoughts and latent_active:
                return gen_ids, states, stats
            return gen_ids, stats

        if return_thoughts and latent_active:
            return gen_ids, states
        return gen_ids
