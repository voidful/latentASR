# Method

LatentASR adds continuous latent test-time scaling to a frozen ASR backbone.
The implementation wraps `Qwen/Qwen3-ASR-0.6B` with two lightweight modules:

- **Latent Adapter**: converts decoder hidden states into bounded latent
  deltas.
- **Value Head**: predicts whether additional latent compute is useful and
  enables dynamic halting.

## Prefix Layout

The default layout is prefix mode:

```text
system/user/audio prompt
assistant language <asr_text>
<|latent|> <|latent|> <|latent|> <|latent|>
transcript tokens
```

Transcript loss masks all prompt and latent positions.

## Stable Injection

Each latent delta is stabilized by three mechanisms:

1. **Bounded delta**: `delta_proj(h_k)` is L2-normalized and scaled by a
   learned per-step scalar.
2. **Sigmoid gate**: a zero-initialized gate starts at `0.5` and learns how
   much of the delta to apply.
3. **Fixed embedding anchor**: the injected vector is
   `embedding(<|latent|>) + gate * delta`, keeping the input near a real token
   embedding.

These mechanisms correspond to the ablation flags:

```bash
LATENT_USE_BOUNDED_DELTA=1
LATENT_USE_INJECTION_GATE=1
LATENT_USE_EMBEDDING_ANCHOR=1
```

## Value Head Target

For each training utterance, the value head predicts the latent-vs-baseline
accuracy gain. The default target is:

```text
0.9 * tanh(3.0 * (latent_accuracy - baseline_accuracy))
```

If both baseline and latent accuracies are zero for an utterance, the target
uses a clamped CE-difference fallback:

```text
0.9 * tanh(0.5 * clamp(baseline_ce - latent_ce, -2, 2))
```

With probability `VALUE_FORCED_NEG_PROB=0.3` per minibatch, the target is
flipped to `-|target|` for conservative calibration.

The current implementation supervises the value prediction at every produced
latent state, matching the fact that inference can halt after any step.

## Dynamic Halting

At inference time:

1. The initial value prediction gates the whole loop.
2. If `v_0 < theta`, all latent tokens are removed and generation falls back to
   the frozen baseline.
3. After each latent step, the value head is re-evaluated.
4. If `v_k < theta`, unused latent tokens are removed before decoding.

The paper uses `theta=0.0` for the deployed setting.

