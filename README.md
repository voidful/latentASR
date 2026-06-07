---
license: apache-2.0
library_name: transformers
base_model: Qwen/Qwen3-ASR-0.6B
tags:
  - automatic-speech-recognition
  - qwen3-asr
  - latent-reasoning
  - test-time-compute
  - parameter-efficient
datasets:
  - google/fleurs
  - facebook/voxpopuli
metrics:
  - wer
  - cer
---

# latentASR

`latentASR` is the release repo for **Listen, Think, Transcribe: Continuous
Latent Test-Time Scaling for ASR**. The method adds a small trainable latent
adapter and value head on top of a frozen `Qwen/Qwen3-ASR-0.6B` backbone.

The released checkpoint is an adapter-only artifact. It does not redistribute
the Qwen3-ASR base weights.

## What Is Included

- Core model/training/evaluation code: `model.py`, `train.py`, `eval.py`.
- Paper-aligned adapter checkpoint: `checkpoints/latentASR_adapter.pth`.
- Experiment runners: `experiments/`.
- Analysis and summarization scripts: `scripts/`.
- Reproducibility artifacts: `results/`.
- Documentation: `docs/`.

## Method Summary

LatentASR inserts `N=4` latent prefix positions before transcript generation.
At each latent step, a frozen decoder hidden state is projected into a bounded
delta, gated, and added around a fixed latent-token embedding. A value head
predicts whether further latent computation is useful and halts dynamically.

The frozen ASR backbone is never updated in the LatentASR setting. The adapter
has 5,251,077 trainable parameters.

## Install

```bash
git clone https://huggingface.co/voidful/latentASR
cd latentASR
python -m pip install -r requirements.txt
```

The code expects the `qwen-asr` package, which provides
`Qwen3ASRModel.from_pretrained`.

## Quick Evaluation

Evaluate the released adapter on FLEURS English:

```bash
python eval.py \
  --model-id Qwen/Qwen3-ASR-0.6B \
  --dataset-name google/fleurs \
  --configs en_us \
  --split test \
  --latent-ckpt checkpoints/latentASR_adapter.pth \
  --dynamic-halt-threshold 0.0 \
  --skip-baseline-ft \
  --skip-prompt-tuning \
  --skip-lora-r16 \
  --output-json eval_runs/fleurs_en_us.json
```

Use `--max-samples-per-config 100` for a quick smoke test.

## Single-File Transcription

```bash
python examples/transcribe_file.py path/to/audio.wav \
  --latent-ckpt checkpoints/latentASR_adapter.pth \
  --theta 0.0
```

## Training

The paper-aligned minimal-data activation setting uses:

- `N_LATENT=4`
- `TRAIN_MAX_SAMPLES=500`
- effective batch size 16
- 10 epochs
- value forced-negative probability `0.3`
- latent input noise `0.5`
- latent-loop dropout `0.15`

Run the maintained sequential baseline/adapter trainer:

```bash
./experiments/run_all_training_modes.sh
```

Run the paper ablation checkpoint retraining suite:

```bash
./experiments/run_paper_tbd_retrain.sh
```

## Result Scope

The clean benchmark deltas are single-run point estimates. They are reported as
directional, not statistically established, and should be confirmed with paired
multi-seed testing before making strong claims.

ASCEND is included only as an overlap-confounded in-domain diagnostic because
ASCEND contributes 5/500 activation utterances and speaker-disjointness was not
enforced.

## Important Files

- `docs/METHOD.md`: architecture and loss details.
- `docs/REPRODUCIBILITY.md`: exact commands and evaluation workflow.
- `docs/RESULTS.md`: headline tables and caveats.
- `model_metadata.json`: released checkpoint metadata.

