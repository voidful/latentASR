# Reproducibility

This document describes how to run the released code and adapter checkpoint.

## Environment

```bash
python -m pip install -r requirements.txt
```

Recommended runtime:

- Python 3.10+
- CUDA GPU with bfloat16 support for practical speed
- `qwen-asr>=0.0.6`

## Released Adapter

The released checkpoint is:

```text
checkpoints/latentASR_adapter.pth
```

It was copied from:

```text
activation_500_epoch10.pth
```

Metadata:

- Base model: `Qwen/Qwen3-ASR-0.6B`
- Latent budget: `N=4`
- Activation size: 500 utterances
- Adapter parameters: 5,251,077
- SHA256: `f0ce39fa5e6952fced6992508f3e2b32ea8467442b545678781a0f04e64f2430`

## Main Evaluation

FLEURS English:

```bash
python eval.py \
  --model-id Qwen/Qwen3-ASR-0.6B \
  --dataset-name google/fleurs \
  --configs en_us \
  --split test \
  --latent-ckpt checkpoints/latentASR_adapter.pth \
  --dynamic-halt-threshold 0.0 \
  --max-samples-per-config 0 \
  --skip-baseline-ft \
  --skip-prompt-tuning \
  --skip-lora-r16 \
  --output-json eval_runs/fleurs_en_us.json
```

VoxPopuli English:

```bash
python eval.py \
  --model-id Qwen/Qwen3-ASR-0.6B \
  --dataset-name facebook/voxpopuli \
  --configs en \
  --split test \
  --latent-ckpt checkpoints/latentASR_adapter.pth \
  --dynamic-halt-threshold 0.0 \
  --max-samples-per-config 0 \
  --skip-baseline-ft \
  --skip-prompt-tuning \
  --skip-lora-r16 \
  --output-json eval_runs/voxpopuli_en.json
```

## Threshold Sweep

```bash
LATENT_CKPT=checkpoints/latentASR_adapter.pth \
./experiments/run_threshold_sweep.sh
```

## Minimal-Data Activation Training

```bash
TRAIN_MODE=latent \
N_LATENT=4 \
TRAIN_MAX_SAMPLES=500 \
BATCH_SIZE=4 \
GRAD_ACCUM_STEPS=4 \
NUM_EPOCHS=10 \
python train.py
```

The sequential comparison runner defaults to the paper-aligned protocol:

```bash
./experiments/run_all_training_modes.sh
```

## Notes

- The clean benchmark improvements are small single-run point estimates.
- The ASCEND diagnostic is overlap-confounded and should not be treated as
  speaker-independent generalization evidence.
- Results in `results/` are included for transparency and summarization, not as
  a substitute for rerunning evaluation in your environment.

