# LatentASR

LatentASR adds continuous latent test-time scaling to a frozen Qwen3-ASR
backbone. A lightweight adapter refines a small prefix of latent tokens before
transcription, and a value head dynamically halts the latent loop when extra
compute is not useful.

The repository is organized around reproducible ASR experiments for the paper:
clean English evaluation, multilingual streaming evaluation, robustness/noise
evaluation, threshold sweeps, and baseline adaptation comparisons.

## Repository Layout

- `model.py`, `train.py`, `eval.py`: core LatentASR model, training, and evaluation code.
- `config.py`, `data.py`, `losses.py`, `peft_utils.py`, `utils.py`: shared training/evaluation utilities.
- `experiments/`: maintained experiment entrypoints.
- `scripts/`: report generation, summarization, analysis, and hyperparameter helpers.
- `METHODOLOGY.md`: method details.

Generated outputs, checkpoints, and local caches are intentionally ignored by
Git. The private GitHub repository contains code and documentation, not large
model artifacts.

## Main Experiment Entrypoints

The root-level scripts are compatibility wrappers. The maintained scripts live
under `experiments/`.

```bash
# English clean suite plus optional SNR noise sweeps
./run_lr_hf_asr_showcase.sh

# FLEURS 30 and MLS public7 with HuggingFace streaming
./run_lr_multilingual_asr_streaming.sh

# Value-head halting threshold sweep
./run_threshold_sweep.sh

# Sequential training of baseline, prompt tuning, LoRA, and LatentASR
./run_all_modes.sh
```

Common environment variables:

```bash
LATENT_CKPT=./latent_qwen_asr_best.pth
MODEL_ID=Qwen/Qwen3-ASR-0.6B
MAX_SAMPLES_PER_CONFIG=0
OUT_DIR=./eval_runs/my_run
```

Use streaming for large HuggingFace datasets to avoid materializing full audio
splits on disk:

```bash
MAX_SAMPLES_PER_CONFIG=0 CASE_FILTER='' ./run_lr_multilingual_asr_streaming.sh
```

## Paper Reports

Summaries are generated from existing JSON outputs:

```bash
python scripts/summarize_lr_showcase.py eval_runs/hf_asr_showcase_full_20260503_152506
python scripts/summarize_threshold_sweep.py eval_runs/threshold_sweep_20260504_0849
python scripts/write_full_asr_experiment_report.py
```

## Artifact Policy

The following are excluded from Git:

- `*.pth`, `*.bin`, `*.safetensors`, adapter/checkpoint directories.
- `eval_runs/`, `logs/`, cache folders, and core dumps.
- local agent/editor metadata.

Place checkpoints in the repository root when running experiments locally, or
set `LATENT_CKPT` to an external path.
