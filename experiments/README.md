# Experiment Runners

This directory contains the maintained entrypoints for LatentASR experiments.
All runners default to Qwen/Qwen3-ASR-0.6B and `./latent_qwen_asr_best.pth`.
The checkpoint itself is intentionally ignored by Git.

## Runners

- `run_english_clean_and_noise.sh`: English clean suite plus optional SNR noise sweeps.
- `run_multilingual_streaming.sh`: FLEURS 30 and MLS public7 streaming evaluation.
- `run_threshold_sweep.sh`: Value-head halting threshold sweep.
- `run_all_training_modes.sh`: Sequential baseline, prompt tuning, LoRA, and LatentASR training.
- `run_legacy_generalization.sh`: older SpeechTest generalization/noise runner kept for reproducibility.

Root-level scripts with the old names are thin compatibility wrappers around
these maintained files.

## Common Environment Variables

- `LATENT_CKPT`: latent adapter checkpoint path.
- `MODEL_ID`: HuggingFace model ID.
- `PYTHON_BIN`: Python executable override.
- `MAX_SAMPLES_PER_CONFIG`: per-config cap; `0` means full split.
- `OUT_DIR`: output directory.
- `RESUME=1`: skip JSON outputs that already exist.

Full experimental JSON/log outputs are written under `eval_runs/` and are not
committed to Git.
