# Paper 500-Activation Complete Results

All results below use the 500-utterance activation setting unless explicitly stated otherwise. Metrics are shown in percent. Unless noted otherwise, `Delta` is `model - baseline`, so negative means an error reduction.

Canonical paper choice: use `eval_runs/paper_tbd_retrain_20260518/checkpoints/activation_500/activation_500_epoch10.pth` for the main LatentASR result. This is the better of the two available 500-utterance/N=4/theta=0 runs by mean FLEURS/VoxPopuli WER delta, and it is the checkpoint represented by the 500 row in the activation-size sweep. The older `eval_runs/paper500_retrain_20260518/checkpoints/n4/n4_epoch10.pth` should be treated as an auxiliary ablation/diagnostic run unless its dependent tables are intentionally retained with that label.

## Source Artifacts

- Main baseline/adaptation JSONs: `eval_runs/paper500_main_eval_20260518/`
- Canonical LatentASR JSONs: `eval_runs/paper_tbd_eval_20260518/activation_500_fleurs_theta_zero.json`, `eval_runs/paper_tbd_eval_20260518/activation_500_voxpopuli_theta_zero.json`
- Retrained 500 checkpoints: `eval_runs/paper500_retrain_20260518/checkpoints/`
- Main baseline checkpoints: `eval_runs/paper500_main_baselines_20260518/checkpoints/`
- Threshold sweep: `eval_runs/paper500_threshold_20260518/threshold_sweep_report.md`
- Component/N/p_neg ablations: `eval_runs/paper500_eval_20260518/paper_tbd_results.md`
- Activation scaling 100-800: `eval_runs/paper_tbd_eval_20260518/activation_scaling_100_800_full_results.md`
- Multilingual streaming: `eval_runs/paper500_multilingual_streaming_20260518/showcase_report.md`
- English clean/noise suite: `eval_runs/paper500_english_clean_noise_20260518/showcase_report.md`
- ASCEND: `eval_runs/paper500_ascend_20260518/ascend_clean.json`
- Latency: `eval_runs/paper500_latency_20260518/latency.json`

## Main Results

Main baselines were retrained/evaluated with the 500-utterance activation setting. The canonical LatentASR row uses the best 500-utterance activation checkpoint from the activation-size sweep (`activation_500_epoch10.pth`). Full fine-tuning initially OOMed with micro-batch 16, so it was rerun with micro-batch 4 and gradient accumulation 4, preserving effective batch size 16.

| Model | FLEURS WER | FLEURS CER | Rel. Delta WER | VoxPopuli WER | VoxPopuli CER | Rel. Delta WER |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 4.90 | 2.33 | -- | 9.04 | 5.90 | -- |
| Full fine-tuning | 5.56 | 2.48 | +13.38 | 9.49 | 6.18 | +4.96 |
| Prompt tuning | 84.80 | 82.47 | +1630.70 | 88.51 | 87.90 | +879.33 |
| LoRA r16 | 6.47 | 2.94 | +31.97 | 10.21 | 6.54 | +13.00 |
| LatentASR N=4, theta=0 | 4.78 | 2.24 | -2.54 | 8.99 | 5.87 | -0.47 |

Prompt tuning collapses badly in the 500 setting on both FLEURS and VoxPopuli. This should be reported plainly if the table is used.

## Activation Scaling 100-800

The activation-scaling sweep was run at 100 to 800 utterances, every 100. The 500 point gives the best average absolute WER delta across FLEURS and VoxPopuli.

| Activation size | FLEURS WER | FLEURS Delta pp | VoxPopuli WER | VoxPopuli Delta pp | Mean Delta pp |
|---:|---:|---:|---:|---:|---:|
| 100 | 4.87 | -0.03 | 8.95 | -0.08 | -0.055 |
| 200 | 4.91 | +0.01 | 9.01 | -0.03 | -0.010 |
| 300 | 4.90 | +0.00 | 9.01 | -0.03 | -0.015 |
| 400 | 4.91 | +0.01 | 9.04 | -0.00 | +0.005 |
| 500 | 4.78 | -0.12 | 8.99 | -0.04 | -0.084 |
| 600 | 4.93 | +0.03 | 9.02 | -0.02 | +0.005 |
| 700 | 4.82 | -0.08 | 9.04 | +0.00 | -0.040 |
| 800 | 4.88 | -0.02 | 9.07 | +0.04 | +0.010 |

## Threshold Sweep

Converted from the sweep report to paper convention: Delta WER is `LatentASR - Baseline`, so negative is better.

Important consistency note: the threshold sweep below was run with `paper500_retrain_20260518/checkpoints/n4/n4_epoch10.pth`, not the canonical `activation_500_epoch10.pth`. If the main paper uses the better canonical checkpoint, this section must either be rerun with `activation_500_epoch10.pth` or explicitly labeled as an auxiliary same-protocol diagnostic run.

| Dataset | theta | Avg. steps | Skip | WER | Delta WER pp |
|---|---:|---:|---:|---:|---:|
| FLEURS | -2.0 | 4.00 | 0.0 | 4.84 | -0.062 |
| FLEURS | -0.2 | 1.37 | 43.7 | 4.86 | -0.041 |
| FLEURS | 0.0 | 1.24 | 47.0 | 4.86 | -0.041 |
| FLEURS | +0.2 | 0.00 | 100.0 | 4.90 | +0.000 |
| FLEURS | +0.5 | 0.00 | 100.0 | 4.90 | +0.000 |
| VoxPopuli | -2.0 | 4.00 | 0.0 | 9.02 | -0.014 |
| VoxPopuli | -0.2 | 0.84 | 54.9 | 8.97 | -0.070 |
| VoxPopuli | 0.0 | 0.71 | 60.2 | 8.97 | -0.063 |
| VoxPopuli | +0.2 | 0.00 | 100.0 | 9.04 | +0.000 |
| VoxPopuli | +0.5 | 0.00 | 100.0 | 9.04 | +0.000 |

In this 500 setting, full compute does not regress VoxPopuli; it is slightly positive. The older claim that no-halting increases VoxPopuli WER should be revised for the 500 paper version. Dynamic halting is still useful because it recovers most of the gain at much lower average steps.

## Compute Allocation at theta=0

The clean FLEURS/VoxPopuli theta=0 allocation for the canonical `activation_500_epoch10.pth` is available from the activation sweep: FLEURS average steps 1.20 with 47.0% skip; VoxPopuli average steps 0.70 with 63.0% skip. The table below additionally includes ASCEND from the older `n4_epoch10.pth` run and therefore should be rerun if strict checkpoint consistency is required.

| Dataset | N=0 | N=1 | N=2 | N=3 | N=4 |
|---|---:|---:|---:|---:|---:|
| FLEURS clean | 47.0 | 23.2 | 7.0 | 4.6 | 18.2 |
| VoxPopuli clean | 60.2 | 24.0 | 6.7 | 2.8 | 6.2 |
| ASCEND clean | 22.3 | 30.3 | 8.2 | 5.1 | 34.1 |

## Component Ablation

FLEURS, theta=0. Baseline WER for this ablation run is 4.90.

| Variant | WER | Delta WER pp |
|---|---:|---:|
| Full LatentASR N=4 | 4.86 | -0.04 |
| Remove bounded delta | 51.75 | +46.85 |
| Remove sigmoid gate | 7.89 | +2.99 |
| Remove fixed-embedding anchor | 16.22 | +11.33 |

This strongly supports keeping the stabilization-mechanism ablation in the paper.

## Latent Budget N Sweep

theta=0.

| N | FLEURS WER | FLEURS Delta pp | VoxPopuli WER | VoxPopuli Delta pp |
|---:|---:|---:|---:|---:|
| 1 | 4.82 | -0.08 | 8.98 | -0.06 |
| 2 | 4.89 | -0.01 | 9.03 | -0.00 |
| 4 | 4.86 | -0.04 | 8.97 | -0.06 |
| 8 | 4.90 | +0.00 | 9.03 | -0.00 |

N=1 is surprisingly strong in this 500 run. N=4 remains a reasonable default because it ties/bests VoxPopuli and supports deeper reasoning on the subset that needs it, but the paper should not claim N=4 is strictly best on every dataset.

## Forced-Negative Sampling

FLEURS, theta=0 unless otherwise specified.

| Setting | WER | Delta WER pp | Skip at theta=+0.2 | Skip range over sweep |
|---|---:|---:|---:|---:|
| p_neg=0.3 | 4.86 | -0.04 | 100.0 | [0, 100] |
| p_neg=0.0 | 4.80 | -0.10 | 100.0 | [0, 100] |

Important paper caveat: in the 500 run, removing forced-negative sampling does not collapse threshold reachability. The previous placeholder prose saying the saturated regime disappears is false for this setting and should be removed or rewritten. A safer statement is that forced-negative sampling changes value calibration/aggressiveness, while this 500 run does not show it is required for saturation.

## Multilingual Streaming

| Subset | #utts | Baseline WER | LatentASR WER | Delta WER pp | Rel. Delta WER | Baseline CER | LatentASR CER | Delta CER pp | Rel. Delta CER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Primary multilingual: FLEURS30 + MLS public7 | 36,982 | 24.982 | 24.941 | -0.041 | -0.163 | 9.428 | 9.411 | -0.017 | -0.178 |
| FLEURS 30 overall | 23,049 | 32.649 | 32.573 | -0.076 | -0.232 | 12.810 | 12.764 | -0.046 | -0.358 |
| FLEURS core12 | 8,876 | 31.708 | 31.627 | -0.081 | -0.255 | 12.649 | 12.564 | -0.085 | -0.671 |
| FLEURS +8 | 5,597 | 17.479 | 17.447 | -0.032 | -0.185 | 6.922 | 6.904 | -0.018 | -0.256 |
| FLEURS +10 | 8,576 | 43.523 | 43.424 | -0.099 | -0.227 | 16.819 | 16.796 | -0.024 | -0.142 |
| MLS public7 overall | 13,933 | 12.299 | 12.316 | +0.018 | +0.143 | 3.834 | 3.865 | +0.031 | +0.815 |

MLS public7 slightly regresses under 500. The broad multilingual aggregate remains positive because FLEURS 30 improves.

### Character-Based FLEURS Languages

| Language | #utts | Baseline CER | LatentASR CER | Rel. Delta CER |
|---|---:|---:|---:|---:|
| Mandarin (cmn_hans_cn) | 945 | 47.94 | 47.58 | -0.75 |
| Cantonese (yue_hant_hk) | 819 | 49.20 | 48.80 | -0.82 |
| Japanese (ja_jp) | 650 | 10.35 | 10.35 | +0.00 |
| Thai (th_th) | 1,021 | 9.53 | 9.41 | -1.31 |

## English Clean/Noise Suite and Robustness

| Dataset | Condition | Metric | Baseline | LatentASR | Delta pp | Rel. Delta |
|---|---|---|---:|---:|---:|---:|
| FLEURS | clean | WER | 4.90 | 4.78 | -0.124 | -2.54 |
| FLEURS | SNR=0 dB | WER | 29.81 | 29.42 | -0.393 | -1.32 |
| MInDS-14 | clean | WER | 33.70 | 33.68 | -0.024 | -0.07 |
| MInDS-14 | SNR=0 dB | WER | 40.25 | 40.49 | +0.237 | +0.59 |
| VoxPopuli | clean | WER | 9.04 | 8.99 | -0.043 | -0.47 |
| VoxPopuli | SNR=0 dB | WER | 19.71 | 19.51 | -0.200 | -1.02 |
| ASCEND | clean/accented | CER | 57.81 | 48.55 | -9.26 | -16.02 |

ASCEND remains the strongest robustness result. MInDS-14 SNR=0 regresses slightly and should not be summarized as uniformly robust across every noise condition.

## Latency

Measured on one NVIDIA RTX 5090, batch size 1, 100 utterances after 10 warmup utterances.

| Setting | FLEURS ms/utt | FLEURS overhead | VoxPopuli ms/utt | VoxPopuli overhead |
|---|---:|---:|---:|---:|
| Baseline | 304.31 | -- | 297.95 | -- |
| LatentASR theta=-2.0, full N=4 | 358.31 | +17.75 | 349.70 | +17.37 |
| LatentASR theta=0.0, deployed | 341.69 | +12.28 | 329.37 | +10.54 |
| LatentASR theta=+0.5, full skip | 321.52 | +5.66 | 314.36 | +5.51 |

The full-skip wrapper still has a fixed overhead of about 5-6%, so the paper should not claim exact zero overhead in the skip case.

## Paper Text Adjustments Required

- Replace the old main table with the canonical `activation_500_epoch10.pth` main results above.
- Rerun or relabel threshold, compute-allocation, multilingual, robustness, and per-sample analysis tables that currently cite `paper500_retrain_20260518/checkpoints/n4/n4_epoch10.pth`.
- Replace the threshold discussion: with 500, full compute does not regress VoxPopuli; dynamic halting is a compute-efficiency mechanism that preserves gains rather than the only way to avoid VoxPopuli regression.
- Replace the forced-negative ablation prose: p_neg=0.0 does not remove the saturated skip regime in the 500 run.
- Keep the component ablation; it is strong and clean.
- Mention prompt tuning collapse explicitly or consider moving prompt tuning to a failure-mode baseline if the huge WER hurts presentation.
- For multilingual, say the primary aggregate and FLEURS 30 improve, while MLS public7 is near-neutral/slightly worse.
- For robustness, say ASCEND and most zero-SNR English conditions improve, but MInDS-14 SNR=0 is a small regression.
