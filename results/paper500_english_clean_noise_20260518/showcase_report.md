# LR HuggingFace ASR Showcase Report

- Generated UTC: 2026-05-18T20:13:39+00:00
- Output directory: `/user_data/lr_whisper/eval_runs/paper500_english_clean_noise_20260518`
- Delta WER is `base_model_wer - latent_reasoning_wer`; positive means LR is better.

## Best LR Wins

| Rank | Dataset | Configs | Condition | N | Base WER | LR WER | Delta WER | Relative | Delta CER | JSON |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | google/fleurs | en_us | snr0db | 647 | 0.298137 | 0.294203 | 0.003934 | +1.32% | 0.003533 | `fleurs_en_us_snr0db.json` |
| 2 | facebook/voxpopuli | en | snr0db | 1842 | 0.197111 | 0.195108 | 0.002003 | +1.02% | -0.000284 | `voxpopuli_en_snr0db.json` |
| 3 | facebook/voxpopuli | en | clean | 1842 | 0.090375 | 0.089745 | 0.000630 | +0.70% | 0.000427 | `voxpopuli_en_clean.json` |
| 4 | google/fleurs | en_us | clean | 647 | 0.048999 | 0.048585 | 0.000414 | +0.85% | 0.000573 | `fleurs_en_us_clean.json` |
| 5 | PolyAI/minds14 | en-US | clean | 563 | 0.337011 | 0.336773 | 0.000237 | +0.07% | 0.000154 | `minds14_en_us_clean.json` |

## All Cases

| Dataset | Configs | Condition | N | Base WER | LR WER | Delta WER | Relative | Base CER | LR CER | Delta CER | JSON |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| PolyAI/minds14 | en-US | clean | 563 | 0.337011 | 0.336773 | 0.000237 | +0.07% | 0.295093 | 0.294939 | 0.000154 | `minds14_en_us_clean.json` |
| facebook/voxpopuli | en | clean | 1842 | 0.090375 | 0.089745 | 0.000630 | +0.70% | 0.059004 | 0.058576 | 0.000427 | `voxpopuli_en_clean.json` |
| google/fleurs | en_us | clean | 647 | 0.048999 | 0.048585 | 0.000414 | +0.85% | 0.023260 | 0.022687 | 0.000573 | `fleurs_en_us_clean.json` |
| PolyAI/minds14 | en-US | snr0db | 563 | 0.402491 | 0.404864 | -0.002372 | -0.59% | 0.333086 | 0.333368 | -0.000282 | `minds14_en_us_snr0db.json` |
| facebook/voxpopuli | en | snr0db | 1842 | 0.197111 | 0.195108 | 0.002003 | +1.02% | 0.120512 | 0.120796 | -0.000284 | `voxpopuli_en_snr0db.json` |
| google/fleurs | en_us | snr0db | 647 | 0.298137 | 0.294203 | 0.003934 | +1.32% | 0.192169 | 0.188636 | 0.003533 | `fleurs_en_us_snr0db.json` |

## Regressions To Check

| Dataset | Configs | Condition | N | Delta WER | Relative | JSON |
|---|---|---|---:|---:|---:|---|
| PolyAI/minds14 | en-US | snr0db | 563 | -0.002372 | -0.59% | `minds14_en_us_snr0db.json` |

