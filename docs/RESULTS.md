# Results

This release includes JSON and Markdown outputs under `results/`. The most
important caveat is that the clean benchmark gains are single-run directional
point estimates, not statistically established improvements.

## Clean Main Benchmarks

| Model | FLEURS WER | VoxPopuli WER |
| --- | ---: | ---: |
| Frozen Qwen3-ASR baseline | 4.900 | 9.038 |
| LatentASR, N=4, theta=0 | 4.776 | 8.995 |

Relative WER deltas:

- FLEURS: `-2.54%`
- VoxPopuli: `-0.47%`

## Robustness Diagnostics

At SNR `0` dB:

| Dataset | Baseline WER | LatentASR WER | Relative delta |
| --- | ---: | ---: | ---: |
| FLEURS | 29.81 | 29.42 | -1.32% |
| VoxPopuli | 19.71 | 19.51 | -1.02% |

Stress broad suite weighted aggregate:

- Baseline WER: `36.843`
- LatentASR WER: `36.265`
- Delta: `-0.578` pp (`-1.57%` relative)

## ASCEND Diagnostic

ASCEND CER:

- Baseline: `57.81`
- LatentASR: `48.55`
- Relative CER delta: `-16.02%`

This is explicitly treated as an overlap-confounded in-domain diagnostic.
ASCEND contributes 5/500 activation utterances and speaker-disjointness was not
enforced.

## Included Result Folders

- `results/paper500_main_eval_20260518/`
- `results/paper500_english_clean_noise_20260518/`
- `results/paper500_multilingual_streaming_20260518/`
- `results/paper500_threshold_20260518/`
- `results/paper500_more_testing_stress_20260525/`
- `results/paper500_ascend_20260518/`

