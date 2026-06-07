# LR HuggingFace ASR Showcase Report

- Generated UTC: 2026-05-25T16:47:09+00:00
- Output directory: `/user_data/lr_whisper/eval_runs/paper500_more_testing_stress_20260525`
- Delta WER is `base_model_wer - latent_reasoning_wer`; positive means LR is better.

## Best LR Wins

| Rank | Dataset | Configs | Condition | N | Base WER | LR WER | Delta WER | Relative | Delta CER | JSON |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | SpeechTest/extreme_asr_pony | default | unknown | 499 | 0.722823 | 0.664954 | 0.057869 | +8.01% | 0.033641 | `extreme_asr_pony_500.json` |
| 2 | edinburghcstr/ami | ihm,sdm | unknown | 996 | 0.777838 | 0.755178 | 0.022660 | +2.91% | 0.009691 | `ami_ihm_sdm_snr0_500.json` |
| 3 | TwinkStart/tedlium | release1 | unknown | 500 | 0.148490 | 0.142912 | 0.005578 | +3.76% | 0.001232 | `tedlium_release1_snr0_500.json` |
| 4 | SpeechTest/gigaspeech | test | unknown | 391 | 0.209385 | 0.208748 | 0.000638 | +0.30% | -0.000125 | `gigaspeech_test_snr0_500.json` |

## All Cases

| Dataset | Configs | Condition | N | Base WER | LR WER | Delta WER | Relative | Base CER | LR CER | Delta CER | JSON |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SpeechTest/common_voice_16_0 | en | unknown | 500 | 0.316173 | 0.319362 | -0.003189 | -1.01% | 0.215902 | 0.213539 | 0.002363 | `common_voice_en_snr0_500.json` |
| SpeechTest/extreme_asr_pony | default | unknown | 499 | 0.722823 | 0.664954 | 0.057869 | +8.01% | 0.414491 | 0.380850 | 0.033641 | `extreme_asr_pony_500.json` |
| SpeechTest/gigaspeech | test | unknown | 391 | 0.209385 | 0.208748 | 0.000638 | +0.30% | 0.128731 | 0.128856 | -0.000125 | `gigaspeech_test_snr0_500.json` |
| SpeechTest/peoples_speech | test | unknown | 500 | 0.316871 | 0.317710 | -0.000839 | -0.26% | 0.224545 | 0.229471 | -0.004926 | `peoples_speech_test_snr0_500.json` |
| TwinkStart/tedlium | release1 | unknown | 500 | 0.148490 | 0.142912 | 0.005578 | +3.76% | 0.088532 | 0.087300 | 0.001232 | `tedlium_release1_snr0_500.json` |
| edinburghcstr/ami | ihm,sdm | unknown | 996 | 0.777838 | 0.755178 | 0.022660 | +2.91% | 0.639126 | 0.629435 | 0.009691 | `ami_ihm_sdm_snr0_500.json` |
| openslr/librispeech_asr | clean,other | unknown | 1000 | 0.184713 | 0.185840 | -0.001127 | -0.61% | 0.112295 | 0.111297 | 0.000998 | `librispeech_clean_other_snr0_500.json` |

## Regressions To Check

| Dataset | Configs | Condition | N | Delta WER | Relative | JSON |
|---|---|---|---:|---:|---:|---|
| SpeechTest/common_voice_16_0 | en | unknown | 500 | -0.003189 | -1.01% | `common_voice_en_snr0_500.json` |
| openslr/librispeech_asr | clean,other | unknown | 1000 | -0.001127 | -0.61% | `librispeech_clean_other_snr0_500.json` |
| SpeechTest/peoples_speech | test | unknown | 500 | -0.000839 | -0.26% | `peoples_speech_test_snr0_500.json` |

