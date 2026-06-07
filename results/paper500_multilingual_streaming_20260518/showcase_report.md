# LR HuggingFace ASR Showcase Report

- Generated UTC: 2026-05-19T08:59:14+00:00
- Output directory: `/user_data/lr_whisper/eval_runs/paper500_multilingual_streaming_20260518`
- Delta WER is `base_model_wer - latent_reasoning_wer`; positive means LR is better.

## Best LR Wins

| Rank | Dataset | Configs | Condition | N | Base WER | LR WER | Delta WER | Relative | Delta CER | JSON |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | google/fleurs | cs_cz,da_dk,el_gr,fa_ir,fi_fi,fil_ph,hu_hu,mk_mk,ro_ro,sv_se | clean | 8576 | 0.435226 | 0.434236 | 0.000990 | +0.23% | 0.000238 | `fleurs_extra10_streaming_clean.json` |
| 2 | google/fleurs | en_us,cmn_hans_cn,yue_hant_hk,ar_eg,de_de,es_419,fr_fr,it_it,ja_jp,ko_kr,pt_br,ru_ru | clean | 8876 | 0.317084 | 0.316275 | 0.000809 | +0.26% | 0.000848 | `fleurs_core12_streaming_clean.json` |
| 3 | google/fleurs | hi_in,id_id,ms_my,nl_nl,pl_pl,th_th,tr_tr,vi_vn | clean | 5597 | 0.174793 | 0.174469 | 0.000324 | +0.19% | 0.000177 | `fleurs_extra8_streaming_clean.json` |

## All Cases

| Dataset | Configs | Condition | N | Base WER | LR WER | Delta WER | Relative | Base CER | LR CER | Delta CER | JSON |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| facebook/multilingual_librispeech | german,dutch,spanish,french,italian,polish,portuguese | clean | 13933 | 0.122987 | 0.123163 | -0.000176 | -0.14% | 0.038337 | 0.038650 | -0.000312 | `mls_public7_streaming_clean.json` |
| google/fleurs | cs_cz,da_dk,el_gr,fa_ir,fi_fi,fil_ph,hu_hu,mk_mk,ro_ro,sv_se | clean | 8576 | 0.435226 | 0.434236 | 0.000990 | +0.23% | 0.168194 | 0.167956 | 0.000238 | `fleurs_extra10_streaming_clean.json` |
| google/fleurs | en_us,cmn_hans_cn,yue_hant_hk,ar_eg,de_de,es_419,fr_fr,it_it,ja_jp,ko_kr,pt_br,ru_ru | clean | 8876 | 0.317084 | 0.316275 | 0.000809 | +0.26% | 0.126493 | 0.125644 | 0.000848 | `fleurs_core12_streaming_clean.json` |
| google/fleurs | hi_in,id_id,ms_my,nl_nl,pl_pl,th_th,tr_tr,vi_vn | clean | 5597 | 0.174793 | 0.174469 | 0.000324 | +0.19% | 0.069216 | 0.069039 | 0.000177 | `fleurs_extra8_streaming_clean.json` |

## Regressions To Check

| Dataset | Configs | Condition | N | Delta WER | Relative | JSON |
|---|---|---|---:|---:|---:|---|
| facebook/multilingual_librispeech | german,dutch,spanish,french,italian,polish,portuguese | clean | 13933 | -0.000176 | -0.14% | `mls_public7_streaming_clean.json` |

