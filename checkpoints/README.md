# Checkpoints

This directory contains the released LatentASR adapter checkpoint:

- `latentASR_adapter.pth`

The checkpoint stores only the lightweight LatentASR modules:

- `init_proj`
- `delta_proj`
- `step_proj`
- `step_embed`
- `log_scale`
- `value_head`
- `injection_gate`

It does not include the frozen `Qwen/Qwen3-ASR-0.6B` backbone. Evaluation and
inference load the base model from Hugging Face and then attach this adapter.

Checkpoint metadata:

- Source run: `activation_500_epoch10.pth`
- Base model: `Qwen/Qwen3-ASR-0.6B`
- Latent budget: `N=4`
- Training set size: 500 utterances
- Adapter parameters: 5,251,077
- SHA256: `f0ce39fa5e6952fced6992508f3e2b32ea8467442b545678781a0f04e64f2430`

