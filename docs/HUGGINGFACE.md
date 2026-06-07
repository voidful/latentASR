# Hugging Face Release

The public model repo is intended to be:

```text
voidful/latentASR
```

The repository hosts:

- LatentASR code
- adapter checkpoint
- documentation
- model card
- reproducibility outputs

## Upload

From this project root:

```bash
python hf_upload/upload_to_hf.py --repo-id voidful/latentASR
```

The script creates the model repo if needed and uploads the current folder.

## Download Checkpoint Programmatically

```python
from huggingface_hub import hf_hub_download

ckpt = hf_hub_download(
    repo_id="voidful/latentASR",
    filename="checkpoints/latentASR_adapter.pth",
)
print(ckpt)
```

