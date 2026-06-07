"""Upload the latentASR project and adapter checkpoint to Hugging Face Hub."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload latentASR to Hugging Face Hub.")
    parser.add_argument("--repo-id", default="voidful/latentASR")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--commit-message",
        default="Release latentASR adapter, code, docs, and reproducibility artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(project_root),
        revision=args.revision,
        commit_message=args.commit_message,
        ignore_patterns=[
            ".git/*",
            "__pycache__/*",
            "*.pyc",
            ".pytest_cache/*",
            "eval_runs/*",
            "logs/*",
            "wandb/*",
        ],
    )
    print(f"Uploaded {project_root} to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()

