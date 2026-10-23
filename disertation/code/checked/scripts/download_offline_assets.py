"""Download Hugging Face assets needed by the CHECKED training scripts.

Run this on a machine with Hugging Face access before copying the repo to an
offline machine:

    python scripts/download_offline_assets.py

The training and test scripts automatically prefer these local directories when
they exist:

    offline_models/bert-base-chinese
    offline_models/google--vit-base-patch16-224-in21k
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "offline_models"

MODELS = {
    "bert-base-chinese": "bert-base-chinese",
    "google/vit-base-patch16-224-in21k": "google--vit-base-patch16-224-in21k",
}

BASE_ALLOW_PATTERNS = [
    "*.json",
    "*.txt",
    "*.model",
    "*.safetensors",
    "tokenizer.*",
    "vocab.*",
    "merges.txt",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
]

BIN_ALLOW_PATTERNS = [
    "pytorch_model.bin",
    "*.bin",
]


def build_allow_patterns(include_bin: bool) -> list[str]:
    patterns = list(BASE_ALLOW_PATTERNS)
    if include_bin:
        patterns.extend(BIN_ALLOW_PATTERNS)
    return patterns


def has_weight_file(path: Path, include_bin: bool) -> bool:
    if list(path.glob("*.safetensors")):
        return True
    return include_bin and list(path.glob("*.bin"))


def portable_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def download_model(repo_id: str, target_dir: Path, allow_patterns: Iterable[str], token: str | None) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} -> {target_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        allow_patterns=list(allow_patterns),
        token=token,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download local Hugging Face model assets for offline training.")
    parser.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT), help="Destination directory for offline models.")
    parser.add_argument("--include-bin", action="store_true", help="Also download PyTorch .bin weights if available.")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="Optional Hugging Face token.")
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("HF_ENDPOINT"),
        help="Optional Hugging Face endpoint, e.g. https://hf-mirror.com",
    )
    args = parser.parse_args()

    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint

    model_root = Path(args.model_root).resolve()
    allow_patterns = build_allow_patterns(args.include_bin)
    manifest = {
        "model_root": portable_path(model_root),
        "include_bin": bool(args.include_bin),
        "models": [],
    }

    for repo_id, local_name in MODELS.items():
        target_dir = model_root / local_name
        download_model(repo_id, target_dir, allow_patterns, args.token)
        if not has_weight_file(target_dir, args.include_bin):
            raise RuntimeError(
                f"No model weight file was downloaded for {repo_id}. "
                "Re-run with --include-bin or choose a model that provides safetensors."
            )
        manifest["models"].append({"repo_id": repo_id, "local_dir": portable_path(target_dir)})

    manifest_path = model_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")
    print("Offline assets are ready. Copy the repo, including offline_models, to AutoDL.")


if __name__ == "__main__":
    main()
