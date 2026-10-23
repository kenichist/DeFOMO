#!/usr/bin/env python3
"""Populate and verify relocatable Hugging Face and torchvision caches."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path


HF_REPOSITORIES = (
    "bert-base-uncased",
    "bert-base-chinese",
    "distilbert-base-uncased",
    "distilbert-base-multilingual-cased",
    "hfl/chinese-roberta-wwm-ext",
    "microsoft/deberta-v3-base",
    "microsoft/mdeberta-v3-base",
    "roberta-base",
    "openai/clip-vit-base-patch32",
    "google/vit-base-patch16-224-in21k",
)

COMMON_MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer*",
    "vocab*",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "spm.model",
    "spiece.model",
    "sentencepiece.bpe.model",
    "preprocessor_config.json",
    "processor_config.json",
)


def required_patterns(repo_id: str) -> list[str]:
    # The HFL checkpoint currently publishes PyTorch BIN weights; the other
    # pinned models publish safetensors. Avoid duplicate TF/ONNX/Flax exports.
    bin_repositories = {
        "hfl/chinese-roberta-wwm-ext",
        "microsoft/deberta-v3-base",
        "microsoft/mdeberta-v3-base",
        "openai/clip-vit-base-patch32",
    }
    weights = ("pytorch_model.bin",) if repo_id in bin_repositories else ("*.safetensors",)
    return [*COMMON_MODEL_FILES, *weights]


def validate_repository(repo_id: str, hub_root: Path) -> None:
    """Load each required object without network access, then release it."""
    if repo_id == "openai/clip-vit-base-patch32":
        from transformers import CLIPModel, CLIPProcessor

        processor = CLIPProcessor.from_pretrained(repo_id, cache_dir=hub_root, local_files_only=True)
        model = CLIPModel.from_pretrained(repo_id, cache_dir=hub_root, local_files_only=True)
    elif repo_id == "google/vit-base-patch16-224-in21k":
        from transformers import AutoImageProcessor, ViTModel

        processor = AutoImageProcessor.from_pretrained(repo_id, cache_dir=hub_root, local_files_only=True)
        model = ViTModel.from_pretrained(repo_id, cache_dir=hub_root, local_files_only=True)
    else:
        from transformers import AutoModel, AutoTokenizer

        processor = AutoTokenizer.from_pretrained(repo_id, cache_dir=hub_root, local_files_only=True)
        model = AutoModel.from_pretrained(repo_id, cache_dir=hub_root, local_files_only=True)
    del model, processor
    gc.collect()


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("benchmarks/offline_assets"))
    parser.add_argument("--models", nargs="+", default=list(HF_REPOSITORIES))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--skip-torchvision", action="store_true")
    parser.add_argument("--verify", action="store_true", help="Verify without downloading anything.")
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    hf_root = output_root / "huggingface"
    hub_root = hf_root / "hub"
    torch_root = output_root / "torch"
    hub_root.mkdir(parents=True, exist_ok=True)
    torch_root.mkdir(parents=True, exist_ok=True)
    os.environ.update(HF_HOME=str(hf_root), HF_HUB_CACHE=str(hub_root), TORCH_HOME=str(torch_root))

    from huggingface_hub import snapshot_download

    entries: list[dict] = []
    for repo_id in args.models:
        started = time.time()
        try:
            snapshot_path = Path(
                snapshot_download(
                    repo_id=repo_id,
                    cache_dir=hub_root,
                    local_files_only=(args.local_files_only or args.verify),
                    max_workers=args.max_workers,
                    allow_patterns=required_patterns(repo_id),
                )
            )
            validate_repository(repo_id, hub_root)
            entries.append(
                {
                    "kind": "huggingface",
                    "name": repo_id,
                    "status": "ready",
                    "snapshot": str(snapshot_path),
                    "bytes": directory_size(snapshot_path),
                    "elapsed_seconds": round(time.time() - started, 3),
                }
            )
        except Exception as exc:
            entries.append({"kind": "huggingface", "name": repo_id, "status": "error", "error": repr(exc)})

    if not args.skip_torchvision:
        try:
            import torch
            from torchvision.models import (
                ConvNeXt_Large_Weights,
                ResNet50_Weights,
                VGG19_Weights,
                convnext_large,
                resnet50,
                vgg19,
            )

            for name, constructor, weights in (
                ("resnet50", resnet50, ResNet50_Weights.DEFAULT),
                ("vgg19", vgg19, VGG19_Weights.DEFAULT),
                ("convnext_large", convnext_large, ConvNeXt_Large_Weights.DEFAULT),
            ):
                weight_path = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
                if args.verify:
                    if not weight_path.exists():
                        raise FileNotFoundError(weight_path)
                else:
                    constructor(weights=weights)
                entries.append(
                    {
                        "kind": "torchvision",
                        "name": name,
                        "status": "ready",
                        "path": str(weight_path),
                        "bytes": weight_path.stat().st_size,
                        "sha256": sha256(weight_path),
                    }
                )
        except Exception as exc:
            entries.append(
                {
                    "kind": "torchvision",
                    "name": "resnet50,vgg19,convnext_large",
                    "status": "error",
                    "error": repr(exc),
                }
            )

    manifest = {
        "output_root": str(output_root),
        "environment": {
            "HF_HOME": str(hf_root),
            "HF_HUB_CACHE": str(hub_root),
            "TORCH_HOME": str(torch_root),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        "entries": entries,
        "complete": bool(entries) and all(entry["status"] == "ready" for entry in entries),
        "total_bytes": directory_size(output_root),
    }
    (output_root / "offline_assets_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if not manifest["complete"]:
        raise SystemExit("one or more required offline assets are missing")


if __name__ == "__main__":
    main()
