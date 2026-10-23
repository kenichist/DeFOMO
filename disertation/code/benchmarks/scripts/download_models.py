#!/usr/bin/env python3
"""Download model snapshots into one shared Hugging Face cache.

Run this on the RTX 6000 workstation. The primary profile is modest; DSR is
separate because its Qwen checkpoints require substantially more storage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


PROFILES = {
    "primary": [
        "bert-base-uncased",
        "bert-base-chinese",
        "distilbert-base-uncased",
        "distilbert-base-multilingual-cased",
        "hfl/chinese-roberta-wwm-ext",
        "microsoft/deberta-v3-base",
        "microsoft/mdeberta-v3-base",
        "roberta-base",
        "openai/clip-vit-base-patch32",
    ],
    "defomo": [
        "distilbert-base-uncased",
        "google/vit-base-patch16-224-in21k",
    ],
    "tern": [
        "distilbert-base-uncased",
        "openai/clip-vit-base-patch32",
        "facebook/vit-mae-base",
    ],
    "dsr": [
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2-VL-7B-Instruct",
        "roberta-base",
        "openai/clip-vit-base-patch32",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=[*PROFILES, "all"], default="primary")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--token", default=None)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected = set()
    profiles = PROFILES if args.profile == "all" else {args.profile: PROFILES[args.profile]}
    for model_ids in profiles.values():
        selected.update(model_ids)
    if not args.dry_run:
        args.cache_root.mkdir(parents=True, exist_ok=True)
    records = []
    for model_id in sorted(selected):
        print(model_id, flush=True)
        if args.dry_run:
            records.append({"model_id": model_id, "snapshot_path": None, "status": "dry_run"})
            continue
        snapshot_path = snapshot_download(
            repo_id=model_id,
            cache_dir=str(args.cache_root),
            token=args.token,
            max_workers=args.max_workers,
        )
        records.append(
            {
                "model_id": model_id,
                "snapshot_path": snapshot_path,
                "snapshot_revision": Path(snapshot_path).name,
                "status": "downloaded",
            }
        )
    manifest = {"profile": args.profile, "cache_root": str(args.cache_root.resolve()), "models": records}
    if not args.dry_run:
        output = args.cache_root / f"download_manifest_{args.profile}.json"
        output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
