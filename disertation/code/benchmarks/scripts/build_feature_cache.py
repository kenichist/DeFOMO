#!/usr/bin/env python3
"""Precompute deterministic frozen-encoder features for repeated baseline runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


MODELS = ["clip_linear", "spotfake_faknow", "cafe_faknow"]


def text_model_for(dataset: str) -> str:
    return "bert-base-uncased" if dataset == "fakeddit" else "hfl/chinese-roberta-wwm-ext"


def features_for(model_name, model, batch, device):
    moved = {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}
    if model_name == "clip_linear":
        model.clip.eval()
        text = F.normalize(
            model.clip.get_text_features(input_ids=moved["input_ids"], attention_mask=moved["attention_mask"]),
            dim=-1,
        )
        image = F.normalize(model.clip.get_image_features(pixel_values=moved["pixel_values"]), dim=-1)
        similarity = (text * image).sum(dim=1, keepdim=True)
        return {
            "clip_features": torch.cat(
                [text, image, torch.abs(text - image), similarity, moved["has_image"].unsqueeze(1)], dim=1
            )
        }
    if model_name == "spotfake_faknow":
        core = model.spotfake.model
        core.text_encoder.bert.eval()
        core.vision_encoder.vis_encoder.eval()
        text = core.text_encoder.bert(
            input_ids=moved["input_ids"], attention_mask=moved["attention_mask"]
        ).pooler_output
        image = core.vision_encoder.vis_encoder(moved["pixel_values"])
        return {"spotfake_text_base": text, "spotfake_image_base": image}
    model.text_encoder.eval()
    model.image_encoder.eval()
    hidden = model.text_encoder(
        input_ids=moved["input_ids"], attention_mask=moved["attention_mask"]
    ).last_hidden_state[:, :30]
    if hidden.shape[1] < 30:
        hidden = F.pad(hidden, (0, 0, 0, 30 - hidden.shape[1]))
    text = F.adaptive_avg_pool1d(hidden, 200)
    image = model.image_encoder(pixel_values=moved["pixel_values"]).image_embeds
    image = image * moved["has_image"].unsqueeze(1)
    return {"cafe_text": text, "cafe_image": image}


def build_one(dataset, model_name, split, args, repository_root, device):
    sys.path.insert(0, str(repository_root / "benchmarks" / "src"))
    from train_baseline import BatchCollator, CAFEAdapter, CLIPLinearClassifier, SpotFakeAdapter, StandardDataset

    csv_path = args.prepared_root / dataset / f"{split}.csv"
    cache_dir = args.cache_root / dataset / model_name / split
    meta_path = cache_dir / "meta.json"
    if meta_path.exists() and not args.rebuild:
        print(f"SKIP {dataset}/{model_name}/{split}: {meta_path}", flush=True)
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset_rows = StandardDataset(csv_path, args.smoke_limit)
    collator = BatchCollator(model_name, text_model_for(dataset), args.clip_model, args.max_length)
    loader = DataLoader(
        dataset_rows,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=collator,
    )
    if model_name == "clip_linear":
        model = CLIPLinearClassifier(args.clip_model, freeze=True, cached=False)
    elif model_name == "spotfake_faknow":
        model = SpotFakeAdapter(text_model_for(dataset), True, repository_root, cached=False)
    else:
        model = CAFEAdapter(text_model_for(dataset), args.clip_model, repository_root, cached=False)
    model = model.to(device).eval()

    storage = {}
    filenames = {}
    dtype = np.float16 if args.dtype == "float16" else np.float32
    offset = 0
    with torch.inference_mode():
        for batch in loader:
            outputs = features_for(model_name, model, batch, device)
            if not storage:
                for name, tensor in outputs.items():
                    filename = f"{name}.npy"
                    filenames[name] = filename
                    storage[name] = np.lib.format.open_memmap(
                        cache_dir / filename,
                        mode="w+",
                        dtype=dtype,
                        shape=(len(dataset_rows), *tensor.shape[1:]),
                    )
            count = int(next(iter(outputs.values())).shape[0])
            for name, tensor in outputs.items():
                storage[name][offset : offset + count] = tensor.detach().float().cpu().numpy().astype(dtype, copy=False)
            offset += count
            print(f"{dataset}/{model_name}/{split}: {offset}/{len(dataset_rows)}", flush=True)
    for array in storage.values():
        array.flush()
    ordered_hash = hashlib.sha256("\n".join(dataset_rows.ids).encode("utf-8")).hexdigest()
    meta = {
        "dataset": dataset,
        "model": model_name,
        "split": split,
        "rows": len(dataset_rows),
        "ordered_id_sha256": ordered_hash,
        "dtype": args.dtype,
        "features": filenames,
        "source_csv": str(csv_path.resolve()),
        "smoke_limit": args.smoke_limit,
    }
    temporary = cache_dir / "meta.json.tmp"
    temporary.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    temporary.replace(meta_path)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=["fakeddit", "checked"], default=["fakeddit", "checked"])
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--prepared-root", type=Path, default=Path("benchmarks/prepared"))
    parser.add_argument("--cache-root", type=Path, default=Path("benchmarks/feature_cache/frozen"))
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--smoke-limit", type=int, default=None, help="Build a tiny cache for dependency testing only.")
    args = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for dataset in args.datasets:
        for model_name in args.models:
            for split in ("train", "val", "test"):
                build_one(dataset, model_name, split, args, repository_root, device)


if __name__ == "__main__":
    main()
