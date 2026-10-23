#!/usr/bin/env python3
"""Build immutable split manifests matching the DeFOMO experiments.

The manifests contain identifiers and labels only. Raw text and images remain in
their original dataset directories and are joined later by materialize_dataset.py.
This prevents a baseline from silently changing the evaluation population.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd
from sklearn.model_selection import train_test_split


DEFAULT_FAKEDDIT = Path(r"C:\Users\kenic\OneDrive\Desktop\FYP\mine\fakeddit_all")
DEFAULT_CHECKED = Path("checked/CHECKED/data_checked/processed/splits_group_content")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_ids(ids: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for item_id in ids:
        digest.update(str(item_id).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def summarize(frame: pd.DataFrame) -> Dict[str, object]:
    return {
        "rows": int(len(frame)),
        "unique_ids": int(frame["item_id"].nunique()),
        "duplicate_ids": int(frame["item_id"].duplicated().sum()),
        "label_raw_counts": {
            str(k): int(v) for k, v in frame["label_raw"].value_counts().sort_index().items()
        },
        "label_fake_counts": {
            str(k): int(v) for k, v in frame["label_fake"].value_counts().sort_index().items()
        },
        "ordered_id_sha256": sha256_ids(frame["item_id"]),
    }


def write_split(frame: pd.DataFrame, path: Path) -> Dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")
    info = summarize(frame)
    info["manifest_sha256"] = sha256_file(path)
    return info


def read_fakeddit(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        sep="\t",
        usecols=["id", "clean_title", "2_way_label"],
        dtype={"id": "string", "clean_title": "string", "2_way_label": "Int64"},
        low_memory=False,
    )
    # This is exactly the filtering used by the original DeFOMO Fakeddit code.
    frame = frame.dropna(subset=["2_way_label", "clean_title"]).copy()
    frame["item_id"] = frame["id"].astype(str)
    frame["label_raw"] = frame["2_way_label"].astype(int)
    # Fakeddit encodes 0=fake and 1=real. Standardized outputs use fake=1.
    frame["label_fake"] = 1 - frame["label_raw"]
    return frame[["item_id", "label_raw", "label_fake"]]


def build_fakeddit(root: Path, output: Path, split_seed: int) -> Dict[str, object]:
    source_train = root / "train.tsv"
    source_test = root / "test.tsv"
    full_train = read_fakeddit(source_train)
    train, val = train_test_split(
        full_train,
        test_size=0.20,
        random_state=split_seed,
        stratify=full_train["label_raw"],
    )
    test = read_fakeddit(source_test)
    splits = {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }
    split_info = {
        name: write_split(frame, output / "fakeddit" / f"{name}.csv")
        for name, frame in splits.items()
    }
    sets = {name: set(frame["item_id"]) for name, frame in splits.items()}
    return {
        "dataset": "Fakeddit v2 binary",
        "split_protocol": (
            "drop rows missing 2_way_label or clean_title; stratified 80/20 split of "
            "official train.tsv with sklearn random_state=42; official test.tsv held out"
        ),
        "split_seed": split_seed,
        "label_mapping": {"label_raw": {"0": "fake", "1": "real"}, "label_fake": {"1": "fake", "0": "real"}},
        "sources": {
            "train.tsv": {"path": str(source_train.resolve()), "sha256": sha256_file(source_train)},
            "test.tsv": {"path": str(source_test.resolve()), "sha256": sha256_file(source_test)},
        },
        "splits": split_info,
        "overlap_counts": {
            "train_val": len(sets["train"] & sets["val"]),
            "train_test": len(sets["train"] & sets["test"]),
            "val_test": len(sets["val"] & sets["test"]),
        },
    }


def build_checked(root: Path, output: Path) -> Dict[str, object]:
    splits: Dict[str, pd.DataFrame] = {}
    sources: Dict[str, object] = {}
    for name in ("train", "val", "test"):
        source = root / f"{name}.csv"
        frame = pd.read_csv(source, usecols=["item_id", "label_id"], dtype=str)
        frame["item_id"] = frame["item_id"].astype(str)
        frame["label_raw"] = frame["label_id"].astype(int)
        # CHECKED already encodes fake=1 and real=0.
        frame["label_fake"] = frame["label_raw"]
        frame = frame[["item_id", "label_raw", "label_fake"]]
        splits[name] = frame
        sources[f"{name}.csv"] = {"path": str(source.resolve()), "sha256": sha256_file(source)}
    split_info = {
        name: write_split(frame, output / "checked" / f"{name}.csv")
        for name, frame in splits.items()
    }
    sets = {name: set(frame["item_id"]) for name, frame in splits.items()}
    return {
        "dataset": "CHECKED binary grouped-content split",
        "split_protocol": "existing group_content train/val/test files; no re-splitting",
        "label_mapping": {"label_raw": {"0": "real", "1": "fake"}, "label_fake": {"1": "fake", "0": "real"}},
        "sources": sources,
        "splits": split_info,
        "overlap_counts": {
            "train_val": len(sets["train"] & sets["val"]),
            "train_test": len(sets["train"] & sets["test"]),
            "val_test": len(sets["val"] & sets["test"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fakeddit-root", type=Path, default=DEFAULT_FAKEDDIT)
    parser.add_argument("--checked-root", type=Path, default=DEFAULT_CHECKED)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/manifests"))
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()

    metadata = {
        "schema_version": 1,
        "fixed_split_seed": args.split_seed,
        "run_seeds_are_independent_of_the_fixed_split": True,
        "fakeddit": build_fakeddit(args.fakeddit_root, args.output, args.split_seed),
        "checked": build_checked(args.checked_root, args.output),
    }
    metadata_path = args.output / "split_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
