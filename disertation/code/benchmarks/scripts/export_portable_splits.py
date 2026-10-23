#!/usr/bin/env python3
"""Export portable locked split tables for an offline RunPod transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepared-test",
        type=Path,
        default=Path("benchmarks/prepared/fakeddit/test.csv"),
    )
    parser.add_argument(
        "--manifest-test",
        type=Path,
        default=Path("benchmarks/manifests/fakeddit/test.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/portable_splits/fakeddit/test.csv"),
    )
    args = parser.parse_args()

    prepared = pd.read_csv(
        args.prepared_test,
        usecols=["item_id", "text", "domain", "label_raw", "label_fake"],
        dtype={"item_id": str, "text": str, "domain": str},
        low_memory=False,
    )
    manifest = pd.read_csv(args.manifest_test, dtype={"item_id": str})
    if prepared["item_id"].duplicated().any():
        raise RuntimeError("prepared Fakeddit test contains duplicate item IDs")
    if prepared["item_id"].tolist() != manifest["item_id"].tolist():
        raise RuntimeError("prepared Fakeddit test order does not match the locked manifest")
    for label in ("label_raw", "label_fake"):
        if not (
            prepared[label].astype(int).to_numpy()
            == manifest[label].astype(int).to_numpy()
        ).all():
            raise RuntimeError(f"prepared Fakeddit {label} differs from the locked manifest")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".csv.tmp")
    prepared.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(args.output)
    metadata = {
        "dataset": "fakeddit",
        "split": "test",
        "rows": int(len(prepared)),
        "columns": prepared.columns.tolist(),
        "ordered_id_sha256": hashlib.sha256(
            ("\n".join(prepared["item_id"]) + "\n").encode("utf-8")
        ).hexdigest(),
        "csv_sha256": sha256(args.output),
        "source_manifest_sha256": sha256(args.manifest_test),
        "purpose": (
            "Portable copy of the locked full Fakeddit test text and labels. "
            "Image paths are resolved against --fakeddit-images on the target machine."
        ),
    }
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
