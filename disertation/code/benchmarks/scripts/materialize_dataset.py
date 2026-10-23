#!/usr/bin/env python3
"""Join immutable split IDs to raw data and create a common baseline table."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import List

import pandas as pd


DEFAULT_FAKEDDIT = Path(r"C:\Users\kenic\OneDrive\Desktop\FYP\mine\fakeddit_all")
DEFAULT_FAKEDDIT_IMAGES = Path(r"C:\Users\kenic\OneDrive\Desktop\FYP\mine\src\multimodal\data\images")
DEFAULT_CHECKED = Path("checked/CHECKED/data_checked/processed/splits_group_content")
DEFAULT_CHECKED_ROOT = Path("checked/CHECKED")


def parse_list(value: object) -> List[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    text = str(value).strip()
    if not text:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except Exception:
            pass
    return []


def materialize_fakeddit(
    split: str,
    manifest: pd.DataFrame,
    source_root: Path,
    image_root: Path,
    portable_test: Path,
) -> pd.DataFrame:
    source_name = "test.tsv" if split == "test" else "train.tsv"
    source_path = source_root / source_name
    if split == "test" and not source_path.is_file():
        if not portable_test.is_file():
            raise FileNotFoundError(
                f"{source_path} is missing and no portable locked test table was found at {portable_test}"
            )
        source = pd.read_csv(
            portable_test,
            usecols=["item_id", "text", "domain", "label_raw", "label_fake"],
            dtype=str,
            low_memory=False,
        )
        joined = manifest.merge(
            source,
            on="item_id",
            how="left",
            validate="one_to_one",
            sort=False,
            suffixes=("", "_portable"),
        )
        if joined["text"].isna().any():
            missing = joined.loc[joined["text"].isna(), "item_id"].head().tolist()
            raise RuntimeError(f"manifest IDs missing from portable test table: {missing}")
        for label in ("label_raw", "label_fake"):
            portable_label = f"{label}_portable"
            if not (
                joined[label].astype(int).to_numpy()
                == joined[portable_label].astype(int).to_numpy()
            ).all():
                raise RuntimeError(f"locked {label} differs from portable test table")
        joined["image_path"] = joined["item_id"].map(
            lambda value: str((image_root / f"{value}.jpg").resolve())
        )
        joined["has_image"] = joined["image_path"].map(lambda value: Path(value).is_file())
        return joined[
            ["item_id", "text", "image_path", "has_image", "domain", "label_raw", "label_fake"]
        ]

    source = pd.read_csv(
        source_path,
        sep="\t",
        usecols=["id", "clean_title", "2_way_label", "domain", "subreddit"],
        dtype=str,
        low_memory=False,
    )
    source = source.dropna(subset=["2_way_label", "clean_title"]).copy()
    source["item_id"] = source["id"].astype(str)
    joined = manifest.merge(source, on="item_id", how="left", validate="one_to_one", sort=False)
    if joined["clean_title"].isna().any():
        missing = joined.loc[joined["clean_title"].isna(), "item_id"].head().tolist()
        raise RuntimeError(f"manifest IDs missing from {source_name}: {missing}")
    joined["text"] = joined["clean_title"].astype(str)
    joined["image_path"] = joined["item_id"].map(lambda value: str((image_root / f"{value}.jpg").resolve()))
    joined["has_image"] = joined["image_path"].map(lambda value: Path(value).is_file())
    joined["domain"] = joined["subreddit"].fillna(joined["domain"]).fillna("unknown")
    return joined[["item_id", "text", "image_path", "has_image", "domain", "label_raw", "label_fake"]]


def materialize_checked(split: str, manifest: pd.DataFrame, source_root: Path, checked_root: Path) -> pd.DataFrame:
    source = pd.read_csv(source_root / f"{split}.csv", dtype=str, keep_default_na=False)
    joined = manifest.merge(source, on="item_id", how="left", validate="one_to_one", sort=False, suffixes=("", "_source"))
    if joined["text"].eq("").all():
        raise RuntimeError(f"no CHECKED text found for split {split}")

    def first_image(value: object) -> str:
        paths = parse_list(value)
        if not paths:
            return ""
        path = Path(paths[0])
        if not path.is_absolute():
            path = checked_root / path
        return str(path.resolve())

    joined["text"] = joined["cleaned_text"].where(joined["cleaned_text"].ne(""), joined["text"])
    joined["image_path"] = joined["local_image_paths"].map(first_image)
    joined["has_image"] = joined["image_path"].map(lambda value: bool(value) and Path(value).is_file())
    joined["domain"] = joined["source"].where(joined["source"].ne(""), "checked")
    return joined[["item_id", "text", "image_path", "has_image", "domain", "label_raw", "label_fake"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["fakeddit", "checked"], required=True)
    parser.add_argument("--manifest-root", type=Path, default=Path("benchmarks/manifests"))
    parser.add_argument("--output-root", type=Path, default=Path("benchmarks/prepared"))
    parser.add_argument("--fakeddit-root", type=Path, default=DEFAULT_FAKEDDIT)
    parser.add_argument("--fakeddit-images", type=Path, default=DEFAULT_FAKEDDIT_IMAGES)
    parser.add_argument(
        "--fakeddit-portable-test",
        type=Path,
        default=Path("benchmarks/portable_splits/fakeddit/test.csv"),
        help="Locked test text/labels used when the original Fakeddit test.tsv is unavailable.",
    )
    parser.add_argument("--checked-splits", type=Path, default=DEFAULT_CHECKED)
    parser.add_argument("--checked-root", type=Path, default=DEFAULT_CHECKED_ROOT)
    parser.add_argument("--faknow-json", action="store_true", help="also emit SpotFake-compatible JSON records")
    args = parser.parse_args()

    output_dir = args.output_root / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"dataset": args.dataset, "splits": {}}
    for split in ("train", "val", "test"):
        manifest = pd.read_csv(args.manifest_root / args.dataset / f"{split}.csv", dtype={"item_id": str})
        if args.dataset == "fakeddit":
            frame = materialize_fakeddit(
                split,
                manifest,
                args.fakeddit_root,
                args.fakeddit_images,
                args.fakeddit_portable_test,
            )
        else:
            frame = materialize_checked(split, manifest, args.checked_splits, args.checked_root)
        csv_path = output_dir / f"{split}.csv"
        frame.to_csv(csv_path, index=False, lineterminator="\n")
        if args.faknow_json:
            faknow = frame.rename(columns={"text": "post_text", "image_path": "image_id", "label_fake": "label"})
            faknow[["post_text", "image_id", "label"]].to_json(
                output_dir / f"{split}_spotfake.json", orient="records", force_ascii=False, indent=2
            )
        report["splits"][split] = {
            "rows": int(len(frame)),
            "fake": int(frame["label_fake"].sum()),
            "real": int((1 - frame["label_fake"]).sum()),
            "images_present": int(frame["has_image"].sum()),
            "images_missing": int((~frame["has_image"]).sum()),
            "csv": str(csv_path.resolve()),
        }
    (output_dir / "materialization_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
