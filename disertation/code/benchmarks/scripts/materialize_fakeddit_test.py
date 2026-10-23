#!/usr/bin/env python3
"""Materialize only the fixed Fakeddit test split."""

from pathlib import Path
import os
import pandas as pd

MANIFEST = Path("benchmarks/manifests/fakeddit/test.csv")
SOURCE = Path(r"C:\Users\kenic\OneDrive\Desktop\FYP\mine\fakeddit_all\test.tsv")
IMAGES = Path(r"C:\Users\kenic\OneDrive\Desktop\FYP\mine\src\multimodal\data\images")
OUTPUT = Path("benchmarks/prepared/fakeddit/test.csv")


def main() -> None:
    manifest = pd.read_csv(MANIFEST, dtype={"item_id": str})
    source = pd.read_csv(
        SOURCE,
        sep="\t",
        usecols=["id", "clean_title", "2_way_label", "domain", "subreddit"],
        dtype=str,
        low_memory=False,
    ).dropna(subset=["2_way_label", "clean_title"])
    source["item_id"] = source["id"].astype(str)
    frame = manifest.merge(source, on="item_id", how="left", validate="one_to_one", sort=False)
    available = {
        Path(entry.name).stem
        for entry in os.scandir(IMAGES)
        if entry.is_file() and entry.name.lower().endswith(".jpg")
    }
    image_root = IMAGES.resolve()
    frame["text"] = frame["clean_title"].astype(str)
    frame["image_path"] = frame["item_id"].map(lambda value: str(image_root / f"{value}.jpg"))
    frame["has_image"] = frame["item_id"].isin(available)
    frame["domain"] = frame["subreddit"].fillna(frame["domain"]).fillna("unknown")
    columns = ["item_id", "text", "image_path", "has_image", "domain", "label_raw", "label_fake"]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame[columns].to_csv(OUTPUT, index=False, lineterminator="\n")
    print({"rows": len(frame), "images_present": int(frame["has_image"].sum()), "output": str(OUTPUT.resolve())})


if __name__ == "__main__":
    main()
