# -*- coding: utf-8 -*-
# Converts LIAR train/valid/test TSV (no headers) into FaKnow JSON:
# each item -> {"text": "...", "label": 0/1, "domain": 0}

import pandas as pd, json, sys

MODE = "binary"   # "binary" or "six"

# Path names expected alongside this script:
SPLITS = {"train": "train.tsv", "valid": "valid.tsv", "test": "test.tsv"}

# LIAR is headerless. Canonical column positions:
# 0:id, 1:label, 2:statement, 3:subject, 4:speaker, 5:job_title, 6:state,
# 7:party, 8:barely_true, 9:false, 10:half_true, 11:mostly_true, 12:pants_on_fire, 13:context
# Some mirrors have 12–14 cols; we only need col 1 (label) and col 2 (statement).

BINARY_MAP = {
    "true": 0, "mostly-true": 0, "half-true": 0,
    "false": 1, "barely-true": 1, "pants-fire": 1, "pants on fire": 1,
}

SIX_CLASSES = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
SIX_INDEX = {lab: i for i, lab in enumerate(SIX_CLASSES)}

def norm_label(s):
    s = str(s).strip().lower()
    # normalize variants
    if s == "pants on fire": s = "pants-fire"
    return s

def to_records(df):
    # Ensure enough columns
    if df.shape[1] < 3:
        raise ValueError(f"Expected >=3 columns but got {df.shape[1]}. First row: {df.iloc[0].tolist()}")
    labels = df.iloc[:, 1].astype(str).map(norm_label)
    texts  = df.iloc[:, 2].astype(str)

    recs = []
    for lab, txt in zip(labels, texts):
        if MODE == "binary":
            y = BINARY_MAP.get(lab, 1)  # unknowns default to "fake" = 1
        else:
            y = SIX_INDEX.get(lab, SIX_INDEX["false"])
        recs.append({"text": txt.strip(), "label": int(y), "domain": 0})  # single-domain to start
    return recs

def convert_one(in_path, out_path):
    df = pd.read_csv(in_path, sep="\t", header=None, dtype=str, keep_default_na=False, quoting=3, engine="python")
    recs = to_records(df)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(recs, f, ensure_ascii=False, indent=2)
    from collections import Counter
    c = Counter([r["label"] for r in recs])
    print(f"Wrote {out_path} | rows={len(recs)} | label_dist={dict(c)}")

def main():
    for split, tsv in SPLITS.items():
        convert_one(tsv, f"{split}.json")
    print("Done. Files: train.json, valid.json, test.json")

if __name__ == "__main__":
    main()
