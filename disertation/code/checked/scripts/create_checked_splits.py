"""Create safer CHECKED train/validation/test splits.

Examples:

    python scripts/create_checked_splits.py \
      --input CHECKED/data_checked/processed/checked_multimodal.csv \
      --output_dir CHECKED/data_checked/processed/splits_group_text \
      --mode group --group_col cleaned_text_hash

The script writes a split report and exits with status 2 if post-split leakage
checks fail. Existing split directories are not read or modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


MEDIA_COLUMNS = ["image_urls", "video_urls", "local_image_paths", "local_video_paths"]
ROOT = Path(__file__).resolve().parents[1]


def parse_list(value: Any) -> List[str]:
    text = str(value or "").strip()
    if not text or text.lower() in {"", "nan", "none", "null", "[]", "{}"}:
        return []
    if text[:1] in {"[", "{"}:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
            if isinstance(parsed, dict):
                return [str(item).strip() for item in parsed.values() if str(item).strip()]
        except json.JSONDecodeError:
            pass
    return [text]


def normalize_text(value: Any) -> str:
    return re.sub(r"[\W_]+", "", str(value or "").lower(), flags=re.UNICODE)


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def resolve_media_path(value: str) -> Path | None:
    candidate = Path(value.replace("\\", "/"))
    for path in [candidate, ROOT / candidate, ROOT / "CHECKED" / candidate]:
        if path.exists() and path.is_file():
            return path.resolve()
    return None


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def media_signature(row: pd.Series) -> str:
    values = []
    for column in MEDIA_COLUMNS:
        for value in parse_list(row.get(column, "")):
            values.append(value.replace("\\", "/").split("?", 1)[0].lower())
    return stable_hash("\n".join(sorted(set(values)))) if values else stable_hash(f"no-media:{row.name}")


def content_components(frame: pd.DataFrame) -> List[str]:
    parent = list(range(len(frame)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    first_seen: Dict[str, int] = {}
    hash_cache: Dict[str, str] = {}
    for position, (_, row) in enumerate(frame.iterrows()):
        keys = [f"text:{row.get('cleaned_text_hash', '')}"]
        for column in MEDIA_COLUMNS:
            for value in parse_list(row.get(column, "")):
                normalized = value.replace("\\", "/").split("?", 1)[0].lower()
                if normalized:
                    keys.append(f"{column}:{normalized}")
                if column in {"local_image_paths", "local_video_paths"}:
                    resolved = resolve_media_path(value)
                    if resolved is not None:
                        cache_key = str(resolved)
                        if cache_key not in hash_cache:
                            try:
                                hash_cache[cache_key] = file_hash(resolved)
                            except OSError:
                                hash_cache[cache_key] = ""
                        if hash_cache[cache_key]:
                            keys.append(f"media-bytes:{hash_cache[cache_key]}")
        for key in keys:
            if not key or key.endswith(":"):
                continue
            if key in first_seen:
                union(position, first_seen[key])
            else:
                first_seen[key] = position
    return [f"component:{find(index)}" for index in range(len(frame))]


def add_derived_group_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    text_column = "cleaned_text" if "cleaned_text" in frame else "text"
    if text_column in frame:
        frame["cleaned_text_hash"] = frame[text_column].map(lambda value: stable_hash(normalize_text(value)))
    frame["media_hash"] = frame.apply(media_signature, axis=1)
    frame["content_component"] = content_components(frame)
    return frame


def validate_ratios(train: float, val: float, test: float) -> None:
    if min(train, val, test) <= 0:
        raise ValueError("All split ratios must be positive.")
    if not math_isclose(train + val + test, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {train + val + test:.8f}.")


def math_isclose(left: float, right: float, tolerance: float = 1e-8) -> bool:
    return abs(left - right) <= tolerance


def stratified_split(
    frame: pd.DataFrame,
    ratios: Tuple[float, float, float],
    seed: int,
) -> Dict[str, pd.DataFrame]:
    train_ratio, val_ratio, test_ratio = ratios
    train, remainder = train_test_split(
        frame,
        test_size=val_ratio + test_ratio,
        random_state=seed,
        stratify=frame["label_id"].astype(str),
    )
    val_fraction = val_ratio / (val_ratio + test_ratio)
    val, test = train_test_split(
        remainder,
        train_size=val_fraction,
        random_state=seed,
        stratify=remainder["label_id"].astype(str),
    )
    return {"train": train, "val": val, "test": test}


def time_split(
    frame: pd.DataFrame,
    ratios: Tuple[float, float, float],
    timestamp_col: str,
) -> Dict[str, pd.DataFrame]:
    if timestamp_col not in frame:
        raise ValueError(f"Timestamp column {timestamp_col!r} is not present.")
    parsed = pd.to_datetime(frame[timestamp_col], errors="coerce")
    if parsed.isna().any():
        raise ValueError(
            f"{int(parsed.isna().sum())} rows have invalid/missing timestamps in {timestamp_col!r}."
        )
    ordered = frame.assign(_parsed_timestamp=parsed).sort_values(
        ["_parsed_timestamp", "item_id"], kind="stable"
    )
    train_end = int(round(len(ordered) * ratios[0]))
    val_end = train_end + int(round(len(ordered) * ratios[1]))
    return {
        "train": ordered.iloc[:train_end].drop(columns="_parsed_timestamp"),
        "val": ordered.iloc[train_end:val_end].drop(columns="_parsed_timestamp"),
        "test": ordered.iloc[val_end:].drop(columns="_parsed_timestamp"),
    }


def group_split_candidate(
    frame: pd.DataFrame,
    group_col: str,
    ratios: Tuple[float, float, float],
    seed: int,
) -> Dict[str, pd.DataFrame]:
    train_ratio, val_ratio, test_ratio = ratios
    first = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
    train_idx, remainder_idx = next(first.split(frame, groups=frame[group_col]))
    train = frame.iloc[train_idx]
    remainder = frame.iloc[remainder_idx]
    val_fraction = val_ratio / (val_ratio + test_ratio)
    second = GroupShuffleSplit(n_splits=1, train_size=val_fraction, random_state=seed + 1)
    val_idx, test_idx = next(second.split(remainder, groups=remainder[group_col]))
    return {
        "train": train,
        "val": remainder.iloc[val_idx],
        "test": remainder.iloc[test_idx],
    }


def split_score(
    parts: Mapping[str, pd.DataFrame],
    ratios: Tuple[float, float, float],
    labels: Sequence[str],
) -> float:
    total = sum(len(part) for part in parts.values())
    score = 0.0
    for (name, part), target_ratio in zip(parts.items(), ratios):
        score += abs(len(part) / total - target_ratio) * 3.0
        distribution = part["label_id"].astype(str).value_counts(normalize=True)
        overall = pd.concat(parts.values())["label_id"].astype(str).value_counts(normalize=True)
        for label in labels:
            score += abs(float(distribution.get(label, 0)) - float(overall.get(label, 0)))
        if set(part["label_id"].astype(str)) != set(labels):
            score += 100.0
    return score


def grouped_split(
    frame: pd.DataFrame,
    group_col: str,
    ratios: Tuple[float, float, float],
    seed: int,
    attempts: int,
) -> Dict[str, pd.DataFrame]:
    if group_col not in frame:
        raise ValueError(f"Group column {group_col!r} is not present.")
    if frame[group_col].astype(str).str.strip().eq("").any():
        raise ValueError(f"Group column {group_col!r} contains empty values.")
    if frame[group_col].nunique() < 3:
        raise ValueError(f"Group column {group_col!r} needs at least three unique values.")

    labels = sorted(frame["label_id"].astype(str).unique())
    candidates = []
    for attempt in range(attempts):
        try:
            parts = group_split_candidate(frame, group_col, ratios, seed + attempt * 2)
        except ValueError:
            continue
        candidates.append((split_score(parts, ratios, labels), parts))
    if not candidates:
        raise ValueError("Could not produce a grouped split.")
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def flattened_values(frame: pd.DataFrame, column: str) -> set[str]:
    values: set[str] = set()
    if column not in frame:
        return values
    for raw in frame[column]:
        items = parse_list(raw) if column in MEDIA_COLUMNS else [raw]
        for item in items:
            value = normalize_text(item) if column in {"cleaned_text", "text"} else str(item).strip()
            if value:
                values.add(value)
    return values


def local_media_hashes(frame: pd.DataFrame) -> set[str]:
    hashes: set[str] = set()
    cache: Dict[str, str] = {}
    for column in ["local_image_paths", "local_video_paths"]:
        if column not in frame:
            continue
        for raw in frame[column]:
            for value in parse_list(raw):
                resolved = resolve_media_path(value)
                if resolved is None:
                    continue
                key = str(resolved)
                if key not in cache:
                    try:
                        cache[key] = file_hash(resolved)
                    except OSError:
                        cache[key] = ""
                if cache[key]:
                    hashes.add(cache[key])
    return hashes


def validate_split(
    parts: Mapping[str, pd.DataFrame],
    group_col: str | None,
    allow_content_overlap: bool,
) -> Dict[str, Any]:
    failures: List[str] = []
    warnings: List[str] = []
    names = list(parts)
    checks = []
    media_hash_sets = {name: local_media_hashes(part) for name, part in parts.items()}
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            left_ids = set(parts[left]["item_id"].astype(str))
            right_ids = set(parts[right]["item_id"].astype(str))
            duplicate_ids = left_ids & right_ids
            if duplicate_ids:
                failures.append(f"{left}/{right}: {len(duplicate_ids)} duplicate item_id values")
            if group_col:
                group_overlap = set(parts[left][group_col].astype(str)) & set(
                    parts[right][group_col].astype(str)
                )
                if group_overlap:
                    failures.append(
                        f"{left}/{right}: {len(group_overlap)} {group_col} groups cross splits"
                    )
            for column in ["cleaned_text", "text", *MEDIA_COLUMNS]:
                if column not in parts[left] or column not in parts[right]:
                    continue
                overlap = flattened_values(parts[left], column) & flattened_values(parts[right], column)
                checks.append(
                    {
                        "left": left,
                        "right": right,
                        "column": column,
                        "overlap_count": len(overlap),
                        "examples": sorted(overlap)[:5],
                    }
                )
                if overlap:
                    message = f"{left}/{right}: {len(overlap)} exact {column} values cross splits"
                    if allow_content_overlap:
                        warnings.append(message)
                    else:
                        failures.append(message)
            hash_overlap = media_hash_sets[left] & media_hash_sets[right]
            checks.append(
                {
                    "left": left,
                    "right": right,
                    "column": "local_media_sha256",
                    "overlap_count": len(hash_overlap),
                    "examples": sorted(hash_overlap)[:5],
                }
            )
            if hash_overlap:
                message = f"{left}/{right}: {len(hash_overlap)} local media SHA-256 values cross splits"
                if allow_content_overlap:
                    warnings.append(message)
                else:
                    failures.append(message)
    expected_labels = set(pd.concat(parts.values())["label_id"].astype(str))
    for name, part in parts.items():
        labels = set(part["label_id"].astype(str))
        if labels != expected_labels:
            failures.append(f"{name}: missing labels {sorted(expected_labels - labels)}")
    return {"passed": not failures, "failures": failures, "warnings": warnings, "overlap_checks": checks}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create leakage-checked CHECKED splits.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["stratified", "time", "group"], required=True)
    parser.add_argument("--timestamp_col", default="timestamp")
    parser.add_argument("--group_col", default="")
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-attempts", type=int, default=200)
    parser.add_argument("--remove-exact-duplicate-text", action="store_true")
    parser.add_argument(
        "--allow-content-overlap",
        action="store_true",
        help="Write a diagnostic split despite exact text/media overlap; IDs and grouping still must pass.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_ratios(args.train, args.val, args.test)
    frame = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    if "item_id" not in frame and "id" in frame:
        frame["item_id"] = frame["id"]
    if "item_id" not in frame or "label_id" not in frame:
        raise ValueError("Input must contain item_id (or id) and label_id.")
    if frame["item_id"].duplicated().any():
        duplicates = int(frame["item_id"].duplicated(keep=False).sum())
        raise ValueError(f"Input contains {duplicates} rows with duplicate item_id values.")

    original_rows = len(frame)
    frame = add_derived_group_columns(frame)
    removed_duplicates = 0
    if args.remove_exact_duplicate_text:
        if "cleaned_text_hash" not in frame:
            raise ValueError("Cannot remove duplicate text because no text column is available.")
        conflicting = frame.groupby("cleaned_text_hash")["label_id"].nunique()
        conflicting = set(conflicting[conflicting > 1].index)
        if conflicting:
            raise ValueError(
                f"{len(conflicting)} exact text groups contain conflicting labels; inspect before deduplication."
            )
        before = len(frame)
        frame = frame.drop_duplicates("cleaned_text_hash", keep="first").copy()
        removed_duplicates = before - len(frame)

    ratios = (args.train, args.val, args.test)
    group_col: str | None = None
    if args.mode == "stratified":
        parts = stratified_split(frame, ratios, args.seed)
    elif args.mode == "time":
        parts = time_split(frame, ratios, args.timestamp_col)
    else:
        if not args.group_col:
            raise ValueError("--group_col is required for --mode group.")
        group_col = args.group_col
        parts = grouped_split(frame, group_col, ratios, args.seed, args.group_attempts)

    parts = {name: part.reset_index(drop=True) for name, part in parts.items()}
    validation = validate_split(parts, group_col, args.allow_content_overlap)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "input": str(args.input.resolve()),
        "mode": args.mode,
        "group_col": group_col,
        "timestamp_col": args.timestamp_col if args.mode == "time" else None,
        "seed": args.seed,
        "requested_ratios": {"train": args.train, "val": args.val, "test": args.test},
        "original_rows": original_rows,
        "removed_exact_duplicate_text_rows": removed_duplicates,
        "sizes": {name: len(part) for name, part in parts.items()},
        "class_distribution": {
            name: part["label_id"].value_counts().to_dict() for name, part in parts.items()
        },
        "timestamp_ranges": {
            name: {
                "min": str(pd.to_datetime(part.get(args.timestamp_col), errors="coerce").min()),
                "max": str(pd.to_datetime(part.get(args.timestamp_col), errors="coerce").max()),
            }
            if args.timestamp_col in part
            else None
            for name, part in parts.items()
        },
        "validation": validation,
    }
    for name, part in parts.items():
        derived = [
            column
            for column in ["cleaned_text_hash", "media_hash", "content_component"]
            if column in part
        ]
        part.drop(columns=derived).to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8")
    (output_dir / "split_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    text_lines = [
        "CHECKED safer split report",
        f"Mode: {args.mode}",
        f"Group column: {group_col}",
        f"Sizes: {report['sizes']}",
        f"Class distribution: {report['class_distribution']}",
        f"Validation passed: {validation['passed']}",
    ]
    text_lines.extend(f"FAIL: {message}" for message in validation["failures"])
    text_lines.extend(f"WARNING: {message}" for message in validation["warnings"])
    (output_dir / "split_report.txt").write_text("\n".join(text_lines) + "\n", encoding="utf-8")

    print("\n".join(text_lines))
    print(f"Candidate splits written to: {output_dir}")
    if not validation["passed"]:
        print(
            "Split validation failed. Do not train on this candidate until the reported contamination is resolved.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
