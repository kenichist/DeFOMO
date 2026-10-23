"""Leakage-aware heterogeneous graph utilities for CHECKED.

Runtime graphs are built independently for each split. Raw labels and source
paths are never stored in graph node features. In fair mode, raw user identity
nodes and edges are disabled.
"""

from __future__ import annotations

import json
import math
from bisect import insort
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, Linear, SAGEConv


PROJECT_ROOT = Path(__file__).resolve().parent
GRAPH_VARIANTS = {
    "media_only",
    "response_only",
    "metadata_only",
    "response_metadata",
    "full_safe",
    "upper_bound",
}
CHECKED_EDGE_TYPES = [
    ("user", "posted", "post"),
    ("post", "rev_posted", "user"),
    ("post", "has_image", "image"),
    ("image", "rev_has_image", "post"),
    ("post", "has_video", "video"),
    ("video", "rev_has_video", "post"),
    ("post", "has_comment", "comment"),
    ("comment", "belongs_to_post", "post"),
    ("post", "has_repost", "repost"),
    ("repost", "belongs_to_post", "post"),
    ("user", "wrote_comment", "comment"),
    ("comment", "rev_wrote_comment", "user"),
    ("user", "made_repost", "repost"),
    ("repost", "rev_made_repost", "user"),
]
SAFE_POST_METADATA = [
    "text_length",
    "media_count",
    "has_image",
    "has_video",
    "early_comment_count",
    "early_repost_count",
    "has_early_comments",
    "has_early_reposts",
    "unique_early_commenters",
    "unique_early_reposters",
    "time_to_first_comment_hours",
    "time_to_first_repost_hours",
    "mean_comment_delay_hours",
    "mean_repost_delay_hours",
]
UPPER_BOUND_POST_METADATA = SAFE_POST_METADATA + [
    "comment_count",
    "repost_count",
    "like_count",
    "num_comments",
    "num_reposts",
]
RESPONSE_FEATURE_NAMES = [
    "log_text_length",
    "log_hours_since_post",
    "within_window",
    "has_text",
    "has_media",
    "hour_sin",
    "hour_cos",
    "missing_timestamp",
]
MEDIA_FEATURE_NAMES = [
    "is_local",
    "is_remote_url",
    "is_image",
    "is_video",
    "extension_known",
    "file_available",
]


def parse_json_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "[]", "{}"}:
        return []
    if text[:1] in {"[", "{"}:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if str(item).strip()]
            if isinstance(parsed, dict):
                return [str(item) for item in parsed.values() if str(item).strip()]
        except json.JSONDecodeError:
            pass
    return [text]


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def normalize_checked_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    if "item_id" in frame and "id" not in frame:
        frame["id"] = frame["item_id"]
    if "label_id" in frame and "2_way_label" not in frame:
        frame["2_way_label"] = frame["label_id"]
    if "cleaned_text" in frame and "clean_title" not in frame:
        frame["clean_title"] = frame["cleaned_text"]
    defaults = {
        "clean_title": "",
        "timestamp": "",
        "user_id": "",
        "local_image_paths": "[]",
        "local_video_paths": "[]",
        "image_urls": "[]",
        "video_urls": "[]",
        "repost_count": 0,
        "comment_count": 0,
        "like_count": 0,
        "num_comments": 0,
        "num_reposts": 0,
        "has_image": False,
        "has_video": False,
        "has_comments": False,
        "has_reposts": False,
    }
    for column, default in defaults.items():
        if column not in frame:
            frame[column] = default
    frame["id"] = frame["id"].astype(str)
    return frame


@dataclass
class ResponseGraphConfig:
    enabled: bool = False
    dataset_root: str = "CHECKED/dataset"
    comment_csv: str = ""
    repost_csv: str = ""
    response_window_hours: float = 24.0
    include_late_responses: bool = False
    max_comments_per_post: int = 8
    max_reposts_per_post: int = 8
    fair_mode: bool = False


@dataclass
class ResponseLoadResult:
    comments: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    reposts: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    report: Dict[str, Any] = field(default_factory=dict)


def _resolve_dataset_root(value: str) -> Path:
    path = Path(value)
    candidates = [path, PROJECT_ROOT / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def _response_hours(post_time: Any, response_time: Any) -> Optional[float]:
    def parse_timestamp(value: Any) -> Optional[datetime]:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            for pattern in ("%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M:%S"):
                try:
                    return datetime.strptime(text, pattern)
                except ValueError:
                    continue
        return None

    post_timestamp = parse_timestamp(post_time)
    response_timestamp = parse_timestamp(response_time)
    if post_timestamp is None or response_timestamp is None:
        return None
    return float((response_timestamp - post_timestamp).total_seconds() / 3600.0)


def _normalize_response(
    raw: Mapping[str, Any],
    post_id: str,
    post_time: Any,
    response_type: str,
    sequence: int,
) -> Dict[str, Any]:
    hours = _response_hours(post_time, raw.get("date", raw.get("timestamp", "")))
    raw_response_id = str(
        raw.get("id") or raw.get(f"{response_type}_id") or sequence
    )
    response_id = f"{post_id}:{response_type}:{raw_response_id}"
    media = parse_json_list(raw.get("pic_url", raw.get("image_urls", "[]")))
    return {
        "id": response_id,
        "post_id": post_id,
        "text": str(raw.get("text", "") or ""),
        "timestamp": str(raw.get("date", raw.get("timestamp", "")) or ""),
        "user_id": str(raw.get("user_id", "") or ""),
        "hours_since_post": hours,
        "has_media": bool(media),
    }


def _filter_and_cap_responses(
    records: Iterable[Dict[str, Any]],
    config: ResponseGraphConfig,
    cap: int,
    counters: Dict[str, int],
) -> List[Dict[str, Any]]:
    eligible: List[Tuple[Tuple[bool, float, str, int], Dict[str, Any]]] = []
    eligible_count = 0
    for record in records:
        hours = record["hours_since_post"]
        if hours is None:
            counters["missing_timestamp"] += 1
            if config.fair_mode and not config.include_late_responses:
                counters["removed_missing_timestamp"] += 1
                continue
        elif hours < 0:
            counters["removed_negative_time"] += 1
            continue
        elif hours > config.response_window_hours and not config.include_late_responses:
            counters["removed_after_window"] += 1
            continue
        eligible_count += 1
        key = (
            record["hours_since_post"] is None,
            record["hours_since_post"] if record["hours_since_post"] is not None else math.inf,
            record["id"],
            eligible_count,
        )
        insort(eligible, (key, record))
        if cap > 0 and len(eligible) > cap:
            eligible.pop()
    selected = [record for _, record in eligible]
    counters["removed_by_cap"] += max(0, eligible_count - len(selected))
    counters["included"] += len(selected)
    return selected


def _load_response_csv(
    csv_path: str,
    response_type: str,
    post_times: Mapping[str, Any],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    path = Path(csv_path)
    if not path.exists():
        return {}, {"available": False, "reason": f"File not found: {path}"}
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    post_column = next(
        (column for column in ["microblog_id", "post_id", "item_id"] if column in frame),
        None,
    )
    if post_column is None:
        return {}, {"available": False, "reason": f"{path} has no post ID column"}
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sequence, row in frame.iterrows():
        post_id = str(row.get(post_column, ""))
        if post_id not in post_times:
            continue
        grouped[post_id].append(
            _normalize_response(row.to_dict(), post_id, post_times[post_id], response_type, int(sequence))
        )
    has_timestamps = any(
        record["timestamp"]
        for records in grouped.values()
        for record in records
    )
    has_text = any(record["text"] for records in grouped.values() for record in records)
    return dict(grouped), {
        "available": True,
        "path": str(path),
        "rows_for_split": sum(map(len, grouped.values())),
        "has_timestamps": has_timestamps,
        "has_text": has_text,
    }


def load_checked_responses(
    df: pd.DataFrame,
    config: ResponseGraphConfig,
) -> ResponseLoadResult:
    frame = normalize_checked_dataframe(df)
    post_times = dict(zip(frame["id"].astype(str), frame["timestamp"]))
    report: Dict[str, Any] = {
        "requested": bool(config.enabled),
        "available": False,
        "source": None,
        "response_window_hours": config.response_window_hours,
        "include_late_responses": config.include_late_responses,
        "max_comments_per_post": config.max_comments_per_post,
        "max_reposts_per_post": config.max_reposts_per_post,
        "warnings": [],
    }
    if not config.enabled:
        report["reason"] = "Response graph was not requested."
        return ResponseLoadResult(report=report)

    raw_comments: Dict[str, List[Dict[str, Any]]] = {}
    raw_reposts: Dict[str, List[Dict[str, Any]]] = {}
    selected_comments: Dict[str, List[Dict[str, Any]]] = {}
    selected_reposts: Dict[str, List[Dict[str, Any]]] = {}
    counters_by_type = {
        response_type: {
            "available": 0,
            "included": 0,
            "missing_timestamp": 0,
            "removed_missing_timestamp": 0,
            "removed_negative_time": 0,
            "removed_after_window": 0,
            "removed_by_cap": 0,
        }
        for response_type in ["comments", "reposts"]
    }
    source_details: Dict[str, Any] = {}
    responses_already_selected = False
    if config.comment_csv or config.repost_csv:
        if config.comment_csv:
            raw_comments, source_details["comments"] = _load_response_csv(
                config.comment_csv, "comment", post_times
            )
        if config.repost_csv:
            raw_reposts, source_details["reposts"] = _load_response_csv(
                config.repost_csv, "repost", post_times
            )
        report["source"] = "csv"
        report["source_details"] = source_details
        for response_type, details in source_details.items():
            if details.get("available") and not details.get("has_timestamps"):
                report["warnings"].append(
                    f"{response_type} CSV has no timestamps; fair mode will exclude those responses."
                )
    else:
        responses_already_selected = True
        dataset_root = _resolve_dataset_root(config.dataset_root)
        report["source"] = "raw_json"
        report["dataset_root"] = str(dataset_root)
        missing_files = 0
        post_items = list(post_times.items())
        print(
            f"Loading and filtering response graph from {dataset_root} "
            f"for {len(post_items)} split-local posts...",
            flush=True,
        )
        for post_number, (post_id, post_time) in enumerate(post_items, start=1):
            if post_number == 1 or post_number % 100 == 0 or post_number == len(post_items):
                print(
                    f"  Response graph progress: {post_number}/{len(post_items)} posts",
                    flush=True,
                )
            candidates = [
                dataset_root / "fake_news" / f"{post_id}.json",
                dataset_root / "real_news" / f"{post_id}.json",
            ]
            path = next((candidate for candidate in candidates if candidate.exists()), None)
            if path is None:
                missing_files += 1
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                report["warnings"].append(f"Could not parse {path}: {exc}")
                continue
            for response_type, target, cap in [
                ("comment", selected_comments, config.max_comments_per_post),
                ("repost", selected_reposts, config.max_reposts_per_post),
            ]:
                key = f"{response_type}s"
                raw_records = payload.get(key) or []
                counters = counters_by_type[key]
                counters["available"] += len(raw_records)
                records = (
                    _normalize_response(raw, post_id, post_time, response_type, sequence)
                    for sequence, raw in enumerate(payload.get(key) or [])
                )
                selected = _filter_and_cap_responses(records, config, cap, counters)
                if selected:
                    target[post_id] = selected
        report["missing_post_files"] = missing_files

    if not responses_already_selected:
        for response_type, source, target, cap in [
            ("comments", raw_comments, selected_comments, config.max_comments_per_post),
            ("reposts", raw_reposts, selected_reposts, config.max_reposts_per_post),
        ]:
            counters = counters_by_type[response_type]
            counters["available"] = sum(map(len, source.values()))
            for post_id, records in source.items():
                selected = _filter_and_cap_responses(records, config, cap, counters)
                if selected:
                    target[post_id] = selected

    report["counts"] = counters_by_type
    included = sum(counters["included"] for counters in counters_by_type.values())
    report["available"] = included > 0
    if not report["available"]:
        report["reason"] = "No responses remained after availability, timing, and cap checks."
    if config.fair_mode:
        missing = sum(counters["missing_timestamp"] for counters in counters_by_type.values())
        if missing:
            report["warnings"].append(
                f"{missing} responses lacked usable timestamps and were excluded in fair mode."
            )
    return ResponseLoadResult(selected_comments, selected_reposts, report)


class CheckedGraphBuilder:
    """Build a split-local CHECKED graph with explicit safe graph variants."""

    def __init__(
        self,
        post_feature_dim: int = 768,
        user_embed_dim: int = 64,
        media_embed_dim: int = len(MEDIA_FEATURE_NAMES),
        response_feature_dim: int = len(RESPONSE_FEATURE_NAMES),
        seed: int = 42,
        use_user_graph: bool = True,
        graph_variant: str = "media_only",
        metadata_mode: str = "none",
        response_config: Optional[ResponseGraphConfig] = None,
    ):
        if graph_variant not in GRAPH_VARIANTS:
            raise ValueError(f"Unknown graph variant {graph_variant!r}; choose from {sorted(GRAPH_VARIANTS)}")
        self.base_post_feature_dim = post_feature_dim
        self.user_embed_dim = user_embed_dim
        self.media_embed_dim = media_embed_dim
        self.response_feature_dim = response_feature_dim
        self.seed = seed
        self.use_user_graph = use_user_graph
        self.graph_variant = graph_variant
        self.metadata_mode = metadata_mode
        self.response_config = response_config or ResponseGraphConfig()
        self.post_to_idx: Dict[str, int] = {}
        self.user_to_idx: Dict[str, int] = {}
        self.image_to_idx: Dict[str, int] = {}
        self.video_to_idx: Dict[str, int] = {}
        self.comment_to_idx: Dict[str, int] = {}
        self.repost_to_idx: Dict[str, int] = {}
        self.last_report: Dict[str, Any] = {}
        self.response_result = ResponseLoadResult(report={"available": False, "requested": False})
        self.post_metadata_names = self._post_metadata_names()
        self.include_post_text = self.graph_variant not in {"metadata_only", "response_metadata"}
        self.post_feature_dim = (
            self.base_post_feature_dim if self.include_post_text else 0
        ) + len(self.post_metadata_names)
        if self.post_feature_dim <= 0:
            raise ValueError(
                f"Graph variant {self.graph_variant!r} has no post features under "
                f"metadata_mode={self.metadata_mode!r}."
            )

    @property
    def uses_media(self) -> bool:
        return self.graph_variant in {"media_only", "full_safe", "upper_bound"}

    @property
    def uses_responses(self) -> bool:
        return self.graph_variant in {"response_only", "response_metadata", "full_safe", "upper_bound"}

    @property
    def uses_metadata(self) -> bool:
        return self.graph_variant in {"metadata_only", "response_metadata", "full_safe", "upper_bound"}

    def _post_metadata_names(self) -> List[str]:
        if not self.uses_metadata:
            return []
        if self.graph_variant == "upper_bound" or self.metadata_mode == "upper_bound":
            return list(UPPER_BOUND_POST_METADATA)
        if self.metadata_mode == "none":
            return []
        return list(SAFE_POST_METADATA)

    def build_graph(
        self,
        df: pd.DataFrame,
        post_features: Optional[torch.Tensor] = None,
        responses: Optional[ResponseLoadResult] = None,
    ) -> HeteroData:
        frame = normalize_checked_dataframe(df).reset_index(drop=True)
        self.response_result = responses or (
            load_checked_responses(frame, self.response_config)
            if self.uses_responses or (self.uses_metadata and self.response_config.enabled)
            else ResponseLoadResult(report={"requested": False, "available": False, "reason": "Variant excludes responses."})
        )
        self._build_mappings(frame)
        data = HeteroData()

        if self.include_post_text:
            base_post = (
                post_features.float()
                if post_features is not None and post_features.size(0) == len(self.post_to_idx)
                else self._numeric_post_features(frame, self.base_post_feature_dim)
            )
        else:
            base_post = torch.empty((len(frame), 0), dtype=torch.float32)
        metadata = self._post_metadata_features(frame)
        data["post"].x = torch.cat([base_post, metadata], dim=1) if metadata.numel() else base_post
        data["user"].x = self._user_features()
        data["image"].x = self._media_features(self.image_to_idx, "image")
        data["video"].x = self._media_features(self.video_to_idx, "video")
        data["comment"].x = self._response_features(self.comment_to_idx, self.response_result.comments)
        data["repost"].x = self._response_features(self.repost_to_idx, self.response_result.reposts)

        edge_index = self._build_edges(frame)
        for edge_type in CHECKED_EDGE_TYPES:
            data[edge_type].edge_index = edge_index.get(
                edge_type, torch.empty((2, 0), dtype=torch.long)
            )
        data["post"].post_ids = list(self.post_to_idx.keys())
        self.last_report = self._build_report(frame, data)
        self._print_report(self.last_report)
        return data

    def _build_mappings(self, frame: pd.DataFrame) -> None:
        self.post_to_idx = {}
        self.user_to_idx = {}
        self.image_to_idx = {}
        self.video_to_idx = {}
        self.comment_to_idx = {}
        self.repost_to_idx = {}
        for _, row in frame.iterrows():
            post_id = str(row["id"])
            self.post_to_idx.setdefault(post_id, len(self.post_to_idx))
            user_id = str(row.get("user_id", "")).strip()
            if self.use_user_graph and user_id:
                self.user_to_idx.setdefault(user_id, len(self.user_to_idx))
            if self.uses_media:
                for image_ref in self._media_refs(row, "image"):
                    self.image_to_idx.setdefault(image_ref, len(self.image_to_idx))
                for video_ref in self._media_refs(row, "video"):
                    self.video_to_idx.setdefault(video_ref, len(self.video_to_idx))
            if self.uses_responses:
                for response in self.response_result.comments.get(post_id, []):
                    self.comment_to_idx.setdefault(response["id"], len(self.comment_to_idx))
                    if self.use_user_graph and response["user_id"]:
                        self.user_to_idx.setdefault(response["user_id"], len(self.user_to_idx))
                for response in self.response_result.reposts.get(post_id, []):
                    self.repost_to_idx.setdefault(response["id"], len(self.repost_to_idx))
                    if self.use_user_graph and response["user_id"]:
                        self.user_to_idx.setdefault(response["user_id"], len(self.user_to_idx))

    def _media_refs(self, row: pd.Series, media_type: str) -> List[str]:
        if media_type == "image":
            local = parse_json_list(row.get("local_image_paths", "[]"))
            urls = parse_json_list(row.get("image_urls", "[]"))
        else:
            local = parse_json_list(row.get("local_video_paths", "[]"))
            urls = parse_json_list(row.get("video_urls", "[]"))
        return local or urls

    def _build_edges(self, frame: pd.DataFrame) -> Dict[Tuple[str, str, str], torch.Tensor]:
        edge_lists: Dict[Tuple[str, str, str], Tuple[List[int], List[int]]] = {
            edge_type: ([], []) for edge_type in CHECKED_EDGE_TYPES
        }

        def add(edge_type: Tuple[str, str, str], source: int, target: int) -> None:
            edge_lists[edge_type][0].append(source)
            edge_lists[edge_type][1].append(target)

        for _, row in frame.iterrows():
            post_id = str(row["id"])
            post_idx = self.post_to_idx[post_id]
            user_id = str(row.get("user_id", "")).strip()
            if self.use_user_graph and user_id in self.user_to_idx:
                user_idx = self.user_to_idx[user_id]
                add(("user", "posted", "post"), user_idx, post_idx)
                add(("post", "rev_posted", "user"), post_idx, user_idx)
            if self.uses_media:
                for image_ref in self._media_refs(row, "image"):
                    image_idx = self.image_to_idx[image_ref]
                    add(("post", "has_image", "image"), post_idx, image_idx)
                    add(("image", "rev_has_image", "post"), image_idx, post_idx)
                for video_ref in self._media_refs(row, "video"):
                    video_idx = self.video_to_idx[video_ref]
                    add(("post", "has_video", "video"), post_idx, video_idx)
                    add(("video", "rev_has_video", "post"), video_idx, post_idx)
            if self.uses_responses:
                for response in self.response_result.comments.get(post_id, []):
                    response_idx = self.comment_to_idx[response["id"]]
                    add(("post", "has_comment", "comment"), post_idx, response_idx)
                    add(("comment", "belongs_to_post", "post"), response_idx, post_idx)
                    if self.use_user_graph and response["user_id"] in self.user_to_idx:
                        user_idx = self.user_to_idx[response["user_id"]]
                        add(("user", "wrote_comment", "comment"), user_idx, response_idx)
                        add(("comment", "rev_wrote_comment", "user"), response_idx, user_idx)
                for response in self.response_result.reposts.get(post_id, []):
                    response_idx = self.repost_to_idx[response["id"]]
                    add(("post", "has_repost", "repost"), post_idx, response_idx)
                    add(("repost", "belongs_to_post", "post"), response_idx, post_idx)
                    if self.use_user_graph and response["user_id"] in self.user_to_idx:
                        user_idx = self.user_to_idx[response["user_id"]]
                        add(("user", "made_repost", "repost"), user_idx, response_idx)
                        add(("repost", "rev_made_repost", "user"), response_idx, user_idx)
        return {
            edge_type: (
                torch.tensor([source, target], dtype=torch.long)
                if source
                else torch.empty((2, 0), dtype=torch.long)
            )
            for edge_type, (source, target) in edge_lists.items()
        }

    def _numeric_post_features(self, frame: pd.DataFrame, dimension: int) -> torch.Tensor:
        rows = []
        for _, row in frame.iterrows():
            values = [
                math.log1p(max(0.0, safe_float(row.get(column, 0))))
                for column in ["repost_count", "comment_count", "like_count", "num_comments", "num_reposts"]
            ]
            values.extend(
                1.0 if str(row.get(column, "")).lower() == "true" else 0.0
                for column in ["has_image", "has_video", "has_comments", "has_reposts"]
            )
            rows.append(values)
        base = torch.tensor(rows, dtype=torch.float32) if rows else torch.empty((0, 9))
        if base.size(1) >= dimension:
            return base[:, :dimension]
        return torch.cat([base, torch.zeros((base.size(0), dimension - base.size(1)))], dim=1)

    def _post_metadata_features(self, frame: pd.DataFrame) -> torch.Tensor:
        if not self.post_metadata_names:
            return torch.empty((len(frame), 0), dtype=torch.float32)
        rows = []
        for _, row in frame.iterrows():
            post_id = str(row["id"])
            comments = self.response_result.comments.get(post_id, [])
            reposts = self.response_result.reposts.get(post_id, [])
            values = {
                "text_length": math.log1p(len(str(row.get("clean_title", "")))),
                "media_count": math.log1p(
                    len(self._media_refs(row, "image")) + len(self._media_refs(row, "video"))
                ),
                "has_image": float(bool(self._media_refs(row, "image"))),
                "has_video": float(bool(self._media_refs(row, "video"))),
                "early_comment_count": math.log1p(len(comments)),
                "early_repost_count": math.log1p(len(reposts)),
                "has_early_comments": float(bool(comments)),
                "has_early_reposts": float(bool(reposts)),
                "unique_early_commenters": math.log1p(
                    len({response["user_id"] for response in comments if response["user_id"]})
                ),
                "unique_early_reposters": math.log1p(
                    len({response["user_id"] for response in reposts if response["user_id"]})
                ),
                "time_to_first_comment_hours": math.log1p(
                    max(0.0, comments[0]["hours_since_post"])
                )
                if comments and comments[0]["hours_since_post"] is not None
                else 0.0,
                "mean_comment_delay_hours": math.log1p(
                    np.mean(
                        [
                            max(0.0, response["hours_since_post"])
                            for response in comments
                            if response["hours_since_post"] is not None
                        ]
                    )
                )
                if any(response["hours_since_post"] is not None for response in comments)
                else 0.0,
                "mean_repost_delay_hours": math.log1p(
                    np.mean(
                        [
                            max(0.0, response["hours_since_post"])
                            for response in reposts
                            if response["hours_since_post"] is not None
                        ]
                    )
                )
                if any(response["hours_since_post"] is not None for response in reposts)
                else 0.0,
                "time_to_first_repost_hours": math.log1p(
                    max(0.0, reposts[0]["hours_since_post"])
                )
                if reposts and reposts[0]["hours_since_post"] is not None
                else 0.0,
                "comment_count": math.log1p(max(0.0, safe_float(row.get("comment_count", 0)))),
                "repost_count": math.log1p(max(0.0, safe_float(row.get("repost_count", 0)))),
                "like_count": math.log1p(max(0.0, safe_float(row.get("like_count", 0)))),
                "num_comments": math.log1p(max(0.0, safe_float(row.get("num_comments", 0)))),
                "num_reposts": math.log1p(max(0.0, safe_float(row.get("num_reposts", 0)))),
            }
            rows.append([values[name] for name in self.post_metadata_names])
        return torch.tensor(rows, dtype=torch.float32)

    def _user_features(self) -> torch.Tensor:
        count = len(self.user_to_idx)
        if count == 0:
            return torch.empty((0, self.user_embed_dim), dtype=torch.float32)
        # Identity nodes are upper-bound only. Their feature is deliberately
        # constant so identity enters only through topology, never label rates.
        return torch.ones((count, self.user_embed_dim), dtype=torch.float32)

    def _media_features(self, mapping: Mapping[str, int], media_type: str) -> torch.Tensor:
        rows = []
        for reference, _ in sorted(mapping.items(), key=lambda item: item[1]):
            normalized = reference.replace("\\", "/")
            is_remote = normalized.lower().startswith(("http://", "https://"))
            path = Path(normalized)
            candidates = [path, PROJECT_ROOT / path, PROJECT_ROOT / "CHECKED" / path]
            file_available = any(candidate.exists() and candidate.is_file() for candidate in candidates)
            extension_known = bool(path.suffix)
            base = [
                float(not is_remote),
                float(is_remote),
                float(media_type == "image"),
                float(media_type == "video"),
                float(extension_known),
                float(file_available),
            ]
            if self.media_embed_dim > len(base):
                base.extend([0.0] * (self.media_embed_dim - len(base)))
            rows.append(base[: self.media_embed_dim])
        return (
            torch.tensor(rows, dtype=torch.float32)
            if rows
            else torch.empty((0, self.media_embed_dim), dtype=torch.float32)
        )

    def _response_features(
        self,
        mapping: Mapping[str, int],
        by_post: Mapping[str, Sequence[Mapping[str, Any]]],
    ) -> torch.Tensor:
        records = {
            str(record["id"]): record
            for responses in by_post.values()
            for record in responses
        }
        rows = []
        for response_id, _ in sorted(mapping.items(), key=lambda item: item[1]):
            record = records[response_id]
            hours = record["hours_since_post"]
            timestamp = pd.to_datetime(record["timestamp"], errors="coerce")
            hour = int(timestamp.hour) if not pd.isna(timestamp) else 0
            base = [
                math.log1p(len(record["text"])),
                math.log1p(max(0.0, hours)) if hours is not None else 0.0,
                float(hours is not None and hours <= self.response_config.response_window_hours),
                float(bool(record["text"].strip())),
                float(record["has_media"]),
                math.sin(2.0 * math.pi * hour / 24.0),
                math.cos(2.0 * math.pi * hour / 24.0),
                float(hours is None),
            ]
            if self.response_feature_dim > len(base):
                base.extend([0.0] * (self.response_feature_dim - len(base)))
            rows.append(base[: self.response_feature_dim])
        return (
            torch.tensor(rows, dtype=torch.float32)
            if rows
            else torch.empty((0, self.response_feature_dim), dtype=torch.float32)
        )

    def _build_report(self, frame: pd.DataFrame, data: HeteroData) -> Dict[str, Any]:
        post_count = len(frame)
        comments = len(self.comment_to_idx)
        reposts = len(self.repost_to_idx)
        posts_with_comments = sum(bool(self.response_result.comments.get(str(row["id"]))) for _, row in frame.iterrows())
        posts_with_reposts = sum(bool(self.response_result.reposts.get(str(row["id"]))) for _, row in frame.iterrows())
        response_counts = self.response_result.report.get("counts", {})
        return {
            "graph_variant": self.graph_variant,
            "metadata_mode": self.metadata_mode,
            "split_local": True,
            "raw_labels_stored_in_graph": False,
            "source_feature_used": False,
            "node_counts": {
                "post": post_count,
                "image": len(self.image_to_idx),
                "video": len(self.video_to_idx),
                "comment": comments,
                "repost": reposts,
                "raw_user": len(self.user_to_idx),
            },
            "edge_counts": {
                "|".join(edge_type): int(data[edge_type].edge_index.size(1))
                for edge_type in CHECKED_EDGE_TYPES
            },
            "average_comments_per_post": comments / max(1, post_count),
            "average_reposts_per_post": reposts / max(1, post_count),
            "posts_with_no_comments_percent": 100.0 * (post_count - posts_with_comments) / max(1, post_count),
            "posts_with_no_reposts_percent": 100.0 * (post_count - posts_with_reposts) / max(1, post_count),
            "posts_with_image_percent": 100.0 * sum(bool(self._media_refs(row, "image")) for _, row in frame.iterrows()) / max(1, post_count),
            "posts_with_video_percent": 100.0 * sum(bool(self._media_refs(row, "video")) for _, row in frame.iterrows()) / max(1, post_count),
            "responses_removed_by_window": sum(
                counts.get("removed_after_window", 0) for counts in response_counts.values()
            ),
            "responses_removed_missing_timestamp": sum(
                counts.get("removed_missing_timestamp", 0) for counts in response_counts.values()
            ),
            "responses_removed_by_cap": sum(
                counts.get("removed_by_cap", 0) for counts in response_counts.values()
            ),
            "post_feature_dimension": self.post_feature_dim,
            "frozen_post_text_embedding_enabled": self.include_post_text,
            "post_metadata_features": self.post_metadata_names,
            "response_feature_names": RESPONSE_FEATURE_NAMES if self.uses_responses else [],
            "media_feature_names": MEDIA_FEATURE_NAMES if self.uses_media else [],
            "graph_feature_count": (
                self.post_feature_dim
                + (len(RESPONSE_FEATURE_NAMES) if self.uses_responses else 0)
                + (len(MEDIA_FEATURE_NAMES) if self.uses_media else 0)
            ),
            "unsafe_features_disabled": [
                "source",
                "raw labels",
                "label-derived credibility",
                *([] if self.use_user_graph else ["raw user identity"]),
                *([] if self.metadata_mode == "upper_bound" else ["lifetime engagement counts"]),
                *([] if self.response_config.include_late_responses else ["late responses"]),
            ],
            "response_graph": self.response_result.report,
            "image_graph_features": "Media availability metadata; no ViT embeddings in graph nodes.",
            "post_graph_features": (
                "Frozen transformer embeddings plus declared graph metadata."
                if self.include_post_text and self.post_metadata_names
                else "Frozen transformer embeddings."
                if self.include_post_text
                else "Declared graph metadata only; frozen post text excluded for this ablation."
            ),
            "comment_repost_text_embeddings": False,
        }

    @staticmethod
    def _print_report(report: Mapping[str, Any]) -> None:
        counts = report["node_counts"]
        print("CHECKED graph statistics:")
        for node_type in ["post", "raw_user", "image", "video", "comment", "repost"]:
            print(f"  {node_type.replace('_', ' ').title()}: {counts[node_type]}")
        for relation, count in report["edge_counts"].items():
            print(f"  Edges {tuple(relation.split('|'))}: {count}")
        print(f"  Variant: {report['graph_variant']}")
        print(f"  Post metadata: {report['post_metadata_features']}")
        if report["response_graph"].get("requested") and not report["response_graph"].get("available"):
            print(f"  WARNING: response graph unavailable: {report['response_graph'].get('reason')}")


class CheckedGNNEncoder(nn.Module):
    """Heterogeneous message-passing encoder returning post embeddings."""

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        post_in_channels: int = 768,
        user_in_channels: int = 64,
        media_in_channels: int = len(MEDIA_FEATURE_NAMES),
        response_in_channels: int = len(RESPONSE_FEATURE_NAMES),
    ):
        super().__init__()
        self.dropout = dropout
        self.input_projections = nn.ModuleDict(
            {
                "post": Linear(post_in_channels, hidden_channels),
                "user": Linear(user_in_channels, hidden_channels),
                "image": Linear(media_in_channels, hidden_channels),
                "video": Linear(media_in_channels, hidden_channels),
                "comment": Linear(response_in_channels, hidden_channels),
                "repost": Linear(response_in_channels, hidden_channels),
            }
        )
        self.convs = nn.ModuleList(
            [
                HeteroConv(
                    {
                        edge_type: SAGEConv(
                            (hidden_channels, hidden_channels), hidden_channels
                        )
                        for edge_type in CHECKED_EDGE_TYPES
                    },
                    aggr="sum",
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict) -> torch.Tensor:
        hidden = {
            node_type: F.relu(self.input_projections[node_type](features.float()))
            for node_type, features in x_dict.items()
            if node_type in self.input_projections
        }
        for conv in self.convs:
            output = conv(hidden, edge_index_dict)
            hidden = {
                node_type: F.dropout(
                    F.relu(output.get(node_type, value)),
                    p=self.dropout,
                    training=self.training,
                )
                for node_type, value in hidden.items()
            }
        return hidden["post"]


class HeteroGNN(nn.Module):
    def __init__(
        self,
        hidden_channels: int = 128,
        out_channels: int = 2,
        num_layers: int = 2,
        dropout: float = 0.2,
        post_in_channels: int = 768,
        user_in_channels: int = 64,
        media_in_channels: int = len(MEDIA_FEATURE_NAMES),
        response_in_channels: int = len(RESPONSE_FEATURE_NAMES),
        **_: Any,
    ):
        super().__init__()
        self.encoder = CheckedGNNEncoder(
            hidden_channels,
            num_layers,
            dropout,
            post_in_channels,
            user_in_channels,
            media_in_channels,
            response_in_channels,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels),
        )

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict) -> torch.Tensor:
        return self.classifier(self.encoder(x_dict, edge_index_dict))


class HGTModel(HeteroGNN):
    pass


FakedditGraphBuilder = CheckedGraphBuilder


if __name__ == "__main__":
    csv_path = Path("CHECKED/data_checked/processed/train.csv")
    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False).head(8)
    graph_builder = CheckedGraphBuilder(post_feature_dim=16, use_user_graph=False)
    graph = graph_builder.build_graph(frame)
    model = HeteroGNN(hidden_channels=32, post_in_channels=graph["post"].x.size(1))
    output = model(graph.x_dict, graph.edge_index_dict)
    print(f"Logits shape: {tuple(output.shape)}")
