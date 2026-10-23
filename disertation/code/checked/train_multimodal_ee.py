"""CHECKED-compatible multimodal training components.

The original scripts expected Fakeddit/Reddit TSV fields. This module keeps the
same public names used by those scripts while adapting the data path to CHECKED:
`item_id`, `cleaned_text`, `label_id`, `user_id`, local media paths, and
microblog/media graph relations.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from gnn_module import (
    MEDIA_FEATURE_NAMES,
    RESPONSE_FEATURE_NAMES,
    CheckedGraphBuilder,
    CheckedGNNEncoder,
    ResponseGraphConfig,
    ResponseLoadResult,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OFFLINE_MODEL_ROOT = Path(os.environ.get("CHECKED_OFFLINE_MODEL_ROOT", PROJECT_ROOT / "offline_models"))
OFFLINE_TEXT_MODEL_DIR = DEFAULT_OFFLINE_MODEL_ROOT / "bert-base-chinese"
OFFLINE_VIT_MODEL_DIR = DEFAULT_OFFLINE_MODEL_ROOT / "google--vit-base-patch16-224-in21k"


def default_model_name(repo_id: str, local_dir: Path) -> str:
    return str(local_dir) if local_dir.exists() else repo_id


def use_hf_local_files_only(model_name: str) -> bool:
    setting = os.environ.get("CHECKED_HF_LOCAL_FILES_ONLY", "auto").strip().lower()
    if setting in {"1", "true", "yes", "on"}:
        return True
    if setting in {"0", "false", "no", "off"}:
        return False
    return Path(model_name).exists()


def hf_load_kwargs(model_name: str, **kwargs):
    if use_hf_local_files_only(model_name) and "local_files_only" not in kwargs:
        kwargs["local_files_only"] = True
    return kwargs


def parse_json_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []
    if text[:1] in {"[", "{"}:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(v) for v in parsed if str(v).strip()]
            if isinstance(parsed, dict):
                return [str(v) for v in parsed.values() if str(v).strip()]
        except json.JSONDecodeError:
            pass
    return [text]


def read_checked_table(path: os.PathLike[str] | str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() and path.parts and path.parts[0].lower() == "checked":
        alt = Path(*path.parts[1:])
        if alt.exists():
            path = alt
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    return pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)


def normalize_checked_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the legacy column names expected by the old scripts."""
    df = df.copy()
    if "item_id" in df.columns and "id" not in df.columns:
        df["id"] = df["item_id"]
    if "cleaned_text" in df.columns and "clean_title" not in df.columns:
        df["clean_title"] = df["cleaned_text"]
    elif "text" in df.columns and "clean_title" not in df.columns:
        df["clean_title"] = df["text"]
    if "label_id" in df.columns and "2_way_label" not in df.columns:
        df["2_way_label"] = df["label_id"]
    if "label" not in df.columns and "2_way_label" in df.columns:
        df["label"] = df["2_way_label"].astype(str).map({"0": "real", "1": "fake"}).fillna("")

    defaults = {
        "local_image_paths": "[]",
        "local_video_paths": "[]",
        "image_urls": "[]",
        "video_urls": "[]",
        "user_id": "",
        "repost_count": "0",
        "comment_count": "0",
        "like_count": "0",
        "num_comments": "0",
        "num_reposts": "0",
        "has_image": "False",
        "has_video": "False",
        "has_comments": "False",
        "has_reposts": "False",
    }
    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default

    df["id"] = df["id"].astype(str)
    df["item_id"] = df.get("item_id", df["id"]).astype(str)
    df["clean_title"] = df["clean_title"].fillna("").astype(str)
    df["label"] = df["label"].fillna("").astype(str).str.lower()
    df["2_way_label"] = df["2_way_label"].astype(int)
    df["label_id"] = df["2_way_label"].astype(int)
    return df


class Config:
    TEXT_MODEL_NAME = os.environ.get("CHECKED_TEXT_MODEL", default_model_name("bert-base-chinese", OFFLINE_TEXT_MODEL_DIR))
    VIT_MODEL_NAME = os.environ.get("CHECKED_VIT_MODEL", default_model_name("google/vit-base-patch16-224-in21k", OFFLINE_VIT_MODEL_DIR))
    MAX_LEN = int(os.environ.get("CHECKED_MAX_LEN", "192"))
    IMAGE_ROOT = os.environ.get("CHECKED_IMAGE_ROOT", "CHECKED")
    DOMAIN_WEIGHTS_JSON = os.environ.get("CHECKED_CREDIBILITY_JSON", "CHECKED/data_checked/reports/user_credibility.json")

    USE_TEXT = True
    USE_IMAGE = True
    USE_GRAPH = True
    USE_PROPAGATION = False
    USE_CREDIBILITY = False
    USE_SOCIAL_GRAPH = False
    USE_USER_GRAPH = True
    USE_RESPONSE_GRAPH = False
    GRAPH_VARIANT = "media_only"
    METADATA_MODE = "none"

    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D_PROJ = 256
    GNN_HIDDEN_DIM = 128
    POST_FEATURE_DIM = 768
    USER_EMBED_DIM = 64
    SUBREDDIT_EMBED_DIM = 64
    DOMAIN_EMBED_DIM = 64
    MEDIA_EMBED_DIM = len(MEDIA_FEATURE_NAMES)
    RESPONSE_FEATURE_DIM = len(RESPONSE_FEATURE_NAMES)
    FUSION_TYPE = "concat"
    CREDIBILITY_WEIGHT = 1.0
    SOCIAL_GRAPH_ROOT = ""
    PROPAGATION_FEATURE_COLUMNS = [
        "repost_count",
        "comment_count",
        "like_count",
        "num_comments",
        "num_reposts",
    ]
    PROPAGATION_FEATURE_DIM = len(PROPAGATION_FEATURE_COLUMNS)

    @classmethod
    def validate(cls) -> None:
        if not any([cls.USE_TEXT, cls.USE_IMAGE, cls.USE_GRAPH, cls.USE_PROPAGATION, cls.USE_CREDIBILITY]):
            raise ValueError("At least one modality must be enabled.")

    @classmethod
    def get_modality_string(cls) -> str:
        parts = []
        if cls.USE_TEXT:
            parts.append("text")
        if cls.USE_IMAGE:
            parts.append("image")
        if cls.USE_GRAPH:
            parts.append("graph")
        if cls.USE_PROPAGATION:
            parts.append("prop")
        if cls.USE_CREDIBILITY:
            parts.append("cred")
        return "_".join(parts) if parts else "none"


def load_hf_model(model_cls, model_name: str, **kwargs):
    """Load Hugging Face models through safetensors when available.

    Newer transformers versions block loading `.bin` weights with torch < 2.6.
    This helper makes the failure actionable instead of surfacing a long stack
    trace from torch.load.
    """
    kwargs = hf_load_kwargs(model_name, **kwargs)
    try:
        return model_cls.from_pretrained(model_name, use_safetensors=True, **kwargs)
    except (OSError, ValueError) as exc:
        message = str(exc)
        if "safetensors" in message.lower() or "torch.load" in message or "CVE-2025-32434" in message:
            raise RuntimeError(
                "Could not load Hugging Face model safely. Use a model with safetensors "
                "(default: bert-base-chinese) or upgrade this environment to torch>=2.6. "
                f"Original error for {model_name}: {exc}"
            ) from exc
        raise


def load_domain_weights(path: str) -> Tuple[Dict[str, float], float]:
    weight_path = Path(path)
    if not weight_path.exists():
        return {}, 0.5
    payload = json.loads(weight_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "weights" in payload:
        weights = {str(k): float(v) for k, v in payload.get("weights", {}).items()}
    else:
        weights = {str(k): float(v) for k, v in payload.items()}
    mean_weight = float(np.mean(list(weights.values()))) if weights else 0.5
    return weights, mean_weight


def _text_series(df: pd.DataFrame) -> List[str]:
    column = "clean_title" if "clean_title" in df.columns else "cleaned_text"
    return df[column].fillna("").astype(str).tolist()


@torch.no_grad()
def extract_post_text_features(
    df: pd.DataFrame,
    tokenizer,
    text_model,
    device: torch.device,
    batch_size: int = 32,
    max_len: int = 192,
) -> torch.Tensor:
    texts = _text_series(df)
    outputs: List[torch.Tensor] = []
    text_model.eval()
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        model_out = text_model(**encoded)
        outputs.append(model_out.last_hidden_state[:, 0].detach().cpu())
    if not outputs:
        return torch.empty((0, Config.POST_FEATURE_DIM), dtype=torch.float32)
    return torch.cat(outputs, dim=0)


class FakedditGraphBuilder(CheckedGraphBuilder):
    """Backward-compatible name for the CHECKED graph builder."""

    def __init__(
        self,
        post_feature_dim: int = 768,
        user_embed_dim: int = 64,
        subreddit_embed_dim: int = 64,
        domain_embed_dim: int = 64,
        seed: int = 42,
        social_graph_root: str = "",
        use_social_graph: bool = False,
        use_user_graph: bool = True,
        graph_variant: str = "media_only",
        metadata_mode: str = "none",
        response_config: Optional[ResponseGraphConfig] = None,
        **_: Any,
    ):
        super().__init__(
            post_feature_dim=post_feature_dim,
            user_embed_dim=user_embed_dim,
            media_embed_dim=Config.MEDIA_EMBED_DIM,
            response_feature_dim=Config.RESPONSE_FEATURE_DIM,
            seed=seed,
            use_user_graph=use_user_graph,
            graph_variant=graph_variant,
            metadata_mode=metadata_mode,
            response_config=response_config,
        )


class FakedditMultimodalGNNDataset(torch.utils.data.Dataset):
    """Dataset adapter for CHECKED multimodal rows."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer=None,
        image_processor=None,
        image_root: str = "CHECKED",
        max_len: int = 192,
        post_to_idx: Optional[Dict[str, int]] = None,
        domain_weights: Optional[Dict[str, float]] = None,
        default_weight: float = 0.5,
    ):
        self.df = normalize_checked_dataframe(df)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_root = Path(image_root)
        self.max_len = max_len
        self.post_to_idx = post_to_idx or {str(row["id"]): idx for idx, row in self.df.iterrows()}
        self.domain_weights = domain_weights or {}
        self.default_weight = float(default_weight)
        self._blank_image = Image.new("RGB", (224, 224), color=(0, 0, 0))

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_media_path(self, raw_path: str) -> Optional[Path]:
        if not raw_path:
            return None
        candidate = Path(raw_path.replace("\\", "/"))
        candidates = [
            candidate,
            self.image_root / candidate,
            Path.cwd() / candidate,
            Path.cwd() / self.image_root / candidate,
        ]
        for path in candidates:
            if path.exists() and path.is_file():
                return path
        return None

    def _load_first_image(self, row: pd.Series) -> Tuple[Image.Image, float]:
        for raw_path in parse_json_list(row.get("local_image_paths", "[]")):
            path = self._resolve_media_path(raw_path)
            if path is None:
                continue
            try:
                return Image.open(path).convert("RGB"), 1.0
            except OSError:
                continue
        return self._blank_image, 0.0

    def _propagation_features(self, row: pd.Series) -> torch.Tensor:
        values = []
        for column in Config.PROPAGATION_FEATURE_COLUMNS:
            if column == "text_length":
                value = float(len(str(row.get("clean_title", ""))))
            elif column == "media_count":
                value = float(
                    len(parse_json_list(row.get("local_image_paths", "[]")) or parse_json_list(row.get("image_urls", "[]")))
                    + len(parse_json_list(row.get("local_video_paths", "[]")) or parse_json_list(row.get("video_urls", "[]")))
                )
            else:
                raw_value = row.get(column, 0)
                normalized = str(raw_value).strip().lower()
                if normalized in {"true", "yes"}:
                    value = 1.0
                elif normalized in {"false", "no", "", "none", "nan"}:
                    value = 0.0
                else:
                    try:
                        value = float(raw_value)
                    except (TypeError, ValueError):
                        value = 0.0
            values.append(math.log1p(max(0.0, value)))
        return torch.tensor(values, dtype=torch.float32)

    def _credibility_score(self, row: pd.Series) -> torch.Tensor:
        key = str(row.get("user_id", "")).lower().strip()
        value = self.domain_weights.get(key, self.default_weight)
        return torch.tensor(float(value), dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        item_id = str(row["id"])
        item: Dict[str, Any] = {
            "labels": torch.tensor(int(row["2_way_label"]), dtype=torch.long),
            "graph_idx": torch.tensor(int(self.post_to_idx.get(item_id, 0)), dtype=torch.long),
            "propagation_features": self._propagation_features(row),
            "credibility_score": self._credibility_score(row),
            "id": item_id,
        }

        if self.tokenizer is not None:
            encoded = self.tokenizer(
                str(row.get("clean_title", "")),
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )
            item["input_ids"] = encoded["input_ids"].squeeze(0)
            item["attention_mask"] = encoded["attention_mask"].squeeze(0)

        if self.image_processor is not None:
            image, has_image = self._load_first_image(row)
            processed = self.image_processor(images=image, return_tensors="pt")
            item["pixel_values"] = processed["pixel_values"].squeeze(0)
            item["has_image"] = torch.tensor(has_image, dtype=torch.float32)

        return item


class MultiModalGNNModel(nn.Module):
    def __init__(
        self,
        text_model=None,
        vit_model=None,
        d_proj: int = 256,
        gnn_dim: int = 128,
        num_classes: int = 2,
        fusion_type: str = "concat",
        credibility_weight: float = 1.0,
        use_text: bool = True,
        use_image: bool = True,
        use_graph: bool = True,
        use_propagation: bool = False,
        use_credibility: bool = False,
    ):
        super().__init__()
        self.text_model = text_model
        self.vit_model = vit_model
        self.use_text = use_text
        self.use_image = use_image
        self.use_graph = use_graph
        self.use_propagation = use_propagation
        self.use_credibility = use_credibility
        self.fusion_type = fusion_type
        self.credibility_weight = credibility_weight

        self.text_proj = None
        if use_text:
            text_dim = int(text_model.config.hidden_size)
            self.text_proj = nn.Linear(text_dim, d_proj)

        self.image_proj = None
        if use_image:
            image_dim = int(vit_model.config.hidden_size)
            self.image_proj = nn.Linear(image_dim, d_proj)

        self.graph_encoder = None
        self.graph_proj = None
        if use_graph:
            self.graph_encoder = CheckedGNNEncoder(
                hidden_channels=gnn_dim,
                num_layers=2,
                dropout=0.2,
                post_in_channels=Config.POST_FEATURE_DIM,
                user_in_channels=Config.USER_EMBED_DIM,
                media_in_channels=Config.MEDIA_EMBED_DIM,
                response_in_channels=Config.RESPONSE_FEATURE_DIM,
            )
            self.graph_proj = nn.Linear(gnn_dim, d_proj)

        self.prop_proj = nn.Linear(Config.PROPAGATION_FEATURE_DIM, d_proj) if use_propagation else None
        self.cred_proj = nn.Linear(1, d_proj) if use_credibility else None

        parts = sum([use_text, use_image, use_graph, use_propagation, use_credibility])
        if parts == 0:
            raise ValueError("No modalities enabled.")
        self.classifier = nn.Sequential(
            nn.Linear(d_proj * parts, d_proj),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_proj, num_classes),
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        has_image=None,
        graph_idx=None,
        graph_data=None,
        propagation_features=None,
        credibility_score=None,
        **_: Any,
    ):
        features: List[torch.Tensor] = []
        if self.use_text:
            out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
            features.append(F.relu(self.text_proj(out.last_hidden_state[:, 0])))

        if self.use_image:
            out = self.vit_model(pixel_values=pixel_values)
            image_feature = F.relu(self.image_proj(out.last_hidden_state[:, 0]))
            if has_image is not None:
                image_feature = image_feature * has_image.float().view(-1, 1)
            features.append(image_feature)

        if self.use_graph:
            if graph_data is None or graph_idx is None:
                raise ValueError("graph_data and graph_idx are required when graph modality is enabled")
            post_embeddings = self.graph_encoder(graph_data.x_dict, graph_data.edge_index_dict)
            features.append(F.relu(self.graph_proj(post_embeddings[graph_idx])))

        if self.use_propagation:
            features.append(F.relu(self.prop_proj(propagation_features.float())))

        if self.use_credibility:
            cred = credibility_score.float().view(-1, 1) * float(self.credibility_weight)
            features.append(F.relu(self.cred_proj(cred)))

        logits = self.classifier(torch.cat(features, dim=-1))
        return type("ModelOutput", (), {"logits": logits, "loss": None})


@torch.no_grad()
def eval_model(model, data_loader, device, config: Config, graph_data=None):
    model.eval()
    preds: List[int] = []
    labels: List[int] = []
    total_loss = 0.0
    total = 0
    for batch in data_loader:
        y = batch["labels"].to(device)
        kwargs: Dict[str, Any] = {}
        if config.USE_TEXT:
            kwargs["input_ids"] = batch["input_ids"].to(device)
            kwargs["attention_mask"] = batch["attention_mask"].to(device)
        if config.USE_IMAGE:
            kwargs["pixel_values"] = batch["pixel_values"].to(device)
            kwargs["has_image"] = batch["has_image"].to(device)
        if config.USE_GRAPH:
            kwargs["graph_idx"] = batch["graph_idx"].to(device)
            kwargs["graph_data"] = graph_data
        if config.USE_PROPAGATION:
            kwargs["propagation_features"] = batch["propagation_features"].to(device)
        if config.USE_CREDIBILITY:
            kwargs["credibility_score"] = batch["credibility_score"].to(device)
        logits = model(**kwargs).logits
        loss = F.cross_entropy(logits, y)
        total_loss += float(loss.item()) * y.size(0)
        total += y.size(0)
        preds.extend(logits.argmax(dim=-1).detach().cpu().tolist())
        labels.extend(y.detach().cpu().tolist())
    correct = sum(int(p == y) for p, y in zip(preds, labels))
    return correct / max(1, len(labels)), total_loss / max(1, total), preds, labels
