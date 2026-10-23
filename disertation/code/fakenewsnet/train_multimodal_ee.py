"""
Trainable Multimodal + Heterogeneous GNN for Fakeddit Misinformation Detection

Features:
- Multi-GPU support with DistributedDataParallel (DDP)
- Learnable credibility alpha (gating mechanism) for adaptive domain weighting
- GNN is part of the trainable model
- Text branch DistilBERT is trainable
- Image branch ViT is trainable end-to-end
- Graph branch uses a separate frozen DistilBERT for static post-node features
- No frozen/random cached GNN embeddings
- Graph branch is saved inside the checkpoint
- Train/val graphs are built separately to avoid leakage

Supported modalities:
- Text (DistilBERT)
- Image (ViT, end-to-end trainable)
- Graph (heterogeneous GNN)
- Credibility weighting (learnable)

Example usage:
    python train_multimodal_gnn.py
    python train_multimodal_gnn.py --text
    python train_multimodal_gnn.py --text --graph
    python train_multimodal_gnn.py --text --graph --cred
    python train_multimodal_gnn.py --image --gpus 2
"""

import os
import json
import hashlib
import argparse
import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import (
    AutoTokenizer,
    AutoModel,
    ViTModel,
    AutoImageProcessor,
    get_linear_schedule_with_warmup,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tqdm import tqdm

from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATConv, Linear


# ==========================
# Config
# ==========================

def default_model_name(local_path: str, fallback_name: str) -> str:
    return local_path if os.path.isdir(local_path) else fallback_name


class Config:
    MAX_LEN = int(os.environ.get("FAKENEWSNET_MAX_LEN", "128"))
    BATCH_SIZE = 8
    EPOCHS = int(os.environ.get("FAKENEWSNET_EPOCHS", "10"))
    LEARNING_RATE = 2e-5
    SEED = 42
    WORLD_SIZE = 5

    TEXT_MODEL_NAME = os.environ.get(
        "FAKENEWSNET_TEXT_MODEL",
        default_model_name(
            os.path.join(os.path.expanduser("~"), "Downloads", "roberta-base"),
            default_model_name("/root/autodl-tmp/models/roberta-base", "roberta-base"),
        ),
    )
    VIT_MODEL_NAME = os.environ.get(
        "FAKENEWSNET_VIT_MODEL",
        default_model_name("/root/autodl-tmp/models/vit-base-patch16-224-in21k", "google/vit-base-patch16-224-in21k"),
    )

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    TRAIN_TSV = os.path.join("data", "train.tsv")
    IMAGE_ROOT = os.path.join("FakeNewsNet", "data", "images")
    DOMAIN_WEIGHTS_JSON = os.path.join("data", "domain_weights.json")

    USE_TEXT = True
    USE_IMAGE = True
    USE_GRAPH = True
    USE_CREDIBILITY = True
    USE_PROPAGATION = False

    FUSION_TYPE = "concat"   # concat, gated, attention, sum
    D_PROJ = 256
    PROPAGATION_FEATURE_COLUMNS = [
        "prop_log_tweet_count",
        "prop_log_valid_tweet_count",
        "prop_log_duration_hours",
        "prop_log_mean_gap_minutes",
        "prop_log_median_gap_minutes",
        "prop_log_std_gap_minutes",
        "prop_burstiness",
        "prop_first_1h_ratio",
        "prop_first_6h_ratio",
        "prop_first_24h_ratio",
        "prop_log_first_1h_count",
        "prop_log_first_24h_count",
    ]
    PROPAGATION_FEATURE_DIM = len(PROPAGATION_FEATURE_COLUMNS)

    GNN_HIDDEN_DIM = 128
    GNN_LAYERS = 2
    GNN_HEADS = 4
    GNN_DROPOUT = 0.3

    USER_EMBED_DIM = 64
    SUBREDDIT_EMBED_DIM = 64
    DOMAIN_EMBED_DIM = 64
    POST_FEATURE_DIM = 768
    SOCIAL_GRAPH_ROOT = os.environ.get(
        "FAKENEWSNET_SOCIAL_GRAPH_ROOT",
        os.path.join("FakeNewsNet", "fakenewsnet_dataset"),
    )
    USE_SOCIAL_GRAPH = os.environ.get("FAKENEWSNET_USE_SOCIAL_GRAPH", "0") == "1"

    CREDIBILITY_WEIGHT = 0.25  # Not used when learnable alpha is enabled

    EARLY_STOPPING_PATIENCE = 3
    EARLY_STOPPING_MIN_DELTA = 0.001
    EARLY_STOPPING_MODE = "max"

    SAVE_PATH = None

    MASTER_ADDR = "127.0.0.1"
    MASTER_PORT = "12355"

    NUM_WORKERS = int(os.environ.get("FAKENEWSNET_NUM_WORKERS", "0" if os.name == "nt" else "4"))

    @classmethod
    def get_modality_string(cls):
        modalities = []
        if cls.USE_TEXT:
            modalities.append("text")
        if cls.USE_IMAGE:
            modalities.append("image")
        if cls.USE_GRAPH:
            modalities.append("graph")
        if cls.USE_PROPAGATION:
            modalities.append("propagation")
        if cls.USE_CREDIBILITY:
            modalities.append("cred_learnable")
        return "_".join(modalities) if modalities else "none"

    @classmethod
    def validate(cls):
        if not (cls.USE_TEXT or cls.USE_IMAGE or cls.USE_GRAPH or cls.USE_PROPAGATION):
            raise ValueError(
                "At least one content modality (text, image, graph, or propagation) must be enabled. "
                "Credibility alone cannot make predictions."
            )


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(Config.SEED)


# ==========================
# Helpers
# ==========================

def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank_safe() -> int:
    return dist.get_rank() if is_dist_initialized() else 0


def get_world_size_safe() -> int:
    return dist.get_world_size() if is_dist_initialized() else 1


def is_main_process() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def safe_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


# ==========================
# Early stopping
# ==========================

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0, mode="min", verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, metric, model):
        if not is_main_process():
            return False

        score = -metric if self.mode == "min" else metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
            return True

        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
            return True

        self.counter += 1
        if self.verbose:
            print(f"  EarlyStopping counter: {self.counter}/{self.patience}")
        if self.counter >= self.patience:
            self.early_stop = True
        return False

    def save_checkpoint(self, model):
        if isinstance(model, DDP):
            self.best_model_state = {
                k: v.detach().cpu().clone()
                for k, v in model.module.state_dict().items()
            }
        else:
            self.best_model_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    def load_best_model(self, model):
        if self.best_model_state is None:
            return

        if isinstance(model, DDP):
            model.module.load_state_dict(self.best_model_state)
        else:
            model.load_state_dict(self.best_model_state)

        if self.verbose and is_main_process():
            print("Loaded best model from early stopping checkpoint.")


# ==========================
# Utilities
# ==========================

def stable_hash_vector(key: str, dim: int, seed: int = 42) -> torch.Tensor:
    raw = f"{seed}:{key}".encode("utf-8")
    digest = hashlib.md5(raw).hexdigest()
    int_seed = int(digest[:8], 16)
    rng = np.random.RandomState(int_seed)
    vec = rng.normal(loc=0.0, scale=1.0, size=dim).astype(np.float32)
    return torch.tensor(vec, dtype=torch.float32)


def load_domain_weights(json_path: str):
    if not os.path.exists(json_path):
        safe_print(f"[load_domain_weights] WARNING: {json_path} not found, using neutral weights.")
        return {}, 0.5

    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"[load_domain_weights] Expected dict in {json_path}, got {type(raw)}")

    scores = np.array(list(raw.values()), dtype=np.float32)
    min_s = float(scores.min())
    max_s = float(scores.max())

    domain_to_weight = {}
    if max_s > min_s:
        for dom, s in raw.items():
            domain_to_weight[dom.lower()] = float((float(s) - min_s) / (max_s - min_s))
    else:
        for dom in raw.keys():
            domain_to_weight[dom.lower()] = 0.5

    mean_weight = float(np.mean(list(domain_to_weight.values()))) if domain_to_weight else 0.5
    safe_print(f"[load_domain_weights] Loaded {len(domain_to_weight)} domains from {json_path}")
    return domain_to_weight, mean_weight


def extract_text_for_row(row: pd.Series) -> str:
    title = str(row.get("clean_title", ""))
    if title == "nan":
        title = ""
    body = str(row.get("text", ""))
    if body == "nan":
        body = ""
    return (title + " " + body).strip()


@torch.no_grad()
def extract_post_text_features(
    df: pd.DataFrame,
    tokenizer: Any,
    text_model: nn.Module,
    device: torch.device,
    batch_size: int = 32,
    max_len: int = 128,
) -> torch.Tensor:
    """
    Extract frozen DistilBERT features for graph post nodes.
    These are static post-node inputs to the trainable GNN.
    """
    text_model.eval()
    all_features = []

    texts = [extract_text_for_row(row) for _, row in df.iterrows()]

    iterator = range(0, len(texts), batch_size)
    if is_main_process():
        iterator = tqdm(iterator, desc="Extracting graph post text features")

    for start in iterator:
        batch_texts = texts[start:start + batch_size]

        enc = tokenizer(
            batch_texts,
            add_special_tokens=True,
            max_length=max_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        outputs = text_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_feats = outputs.last_hidden_state[:, 0, :].cpu()
        all_features.append(cls_feats)

    return torch.cat(all_features, dim=0)


# ==========================
# Graph builder
# ==========================

class FakedditGraphBuilder:
    """
    Builds a heterogeneous graph from a split DataFrame.

    Node types:
    - post
    - user
    - domain
    """

    def __init__(
        self,
        post_feature_dim: int = 768,
        user_embed_dim: int = 64,
        subreddit_embed_dim: int = 64,
        domain_embed_dim: int = 64,
        seed: int = 42,
        social_graph_root: Optional[str] = None,
        use_social_graph: bool = False,
    ):
        self.post_feature_dim = post_feature_dim
        self.user_embed_dim = user_embed_dim
        self.subreddit_embed_dim = subreddit_embed_dim
        self.domain_embed_dim = domain_embed_dim
        self.seed = seed
        self.social_graph_root = social_graph_root
        self.use_social_graph = use_social_graph

        self.user_to_idx: Dict[str, int] = {}
        self.subreddit_to_idx: Dict[str, int] = {}
        self.domain_to_idx: Dict[str, int] = {}
        self.post_to_idx: Dict[str, int] = {}
        self._social_user_cache: Dict[str, list] = {}

    def build_graph(
        self,
        df: pd.DataFrame,
        post_features: torch.Tensor,
    ) -> HeteroData:
        self.user_to_idx = {}
        self.subreddit_to_idx = {}
        self.domain_to_idx = {}
        self.post_to_idx = {}

        for _, row in df.iterrows():
            post_id = str(row["id"])
            if post_id not in self.post_to_idx:
                self.post_to_idx[post_id] = len(self.post_to_idx)

            author = str(row.get("author", ""))
            if author and author != "nan":
                if author not in self.user_to_idx:
                    self.user_to_idx[author] = len(self.user_to_idx)

            for social_user in self._social_user_ids_for_row(row):
                if social_user not in self.user_to_idx:
                    self.user_to_idx[social_user] = len(self.user_to_idx)

            domain = str(row.get("domain", ""))
            if domain and domain != "nan":
                if domain not in self.domain_to_idx:
                    self.domain_to_idx[domain] = len(self.domain_to_idx)

        edge_post_user = self._build_edges(df, "author", self.user_to_idx)
        edge_post_social_user = self._build_social_edges(df, self.user_to_idx)
        if edge_post_social_user.numel() > 0:
            edge_post_user = torch.cat([edge_post_user, edge_post_social_user], dim=1)
        edge_post_domain = self._build_edges(df, "domain", self.domain_to_idx)

        data = HeteroData()

        num_posts = len(self.post_to_idx)
        num_users = len(self.user_to_idx)
        num_domains = len(self.domain_to_idx)

        if post_features.shape[0] != num_posts:
            raise ValueError(
                f"post_features rows ({post_features.shape[0]}) != num_posts ({num_posts})"
            )

        data["post"].x = post_features

        user_features = [
            stable_hash_vector(f"user::{user}", self.user_embed_dim, self.seed)
            for user, _ in sorted(self.user_to_idx.items(), key=lambda x: x[1])
        ]
        domain_features = [
            stable_hash_vector(f"domain::{dom}", self.domain_embed_dim, self.seed)
            for dom, _ in sorted(self.domain_to_idx.items(), key=lambda x: x[1])
        ]

        data["user"].x = (
            torch.stack(user_features)
            if user_features
            else torch.zeros((0, self.user_embed_dim), dtype=torch.float32)
        )
        data["domain"].x = (
            torch.stack(domain_features)
            if domain_features
            else torch.zeros((0, self.domain_embed_dim), dtype=torch.float32)
        )

        data["post", "authored_by", "user"].edge_index = edge_post_user
        data["user", "authored", "post"].edge_index = edge_post_user.flip(0)

        data["post", "from_domain", "domain"].edge_index = edge_post_domain
        data["domain", "sources", "post"].edge_index = edge_post_domain.flip(0)

        labels = df["2_way_label"].values
        data["post"].y = torch.tensor(labels, dtype=torch.long)
        data["post"].post_ids = list(self.post_to_idx.keys())

        safe_print("Graph Statistics:")
        safe_print(f"  Posts: {num_posts}, Users: {num_users}, Domains: {num_domains}")
        if self.use_social_graph:
            safe_print(f"  Social graph root: {self.social_graph_root}")
            safe_print(f"  Post-user edges including social users: {edge_post_user.size(1)}")

        return data

    def _news_dir_for_row(self, row: pd.Series) -> Optional[str]:
        if not self.use_social_graph or not self.social_graph_root:
            return None
        source = str(row.get("source", ""))
        label = str(row.get("label", ""))
        news_id = str(row.get("id", ""))
        if not source or source == "nan" or not label or label == "nan" or not news_id:
            return None
        return os.path.join(self.social_graph_root, source, label, news_id)

    @staticmethod
    def _read_json(path: str):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return None

    @staticmethod
    def _user_id_from_tweet(tweet_obj) -> Optional[str]:
        if not isinstance(tweet_obj, dict):
            return None
        user_obj = tweet_obj.get("user")
        if isinstance(user_obj, dict) and user_obj.get("id") is not None:
            return f"twitter_user::{user_obj['id']}"
        author_id = tweet_obj.get("author_id")
        if author_id is not None:
            return f"twitter_user::{author_id}"
        if tweet_obj.get("id") is not None and ("username" in tweet_obj or "name" in tweet_obj):
            return f"twitter_user::{tweet_obj['id']}"
        return None

    def _social_user_ids_for_row(self, row: pd.Series):
        news_dir = self._news_dir_for_row(row)
        if not news_dir or not os.path.isdir(news_dir):
            return []
        if news_dir in self._social_user_cache:
            return self._social_user_cache[news_dir]

        users = set()
        tweets_dir = os.path.join(news_dir, "tweets")
        if os.path.isdir(tweets_dir):
            for name in os.listdir(tweets_dir):
                if name.endswith(".json"):
                    user_id = self._user_id_from_tweet(self._read_json(os.path.join(tweets_dir, name)))
                    if user_id:
                        users.add(user_id)

        retweets_dir = os.path.join(news_dir, "retweets")
        if os.path.isdir(retweets_dir):
            for name in os.listdir(retweets_dir):
                if not name.endswith(".json"):
                    continue
                payload = self._read_json(os.path.join(retweets_dir, name))
                retweets = payload.get("retweets", []) if isinstance(payload, dict) else []
                for retweet in retweets:
                    user_id = self._user_id_from_tweet(retweet)
                    if user_id:
                        users.add(user_id)
        result = sorted(users)
        self._social_user_cache[news_dir] = result
        return result

    def _build_social_edges(self, df: pd.DataFrame, mapping: Dict[str, int]) -> torch.Tensor:
        src, dst = [], []
        if not self.use_social_graph:
            return torch.empty((2, 0), dtype=torch.long)
        for _, row in df.iterrows():
            post_id = str(row["id"])
            if post_id not in self.post_to_idx:
                continue
            for user_id in self._social_user_ids_for_row(row):
                if user_id in mapping:
                    src.append(self.post_to_idx[post_id])
                    dst.append(mapping[user_id])
        if not src:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor([src, dst], dtype=torch.long)

    def _build_edges(self, df: pd.DataFrame, col: str, mapping: Dict[str, int]) -> torch.Tensor:
        src, dst = [], []
        for _, row in df.iterrows():
            post_id = str(row["id"])
            entity = str(row.get(col, ""))
            if entity and entity != "nan" and entity in mapping:
                src.append(self.post_to_idx[post_id])
                dst.append(mapping[entity])

        if len(src) == 0:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor([src, dst], dtype=torch.long)


# ==========================
# Dataset
# ==========================

class FakedditMultimodalGNNDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: Optional[Any],
        image_processor: Optional[AutoImageProcessor],
        image_root: str,
        max_len: int,
        post_to_idx: Dict[str, int],
        domain_weights: Optional[Dict[str, float]] = None,
        default_weight: float = 0.5,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_root = image_root
        self.max_len = max_len
        self.post_to_idx = post_to_idx
        self.domain_weights = domain_weights or {}
        self.default_weight = float(default_weight)

        if image_processor is not None:
            size = image_processor.size
            if isinstance(size, dict):
                self.height = size.get("height", 224)
                self.width = size.get("width", 224)
            else:
                self.height = self.width = size
        else:
            self.height = self.width = 224

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _normalize_domain(domain: str) -> str:
        d = domain.lower().strip()
        if d.startswith("www."):
            d = d[4:]
        return d

    def _extract_domain_from_row(self, row: pd.Series):
        if "domain" in row:
            val = row["domain"]
            if isinstance(val, str) and val.strip():
                return self._normalize_domain(val)
        return None

    @staticmethod
    def _to_float(value) -> float:
        try:
            if pd.isna(value):
                return 0.0
            return float(value)
        except Exception:
            return 0.0

    def _propagation_features_from_row(self, row: pd.Series) -> torch.Tensor:
        values = [
            self._to_float(row.get(column, 0.0))
            for column in Config.PROPAGATION_FEATURE_COLUMNS
        ]
        return torch.tensor(values, dtype=torch.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if self.tokenizer is not None:
            text = extract_text_for_row(row)
            enc = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
        else:
            input_ids = torch.zeros(self.max_len, dtype=torch.long)
            attention_mask = torch.zeros(self.max_len, dtype=torch.long)

        img_id = str(row["id"])
        img_path = os.path.join(self.image_root, img_id + ".jpg")

        has_image = 1.0
        if self.image_processor is not None:
            try:
                image = Image.open(img_path).convert("RGB")
                img_enc = self.image_processor(image, return_tensors="pt")
                pixel_values = img_enc["pixel_values"].squeeze(0)
            except Exception:
                has_image = 0.0
                pixel_values = torch.zeros(3, self.height, self.width, dtype=torch.float32)
        else:
            has_image = 0.0
            pixel_values = torch.zeros(3, self.height, self.width, dtype=torch.float32)

        label = int(row["2_way_label"])

        domain = self._extract_domain_from_row(row)
        if domain is not None:
            credibility_score = self.domain_weights.get(domain, self.default_weight)
        else:
            credibility_score = self.default_weight

        post_id = str(row["id"])
        graph_idx = self.post_to_idx.get(post_id, 0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "has_image": torch.tensor(has_image, dtype=torch.float32),
            "credibility_score": torch.tensor(credibility_score, dtype=torch.float32),
            "propagation_features": self._propagation_features_from_row(row),
            "graph_idx": torch.tensor(graph_idx, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ==========================
# GNN encoder
# ==========================

class HeteroGNNEncoder(nn.Module):
    """
    Trainable graph encoder that outputs post node embeddings.
    """

    def __init__(
        self,
        post_in_channels: int = 768,
        user_in_channels: int = 64,
        subreddit_in_channels: int = 64,
        domain_in_channels: int = 64,
        hidden_channels: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout

        self.input_projections = nn.ModuleDict({
            "post": Linear(post_in_channels, hidden_channels),
            "user": Linear(user_in_channels, hidden_channels),
            "domain": Linear(domain_in_channels, hidden_channels),
        })

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ("post", "authored_by", "user"): GATConv(
                    hidden_channels, hidden_channels // num_heads,
                    heads=num_heads, dropout=dropout, add_self_loops=False
                ),
                ("user", "authored", "post"): GATConv(
                    hidden_channels, hidden_channels // num_heads,
                    heads=num_heads, dropout=dropout, add_self_loops=False
                ),
                ("post", "from_domain", "domain"): GATConv(
                    hidden_channels, hidden_channels // num_heads,
                    heads=num_heads, dropout=dropout, add_self_loops=False
                ),
                ("domain", "sources", "post"): GATConv(
                    hidden_channels, hidden_channels // num_heads,
                    heads=num_heads, dropout=dropout, add_self_loops=False
                ),
            }, aggr="sum")
            self.convs.append(conv)

    def forward(self, data: HeteroData) -> torch.Tensor:
        x_dict = {
            node_type: self.input_projections[node_type](x)
            for node_type, x in data.x_dict.items()
        }

        edge_index_dict = data.edge_index_dict

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
            x_dict = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in x_dict.items()}

        return x_dict["post"]


# ==========================
# Multimodal model
# ==========================

class MultiModalGNNModel(nn.Module):
    def __init__(
        self,
        text_model=None,
        vit_model=None,
        d_proj=256,
        gnn_dim=128,
        num_classes=2,
        fusion_type="concat",
        credibility_weight=0.25,
        use_text=True,
        use_image=True,
        use_graph=True,
        use_propagation=False,
        use_credibility=True,
    ):
        super().__init__()

        self.use_text = use_text
        self.use_image = use_image
        self.use_graph = use_graph
        self.use_propagation = use_propagation
        self.use_credibility = use_credibility
        self.fusion_type = fusion_type
        self.d_proj = d_proj
        self.credibility_weight = credibility_weight

        self.text_model = text_model
        self.vit_model = vit_model

        self.num_modalities = sum([use_text, use_image, use_graph, use_propagation])
        if self.num_modalities == 0:
            raise ValueError("At least one content modality must be enabled.")

        if use_text:
            d_t = text_model.config.hidden_size
            self.proj_t = nn.Linear(d_t, d_proj)

        if use_image:
            d_v = vit_model.config.hidden_size
            self.proj_v = nn.Linear(d_v, d_proj)

        if use_graph:
            self.gnn_encoder = HeteroGNNEncoder(
                post_in_channels=Config.POST_FEATURE_DIM,
                user_in_channels=Config.USER_EMBED_DIM,
                subreddit_in_channels=Config.SUBREDDIT_EMBED_DIM,
                domain_in_channels=Config.DOMAIN_EMBED_DIM,
                hidden_channels=Config.GNN_HIDDEN_DIM,
                num_layers=Config.GNN_LAYERS,
                num_heads=Config.GNN_HEADS,
                dropout=Config.GNN_DROPOUT,
            )
            self.proj_g = nn.Linear(gnn_dim, d_proj)

        if use_propagation:
            self.proj_p = nn.Sequential(
                nn.Linear(Config.PROPAGATION_FEATURE_DIM, d_proj),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(d_proj, d_proj),
            )

        if self.use_credibility:
            gate_in_dim = self.num_modalities * d_proj if fusion_type == "concat" else d_proj
            self.cred_gate = nn.Sequential(
                nn.Linear(gate_in_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

        if fusion_type == "concat":
            fusion_dim = self.num_modalities * d_proj
        elif fusion_type == "gated":
            if self.num_modalities > 1:
                self.gate_fc = nn.Linear(self.num_modalities * d_proj, self.num_modalities)
            fusion_dim = d_proj
        elif fusion_type == "attention":
            if self.num_modalities > 1:
                self.mod_attn = nn.MultiheadAttention(d_proj, num_heads=4, batch_first=True)
            fusion_dim = d_proj
        else:
            fusion_dim = d_proj

        self.fc1 = nn.Linear(fusion_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        self.last_alpha = None

    def encode_text(self, input_ids, attention_mask):
        out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]

    def encode_image(self, pixel_values):
        out = self.vit_model(pixel_values=pixel_values)
        return out.last_hidden_state[:, 0, :]

    def fuse(self, features, has_image=None):
        if len(features) == 1:
            return features[0]

        if self.fusion_type == "concat":
            return torch.cat(features, dim=-1)

        elif self.fusion_type == "gated":
            concat = torch.cat(features, dim=-1)
            alpha = torch.softmax(self.gate_fc(concat), dim=-1)

            if has_image is not None and self.use_image:
                mask = has_image.view(-1, 1)
                img_idx = 1 if self.use_text else 0
                alpha_list = list(alpha.split(1, dim=-1))
                img_weight = alpha_list[img_idx]
                redistribute = img_weight / max(1, len(features) - 1)

                for i in range(len(alpha_list)):
                    if i == img_idx:
                        alpha_list[i] = alpha_list[i] * mask
                    else:
                        alpha_list[i] = alpha_list[i] + redistribute * (1.0 - mask)
                alpha = torch.cat(alpha_list, dim=-1)

            fused = sum(alpha[:, i:i+1] * f for i, f in enumerate(features))
            return fused

        elif self.fusion_type == "attention":
            stacked = torch.stack(features, dim=1)
            attended, _ = self.mod_attn(stacked, stacked, stacked)
            return attended.mean(dim=1)

        else:
            return sum(features)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        has_image=None,
        credibility_score=None,
        labels=None,
        graph_idx=None,
        graph_data=None,
        propagation_features=None,
    ):
        features = []

        if self.use_text:
            t = self.encode_text(input_ids, attention_mask)
            t = self.proj_t(t)
            features.append(t)

        if self.use_image:
            v = self.encode_image(pixel_values)
            v = self.proj_v(v)
            if has_image is not None:
                v = v * has_image.view(-1, 1).to(v.device)
            features.append(v)

        if self.use_graph:
            if graph_data is None or graph_idx is None:
                raise ValueError("graph_data and graph_idx are required when graph modality is enabled")
            full_graph_embeddings = self.gnn_encoder(graph_data)
            g = full_graph_embeddings[graph_idx.to(full_graph_embeddings.device)]
            g = self.proj_g(g)
            features.append(g)

        if self.use_propagation:
            if propagation_features is None:
                raise ValueError("propagation_features are required when propagation modality is enabled")
            p = propagation_features.to(next(self.parameters()).device).float()
            p = self.proj_p(p)
            features.append(p)

        z = self.fuse(features, has_image=has_image if self.use_image else None)

        h = self.dropout(self.act(self.fc1(z)))
        content_logits = self.fc2(h)

        if self.use_credibility and credibility_score is not None:
            alpha = self.cred_gate(z)
            self.last_alpha = alpha.detach()

            content_probs = F.softmax(content_logits, dim=-1)

            cred = credibility_score.view(-1, 1).to(content_logits.device)
            cred_probs = torch.cat([1.0 - cred, cred], dim=-1)

            final_probs = (1 - alpha) * content_probs + alpha * cred_probs

            eps = 1e-7
            final_probs = torch.clamp(final_probs, min=eps, max=1.0 - eps)
            logits = torch.log(final_probs)
        else:
            self.last_alpha = None
            logits = content_logits

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)


# ==========================
# DDP metric helpers
# ==========================

def reduce_scalar_sum(value: float, device: torch.device) -> float:
    if not is_dist_initialized():
        return float(value)
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item()


def reduce_int_sum(value: int, device: torch.device) -> int:
    if not is_dist_initialized():
        return int(value)
    tensor = torch.tensor(value, dtype=torch.long, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return int(tensor.item())


# ==========================
# Training / evaluation
# ==========================

def train_epoch(model, data_loader, optimizer, scheduler, device, config, graph_data=None, epoch=0):
    model.train()

    local_loss_sum = 0.0
    local_correct = 0
    local_total = 0
    local_alpha_sum = 0.0
    local_alpha_count = 0

    if hasattr(data_loader.sampler, "set_epoch"):
        data_loader.sampler.set_epoch(epoch)

    progress_bar = tqdm(
        data_loader,
        desc=f"Training (GPU {get_rank_safe()})",
        disable=not is_main_process()
    )

    for batch in progress_bar:
        labels = batch["labels"].to(device)
        kwargs = {"labels": labels}

        if config.USE_TEXT:
            kwargs["input_ids"] = batch["input_ids"].to(device, non_blocking=True)
            kwargs["attention_mask"] = batch["attention_mask"].to(device, non_blocking=True)

        if config.USE_IMAGE:
            kwargs["pixel_values"] = batch["pixel_values"].to(device, non_blocking=True)
            kwargs["has_image"] = batch["has_image"].to(device, non_blocking=True)

        if config.USE_GRAPH:
            kwargs["graph_idx"] = batch["graph_idx"].to(device, non_blocking=True)
            kwargs["graph_data"] = graph_data

        if config.USE_PROPAGATION:
            kwargs["propagation_features"] = batch["propagation_features"].to(device, non_blocking=True)

        if config.USE_CREDIBILITY:
            kwargs["credibility_score"] = batch["credibility_score"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(**kwargs)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        preds = torch.argmax(logits, dim=1)

        batch_size = labels.size(0)
        local_correct += torch.sum(preds == labels).item()
        local_total += batch_size
        local_loss_sum += loss.item() * batch_size

        module_ref = model.module if isinstance(model, DDP) else model
        if config.USE_CREDIBILITY and module_ref.last_alpha is not None:
            local_alpha_sum += module_ref.last_alpha.sum().item()
            local_alpha_count += module_ref.last_alpha.size(0)

        if is_main_process():
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    global_correct = reduce_int_sum(local_correct, device)
    global_total = reduce_int_sum(local_total, device)
    global_loss_sum = reduce_scalar_sum(local_loss_sum, device)

    accuracy = global_correct / max(1, global_total)
    avg_loss = global_loss_sum / max(1, global_total)

    if config.USE_CREDIBILITY:
        global_alpha_sum = reduce_scalar_sum(local_alpha_sum, device)
        global_alpha_count = reduce_int_sum(local_alpha_count, device)
        if is_main_process() and global_alpha_count > 0:
            avg_alpha = global_alpha_sum / global_alpha_count
            print(f"  Average alpha (credibility weight): {avg_alpha:.4f}")

    return accuracy, avg_loss


@torch.no_grad()
def eval_model(model, data_loader, device, config, graph_data=None):
    model.eval()

    local_loss_sum = 0.0
    local_correct = 0
    local_total = 0
    local_alpha_sum = 0.0
    local_alpha_count = 0

    local_preds = []
    local_labels = []

    progress_bar = tqdm(
        data_loader,
        desc=f"Evaluating (GPU {get_rank_safe()})",
        disable=not is_main_process()
    )

    for batch in progress_bar:
        labels = batch["labels"].to(device)
        kwargs = {"labels": labels}

        if config.USE_TEXT:
            kwargs["input_ids"] = batch["input_ids"].to(device, non_blocking=True)
            kwargs["attention_mask"] = batch["attention_mask"].to(device, non_blocking=True)

        if config.USE_IMAGE:
            kwargs["pixel_values"] = batch["pixel_values"].to(device, non_blocking=True)
            kwargs["has_image"] = batch["has_image"].to(device, non_blocking=True)

        if config.USE_GRAPH:
            kwargs["graph_idx"] = batch["graph_idx"].to(device, non_blocking=True)
            kwargs["graph_data"] = graph_data

        if config.USE_PROPAGATION:
            kwargs["propagation_features"] = batch["propagation_features"].to(device, non_blocking=True)

        if config.USE_CREDIBILITY:
            kwargs["credibility_score"] = batch["credibility_score"].to(device, non_blocking=True)

        outputs = model(**kwargs)
        loss = outputs.loss
        logits = outputs.logits

        preds = torch.argmax(logits, dim=1)
        batch_size = labels.size(0)

        local_correct += torch.sum(preds == labels).item()
        local_total += batch_size
        local_loss_sum += loss.item() * batch_size

        module_ref = model.module if isinstance(model, DDP) else model
        if config.USE_CREDIBILITY and module_ref.last_alpha is not None:
            local_alpha_sum += module_ref.last_alpha.sum().item()
            local_alpha_count += module_ref.last_alpha.size(0)

        local_preds.extend(preds.detach().cpu().tolist())
        local_labels.extend(labels.detach().cpu().tolist())

    global_correct = reduce_int_sum(local_correct, device)
    global_total = reduce_int_sum(local_total, device)
    global_loss_sum = reduce_scalar_sum(local_loss_sum, device)

    accuracy = global_correct / max(1, global_total)
    avg_loss = global_loss_sum / max(1, global_total)

    if config.USE_CREDIBILITY:
        global_alpha_sum = reduce_scalar_sum(local_alpha_sum, device)
        global_alpha_count = reduce_int_sum(local_alpha_count, device)
        if is_main_process() and global_alpha_count > 0:
            avg_alpha = global_alpha_sum / global_alpha_count
            print(f"  Validation average alpha: {avg_alpha:.4f}")

    if is_dist_initialized():
        gathered_preds = [None for _ in range(get_world_size_safe())]
        gathered_labels = [None for _ in range(get_world_size_safe())]
        dist.all_gather_object(gathered_preds, local_preds)
        dist.all_gather_object(gathered_labels, local_labels)

        if is_main_process():
            all_preds = [p for sub in gathered_preds for p in sub]
            all_labels = [l for sub in gathered_labels for l in sub]
            return accuracy, avg_loss, all_preds, all_labels

        return accuracy, avg_loss, [], []

    return accuracy, avg_loss, local_preds, local_labels


# ==========================
# Main worker
# ==========================

def main_worker(rank, world_size, args):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    is_distributed = world_size > 1

    try:
        if is_distributed:
            backend = "gloo" if os.name == "nt" else "nccl"
            dist.init_process_group(
                backend=backend,
                init_method=f"tcp://{args.master_addr}:{args.master_port}",
                world_size=world_size,
                rank=rank,
                timeout=datetime.timedelta(minutes=30),
            )

        Config.USE_TEXT = args.text
        Config.USE_IMAGE = args.image
        Config.USE_GRAPH = args.graph
        Config.USE_CREDIBILITY = args.cred
        Config.WORLD_SIZE = world_size

        # Safer defaults now that both DistilBERT and ViT can be trainable.
        # Per-GPU batch size is effectively base_batch_size below.
        if Config.USE_IMAGE:
            base_batch_size = 4
        elif Config.USE_TEXT and Config.USE_GRAPH:
            base_batch_size = 8
        elif Config.USE_TEXT:
            base_batch_size = 16
        elif Config.USE_GRAPH:
            base_batch_size = 128
        else:
            base_batch_size = 32

        Config.BATCH_SIZE = base_batch_size * Config.WORLD_SIZE

        modality_str = Config.get_modality_string()
        Config.SAVE_PATH = f"best_model_{modality_str}_{world_size}gpu.pt"

        Config.validate()

        if rank == 0:
            print("=" * 60)
            print("MULTIMODAL MISINFORMATION DETECTION")
            print("=" * 60)
            print(f"Text:  {Config.USE_TEXT}")
            print(f"Image: {Config.USE_IMAGE} (END-TO-END TRAINABLE ViT)")
            print(f"Graph: {Config.USE_GRAPH}")
            print(f"Propagation: {Config.USE_PROPAGATION} (tweet-ID temporal features)")
            print(f"Cred:  {Config.USE_CREDIBILITY} (LEARNABLE ALPHA)")
            print(f"Save path: {Config.SAVE_PATH}")
            print("=" * 60)

        tokenizer = None
        image_processor = None
        graph_text_model = None      # frozen, only for graph feature extraction
        trainable_text_model = None  # trainable, only for direct text branch
        vit_model = None
        domain_weights = {}
        mean_weight = 0.5

        if Config.USE_CREDIBILITY:
            domain_weights, mean_weight = load_domain_weights(Config.DOMAIN_WEIGHTS_JSON)

        if Config.USE_TEXT or Config.USE_GRAPH:
            if rank == 0:
                print("\nLoading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(Config.TEXT_MODEL_NAME)

        if Config.USE_GRAPH:
            if rank == 0:
                print(f"Loading frozen text encoder for graph features: {Config.TEXT_MODEL_NAME}")
            graph_text_model = AutoModel.from_pretrained(Config.TEXT_MODEL_NAME)
            Config.POST_FEATURE_DIM = int(graph_text_model.config.hidden_size)
            for p in graph_text_model.parameters():
                p.requires_grad = False
            graph_text_model = graph_text_model.to(device)
            graph_text_model.eval()

        if Config.USE_TEXT:
            if rank == 0:
                print(f"Loading trainable text encoder for text branch: {Config.TEXT_MODEL_NAME}")
            trainable_text_model = AutoModel.from_pretrained(Config.TEXT_MODEL_NAME)
            try:
                trainable_text_model.gradient_checkpointing_enable()
                if rank == 0:
                    print("Enabled gradient checkpointing for text encoder.")
            except Exception:
                if rank == 0:
                    print("Could not enable gradient checkpointing for text encoder.")
            trainable_text_model = trainable_text_model.to(device)

        if Config.USE_IMAGE:
            if rank == 0:
                print("\nLoading image processor...")
            image_processor = AutoImageProcessor.from_pretrained(Config.VIT_MODEL_NAME, use_fast=True)

            if rank == 0:
                print("Loading trainable ViT...")
            vit_model = ViTModel.from_pretrained(Config.VIT_MODEL_NAME)

            # IMPORTANT: ViT is now trainable end-to-end.
            for p in vit_model.parameters():
                p.requires_grad = True

            try:
                vit_model.gradient_checkpointing_enable()
                if rank == 0:
                    print("Enabled gradient checkpointing for ViT.")
            except Exception:
                if rank == 0:
                    print("Could not enable gradient checkpointing for ViT.")

            vit_model = vit_model.to(device)

        if rank == 0:
            print("\nLoading data...")
        df = pd.read_csv(Config.TRAIN_TSV, sep="\t")
        df = df.dropna(subset=["2_way_label", "clean_title"]).copy()
        df["id"] = df["id"].astype(str)
        df["2_way_label"] = df["2_way_label"].astype(int)

        train_df, val_df = train_test_split(
            df,
            test_size=0.2,
            random_state=Config.SEED,
            stratify=df["2_way_label"].values,
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        if rank == 0:
            print(f"Total samples: {len(df)}")
            print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

        train_post_to_idx = {str(row["id"]): idx for idx, row in train_df.iterrows()}
        val_post_to_idx = {str(row["id"]): idx for idx, row in val_df.iterrows()}

        train_graph_data = None
        val_graph_data = None

        if Config.USE_GRAPH:
            if rank == 0:
                print("\nPreparing TRAIN graph...")
                train_post_features = extract_post_text_features(
                    train_df, tokenizer, graph_text_model, device, batch_size=32, max_len=Config.MAX_LEN
                )
                train_builder = FakedditGraphBuilder(
                    post_feature_dim=Config.POST_FEATURE_DIM,
                    user_embed_dim=Config.USER_EMBED_DIM,
                    subreddit_embed_dim=Config.SUBREDDIT_EMBED_DIM,
                    domain_embed_dim=Config.DOMAIN_EMBED_DIM,
                    seed=Config.SEED,
                    social_graph_root=Config.SOCIAL_GRAPH_ROOT,
                    use_social_graph=Config.USE_SOCIAL_GRAPH,
                )
                train_graph_data = train_builder.build_graph(train_df, train_post_features).cpu()
                train_post_to_idx = train_builder.post_to_idx

                print("\nPreparing VAL graph...")
                val_post_features = extract_post_text_features(
                    val_df, tokenizer, graph_text_model, device, batch_size=32, max_len=Config.MAX_LEN
                )
                val_builder = FakedditGraphBuilder(
                    post_feature_dim=Config.POST_FEATURE_DIM,
                    user_embed_dim=Config.USER_EMBED_DIM,
                    subreddit_embed_dim=Config.SUBREDDIT_EMBED_DIM,
                    domain_embed_dim=Config.DOMAIN_EMBED_DIM,
                    seed=Config.SEED,
                    social_graph_root=Config.SOCIAL_GRAPH_ROOT,
                    use_social_graph=Config.USE_SOCIAL_GRAPH,
                )
                val_graph_data = val_builder.build_graph(val_df, val_post_features).cpu()
                val_post_to_idx = val_builder.post_to_idx

            if is_distributed:
                objects_to_broadcast = [train_graph_data, val_graph_data, train_post_to_idx, val_post_to_idx]
                dist.broadcast_object_list(objects_to_broadcast, src=0)
                train_graph_data, val_graph_data, train_post_to_idx, val_post_to_idx = objects_to_broadcast

            train_graph_data = train_graph_data.to(device)
            val_graph_data = val_graph_data.to(device)

        train_dataset = FakedditMultimodalGNNDataset(
            train_df,
            tokenizer,
            image_processor,
            Config.IMAGE_ROOT,
            Config.MAX_LEN,
            train_post_to_idx,
            domain_weights,
            mean_weight,
        )
        val_dataset = FakedditMultimodalGNNDataset(
            val_df,
            tokenizer,
            image_processor,
            Config.IMAGE_ROOT,
            Config.MAX_LEN,
            val_post_to_idx,
            domain_weights,
            mean_weight,
        )

        if is_distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=Config.SEED,
                drop_last=False,
            )
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                seed=Config.SEED,
                drop_last=False,
            )
        else:
            train_sampler = None
            val_sampler = None

        per_gpu_batch_size = max(1, Config.BATCH_SIZE // world_size)

        if rank == 0 and Config.BATCH_SIZE < world_size:
            print(
                f"Warning: total batch size {Config.BATCH_SIZE} is smaller than number of GPUs {world_size}. "
                f"Using batch size 1 per GPU."
            )

        num_workers = Config.NUM_WORKERS

        train_loader = DataLoader(
            train_dataset,
            batch_size=per_gpu_batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=per_gpu_batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )

        if rank == 0:
            print(f"\nPer-GPU batch size: {per_gpu_batch_size}")
            print(f"Total effective batch size: {per_gpu_batch_size * world_size}")

        model = MultiModalGNNModel(
            text_model=trainable_text_model,
            vit_model=vit_model,
            d_proj=Config.D_PROJ,
            gnn_dim=Config.GNN_HIDDEN_DIM,
            num_classes=2,
            fusion_type=Config.FUSION_TYPE,
            credibility_weight=Config.CREDIBILITY_WEIGHT,
            use_text=Config.USE_TEXT,
            use_image=Config.USE_IMAGE,
            use_graph=Config.USE_GRAPH,
            use_propagation=Config.USE_PROPAGATION,
            use_credibility=Config.USE_CREDIBILITY,
        ).to(device)

        if is_distributed:
            model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

        if rank == 0:
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Trainable parameters: {trainable_params:,}")

        optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
        total_steps = len(train_loader) * Config.EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )

        early_stopping = EarlyStopping(
            patience=Config.EARLY_STOPPING_PATIENCE,
            min_delta=Config.EARLY_STOPPING_MIN_DELTA,
            mode=Config.EARLY_STOPPING_MODE,
            verbose=True,
        )

        modalities_desc = []
        if Config.USE_TEXT:
            modalities_desc.append("Text")
        if Config.USE_IMAGE:
            modalities_desc.append("Image (ViT E2E)")
        if Config.USE_GRAPH:
            modalities_desc.append("Graph")
        if Config.USE_CREDIBILITY:
            modalities_desc.append("Cred (Learnable α)")

        if rank == 0:
            print("\n" + "=" * 60)
            print(f"Starting Training: {' + '.join(modalities_desc)}")
            print("=" * 60)

        best_accuracy = 0.0
        epochs_completed = 0

        for epoch in range(Config.EPOCHS):
            if rank == 0:
                print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")
                print("-" * 60)

            train_acc, train_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                device,
                Config,
                graph_data=train_graph_data,
                epoch=epoch,
            )

            if rank == 0:
                print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

            val_acc, val_loss, val_preds, val_labels = eval_model(
                model,
                val_loader,
                device,
                Config,
                graph_data=val_graph_data,
            )

            if rank == 0:
                print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

            epochs_completed = epoch + 1

            stop_tensor = torch.tensor([0], device=device, dtype=torch.long)

            if rank == 0:
                is_best = early_stopping(val_acc, model)
                if is_best:
                    best_accuracy = val_acc
                    print(f"  New best model! Accuracy: {best_accuracy:.4f}")
                    model_to_save = model.module if isinstance(model, DDP) else model
                    torch.save(model_to_save.state_dict(), Config.SAVE_PATH)

                if early_stopping.early_stop:
                    print(f"\nEarly stopping triggered after epoch {epoch + 1}")
                    stop_tensor[0] = 1

            if is_distributed:
                dist.broadcast(stop_tensor, src=0)

            if stop_tensor.item() == 1:
                break

        if rank == 0:
            print("\n" + "=" * 60)
            print("Training complete.")
            print(f"Modalities: {' + '.join(modalities_desc)}")
            print(f"Epochs completed: {epochs_completed}/{Config.EPOCHS}")
            print(f"Best Validation Accuracy: {best_accuracy:.4f}")
            print(f"Model saved to: {Config.SAVE_PATH}")
            print("=" * 60)

        if rank == 0:
            early_stopping.load_best_model(model)
            model_to_save = model.module if isinstance(model, DDP) else model
            torch.save(model_to_save.state_dict(), Config.SAVE_PATH + ".best")

        if is_distributed:
            dist.barrier()

        best_state = torch.load(
            Config.SAVE_PATH + ".best",
            map_location=f"cuda:{rank}"
        )
        model_to_load = model.module if isinstance(model, DDP) else model
        model_to_load.load_state_dict(best_state)

        _, _, val_preds, val_labels = eval_model(
            model,
            val_loader,
            device,
            Config,
            graph_data=val_graph_data,
        )

        if rank == 0:
            print("\nClassification Report:")
            print(
                classification_report(
                    val_labels,
                    val_preds,
                    labels=[0, 1],
                    target_names=["Real", "Fake"],
                    digits=4,
                    zero_division=0
                )
            )

        if is_distributed:
            dist.barrier()

    finally:
        try:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass


# ==========================
# Main
# ==========================

def main():
    parser = argparse.ArgumentParser(description="Train Multimodal + GNN Model")
    parser.add_argument("--text", action="store_true", help="Enable text modality")
    parser.add_argument("--image", action="store_true", help="Enable image modality")
    parser.add_argument("--graph", action="store_true", help="Enable graph modality")
    parser.add_argument("--prop", action="store_true", help="Enable tweet-ID temporal propagation features")
    parser.add_argument("--cred", action="store_true", help="Enable credibility modality (learnable alpha)")
    parser.add_argument("--gpus", type=int, default=5, help="Number of GPUs to use")
    parser.add_argument(
        "--master-addr",
        type=str,
        default=os.environ.get("MASTER_ADDR", "127.0.0.1"),
        help="DDP master address",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=int(os.environ.get("MASTER_PORT", "29500")),
        help="DDP master port",
    )
    args = parser.parse_args()

    if any([args.text, args.image, args.graph, args.prop, args.cred]):
        Config.USE_TEXT = args.text
        Config.USE_IMAGE = args.image
        Config.USE_GRAPH = args.graph
        Config.USE_PROPAGATION = args.prop
        Config.USE_CREDIBILITY = args.cred

    Config.WORLD_SIZE = args.gpus
    Config.validate()

    # Match the safer batch sizes used in main_worker.
    if Config.USE_IMAGE:
        base_batch_size = 4
    elif Config.USE_TEXT and Config.USE_GRAPH:
        base_batch_size = 8
    elif Config.USE_TEXT:
        base_batch_size = 16
    elif Config.USE_GRAPH:
        base_batch_size = 128
    else:
        base_batch_size = 32

    Config.BATCH_SIZE = base_batch_size * Config.WORLD_SIZE

    if not torch.cuda.is_available():
        print("CUDA is not available. Multi-GPU training requires CUDA.")
        return

    available_gpus = torch.cuda.device_count()
    if available_gpus < Config.WORLD_SIZE:
        print(f"Warning: Requested {Config.WORLD_SIZE} GPUs but only {available_gpus} available.")
        print(f"Using {available_gpus} GPUs instead.")
        Config.WORLD_SIZE = available_gpus

    if Config.WORLD_SIZE < 1:
        print("No GPUs available.")
        return

    if Config.WORLD_SIZE == 1:
        main_worker(0, Config.WORLD_SIZE, args)
        return

    mp.set_start_method("spawn", force=True)

    mp.spawn(
        main_worker,
        args=(Config.WORLD_SIZE, args),
        nprocs=Config.WORLD_SIZE,
        join=True,
    )


if __name__ == "__main__":
    main()
