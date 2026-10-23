"""
Trainable Multimodal + Heterogeneous GNN for Fakeddit Misinformation Detection

Single-GPU / non-DDP version with learnable credibility weighting.

Features:
- Trainable graph encoder
- No frozen/random cached GNN embeddings
- Graph branch is saved inside the checkpoint
- Train/val graphs are built separately to avoid leakage
- Learnable credibility alpha (adaptive domain weighting)

Supported modalities:
- Text (DistilBERT)
- Image (ViT)
- Graph (heterogeneous GNN)
- Credibility weighting (learnable)
"""

import os
import json
import math
import hashlib
import argparse
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from transformers import (
    DistilBertTokenizer,
    DistilBertModel,
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

class Config:
    MAX_LEN = 128
    BATCH_SIZE = 8
    EPOCHS = 10
    LEARNING_RATE = 2e-5
    SEED = 42

    TEXT_MODEL_NAME = "/root/autodl-tmp/models/distilbert-base-uncased/"
    VIT_MODEL_NAME = "/root/autodl-tmp/models/vit-base-patch16-224-in21k/"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    TRAIN_TSV = os.path.join("data", "train.tsv")
    IMAGE_ROOT = "/root/autodl-tmp/images"
    DOMAIN_WEIGHTS_JSON = "domain_weights.json"

    USE_TEXT = True
    USE_IMAGE = True
    USE_GRAPH = True
    USE_CREDIBILITY = True

    FUSION_TYPE = "concat"   # concat, gated, attention, sum
    D_PROJ = 256

    GNN_HIDDEN_DIM = 128
    GNN_LAYERS = 2
    GNN_HEADS = 4
    GNN_DROPOUT = 0.3

    USER_EMBED_DIM = 64
    SUBREDDIT_EMBED_DIM = 64
    DOMAIN_EMBED_DIM = 64
    POST_FEATURE_DIM = 768

    EARLY_STOPPING_PATIENCE = 3
    EARLY_STOPPING_MIN_DELTA = 0.001
    EARLY_STOPPING_MODE = "max"

    SAVE_PATH = None

    @classmethod
    def get_modality_string(cls):
        modalities = []
        if cls.USE_TEXT:
            modalities.append("text")
        if cls.USE_IMAGE:
            modalities.append("image")
        if cls.USE_GRAPH:
            modalities.append("graph")
        if cls.USE_CREDIBILITY:
            modalities.append("cred_learnable")
        return "_".join(modalities) if modalities else "none"

    @classmethod
    def validate(cls):
        if not (cls.USE_TEXT or cls.USE_IMAGE or cls.USE_GRAPH):
            raise ValueError(
                "At least one content modality (text, image, or graph) must be enabled. "
                "Credibility alone cannot make predictions."
            )


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(Config.SEED)


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
        score = -metric if self.mode == "min" else metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
            return True
        elif score > self.best_score + self.min_delta:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
            return False

    def save_checkpoint(self, model):
        self.best_model_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }

    def load_best_model(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            if self.verbose:
                print("Loaded best model from early stopping checkpoint.")


# ==========================
# Utilities
# ==========================

def stable_hash_vector(key: str, dim: int, seed: int = 42) -> torch.Tensor:
    """
    Deterministic pseudo-random vector from a string key.
    This makes entity features stable across runs/splits.
    """
    raw = f"{seed}:{key}".encode("utf-8")
    digest = hashlib.md5(raw).hexdigest()
    int_seed = int(digest[:8], 16)
    rng = np.random.RandomState(int_seed)
    vec = rng.normal(loc=0.0, scale=1.0, size=dim).astype(np.float32)
    return torch.tensor(vec, dtype=torch.float32)


def load_domain_weights(json_path: str):
    if not os.path.exists(json_path):
        print(f"[load_domain_weights] WARNING: {json_path} not found, using neutral weights.")
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
    print(f"[load_domain_weights] Loaded {len(domain_to_weight)} domains from {json_path}")
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
    tokenizer: DistilBertTokenizer,
    text_model: DistilBertModel,
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

    for start in tqdm(range(0, len(texts), batch_size), desc="Extracting graph post text features"):
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
    ):
        self.post_feature_dim = post_feature_dim
        self.user_embed_dim = user_embed_dim
        self.subreddit_embed_dim = subreddit_embed_dim
        self.domain_embed_dim = domain_embed_dim
        self.seed = seed

        self.user_to_idx: Dict[str, int] = {}
        self.subreddit_to_idx: Dict[str, int] = {}
        self.domain_to_idx: Dict[str, int] = {}
        self.post_to_idx: Dict[str, int] = {}

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
            if author and author != "nan" and author != "":
                if author not in self.user_to_idx:
                    self.user_to_idx[author] = len(self.user_to_idx)

            domain = str(row.get("domain", ""))
            if domain and domain != "nan" and domain != "":
                if domain not in self.domain_to_idx:
                    self.domain_to_idx[domain] = len(self.domain_to_idx)

        edge_post_user = self._build_edges(df, "author", self.user_to_idx)
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

        print("Graph Statistics:")
        print(f"  Posts: {num_posts}, Users: {num_users}, Domains: {num_domains}")

        return data

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
        tokenizer: Optional[DistilBertTokenizer],
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

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # text
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

        # image
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

        # label
        label = int(row["2_way_label"])

        # credibility
        domain = self._extract_domain_from_row(row)
        if domain is not None:
            credibility_score = self.domain_weights.get(domain, self.default_weight)
        else:
            credibility_score = self.default_weight

        # graph idx
        post_id = str(row["id"])
        graph_idx = self.post_to_idx.get(post_id, 0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "has_image": torch.tensor(has_image, dtype=torch.float32),
            "credibility_score": torch.tensor(credibility_score, dtype=torch.float32),
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
        use_text=True,
        use_image=True,
        use_graph=True,
        use_credibility=True,
    ):
        super().__init__()

        self.use_text = use_text
        self.use_image = use_image
        self.use_graph = use_graph
        self.use_credibility = use_credibility
        self.fusion_type = fusion_type
        self.d_proj = d_proj

        self.text_model = text_model
        self.vit_model = vit_model

        self.num_modalities = sum([use_text, use_image, use_graph])
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

        z = self.fuse(features, has_image=has_image if self.use_image else None)

        h = self.dropout(self.act(self.fc1(z)))
        content_logits = self.fc2(h)

        if self.use_credibility and credibility_score is not None:
            alpha = self.cred_gate(z)
            self.last_alpha = alpha.detach()

            content_probs = F.softmax(content_logits, dim=-1)

            cred = credibility_score.view(-1, 1).to(content_logits.device)
            cred_probs = torch.cat([1.0 - cred, cred], dim=-1)

            final_probs = (1.0 - alpha) * content_probs + alpha * cred_probs

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
# Training / evaluation
# ==========================

def train_epoch(model, data_loader, optimizer, scheduler, device, config, graph_data=None):
    model.train()
    losses = []
    correct_predictions = 0
    total_predictions = 0

    alpha_sum = 0.0
    alpha_count = 0

    progress_bar = tqdm(data_loader, desc="Training")

    for batch in progress_bar:
        labels = batch["labels"].to(device)
        kwargs = {"labels": labels}

        if config.USE_TEXT:
            kwargs["input_ids"] = batch["input_ids"].to(device)
            kwargs["attention_mask"] = batch["attention_mask"].to(device)

        if config.USE_IMAGE:
            kwargs["pixel_values"] = batch["pixel_values"].to(device)
            kwargs["has_image"] = batch["has_image"].to(device)

        if config.USE_GRAPH:
            kwargs["graph_idx"] = batch["graph_idx"].to(device)
            kwargs["graph_data"] = graph_data

        if config.USE_CREDIBILITY:
            kwargs["credibility_score"] = batch["credibility_score"].to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(**kwargs)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        preds = torch.argmax(logits, dim=1)
        correct_predictions += torch.sum(preds == labels).item()
        total_predictions += labels.size(0)

        losses.append(loss.item())

        if config.USE_CREDIBILITY and model.last_alpha is not None:
            alpha_sum += model.last_alpha.sum().item()
            alpha_count += model.last_alpha.size(0)

        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    accuracy = correct_predictions / max(1, total_predictions)

    if config.USE_CREDIBILITY and alpha_count > 0:
        avg_alpha = alpha_sum / alpha_count
        print(f"  Average alpha (credibility weight): {avg_alpha:.4f}")

    return accuracy, float(np.mean(losses))


@torch.no_grad()
def eval_model(model, data_loader, device, config, graph_data=None):
    model.eval()
    losses = []
    correct_predictions = 0
    total_predictions = 0

    all_preds = []
    all_labels = []

    alpha_sum = 0.0
    alpha_count = 0

    progress_bar = tqdm(data_loader, desc="Evaluating")

    for batch in progress_bar:
        labels = batch["labels"].to(device)
        kwargs = {"labels": labels}

        if config.USE_TEXT:
            kwargs["input_ids"] = batch["input_ids"].to(device)
            kwargs["attention_mask"] = batch["attention_mask"].to(device)

        if config.USE_IMAGE:
            kwargs["pixel_values"] = batch["pixel_values"].to(device)
            kwargs["has_image"] = batch["has_image"].to(device)

        if config.USE_GRAPH:
            kwargs["graph_idx"] = batch["graph_idx"].to(device)
            kwargs["graph_data"] = graph_data

        if config.USE_CREDIBILITY:
            kwargs["credibility_score"] = batch["credibility_score"].to(device)

        outputs = model(**kwargs)
        loss = outputs.loss
        logits = outputs.logits

        preds = torch.argmax(logits, dim=1)
        correct_predictions += torch.sum(preds == labels).item()
        total_predictions += labels.size(0)

        losses.append(loss.item())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if config.USE_CREDIBILITY and model.last_alpha is not None:
            alpha_sum += model.last_alpha.sum().item()
            alpha_count += model.last_alpha.size(0)

    accuracy = correct_predictions / max(1, total_predictions)

    if config.USE_CREDIBILITY and alpha_count > 0:
        avg_alpha = alpha_sum / alpha_count
        print(f"  Validation average alpha: {avg_alpha:.4f}")

    return accuracy, float(np.mean(losses)), all_preds, all_labels


# ==========================
# Main
# ==========================

def main():
    parser = argparse.ArgumentParser(description="Train Multimodal + GNN Model")
    parser.add_argument("--text", action="store_true", help="Enable text modality")
    parser.add_argument("--image", action="store_true", help="Enable image modality")
    parser.add_argument("--graph", action="store_true", help="Enable graph modality")
    parser.add_argument("--cred", action="store_true", help="Enable credibility modality (learnable alpha)")
    args = parser.parse_args()

    if any([args.text, args.image, args.graph, args.cred]):
        Config.USE_TEXT = args.text
        Config.USE_IMAGE = args.image
        Config.USE_GRAPH = args.graph
        Config.USE_CREDIBILITY = args.cred

    if Config.USE_IMAGE:
        Config.BATCH_SIZE = 16
    elif Config.USE_TEXT and Config.USE_GRAPH:
        Config.BATCH_SIZE = 128
    elif Config.USE_TEXT:
        Config.BATCH_SIZE = 256
    elif Config.USE_GRAPH:
        Config.BATCH_SIZE = 512
    else:
        Config.BATCH_SIZE = 64

    Config.validate()

    modality_str = Config.get_modality_string()
    if Config.SAVE_PATH is None:
        Config.SAVE_PATH = f"best_model_{modality_str}.pt"

    print("=" * 60)
    print("MULTIMODAL MISINFORMATION DETECTION")
    print("=" * 60)
    print(f"Device: {Config.DEVICE}")
    print(f"Text:  {Config.USE_TEXT}")
    print(f"Image: {Config.USE_IMAGE}")
    print(f"Graph: {Config.USE_GRAPH}")
    print(f"Cred:  {Config.USE_CREDIBILITY} (LEARNABLE ALPHA)")
    print(f"Save path: {Config.SAVE_PATH}")
    print("=" * 60)

    tokenizer = None
    image_processor = None
    text_model = None
    vit_model = None
    domain_weights = {}
    mean_weight = 0.5

    if Config.USE_CREDIBILITY:
        domain_weights, mean_weight = load_domain_weights(Config.DOMAIN_WEIGHTS_JSON)

    if Config.USE_TEXT or Config.USE_GRAPH:
        print("\nLoading tokenizer...")
        tokenizer = DistilBertTokenizer.from_pretrained(Config.TEXT_MODEL_NAME)

        print("Loading DistilBERT...")
        text_model = DistilBertModel.from_pretrained(Config.TEXT_MODEL_NAME)
        for p in text_model.parameters():
            p.requires_grad = False
        text_model.to(Config.DEVICE)
        text_model.eval()

    if Config.USE_IMAGE:
        print("\nLoading image processor...")
        image_processor = AutoImageProcessor.from_pretrained(Config.VIT_MODEL_NAME, use_fast=True)

        print("Loading ViT...")
        vit_model = ViTModel.from_pretrained(Config.VIT_MODEL_NAME)
        for p in vit_model.parameters():
            p.requires_grad = False
        vit_model.to(Config.DEVICE)
        vit_model.eval()

    print("\nLoading data...")
    df = pd.read_csv(Config.TRAIN_TSV, sep="\t")
    df = df.dropna(subset=["2_way_label", "clean_title"]).copy()
    df["id"] = df["id"].astype(str)
    df["2_way_label"] = df["2_way_label"].astype(int)
    print(f"Total samples: {len(df)}")

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=Config.SEED,
        stratify=df["2_way_label"].values,
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

    train_graph_data = None
    val_graph_data = None
    train_post_to_idx = {str(row['id']): idx for idx, row in train_df.iterrows()}
    val_post_to_idx = {str(row['id']): idx for idx, row in val_df.iterrows()}

    if Config.USE_GRAPH:
        print("\nPreparing TRAIN graph...")
        train_post_features = extract_post_text_features(
            train_df, tokenizer, text_model, Config.DEVICE, batch_size=32, max_len=Config.MAX_LEN
        )
        train_builder = FakedditGraphBuilder(
            post_feature_dim=Config.POST_FEATURE_DIM,
            user_embed_dim=Config.USER_EMBED_DIM,
            subreddit_embed_dim=Config.SUBREDDIT_EMBED_DIM,
            domain_embed_dim=Config.DOMAIN_EMBED_DIM,
            seed=Config.SEED,
        )
        train_graph_data = train_builder.build_graph(train_df, train_post_features)
        train_graph_data = train_graph_data.to(Config.DEVICE)
        train_post_to_idx = train_builder.post_to_idx

        print("\nPreparing VAL graph...")
        val_post_features = extract_post_text_features(
            val_df, tokenizer, text_model, Config.DEVICE, batch_size=32, max_len=Config.MAX_LEN
        )
        val_builder = FakedditGraphBuilder(
            post_feature_dim=Config.POST_FEATURE_DIM,
            user_embed_dim=Config.USER_EMBED_DIM,
            subreddit_embed_dim=Config.SUBREDDIT_EMBED_DIM,
            domain_embed_dim=Config.DOMAIN_EMBED_DIM,
            seed=Config.SEED,
        )
        val_graph_data = val_builder.build_graph(val_df, val_post_features)
        val_graph_data = val_graph_data.to(Config.DEVICE)
        val_post_to_idx = val_builder.post_to_idx

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    print("\nInitializing model...")
    model = MultiModalGNNModel(
        text_model=text_model,
        vit_model=vit_model,
        d_proj=Config.D_PROJ,
        gnn_dim=Config.GNN_HIDDEN_DIM,
        num_classes=2,
        fusion_type=Config.FUSION_TYPE,
        use_text=Config.USE_TEXT,
        use_image=Config.USE_IMAGE,
        use_graph=Config.USE_GRAPH,
        use_credibility=Config.USE_CREDIBILITY,
    ).to(Config.DEVICE)

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
        modalities_desc.append("Image")
    if Config.USE_GRAPH:
        modalities_desc.append("Graph")
    if Config.USE_CREDIBILITY:
        modalities_desc.append("Cred (Learnable α)")

    print("\n" + "=" * 60)
    print(f"Starting Training: {' + '.join(modalities_desc)}")
    print("=" * 60)

    best_accuracy = 0.0
    epochs_completed = 0

    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")
        print("-" * 60)

        train_acc, train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, Config.DEVICE, Config, graph_data=train_graph_data
        )
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

        val_acc, val_loss, val_preds, val_labels = eval_model(
            model, val_loader, Config.DEVICE, Config, graph_data=val_graph_data
        )
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

        epochs_completed = epoch + 1

        is_best = early_stopping(val_acc, model)
        if is_best:
            best_accuracy = val_acc
            print(f"  New best model! Accuracy: {best_accuracy:.4f}")
            torch.save(model.state_dict(), Config.SAVE_PATH)

        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered after epoch {epoch + 1}")
            break

    print("\n" + "=" * 60)
    print("Training complete.")
    print(f"Modalities: {' + '.join(modalities_desc)}")
    print(f"Epochs completed: {epochs_completed}/{Config.EPOCHS}")
    print(f"Best Validation Accuracy: {best_accuracy:.4f}")
    print(f"Model saved to: {Config.SAVE_PATH}")
    print("=" * 60)

    early_stopping.load_best_model(model)
    model.to(Config.DEVICE)

    _, _, val_preds, val_labels = eval_model(
        model, val_loader, Config.DEVICE, Config, graph_data=val_graph_data
    )

    print("\nClassification Report:")
    print(
        classification_report(
            val_labels,
            val_preds,
            target_names=["Fake", "Real"],
            digits=4,
            zero_division=0
        )
    )


if __name__ == "__main__":
    main()