"""
Multimodal + GNN Training Script for Fakeddit Misinformation Detection

This script combines:
1. Text features (DistilBERT)
2. Image features (ViT)
3. Source credibility scores (MediaRank)
4. Graph-based relational features (Heterogeneous GNN)

The model architecture:
- Encode text with DistilBERT
- Encode images with ViT
- Build heterogeneous graph (post -> user, subreddit, domain)
- Extract GNN embeddings for relational context
- Fuse all modalities with credibility weighting

Usage:
    python train_multimodal_gnn.py                              # All 4 modalities (default)
    python train_multimodal_gnn.py --text --image               # Text + Image only
    python train_multimodal_gnn.py --text --graph --cred        # Text + Graph + Cred
    python train_multimodal_gnn.py --image                      # Image only
"""

import os
import argparse
import json
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

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

from typing import Dict, Optional


# ==========================
# Early Stopping
# ==========================

class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.

    Args:
        patience (int): Number of epochs to wait for improvement before stopping.
        min_delta (float): Minimum change to qualify as an improvement.
        mode (str): 'min' for loss (lower is better), 'max' for accuracy (higher is better).
        verbose (bool): If True, prints messages when early stopping is triggered.
    """

    def __init__(self, patience=3, min_delta=0.0, mode='min', verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, metric, model):
        """
        Check if training should stop.

        Args:
            metric: The metric to monitor (loss or accuracy).
            model: The model to save state from.

        Returns:
            bool: True if this is a new best, False otherwise.
        """
        if self.mode == 'min':
            score = -metric
        else:
            score = metric

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
                print(f'  EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            return False

    def save_checkpoint(self, model):
        """Save model state when validation improves."""
        self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    def load_best_model(self, model):
        """Load the best model state."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            if self.verbose:
                print('Loaded best model from early stopping checkpoint.')


# ==========================
# Config
# ==========================

class Config:
    MAX_LEN = 128
    BATCH_SIZE = 5
    EPOCHS = 10

    TEXT_MODEL_NAME = "distilbert-base-uncased"
    VIT_MODEL_NAME = "google/vit-base-patch16-224-in21k"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42

    # paths
    TRAIN_TSV = os.path.join("data", "train.tsv")
    IMAGE_ROOT = os.path.join("data", "images")
    DOMAIN_WEIGHTS_JSON = "domain_weights.json"

    # ===========================================
    # MODALITY SELECTION
    # ===========================================
    # Set each to True/False to enable/disable
    USE_TEXT = True          # DistilBERT text features
    USE_IMAGE = True         # ViT image features
    USE_GRAPH = True         # Heterogeneous GNN features
    USE_CREDIBILITY = True   # Source credibility weighting

    # Fusion settings
    FUSION_TYPE = "concat"   # 'concat', 'gated', 'attention', 'sum'
    D_PROJ = 256             # Projection dimension for each modality

    # GNN settings (only used if USE_GRAPH=True)
    GNN_HIDDEN_DIM = 128
    GNN_LAYERS = 2
    GNN_HEADS = 4
    GNN_DROPOUT = 0.3
    USER_EMBED_DIM = 64
    SUBREDDIT_EMBED_DIM = 64
    DOMAIN_EMBED_DIM = 64

    # Credibility weighting (only used if USE_CREDIBILITY=True)
    CREDIBILITY_WEIGHT = 0.25
    LEARNING_RATE = 2e-5

    # Early stopping configuration
    EARLY_STOPPING_PATIENCE = 3
    EARLY_STOPPING_MIN_DELTA = 0.001
    EARLY_STOPPING_MODE = 'max'

    # Save paths (auto-generated based on modalities)
    SAVE_PATH = None  # Will be set in main()
    GNN_EMBEDDINGS_CACHE = "cached_gnn_embeddings.pt"

    @classmethod
    def get_modality_string(cls):
        """Get string describing enabled modalities."""
        modalities = []
        if cls.USE_TEXT:
            modalities.append("text")
        if cls.USE_IMAGE:
            modalities.append("image")
        if cls.USE_GRAPH:
            modalities.append("graph")
        if cls.USE_CREDIBILITY:
            modalities.append("cred")
        return "_".join(modalities) if modalities else "none"

    @classmethod
    def validate(cls):
        """Validate that at least one content modality is enabled."""
        if not (cls.USE_TEXT or cls.USE_IMAGE or cls.USE_GRAPH):
            raise ValueError(
                "At least one content modality (text, image, or graph) must be enabled. "
                "Credibility alone cannot make predictions."
            )


torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)


# ==========================
# Load domain weights from MediaRank
# ==========================

def load_domain_weights(json_path):
    """
    Load and normalize domain credibility scores.
    """
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
            w = (float(s) - min_s) / (max_s - min_s)
            domain_to_weight[dom.lower()] = float(w)
    else:
        for dom in raw.keys():
            domain_to_weight[dom.lower()] = 0.5

    mean_weight = float(np.mean(list(domain_to_weight.values()))) if domain_to_weight else 0.5

    print(f"[load_domain_weights] Loaded {len(domain_to_weight)} domains from {json_path}")
    return domain_to_weight, mean_weight


# ==========================
# Graph Builder
# ==========================

class FakedditGraphBuilder:
    """
    Builds a heterogeneous graph from Fakeddit data.

    Node types: post, user, subreddit, domain
    Edge types: post->user, post->subreddit, post->domain (bidirectional)
    """

    def __init__(
        self,
        user_embed_dim: int = 64,
        subreddit_embed_dim: int = 64,
        domain_embed_dim: int = 64,
    ):
        self.user_embed_dim = user_embed_dim
        self.subreddit_embed_dim = subreddit_embed_dim
        self.domain_embed_dim = domain_embed_dim

        self.user_to_idx: Dict[str, int] = {}
        self.subreddit_to_idx: Dict[str, int] = {}
        self.domain_to_idx: Dict[str, int] = {}
        self.post_to_idx: Dict[str, int] = {}

    def build_graph(self, df: pd.DataFrame) -> HeteroData:
        """Build heterogeneous graph from DataFrame."""
        self.user_to_idx = {}
        self.subreddit_to_idx = {}
        self.domain_to_idx = {}
        self.post_to_idx = {}

        # Build entity mappings
        for idx, row in df.iterrows():
            post_id = str(row['id'])
            if post_id not in self.post_to_idx:
                self.post_to_idx[post_id] = len(self.post_to_idx)

            author = str(row.get('author', ''))
            if author and author != 'nan' and author != '':
                if author not in self.user_to_idx:
                    self.user_to_idx[author] = len(self.user_to_idx)

            subreddit = str(row.get('subreddit', ''))
            if subreddit and subreddit != 'nan':
                if subreddit not in self.subreddit_to_idx:
                    self.subreddit_to_idx[subreddit] = len(self.subreddit_to_idx)

            domain = str(row.get('domain', ''))
            if domain and domain != 'nan':
                if domain not in self.domain_to_idx:
                    self.domain_to_idx[domain] = len(self.domain_to_idx)

        # Build edges
        edge_post_user = self._build_edges(df, 'author', self.user_to_idx)
        edge_post_subreddit = self._build_edges(df, 'subreddit', self.subreddit_to_idx)
        edge_post_domain = self._build_edges(df, 'domain', self.domain_to_idx)

        # Create HeteroData
        data = HeteroData()

        num_posts = len(self.post_to_idx)
        num_users = len(self.user_to_idx)
        num_subreddits = len(self.subreddit_to_idx)
        num_domains = len(self.domain_to_idx)

        # Initialize node features (will be replaced for posts)
        data['post'].x = torch.zeros(num_posts, 1)  # Placeholder
        data['user'].x = torch.randn(num_users, self.user_embed_dim)
        data['subreddit'].x = torch.randn(num_subreddits, self.subreddit_embed_dim)
        data['domain'].x = torch.randn(num_domains, self.domain_embed_dim)

        # Edge indices (bidirectional)
        data['post', 'authored_by', 'user'].edge_index = edge_post_user
        data['user', 'authored', 'post'].edge_index = edge_post_user.flip(0)

        data['post', 'posted_in', 'subreddit'].edge_index = edge_post_subreddit
        data['subreddit', 'contains', 'post'].edge_index = edge_post_subreddit.flip(0)

        data['post', 'from_domain', 'domain'].edge_index = edge_post_domain
        data['domain', 'sources', 'post'].edge_index = edge_post_domain.flip(0)

        # Labels
        labels = df['2_way_label'].values
        data['post'].y = torch.tensor(labels, dtype=torch.long)

        print(f"Graph Statistics:")
        print(f"  Posts: {num_posts}, Users: {num_users}, Subreddits: {num_subreddits}, Domains: {num_domains}")

        return data

    def _build_edges(self, df: pd.DataFrame, col: str, mapping: Dict[str, int]) -> torch.Tensor:
        """Build post -> entity edges."""
        src, dst = [], []
        for _, row in df.iterrows():
            post_id = str(row['id'])
            entity = str(row.get(col, ''))
            if entity and entity != 'nan' and entity in mapping:
                src.append(self.post_to_idx[post_id])
                dst.append(mapping[entity])
        return torch.tensor([src, dst], dtype=torch.long)


# ==========================
# GNN Module (Lightweight)
# ==========================

class HeteroGNNEncoder(nn.Module):
    """
    Heterogeneous GNN encoder that produces embeddings for post nodes.
    Used as a component in the multimodal model.
    """

    def __init__(
        self,
        post_in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
        user_in_channels: int = 64,
        subreddit_in_channels: int = 64,
        domain_in_channels: int = 64,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_channels = hidden_channels

        # Input projections
        self.input_projections = nn.ModuleDict({
            'post': Linear(post_in_channels, hidden_channels),
            'user': Linear(user_in_channels, hidden_channels),
            'subreddit': Linear(subreddit_in_channels, hidden_channels),
            'domain': Linear(domain_in_channels, hidden_channels),
        })

        # Heterogeneous convolution layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv = HeteroConv({
                ('post', 'authored_by', 'user'): GATConv(
                    hidden_channels, hidden_channels // num_heads,
                    heads=num_heads, dropout=dropout, add_self_loops=False
                ),
                ('user', 'authored', 'post'): GATConv(
                    hidden_channels, hidden_channels // num_heads,
                    heads=num_heads, dropout=dropout, add_self_loops=False
                ),
                ('post', 'posted_in', 'subreddit'): GATConv(
                    hidden_channels, hidden_channels // num_heads,
                    heads=num_heads, dropout=dropout, add_self_loops=False
                ),
                ('subreddit', 'contains', 'post'): GATConv(
                    hidden_channels, hidden_channels // num_heads,
                    heads=num_heads, dropout=dropout, add_self_loops=False
                ),
                ('post', 'from_domain', 'domain'): GATConv(
                    hidden_channels, hidden_channels // num_heads,
                    heads=num_heads, dropout=dropout, add_self_loops=False
                ),
                ('domain', 'sources', 'post'): GATConv(
                    hidden_channels, hidden_channels // num_heads,
                    heads=num_heads, dropout=dropout, add_self_loops=False
                ),
            }, aggr='sum')
            self.convs.append(conv)

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict) -> torch.Tensor:
        """
        Forward pass through the GNN.

        Returns:
            Post node embeddings: (num_posts, hidden_channels)
        """
        # Project inputs
        x_dict = {
            node_type: self.input_projections[node_type](x)
            for node_type, x in x_dict.items()
        }

        # Message passing
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            x_dict = {key: F.dropout(x, p=self.dropout, training=self.training)
                      for key, x in x_dict.items()}

        return x_dict['post']  # Return post embeddings


# ==========================
# Dataset
# ==========================

class FakedditMultimodalGNNDataset(Dataset):
    """
    Dataset that provides text, image, credibility, and GNN embedding indices.
    """

    def __init__(
        self,
        df,
        tokenizer,
        image_processor,
        image_root,
        max_len,
        post_to_idx,
        domain_weights=None,
        default_weight=0.5,
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

        # --- text ---
        if self.tokenizer is not None:
            title = str(row.get("clean_title", ""))
            body = str(row.get("text", ""))
            if body == "nan":
                body = ""
            text = (title + " " + body).strip()

            enc = self.tokenizer.encode_plus(
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

        # --- image ---
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

        # --- label ---
        label = int(row["2_way_label"])

        # --- credibility score ---
        domain = self._extract_domain_from_row(row)
        if domain is not None:
            credibility_score = self.domain_weights.get(domain, self.default_weight)
        else:
            credibility_score = self.default_weight

        # --- graph index ---
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
# Multimodal + GNN Model
# ==========================

class MultiModalGNNModel(nn.Module):
    """
    Configurable multimodal model that can combine any subset of:
    - Text (DistilBERT)
    - Image (ViT)
    - Graph (Heterogeneous GNN)
    - Credibility weighting

    Set use_text, use_image, use_graph, use_credibility to enable/disable each.

    Architecture:
    1. Encode enabled modalities -> project each to d_proj
    2. Fuse all enabled modalities
    3. Apply credibility weighting (if enabled) to final probabilities
    """

    def __init__(
        self,
        text_model=None,
        vit_model=None,
        d_proj=256,
        gnn_dim=128,
        num_classes=2,
        fusion_type="concat",
        credibility_weight=0.2,
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

        self.text_model = text_model
        self.vit_model = vit_model
        self.fusion_type = fusion_type
        self.d_proj = d_proj
        self.num_classes = num_classes
        self.credibility_weight = float(credibility_weight) if use_credibility else 0.0

        if use_credibility and not 0.0 <= self.credibility_weight <= 1.0:
            raise ValueError(f"credibility_weight must be in [0, 1], got {self.credibility_weight}")

        # Count enabled content modalities
        self.num_modalities = sum([use_text, use_image, use_graph])
        if self.num_modalities == 0:
            raise ValueError("At least one content modality (text, image, graph) must be enabled")

        # Create projections only for enabled modalities
        if use_text:
            d_t = text_model.config.hidden_size
            self.proj_t = nn.Linear(d_t, d_proj)

        if use_image:
            d_v = vit_model.config.hidden_size
            self.proj_v = nn.Linear(d_v, d_proj)

        if use_graph:
            self.proj_g = nn.Linear(gnn_dim, d_proj)

        # Fusion layers - dimension depends on number of modalities
        if fusion_type == "concat":
            fusion_dim = self.num_modalities * d_proj
        elif fusion_type == "gated":
            if self.num_modalities > 1:
                self.gate_fc = nn.Linear(self.num_modalities * d_proj, self.num_modalities)
            fusion_dim = d_proj
        elif fusion_type == "attention":
            if self.num_modalities > 1:
                self.mod_attn = nn.MultiheadAttention(
                    embed_dim=d_proj, num_heads=4, batch_first=True
                )
            fusion_dim = d_proj
        else:  # sum
            fusion_dim = d_proj

        self.fc1 = nn.Linear(fusion_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def encode_text(self, input_ids, attention_mask):
        out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]

    def encode_image(self, pixel_values):
        out = self.vit_model(pixel_values=pixel_values)
        return out.last_hidden_state[:, 0, :]

    def fuse(self, features, has_image=None):
        """
        Fuse list of feature tensors.
        features: list of [B, d_proj] tensors for enabled modalities
        """
        if len(features) == 1:
            return features[0]

        if self.fusion_type == "concat":
            return torch.cat(features, dim=-1)

        elif self.fusion_type == "gated":
            concat = torch.cat(features, dim=-1)
            alpha = torch.softmax(self.gate_fc(concat), dim=-1)  # [B, num_modalities]

            # Handle missing images by redistributing weight
            if has_image is not None and self.use_image:
                mask = has_image.view(-1, 1)
                # Find which index is image in the features list
                img_idx = sum([self.use_text])  # image comes after text
                # Zero out image weight and redistribute
                alpha_list = list(alpha.split(1, dim=-1))
                img_weight = alpha_list[img_idx]
                redistribute = img_weight / (len(features) - 1)
                for i in range(len(alpha_list)):
                    if i == img_idx:
                        alpha_list[i] = alpha_list[i] * mask
                    else:
                        alpha_list[i] = alpha_list[i] + redistribute * (1.0 - mask)
                alpha = torch.cat(alpha_list, dim=-1)

            fused = sum(alpha[:, i:i+1] * f for i, f in enumerate(features))
            return fused

        elif self.fusion_type == "attention":
            M = torch.stack(features, dim=1)  # [B, num_modalities, d_proj]
            M_attn, _ = self.mod_attn(M, M, M)
            return M_attn.mean(dim=1)

        else:  # sum
            return sum(features)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        gnn_embeddings=None,
        has_image=None,
        credibility_score=None,
        labels=None,
    ):
        features = []
        batch_size = None

        # 1) Encode text (if enabled)
        if self.use_text:
            t = self.encode_text(input_ids, attention_mask)
            t = self.proj_t(t)
            features.append(t)
            batch_size = t.size(0)

        # 2) Encode image (if enabled)
        if self.use_image:
            v = self.encode_image(pixel_values)
            v = self.proj_v(v)
            # Zero out image features where there is no image
            if has_image is not None:
                mask = has_image.view(-1, 1).to(v.device)
                v = v * mask
            features.append(v)
            batch_size = v.size(0)

        # 3) Project GNN embeddings (if enabled)
        if self.use_graph:
            g = self.proj_g(gnn_embeddings)
            features.append(g)
            batch_size = g.size(0)

        # 4) Fuse all enabled modalities
        z = self.fuse(features, has_image=has_image if self.use_image else None)

        # 5) Classifier head
        h = self.dropout(self.act(self.fc1(z)))
        content_logits = self.fc2(h)

        # 6) Credibility weighting (if enabled)
        if self.use_credibility and credibility_score is not None and self.credibility_weight > 0:
            content_probs = F.softmax(content_logits, dim=-1)
            cred = credibility_score.view(-1, 1).to(content_logits.device)
            cred_probs = torch.cat([1.0 - cred, cred], dim=-1)

            content_weight = 1.0 - self.credibility_weight
            final_probs = content_weight * content_probs + self.credibility_weight * cred_probs

            eps = 1e-7
            final_probs = torch.clamp(final_probs, min=eps, max=1.0 - eps)
            logits = torch.log(final_probs)
        else:
            logits = content_logits

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)


# ==========================
# Training & Evaluation
# ==========================

def extract_gnn_embeddings(
    df: pd.DataFrame,
    graph_builder: FakedditGraphBuilder,
    gnn_encoder: HeteroGNNEncoder,
    text_model,
    image_processor,
    tokenizer,
    device: torch.device,
    cache_path: str = None,
) -> torch.Tensor:
    """
    Pre-compute GNN embeddings for all posts.

    This extracts text+image embeddings, builds the graph,
    and runs the GNN to get contextual post embeddings.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached GNN embeddings from {cache_path}")
        return torch.load(cache_path)

    print("Extracting GNN embeddings...")

    # Build graph
    data = graph_builder.build_graph(df)

    # Extract text embeddings for post nodes
    print("  Extracting text embeddings for graph...")
    text_embeddings = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="  Text"):
        title = str(row.get("clean_title", ""))
        if title == "nan":
            title = ""

        enc = tokenizer.encode_plus(
            title,
            add_special_tokens=True,
            max_length=Config.MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            outputs = text_model(input_ids=input_ids, attention_mask=attention_mask)
            text_emb = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()
            text_embeddings.append(text_emb)

    text_embeddings = torch.stack(text_embeddings)

    # Use text embeddings as post features
    data['post'].x = text_embeddings

    # Move graph to device
    data = data.to(device)
    gnn_encoder = gnn_encoder.to(device)

    # Run GNN
    print("  Running GNN encoder...")
    gnn_encoder.eval()
    with torch.no_grad():
        gnn_embeddings = gnn_encoder(data.x_dict, data.edge_index_dict)

    gnn_embeddings = gnn_embeddings.cpu()

    if cache_path:
        print(f"  Caching GNN embeddings to {cache_path}")
        torch.save(gnn_embeddings, cache_path)

    return gnn_embeddings


def train_epoch(model, data_loader, optimizer, scheduler, gnn_embeddings, device, config):
    model.train()
    losses = []
    correct_predictions = 0
    total_predictions = 0

    progress_bar = tqdm(data_loader, desc="Training")

    for batch in progress_bar:
        labels = batch["labels"].to(device)

        # Prepare inputs based on enabled modalities
        kwargs = {"labels": labels}

        if config.USE_TEXT:
            kwargs["input_ids"] = batch["input_ids"].to(device)
            kwargs["attention_mask"] = batch["attention_mask"].to(device)

        if config.USE_IMAGE:
            kwargs["pixel_values"] = batch["pixel_values"].to(device)
            kwargs["has_image"] = batch["has_image"].to(device)

        if config.USE_GRAPH:
            graph_idx = batch["graph_idx"]
            kwargs["gnn_embeddings"] = gnn_embeddings[graph_idx].to(device)

        if config.USE_CREDIBILITY:
            kwargs["credibility_score"] = batch["credibility_score"].to(device)

        outputs = model(**kwargs)

        loss = outputs.loss
        logits = outputs.logits

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        preds = torch.argmax(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        total_predictions += labels.size(0)

        losses.append(loss.item())
        progress_bar.set_postfix({"loss": f"{loss.item():.4g}"})

    accuracy = correct_predictions.double() / total_predictions
    return accuracy, np.mean(losses)


def eval_model(model, data_loader, gnn_embeddings, device, config):
    model.eval()
    losses = []
    correct_predictions = 0
    total_predictions = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Evaluating")

        for batch in progress_bar:
            labels = batch["labels"].to(device)

            # Prepare inputs based on enabled modalities
            kwargs = {"labels": labels}

            if config.USE_TEXT:
                kwargs["input_ids"] = batch["input_ids"].to(device)
                kwargs["attention_mask"] = batch["attention_mask"].to(device)

            if config.USE_IMAGE:
                kwargs["pixel_values"] = batch["pixel_values"].to(device)
                kwargs["has_image"] = batch["has_image"].to(device)

            if config.USE_GRAPH:
                graph_idx = batch["graph_idx"]
                kwargs["gnn_embeddings"] = gnn_embeddings[graph_idx].to(device)

            if config.USE_CREDIBILITY:
                kwargs["credibility_score"] = batch["credibility_score"].to(device)

            outputs = model(**kwargs)

            loss = outputs.loss
            logits = outputs.logits

            preds = torch.argmax(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_predictions += labels.size(0)
            losses.append(loss.item())

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct_predictions.double() / total_predictions
    return accuracy, np.mean(losses), all_preds, all_labels


# ==========================
# Main
# ==========================

def main():
    # Parse CLI args to override modality flags
    parser = argparse.ArgumentParser(description="Multimodal Misinformation Detection Training")
    parser.add_argument("--text", action="store_true", help="Enable text modality")
    parser.add_argument("--image", action="store_true", help="Enable image modality")
    parser.add_argument("--graph", action="store_true", help="Enable graph modality")
    parser.add_argument("--cred", action="store_true", help="Enable credibility modality")
    args = parser.parse_args()

    # If ANY flag is provided, only those modalities are enabled
    if any([args.text, args.image, args.graph, args.cred]):
        Config.USE_TEXT = args.text
        Config.USE_IMAGE = args.image
        Config.USE_GRAPH = args.graph
        Config.USE_CREDIBILITY = args.cred

    # Validate configuration
    Config.validate()

    # Auto-generate save path based on enabled modalities
    modality_str = Config.get_modality_string()
    if Config.SAVE_PATH is None:
        Config.SAVE_PATH = f"best_model_{modality_str}.pt"

    print("=" * 60)
    print("MULTIMODAL MISINFORMATION DETECTION")
    print("=" * 60)
    print(f"Device: {Config.DEVICE}")
    print(f"\nEnabled Modalities:")
    print(f"  Text (DistilBERT):     {'YES' if Config.USE_TEXT else 'NO'}")
    print(f"  Image (ViT):           {'YES' if Config.USE_IMAGE else 'NO'}")
    print(f"  Graph (GNN):           {'YES' if Config.USE_GRAPH else 'NO'}")
    print(f"  Credibility Weighting: {'YES' if Config.USE_CREDIBILITY else 'NO'}")
    if Config.USE_CREDIBILITY:
        print(f"    -> Content weight:   {(1 - Config.CREDIBILITY_WEIGHT):.0%}")
        print(f"    -> Credibility weight: {Config.CREDIBILITY_WEIGHT:.0%}")
    print(f"\nFusion type: {Config.FUSION_TYPE}")
    print(f"Save path: {Config.SAVE_PATH}")
    print("=" * 60)

    # Initialize variables
    text_model = None
    vit_model = None
    tokenizer = None
    image_processor = None
    gnn_embeddings = None
    domain_weights = {}
    mean_weight = 0.5
    post_to_idx = {}

    # 1) Load domain weights (if using credibility)
    if Config.USE_CREDIBILITY:
        domain_weights, mean_weight = load_domain_weights(Config.DOMAIN_WEIGHTS_JSON)

    # 2) Load tokenizer (needed for text OR graph, since graph uses text embeddings)
    if Config.USE_TEXT or Config.USE_GRAPH:
        print("\nLoading tokenizer...")
        tokenizer = DistilBertTokenizer.from_pretrained(Config.TEXT_MODEL_NAME)

    # 3) Load image processor (if using image)
    if Config.USE_IMAGE:
        print("Loading image processor...")
        image_processor = AutoImageProcessor.from_pretrained(Config.VIT_MODEL_NAME, use_fast=True)

    # 4) Load data
    print("\nLoading data...")
    df = pd.read_csv(Config.TRAIN_TSV, sep="\t")
    df = df.dropna(subset=["2_way_label", "clean_title"])
    df["id"] = df["id"].astype(str)
    df["2_way_label"] = df["2_way_label"].astype(int)
    print(f"Total samples: {len(df)}")

    # 5) Build graph (if using graph)
    if Config.USE_GRAPH:
        print("\nBuilding heterogeneous graph...")
        graph_builder = FakedditGraphBuilder(
            user_embed_dim=Config.USER_EMBED_DIM,
            subreddit_embed_dim=Config.SUBREDDIT_EMBED_DIM,
            domain_embed_dim=Config.DOMAIN_EMBED_DIM,
        )
        _ = graph_builder.build_graph(df)
        post_to_idx = graph_builder.post_to_idx
    else:
        # Still need post_to_idx for dataset
        post_to_idx = {str(row['id']): idx for idx, row in df.iterrows()}

    # 6) Load text encoder (if using text OR graph)
    if Config.USE_TEXT or Config.USE_GRAPH:
        print("\nLoading DistilBERT...")
        text_model = DistilBertModel.from_pretrained(Config.TEXT_MODEL_NAME)
        for p in text_model.parameters():
            p.requires_grad = False
        text_model.to(Config.DEVICE)

    # 7) Load image encoder (if using image)
    if Config.USE_IMAGE:
        print("Loading ViT...")
        vit_model = ViTModel.from_pretrained(Config.VIT_MODEL_NAME)
        for p in vit_model.parameters():
            p.requires_grad = False
        vit_model.to(Config.DEVICE)

    # 8) Extract GNN embeddings (if using graph)
    if Config.USE_GRAPH:
        print("\nInitializing GNN encoder...")
        gnn_encoder = HeteroGNNEncoder(
            post_in_channels=768,
            hidden_channels=Config.GNN_HIDDEN_DIM,
            num_layers=Config.GNN_LAYERS,
            num_heads=Config.GNN_HEADS,
            dropout=Config.GNN_DROPOUT,
            user_in_channels=Config.USER_EMBED_DIM,
            subreddit_in_channels=Config.SUBREDDIT_EMBED_DIM,
            domain_in_channels=Config.DOMAIN_EMBED_DIM,
        )

        gnn_embeddings = extract_gnn_embeddings(
            df=df,
            graph_builder=graph_builder,
            gnn_encoder=gnn_encoder,
            text_model=text_model,
            image_processor=image_processor,
            tokenizer=tokenizer,
            device=Config.DEVICE,
            cache_path=Config.GNN_EMBEDDINGS_CACHE,
        )
        print(f"GNN embeddings shape: {gnn_embeddings.shape}")

    # 9) Create train/val splits
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=Config.SEED,
        stratify=df["2_way_label"].values,
    )
    print(f"\nTrain size: {len(train_df)}, Val size: {len(val_df)}")

    # 10) Create datasets and loaders
    train_dataset = FakedditMultimodalGNNDataset(
        train_df, tokenizer, image_processor, Config.IMAGE_ROOT,
        Config.MAX_LEN, post_to_idx, domain_weights, mean_weight
    )
    val_dataset = FakedditMultimodalGNNDataset(
        val_df, tokenizer, image_processor, Config.IMAGE_ROOT,
        Config.MAX_LEN, post_to_idx, domain_weights, mean_weight
    )

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)

    # 11) Initialize model with enabled modalities
    print("\nInitializing model...")
    model = MultiModalGNNModel(
        text_model=text_model,
        vit_model=vit_model,
        d_proj=Config.D_PROJ,
        gnn_dim=Config.GNN_HIDDEN_DIM,
        num_classes=2,
        fusion_type=Config.FUSION_TYPE,
        credibility_weight=Config.CREDIBILITY_WEIGHT,
        use_text=Config.USE_TEXT,
        use_image=Config.USE_IMAGE,
        use_graph=Config.USE_GRAPH,
        use_credibility=Config.USE_CREDIBILITY,
    ).to(Config.DEVICE)

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    total_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=Config.EARLY_STOPPING_PATIENCE,
        min_delta=Config.EARLY_STOPPING_MIN_DELTA,
        mode=Config.EARLY_STOPPING_MODE,
        verbose=True,
    )

    # Build modality description
    modalities_desc = []
    if Config.USE_TEXT:
        modalities_desc.append("Text")
    if Config.USE_IMAGE:
        modalities_desc.append("Image")
    if Config.USE_GRAPH:
        modalities_desc.append("Graph")
    if Config.USE_CREDIBILITY:
        modalities_desc.append("Credibility")

    print("\n" + "=" * 60)
    print(f"Starting Training: {' + '.join(modalities_desc)}")
    print("=" * 60)

    best_accuracy = 0.0
    epochs_completed = 0

    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")
        print("-" * 60)

        train_acc, train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, gnn_embeddings, Config.DEVICE, Config
        )
        print(f"Train Loss: {train_loss:.4g}, Train Accuracy: {train_acc:.4g}")

        val_acc, val_loss, val_preds, val_labels = eval_model(
            model, val_loader, gnn_embeddings, Config.DEVICE, Config
        )
        print(f"Val Loss: {val_loss:.4g}, Val Accuracy: {val_acc:.4g}")

        epochs_completed = epoch + 1

        is_best = early_stopping(val_acc.item(), model)

        if is_best:
            best_accuracy = val_acc
            print(f"  New best model! Accuracy: {best_accuracy:.4g}")
            torch.save(model.state_dict(), Config.SAVE_PATH)

        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered after epoch {epoch + 1}")
            break

    print("\n" + "=" * 60)
    print("Training complete.")
    print(f"Modalities: {' + '.join(modalities_desc)}")
    print(f"Epochs completed: {epochs_completed}/{Config.EPOCHS}")
    print(f"Best Validation Accuracy: {best_accuracy:.4g}")
    print(f"Model saved to: {Config.SAVE_PATH}")
    print("=" * 60)

    # Reload best model
    early_stopping.load_best_model(model)
    model.to(Config.DEVICE)

    _, _, val_preds, val_labels = eval_model(model, val_loader, gnn_embeddings, Config.DEVICE, Config)

    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds, target_names=["Fake", "Real"], digits=4))


if __name__ == "__main__":
    main()
