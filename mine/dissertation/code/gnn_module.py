"""
Heterogeneous Graph Neural Network for Fakeddit Misinformation Detection

This module constructs a heterogeneous graph from the Fakeddit dataset and
implements a GNN for node classification (credibility prediction).

Node Types:
- post: Reddit posts (main nodes for classification)
- user: Reddit authors
- subreddit: Subreddit communities
- domain: Source domains

Edge Types:
- (post, authored_by, user)
- (post, posted_in, subreddit)
- (post, from_domain, domain)
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATConv, SAGEConv, Linear, HGTConv
from torch_geometric.loader import NeighborLoader
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm


# ==========================
# Graph Construction
# ==========================

class FakedditGraphBuilder:
    """
    Builds a heterogeneous graph from Fakeddit TSV data.

    The graph has:
    - post nodes with text/image features
    - user, subreddit, domain nodes with learnable embeddings
    - edges connecting posts to their metadata entities
    """

    def __init__(
        self,
        post_feature_dim: int = 768,  # DistilBERT hidden size
        user_embed_dim: int = 64,
        subreddit_embed_dim: int = 64,
        domain_embed_dim: int = 64,
    ):
        self.post_feature_dim = post_feature_dim
        self.user_embed_dim = user_embed_dim
        self.subreddit_embed_dim = subreddit_embed_dim
        self.domain_embed_dim = domain_embed_dim

        # Mappings: entity name -> index
        self.user_to_idx: Dict[str, int] = {}
        self.subreddit_to_idx: Dict[str, int] = {}
        self.domain_to_idx: Dict[str, int] = {}
        self.post_to_idx: Dict[str, int] = {}

    def build_graph(
        self,
        df: pd.DataFrame,
        post_features: Optional[torch.Tensor] = None,
    ) -> HeteroData:
        """
        Build heterogeneous graph from DataFrame.

        Args:
            df: DataFrame with columns: id, author, subreddit, domain, 2_way_label
            post_features: Optional tensor of shape (num_posts, feature_dim)
                          If None, uses random initialization

        Returns:
            HeteroData object for PyTorch Geometric
        """
        # Reset mappings
        self.user_to_idx = {}
        self.subreddit_to_idx = {}
        self.domain_to_idx = {}
        self.post_to_idx = {}

        # Build entity mappings
        self._build_mappings(df)

        # Build edge indices
        edge_post_user = self._build_post_user_edges(df)
        edge_post_subreddit = self._build_post_subreddit_edges(df)
        edge_post_domain = self._build_post_domain_edges(df)

        # Create HeteroData
        data = HeteroData()

        # Node features
        num_posts = len(self.post_to_idx)
        num_users = len(self.user_to_idx)
        num_subreddits = len(self.subreddit_to_idx)
        num_domains = len(self.domain_to_idx)

        if post_features is not None:
            data['post'].x = post_features
        else:
            # Placeholder - should be replaced with actual embeddings
            data['post'].x = torch.randn(num_posts, self.post_feature_dim)

        # Initialize other node types with learnable embeddings (random init)
        data['user'].x = torch.randn(num_users, self.user_embed_dim)
        data['subreddit'].x = torch.randn(num_subreddits, self.subreddit_embed_dim)
        data['domain'].x = torch.randn(num_domains, self.domain_embed_dim)

        # Edge indices (bidirectional for message passing)
        data['post', 'authored_by', 'user'].edge_index = edge_post_user
        data['user', 'authored', 'post'].edge_index = edge_post_user.flip(0)

        data['post', 'posted_in', 'subreddit'].edge_index = edge_post_subreddit
        data['subreddit', 'contains', 'post'].edge_index = edge_post_subreddit.flip(0)

        data['post', 'from_domain', 'domain'].edge_index = edge_post_domain
        data['domain', 'sources', 'post'].edge_index = edge_post_domain.flip(0)

        # Node labels (only for posts)
        labels = df['2_way_label'].values
        data['post'].y = torch.tensor(labels, dtype=torch.long)

        # Store post IDs for reference
        data['post'].post_ids = list(self.post_to_idx.keys())

        print(f"Graph Statistics:")
        print(f"  Posts: {num_posts}")
        print(f"  Users: {num_users}")
        print(f"  Subreddits: {num_subreddits}")
        print(f"  Domains: {num_domains}")
        print(f"  Edges (post->user): {edge_post_user.size(1)}")
        print(f"  Edges (post->subreddit): {edge_post_subreddit.size(1)}")
        print(f"  Edges (post->domain): {edge_post_domain.size(1)}")

        return data

    def _build_mappings(self, df: pd.DataFrame):
        """Build entity to index mappings."""
        for idx, row in df.iterrows():
            post_id = str(row['id'])
            if post_id not in self.post_to_idx:
                self.post_to_idx[post_id] = len(self.post_to_idx)

            # Handle missing authors
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

    def _build_post_user_edges(self, df: pd.DataFrame) -> torch.Tensor:
        """Build post -> user edges."""
        src, dst = [], []
        for _, row in df.iterrows():
            post_id = str(row['id'])
            author = str(row.get('author', ''))
            if author and author != 'nan' and author in self.user_to_idx:
                src.append(self.post_to_idx[post_id])
                dst.append(self.user_to_idx[author])
        return torch.tensor([src, dst], dtype=torch.long)

    def _build_post_subreddit_edges(self, df: pd.DataFrame) -> torch.Tensor:
        """Build post -> subreddit edges."""
        src, dst = [], []
        for _, row in df.iterrows():
            post_id = str(row['id'])
            subreddit = str(row.get('subreddit', ''))
            if subreddit and subreddit != 'nan' and subreddit in self.subreddit_to_idx:
                src.append(self.post_to_idx[post_id])
                dst.append(self.subreddit_to_idx[subreddit])
        return torch.tensor([src, dst], dtype=torch.long)

    def _build_post_domain_edges(self, df: pd.DataFrame) -> torch.Tensor:
        """Build post -> domain edges."""
        src, dst = [], []
        for _, row in df.iterrows():
            post_id = str(row['id'])
            domain = str(row.get('domain', ''))
            if domain and domain != 'nan' and domain in self.domain_to_idx:
                src.append(self.post_to_idx[post_id])
                dst.append(self.domain_to_idx[domain])
        return torch.tensor([src, dst], dtype=torch.long)


# ==========================
# GNN Models
# ==========================

class HeteroGNN(nn.Module):
    """
    Heterogeneous Graph Neural Network using HeteroConv with GAT/SAGE layers.

    Supports multiple aggregation methods and can be used for node classification.
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        out_channels: int = 2,  # Binary classification
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
        metadata: Tuple = None,  # (node_types, edge_types)
        post_in_channels: int = 768,
        user_in_channels: int = 64,
        subreddit_in_channels: int = 64,
        domain_in_channels: int = 64,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # Input projections for different node types
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
                # Post -> User edges
                ('post', 'authored_by', 'user'): GATConv(
                    hidden_channels, hidden_channels // num_heads,
                    heads=num_heads, dropout=dropout, add_self_loops=False
                ),
                ('user', 'authored', 'post'): GATConv(
                    hidden_channels, hidden_channels // num_heads,
                    heads=num_heads, dropout=dropout, add_self_loops=False
                ),
                # Post -> Subreddit edges
                ('post', 'posted_in', 'subreddit'): GATConv(
                    hidden_channels, hidden_channels // num_heads,
                    heads=num_heads, dropout=dropout, add_self_loops=False
                ),
                ('subreddit', 'contains', 'post'): GATConv(
                    hidden_channels, hidden_channels // num_heads,
                    heads=num_heads, dropout=dropout, add_self_loops=False
                ),
                # Post -> Domain edges
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

        # Classification head for post nodes
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels),
        )

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict) -> torch.Tensor:
        """
        Forward pass through the heterogeneous GNN.

        Args:
            x_dict: Dictionary of node features {node_type: tensor}
            edge_index_dict: Dictionary of edge indices {edge_type: tensor}

        Returns:
            Logits for post node classification
        """
        # Project input features to common dimension
        x_dict = {
            node_type: self.input_projections[node_type](x)
            for node_type, x in x_dict.items()
        }

        # Message passing layers
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            x_dict = {key: F.dropout(x, p=self.dropout, training=self.training)
                      for key, x in x_dict.items()}

        # Classify post nodes
        out = self.classifier(x_dict['post'])
        return out


class HGTModel(nn.Module):
    """
    Heterogeneous Graph Transformer (HGT) model.

    More powerful than simple HeteroConv but requires more memory.
    Uses attention across different node and edge types.
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        out_channels: int = 2,
        num_layers: int = 2,
        num_heads: int = 4,
        metadata: Tuple = None,
        post_in_channels: int = 768,
        user_in_channels: int = 64,
        subreddit_in_channels: int = 64,
        domain_in_channels: int = 64,
    ):
        super().__init__()

        # Input projections
        self.input_projections = nn.ModuleDict({
            'post': Linear(post_in_channels, hidden_channels),
            'user': Linear(user_in_channels, hidden_channels),
            'subreddit': Linear(subreddit_in_channels, hidden_channels),
            'domain': Linear(domain_in_channels, hidden_channels),
        })

        # HGT layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                HGTConv(hidden_channels, hidden_channels, metadata, num_heads)
            )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels // 2, out_channels),
        )

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict) -> torch.Tensor:
        # Project inputs
        x_dict = {
            node_type: self.input_projections[node_type](x)
            for node_type, x in x_dict.items()
        }

        # HGT layers
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        # Classify posts
        return self.classifier(x_dict['post'])


# ==========================
# Integration with Existing Pipeline
# ==========================

class MultimodalGNNFusion(nn.Module):
    """
    Fuses multimodal features (text + image) from existing ViT/DistilBERT
    encoders with GNN-based relational features.

    This can be used as an additional module in your existing pipeline.
    """

    def __init__(
        self,
        text_dim: int = 768,      # DistilBERT output
        image_dim: int = 768,     # ViT output
        gnn_dim: int = 128,       # GNN hidden dim
        hidden_dim: int = 256,
        num_classes: int = 2,
        fusion_type: str = 'concat',  # 'concat', 'gated', 'attention'
    ):
        super().__init__()

        self.fusion_type = fusion_type

        # Feature projections
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.gnn_proj = nn.Linear(gnn_dim, hidden_dim)

        if fusion_type == 'concat':
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, num_classes),
            )
        elif fusion_type == 'gated':
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 3, 3),
                nn.Softmax(dim=-1),
            )
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim // 2, num_classes),
            )
        elif fusion_type == 'attention':
            self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim // 2, num_classes),
            )

    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        gnn_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse multimodal and graph features.

        Args:
            text_features: (batch_size, text_dim)
            image_features: (batch_size, image_dim)
            gnn_features: (batch_size, gnn_dim)

        Returns:
            Logits: (batch_size, num_classes)
        """
        text_h = self.text_proj(text_features)
        image_h = self.image_proj(image_features)
        gnn_h = self.gnn_proj(gnn_features)

        if self.fusion_type == 'concat':
            fused = torch.cat([text_h, image_h, gnn_h], dim=-1)
            return self.classifier(fused)

        elif self.fusion_type == 'gated':
            stacked = torch.stack([text_h, image_h, gnn_h], dim=1)  # (B, 3, H)
            concat = torch.cat([text_h, image_h, gnn_h], dim=-1)
            gates = self.gate(concat).unsqueeze(-1)  # (B, 3, 1)
            fused = (stacked * gates).sum(dim=1)  # (B, H)
            return self.classifier(fused)

        elif self.fusion_type == 'attention':
            # Stack as sequence for attention
            stacked = torch.stack([text_h, image_h, gnn_h], dim=1)  # (B, 3, H)
            attended, _ = self.attention(stacked, stacked, stacked)
            fused = attended.mean(dim=1)  # (B, H)
            return self.classifier(fused)


# ==========================
# Training Utilities
# ==========================

def train_gnn_epoch(
    model: nn.Module,
    data: HeteroData,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch on full graph."""
    model.train()
    optimizer.zero_grad()

    data = data.to(device)
    out = model(data.x_dict, data.edge_index_dict)

    loss = F.cross_entropy(out, data['post'].y)
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate_gnn(
    model: nn.Module,
    data: HeteroData,
    mask: torch.Tensor,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model accuracy."""
    model.eval()
    data = data.to(device)

    out = model(data.x_dict, data.edge_index_dict)
    pred = out.argmax(dim=-1)

    correct = (pred[mask] == data['post'].y[mask]).sum()
    acc = correct.item() / mask.sum().item()

    loss = F.cross_entropy(out[mask], data['post'].y[mask])

    return loss.item(), acc


def create_train_val_test_masks(
    num_nodes: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create random train/val/test masks."""
    torch.manual_seed(seed)
    perm = torch.randperm(num_nodes)

    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[perm[:train_size]] = True
    val_mask[perm[train_size:train_size + val_size]] = True
    test_mask[perm[train_size + val_size:]] = True

    return train_mask, val_mask, test_mask


# ==========================
# Example Usage
# ==========================

if __name__ == "__main__":
    # Example: Build graph from TSV
    import pandas as pd

    # Load data
    tsv_path = os.path.join("data", "train.tsv")
    if os.path.exists(tsv_path):
        df = pd.read_csv(tsv_path, sep="\t")
    else:
        # Create dummy data for testing
        print("Creating dummy data for demonstration...")
        df = pd.DataFrame({
            'id': [f'post_{i}' for i in range(100)],
            'author': [f'user_{i % 20}' for i in range(100)],
            'subreddit': [f'sub_{i % 5}' for i in range(100)],
            'domain': [f'domain_{i % 10}.com' for i in range(100)],
            '2_way_label': np.random.randint(0, 2, 100),
        })

    # Build graph
    builder = FakedditGraphBuilder()
    data = builder.build_graph(df)

    # Create masks
    num_posts = data['post'].x.size(0)
    train_mask, val_mask, test_mask = create_train_val_test_masks(num_posts)
    data['post'].train_mask = train_mask
    data['post'].val_mask = val_mask
    data['post'].test_mask = test_mask

    # Initialize model
    metadata = (
        ['post', 'user', 'subreddit', 'domain'],
        [
            ('post', 'authored_by', 'user'),
            ('user', 'authored', 'post'),
            ('post', 'posted_in', 'subreddit'),
            ('subreddit', 'contains', 'post'),
            ('post', 'from_domain', 'domain'),
            ('domain', 'sources', 'post'),
        ]
    )

    model = HeteroGNN(
        hidden_channels=128,
        out_channels=2,
        num_layers=2,
        metadata=metadata,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    print("\nTraining GNN...")
    for epoch in range(1, 51):
        loss = train_gnn_epoch(model, data, optimizer, device)

        if epoch % 10 == 0:
            val_loss, val_acc = evaluate_gnn(model, data, val_mask, device)
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f}")

    # Final evaluation
    test_loss, test_acc = evaluate_gnn(model, data, test_mask, device)
    print(f"\nTest Accuracy: {test_acc:.4f}")
