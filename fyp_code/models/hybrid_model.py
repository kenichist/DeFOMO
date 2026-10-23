"""
Hybrid Model for Fake News Detection
Combines BERT text analysis with network graph propagation features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from .bert_classifier import BERTClassifier
from .graph_network import GraphNeuralNetwork, GraphFeatureExtractor


class HybridFakeNewsDetector(nn.Module):
    """
    Hybrid model combining textual and network-based features.

    Architecture:
    - BERT encoder for text content analysis
    - GNN for propagation pattern analysis
    - Statistical graph features
    - Attention-based feature fusion
    - Final classification head
    """

    def __init__(
        self,
        bert_model: str = 'bert-base-uncased',
        text_hidden_dim: int = 768,
        graph_hidden_dim: int = 128,
        graph_output_dim: int = 64,
        graph_feature_dim: int = 15,
        fusion_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.3,
        freeze_bert: bool = False,
        use_attention: bool = True,
        gradient_checkpointing: bool = False
    ):
        """
        Initialize hybrid model.

        Args:
            bert_model: Pre-trained BERT model name
            text_hidden_dim: BERT hidden dimension (768 for base, 1024 for large)
            graph_hidden_dim: GNN hidden dimension
            graph_output_dim: GNN output dimension
            graph_feature_dim: Statistical graph feature dimension
            fusion_dim: Fusion layer dimension
            num_classes: Number of output classes
            dropout: Dropout probability
            freeze_bert: Whether to freeze BERT parameters
            use_attention: Use attention for feature fusion
            gradient_checkpointing: Enable gradient checkpointing in the BERT encoder
        """
        super(HybridFakeNewsDetector, self).__init__()

        self.use_attention = use_attention

        # Text encoder (BERT)
        self.bert_classifier = BERTClassifier(
            pretrained_model=bert_model,
            num_classes=num_classes,
            dropout=dropout,
            freeze_bert=freeze_bert,
            gradient_checkpointing=gradient_checkpointing
        )

        # Graph neural network
        self.gnn = GraphNeuralNetwork(
            node_feature_dim=64,
            hidden_dim=graph_hidden_dim,
            output_dim=graph_output_dim,
            num_layers=2,
            dropout=dropout
        )

        # Statistical graph feature encoder
        self.graph_stat_encoder = nn.Sequential(
            nn.Linear(graph_feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32)
        )

        # Feature projection layers
        self.text_projection = nn.Linear(text_hidden_dim, fusion_dim)
        self.gnn_projection = nn.Linear(graph_output_dim, fusion_dim)
        self.stat_projection = nn.Linear(32, fusion_dim)

        # Attention mechanism for feature fusion
        if use_attention:
            self.attention = MultimodalAttention(
                feature_dim=fusion_dim,
                num_modalities=3,
                dropout=dropout
            )

        # Fusion and classification
        total_fusion_dim = fusion_dim * 3 if not use_attention else fusion_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Linear(fusion_dim // 2, num_classes)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        node_features: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        graph_stats: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hybrid model.

        Args:
            input_ids: BERT input token IDs
            attention_mask: BERT attention mask
            node_features: Graph node features [num_nodes, node_dim]
            edge_index: Graph edge connectivity [2, num_edges]
            graph_stats: Statistical graph features [batch_size, graph_feature_dim]
            token_type_ids: BERT token type IDs

        Returns:
            Dictionary with logits, probabilities, and feature embeddings
        """
        # Text features from BERT
        text_logits, text_features = self.bert_classifier(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # text_features: [batch_size, text_hidden_dim]

        # Initialize graph features
        batch_size = input_ids.size(0)
        device = input_ids.device

        # GNN features
        if node_features is not None and edge_index is not None:
            gnn_features = self.gnn.get_graph_embedding(node_features, edge_index)
            # gnn_features: [batch_size, graph_output_dim]

            # Handle single graph case
            if gnn_features.size(0) == 1 and batch_size > 1:
                gnn_features = gnn_features.repeat(batch_size, 1)
        else:
            # Use zero features if no graph available
            gnn_features = torch.zeros(batch_size, self.gnn.output_dim if hasattr(self.gnn, 'output_dim') else 64).to(device)

        # Statistical graph features
        if graph_stats is not None:
            stat_features = self.graph_stat_encoder(graph_stats)
            # stat_features: [batch_size, 32]
        else:
            # Use zero features if no stats available
            stat_features = torch.zeros(batch_size, 32).to(device)

        # Project all features to same dimension
        text_proj = self.text_projection(text_features)  # [batch_size, fusion_dim]
        gnn_proj = self.gnn_projection(gnn_features)     # [batch_size, fusion_dim]
        stat_proj = self.stat_projection(stat_features)  # [batch_size, fusion_dim]

        # Feature fusion
        if self.use_attention:
            # Stack features for attention
            features_stack = torch.stack([text_proj, gnn_proj, stat_proj], dim=1)
            # features_stack: [batch_size, 3, fusion_dim]

            # Apply multimodal attention
            fused_features = self.attention(features_stack)
            # fused_features: [batch_size, fusion_dim]
        else:
            # Simple concatenation
            fused_features = torch.cat([text_proj, gnn_proj, stat_proj], dim=1)
            # fused_features: [batch_size, fusion_dim * 3]

        # Final fusion and classification
        fused_features = self.fusion_layer(fused_features)
        logits = self.classifier(fused_features)

        # Compute probabilities
        probs = F.softmax(logits, dim=1)

        return {
            'logits': logits,
            'probabilities': probs,
            'text_features': text_features,
            'gnn_features': gnn_features,
            'stat_features': stat_features,
            'fused_features': fused_features
        }

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        node_features: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        graph_stats: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions.

        Returns:
            predictions: Class predictions (0 or 1)
            probabilities: Prediction probabilities
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                node_features=node_features,
                edge_index=edge_index,
                graph_stats=graph_stats,
                token_type_ids=token_type_ids
            )
            predictions = torch.argmax(outputs['probabilities'], dim=1)
            return predictions, outputs['probabilities']


class MultimodalAttention(nn.Module):
    """
    Attention mechanism for fusing multimodal features.

    Learns importance weights for different modalities.
    """

    def __init__(
        self,
        feature_dim: int,
        num_modalities: int = 3,
        dropout: float = 0.1
    ):
        """
        Initialize multimodal attention.

        Args:
            feature_dim: Dimension of each feature modality
            num_modalities: Number of modalities to fuse
            dropout: Dropout probability
        """
        super(MultimodalAttention, self).__init__()

        self.feature_dim = feature_dim
        self.num_modalities = num_modalities

        # Attention query, key, value projections
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([feature_dim]))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply attention to fuse features.

        Args:
            features: Stacked features [batch_size, num_modalities, feature_dim]

        Returns:
            Fused features [batch_size, feature_dim]
        """
        batch_size = features.size(0)

        # Compute queries, keys, values
        Q = self.query(features)  # [batch_size, num_modalities, feature_dim]
        K = self.key(features)
        V = self.value(features)

        # Compute attention scores
        # Use mean of all features as query
        query_mean = Q.mean(dim=1, keepdim=True)  # [batch_size, 1, feature_dim]

        # Attention weights
        scale = self.scale.to(features.device)
        scores = torch.matmul(query_mean, K.transpose(-2, -1)) / scale
        # scores: [batch_size, 1, num_modalities]

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        attended = torch.matmul(attention_weights, V)  # [batch_size, 1, feature_dim]
        attended = attended.squeeze(1)  # [batch_size, feature_dim]

        return attended


class WeightedEnsemble(nn.Module):
    """
    Learnable weighted ensemble of text-only and hybrid predictions.
    """

    def __init__(self, num_classes: int = 2):
        super(WeightedEnsemble, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Learnable weight

    def forward(
        self,
        text_logits: torch.Tensor,
        hybrid_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Combine predictions from text-only and hybrid models.

        Args:
            text_logits: Logits from BERT-only model
            hybrid_logits: Logits from hybrid model

        Returns:
            Combined logits
        """
        alpha_clamped = torch.sigmoid(self.alpha)  # Ensure 0 <= alpha <= 1
        combined = alpha_clamped * text_logits + (1 - alpha_clamped) * hybrid_logits
        return combined


def initialize_hybrid_model(
    device: torch.device,
    bert_model: str = 'bert-base-uncased',
    use_attention: bool = True,
    freeze_bert: bool = False,
    gradient_checkpointing: bool = False
) -> HybridFakeNewsDetector:
    """
    Initialize hybrid model and move to device.

    Args:
        device: Device to use (cuda or cpu)
        bert_model: Pre-trained BERT model name
        use_attention: Use attention for feature fusion
        freeze_bert: Whether to freeze BERT parameters
        gradient_checkpointing: Enable gradient checkpointing in the BERT encoder

    Returns:
        Initialized hybrid model
    """
    model = HybridFakeNewsDetector(
        bert_model=bert_model,
        use_attention=use_attention,
        freeze_bert=freeze_bert,
        gradient_checkpointing=gradient_checkpointing
    )
    model = model.to(device)

    print(f"Hybrid model initialized on device: {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    return model
