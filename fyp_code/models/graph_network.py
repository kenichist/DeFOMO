"""
Network Graph Analysis Module for News Propagation
Analyzes how news spreads through social networks using Graph Neural Networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data


class GraphFeatureExtractor:
    """
    Extracts graph-based features from news propagation patterns.

    Features include:
    - Network topology (degree, centrality, clustering)
    - Temporal propagation patterns
    - User credibility metrics
    - Spread velocity and reach
    """

    def __init__(self):
        self.feature_names = [
            'num_nodes', 'num_edges', 'avg_degree', 'max_degree',
            'density', 'avg_clustering', 'num_components',
            'cascade_depth', 'cascade_breadth', 'structural_virality',
            'avg_path_length', 'diameter', 'betweenness_centrality',
            'closeness_centrality', 'eigenvector_centrality'
        ]

    def extract_graph_features(self, G: nx.Graph) -> np.ndarray:
        """
        Extract statistical features from propagation graph.

        Args:
            G: NetworkX graph representing news propagation

        Returns:
            Feature vector with graph statistics
        """
        features = []

        # Basic statistics
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        features.extend([num_nodes, num_edges])

        if num_nodes == 0:
            return np.zeros(len(self.feature_names))

        # Degree statistics
        degrees = [d for n, d in G.degree()]
        avg_degree = np.mean(degrees) if degrees else 0
        max_degree = np.max(degrees) if degrees else 0
        features.extend([avg_degree, max_degree])

        # Density
        density = nx.density(G)
        features.append(density)

        # Clustering coefficient
        try:
            avg_clustering = nx.average_clustering(G)
        except:
            avg_clustering = 0
        features.append(avg_clustering)

        # Connected components
        num_components = nx.number_connected_components(G)
        features.append(num_components)

        # Cascade metrics
        cascade_depth = self._get_cascade_depth(G)
        cascade_breadth = self._get_cascade_breadth(G)
        structural_virality = self._get_structural_virality(G)
        features.extend([cascade_depth, cascade_breadth, structural_virality])

        # Path length and diameter
        try:
            if nx.is_connected(G):
                avg_path_length = nx.average_shortest_path_length(G)
                diameter = nx.diameter(G)
            else:
                largest_cc = max(nx.connected_components(G), key=len)
                subgraph = G.subgraph(largest_cc)
                avg_path_length = nx.average_shortest_path_length(subgraph)
                diameter = nx.diameter(subgraph)
        except:
            avg_path_length = 0
            diameter = 0
        features.extend([avg_path_length, diameter])

        # Centrality measures (average across all nodes)
        try:
            betweenness = np.mean(list(nx.betweenness_centrality(G).values()))
            closeness = np.mean(list(nx.closeness_centrality(G).values()))
            eigenvector = np.mean(list(nx.eigenvector_centrality(G, max_iter=1000).values()))
        except:
            betweenness = closeness = eigenvector = 0
        features.extend([betweenness, closeness, eigenvector])

        return np.array(features, dtype=np.float32)

    def _get_cascade_depth(self, G: nx.Graph) -> int:
        """Calculate maximum depth of propagation cascade."""
        if G.number_of_nodes() == 0:
            return 0
        try:
            # Assuming node 0 is the source
            if 0 in G.nodes():
                lengths = nx.single_source_shortest_path_length(G, 0)
                return max(lengths.values()) if lengths else 0
        except:
            pass
        return 0

    def _get_cascade_breadth(self, G: nx.Graph) -> float:
        """Calculate average breadth of propagation cascade."""
        if G.number_of_nodes() <= 1:
            return 0
        try:
            if 0 in G.nodes():
                lengths = nx.single_source_shortest_path_length(G, 0)
                depth_counts = {}
                for node, depth in lengths.items():
                    depth_counts[depth] = depth_counts.get(depth, 0) + 1
                return np.mean(list(depth_counts.values()))
        except:
            pass
        return 0

    def _get_structural_virality(self, G: nx.Graph) -> float:
        """
        Calculate structural virality (Goel et al., 2016).
        Measures the branching structure of propagation.
        """
        if G.number_of_nodes() <= 1:
            return 0
        try:
            # Calculate Wiener index (sum of all shortest paths)
            if nx.is_connected(G):
                n = G.number_of_nodes()
                wiener_index = nx.wiener_index(G)
                # Normalize by number of nodes
                return wiener_index / (n * (n - 1) / 2) if n > 1 else 0
        except:
            pass
        return 0


class GraphNeuralNetwork(nn.Module):
    """
    Graph Neural Network for learning propagation patterns.

    Uses Graph Convolutional Networks to learn from network structure.
    """

    def __init__(
        self,
        node_feature_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        graph_type: str = 'gcn'
    ):
        """
        Initialize GNN.

        Args:
            node_feature_dim: Dimension of node features
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            num_layers: Number of GNN layers
            dropout: Dropout probability
            graph_type: Type of graph convolution ('gcn', 'gat', 'sage')
        """
        super(GraphNeuralNetwork, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.graph_type = graph_type

        # Graph convolution layers
        self.convs = nn.ModuleList()

        # First layer
        if graph_type == 'gcn':
            self.convs.append(GCNConv(node_feature_dim, hidden_dim))
        elif graph_type == 'gat':
            self.convs.append(GATConv(node_feature_dim, hidden_dim, heads=4, concat=False))
        elif graph_type == 'sage':
            self.convs.append(SAGEConv(node_feature_dim, hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            if graph_type == 'gcn':
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif graph_type == 'gat':
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
            elif graph_type == 'sage':
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        # Output layer
        if num_layers > 1:
            if graph_type == 'gcn':
                self.convs.append(GCNConv(hidden_dim, output_dim))
            elif graph_type == 'gat':
                self.convs.append(GATConv(hidden_dim, output_dim, heads=4, concat=False))
            elif graph_type == 'sage':
                self.convs.append(SAGEConv(hidden_dim, output_dim))

        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)])
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GNN.

        Args:
            x: Node features [num_nodes, node_feature_dim]
            edge_index: Edge connectivity [2, num_edges]

        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout_layer(x)

        x = self.convs[-1](x, edge_index)
        return x

    def get_graph_embedding(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get graph-level embedding by aggregating node embeddings.

        Args:
            x: Node features
            edge_index: Edge connectivity
            batch: Batch assignment for each node (for batched graphs)

        Returns:
            Graph-level embedding
        """
        node_embeddings = self.forward(x, edge_index)

        # Global pooling
        if batch is None:
            # Single graph - use mean pooling
            graph_embedding = torch.mean(node_embeddings, dim=0, keepdim=True)
        else:
            # Batched graphs - use scatter mean
            from torch_geometric.nn import global_mean_pool
            graph_embedding = global_mean_pool(node_embeddings, batch)

        return graph_embedding


def create_propagation_graph(
    user_ids: List[int],
    timestamps: List[float],
    retweet_edges: List[Tuple[int, int]]
) -> nx.DiGraph:
    """
    Create directed graph from propagation data.

    Args:
        user_ids: List of user IDs involved in propagation
        timestamps: Timestamps of each propagation event
        retweet_edges: List of (source, target) edges representing retweets

    Returns:
        NetworkX directed graph
    """
    G = nx.DiGraph()

    # Add nodes with timestamps
    for user_id, timestamp in zip(user_ids, timestamps):
        G.add_node(user_id, timestamp=timestamp)

    # Add edges
    G.add_edges_from(retweet_edges)

    return G


def convert_to_pyg_data(
    G: nx.Graph,
    node_features: Optional[np.ndarray] = None
) -> Data:
    """
    Convert NetworkX graph to PyTorch Geometric Data object.

    Args:
        G: NetworkX graph
        node_features: Optional node feature matrix

    Returns:
        PyTorch Geometric Data object
    """
    # Get edge index
    edge_list = list(G.edges())
    if len(edge_list) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # Node features
    if node_features is None:
        # Use degree as default feature
        num_nodes = G.number_of_nodes()
        degrees = np.array([G.degree(i) if i in G.nodes() else 0 for i in range(num_nodes)])
        x = torch.tensor(degrees, dtype=torch.float32).unsqueeze(1)
    else:
        x = torch.tensor(node_features, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index)
