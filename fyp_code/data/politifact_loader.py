"""
Data loader for PolitiFact dataset.
Handles loading, preprocessing, and batching of news articles with propagation data.
"""

import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import networkx as nx
from pathlib import Path
import pickle


class PolitiFactDataset(Dataset):
    """
    Dataset class for PolitiFact fake news data.

    Expected data format:
    - CSV or JSON file with columns: id, text, label, [optional: user_ids, timestamps, edges]
    - Label: 0 for real news, 1 for fake news
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512,
        include_graph: bool = True,
        graph_feature_extractor=None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize dataset.

        Args:
            data_path: Path to data file (CSV or JSON)
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
            include_graph: Whether to include graph features
            graph_feature_extractor: GraphFeatureExtractor instance
            cache_dir: Directory to cache preprocessed data
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_graph = include_graph
        self.graph_feature_extractor = graph_feature_extractor
        self.cache_dir = cache_dir

        # Load data
        self.data = self._load_data()

        # Cache preprocessed features if cache_dir is provided
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self._setup_cache()

    def _load_data(self) -> pd.DataFrame:
        """Load data from file."""
        if self.data_path.endswith('.csv'):
            df = pd.read_csv(self.data_path)
        elif self.data_path.endswith('.json'):
            df = pd.read_json(self.data_path)
        elif self.data_path.endswith('.jsonl'):
            df = pd.read_json(self.data_path, lines=True)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path}")

        # Ensure required columns exist
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError("Data must contain 'text' and 'label' columns")

        # Convert labels to binary (0: real, 1: fake)
        if df['label'].dtype == 'object':
            label_map = {'real': 0, 'fake': 1, 'true': 0, 'false': 1}
            df['label'] = df['label'].str.lower().map(label_map)

        return df

    def _setup_cache(self):
        """Setup caching for preprocessed features."""
        cache_file = os.path.join(self.cache_dir, 'preprocessed_features.pkl')
        if os.path.exists(cache_file):
            print(f"Loading cached features from {cache_file}")
            with open(cache_file, 'rb') as f:
                self.cached_features = pickle.load(f)
        else:
            self.cached_features = {}

    def _save_cache(self):
        """Save cached features."""
        if self.cache_dir:
            cache_file = os.path.join(self.cache_dir, 'preprocessed_features.pkl')
            with open(cache_file, 'wb') as f:
                pickle.dump(self.cached_features, f)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from dataset.

        Returns:
            Dictionary with text features, graph features, and label
        """
        row = self.data.iloc[idx]

        # Text encoding
        text = str(row['text'])
        encoded = self.tokenizer.encode_text(text)

        item = {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'label': torch.tensor(row['label'], dtype=torch.long)
        }

        if 'token_type_ids' in encoded:
            item['token_type_ids'] = encoded['token_type_ids'].squeeze(0)

        # Graph features (if available)
        has_graph_data = False
        if self.include_graph and 'graph_data' in row:
            graph_data = row['graph_data']
            # Check if graph_data is not None and not NaN
            if graph_data is not None and not (isinstance(graph_data, float) and pd.isna(graph_data)):
                has_graph_data = True
                graph_stats = self._extract_graph_features(graph_data, idx)
                item['graph_stats'] = torch.tensor(graph_stats, dtype=torch.float32)

        if not has_graph_data:
            # Use zero features if no graph data
            if self.graph_feature_extractor:
                num_features = len(self.graph_feature_extractor.feature_names)
            else:
                num_features = 15  # Default number of graph features
            item['graph_stats'] = torch.zeros(num_features, dtype=torch.float32)

        return item

    def _extract_graph_features(self, graph_data: Dict, idx: int) -> np.ndarray:
        """Extract graph features from propagation data."""
        if hasattr(self, 'cached_features') and idx in self.cached_features:
            return self.cached_features[idx]

        # Build graph from data
        G = self._build_graph(graph_data)

        # Extract features
        if self.graph_feature_extractor:
            features = self.graph_feature_extractor.extract_graph_features(G)
        else:
            # Use basic features if no extractor provided
            features = np.array([
                G.number_of_nodes(),
                G.number_of_edges(),
                np.mean([d for n, d in G.degree()]) if G.number_of_nodes() > 0 else 0,
            ] + [0] * 12, dtype=np.float32)

        # Cache features
        if hasattr(self, 'cached_features'):
            self.cached_features[idx] = features

        return features

    def _build_graph(self, graph_data: Dict) -> nx.Graph:
        """Build NetworkX graph from graph data."""
        G = nx.Graph()

        if 'nodes' in graph_data:
            G.add_nodes_from(graph_data['nodes'])

        if 'edges' in graph_data:
            G.add_edges_from(graph_data['edges'])

        return G


class PolitiFactDataModule:
    """
    Data module for managing train/val/test splits and dataloaders.
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer,
        batch_size: int = 16,
        max_length: int = 512,
        num_workers: int = 4,
        include_graph: bool = True,
        train_split: float = 0.8,
        val_split: float = 0.1,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize data module.

        Args:
            data_dir: Directory containing data files
            tokenizer: BERT tokenizer
            batch_size: Batch size for dataloaders
            max_length: Maximum sequence length
            num_workers: Number of workers for data loading
            include_graph: Whether to include graph features
            train_split: Proportion of data for training
            val_split: Proportion of data for validation
            cache_dir: Directory to cache preprocessed data
        """
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.include_graph = include_graph
        self.train_split = train_split
        self.val_split = val_split
        self.cache_dir = cache_dir

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for different stages.

        Args:
            stage: 'fit' for train/val, 'test' for test, None for all
        """
        try:
            from models.graph_network import GraphFeatureExtractor
        except ImportError:
            from ..models.graph_network import GraphFeatureExtractor
        graph_extractor = GraphFeatureExtractor() if self.include_graph else None

        # Check for pre-split data files
        train_file = os.path.join(self.data_dir, 'train.csv')
        val_file = os.path.join(self.data_dir, 'val.csv')
        test_file = os.path.join(self.data_dir, 'test.csv')

        if os.path.exists(train_file) and os.path.exists(val_file):
            # Use pre-split files
            if stage == 'fit' or stage is None:
                self.train_dataset = PolitiFactDataset(
                    train_file,
                    self.tokenizer,
                    self.max_length,
                    self.include_graph,
                    graph_extractor,
                    os.path.join(self.cache_dir, 'train') if self.cache_dir else None
                )
                self.val_dataset = PolitiFactDataset(
                    val_file,
                    self.tokenizer,
                    self.max_length,
                    self.include_graph,
                    graph_extractor,
                    os.path.join(self.cache_dir, 'val') if self.cache_dir else None
                )

            if (stage == 'test' or stage is None) and os.path.exists(test_file):
                self.test_dataset = PolitiFactDataset(
                    test_file,
                    self.tokenizer,
                    self.max_length,
                    self.include_graph,
                    graph_extractor,
                    os.path.join(self.cache_dir, 'test') if self.cache_dir else None
                )
        else:
            # Load full dataset and split
            data_file = os.path.join(self.data_dir, 'politifact.csv')
            if not os.path.exists(data_file):
                # Try alternative names
                for name in ['data.csv', 'full_data.csv', 'politifact.json']:
                    alt_file = os.path.join(self.data_dir, name)
                    if os.path.exists(alt_file):
                        data_file = alt_file
                        break

            full_dataset = PolitiFactDataset(
                data_file,
                self.tokenizer,
                self.max_length,
                self.include_graph,
                graph_extractor,
                self.cache_dir
            )

            # Split dataset
            total_size = len(full_dataset)
            train_size = int(total_size * self.train_split)
            val_size = int(total_size * self.val_split)
            test_size = total_size - train_size - val_size

            self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
                full_dataset,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42)
            )

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


def create_sample_data(output_dir: str, num_samples: int = 100):
    """
    Create sample PolitiFact data for testing.

    Args:
        output_dir: Directory to save sample data
        num_samples: Number of samples to generate
    """
    os.makedirs(output_dir, exist_ok=True)

    # Sample fake and real news patterns
    fake_patterns = [
        "BREAKING: Shocking truth revealed about",
        "You won't believe what happened when",
        "Scientists HATE this one weird trick",
        "EXPOSED: The real reason behind",
        "URGENT: This will change everything about"
    ]

    real_patterns = [
        "According to recent studies,",
        "Researchers at the university found that",
        "The government announced today that",
        "Data shows an increase in",
        "Officials confirmed that"
    ]

    data = []
    for i in range(num_samples):
        is_fake = i % 2 == 0  # Alternate fake and real
        if is_fake:
            text = np.random.choice(fake_patterns) + f" topic {i}. " + "More sensational claims here. " * 10
            label = 1
        else:
            text = np.random.choice(real_patterns) + f" topic {i}. " + "More factual information here. " * 10
            label = 0

        data.append({
            'id': i,
            'text': text,
            'label': label,
            'graph_data': None  # Can add mock graph data if needed
        })

    # Create DataFrame
    df = pd.DataFrame(data)

    # Split into train/val/test
    train_size = int(0.8 * num_samples)
    val_size = int(0.1 * num_samples)

    train_df = df[:train_size]
    val_df = df[train_size:train_size + val_size]
    test_df = df[train_size + val_size:]

    # Save splits
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

    print(f"Created sample data in {output_dir}")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
