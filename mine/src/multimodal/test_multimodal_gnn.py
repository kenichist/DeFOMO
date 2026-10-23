"""
Standalone Test Script for Multimodal Misinformation Detection

Loads a trained model checkpoint and evaluates on test.tsv.
Uses the same CLI flags as train_multimodal_gnn.py to select modalities.

Usage:
    python test_multimodal_gnn.py --text --image --graph --cred   # All 4
    python test_multimodal_gnn.py --text --image                  # Text + Image
    python test_multimodal_gnn.py --image                         # Image only
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

from transformers import (
    DistilBertTokenizer,
    DistilBertModel,
    ViTModel,
    AutoImageProcessor,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATConv, Linear

from typing import Dict

# Import model and utilities from training script
from train_multimodal_gnn import (
    Config,
    MultiModalGNNModel,
    FakedditMultimodalGNNDataset,
    FakedditGraphBuilder,
    HeteroGNNEncoder,
    extract_gnn_embeddings,
    load_domain_weights,
    eval_model,
)


def main():
    parser = argparse.ArgumentParser(description="Test Multimodal Misinformation Detection Model")
    parser.add_argument("--text", action="store_true", help="Enable text modality")
    parser.add_argument("--image", action="store_true", help="Enable image modality")
    parser.add_argument("--graph", action="store_true", help="Enable graph modality")
    parser.add_argument("--cred", action="store_true", help="Enable credibility modality")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to model checkpoint (auto-detected from modalities if not given)")
    parser.add_argument("--test-tsv", type=str, default=os.path.join("data", "test.tsv"),
                        help="Path to test TSV file (default: data/test.tsv)")
    args = parser.parse_args()

    # Override modality flags if any provided
    if any([args.text, args.image, args.graph, args.cred]):
        Config.USE_TEXT = args.text
        Config.USE_IMAGE = args.image
        Config.USE_GRAPH = args.graph
        Config.USE_CREDIBILITY = args.cred

    Config.validate()

    # Determine model path
    modality_str = Config.get_modality_string()
    model_path = args.model_path or f"best_model_{modality_str}.pt"

    if not os.path.exists(model_path):
        print(f"ERROR: Model checkpoint not found: {model_path}")
        print(f"Train first: python train_multimodal_gnn.py --{' --'.join(modality_str.split('_'))}")
        return

    print("=" * 60)
    print("MULTIMODAL MISINFORMATION DETECTION - TEST")
    print("=" * 60)
    print(f"Device: {Config.DEVICE}")
    print(f"Model:  {model_path}")
    print(f"Test:   {args.test_tsv}")
    print(f"\nEnabled Modalities:")
    print(f"  Text (DistilBERT):     {'YES' if Config.USE_TEXT else 'NO'}")
    print(f"  Image (ViT):           {'YES' if Config.USE_IMAGE else 'NO'}")
    print(f"  Graph (GNN):           {'YES' if Config.USE_GRAPH else 'NO'}")
    print(f"  Credibility Weighting: {'YES' if Config.USE_CREDIBILITY else 'NO'}")
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

    # Load domain weights
    if Config.USE_CREDIBILITY:
        domain_weights, mean_weight = load_domain_weights(Config.DOMAIN_WEIGHTS_JSON)

    # Load tokenizer
    if Config.USE_TEXT or Config.USE_GRAPH:
        print("\nLoading tokenizer...")
        tokenizer = DistilBertTokenizer.from_pretrained(Config.TEXT_MODEL_NAME)

    # Load image processor
    if Config.USE_IMAGE:
        print("Loading image processor...")
        image_processor = AutoImageProcessor.from_pretrained(Config.VIT_MODEL_NAME, use_fast=True)

    # Load test data
    print("\nLoading test data...")
    df = pd.read_csv(args.test_tsv, sep="\t")
    df = df.dropna(subset=["2_way_label", "clean_title"])
    df["id"] = df["id"].astype(str)
    df["2_way_label"] = df["2_way_label"].astype(int)
    print(f"Test samples: {len(df)}")

    # Build graph
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
        post_to_idx = {str(row['id']): idx for idx, row in df.iterrows()}

    # Load text encoder
    if Config.USE_TEXT or Config.USE_GRAPH:
        print("\nLoading DistilBERT...")
        text_model = DistilBertModel.from_pretrained(Config.TEXT_MODEL_NAME)
        for p in text_model.parameters():
            p.requires_grad = False
        text_model.to(Config.DEVICE)

    # Load image encoder
    if Config.USE_IMAGE:
        print("Loading ViT...")
        vit_model = ViTModel.from_pretrained(Config.VIT_MODEL_NAME)
        for p in vit_model.parameters():
            p.requires_grad = False
        vit_model.to(Config.DEVICE)

    # Extract GNN embeddings
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
            cache_path=None,  # Don't cache test embeddings
        )
        print(f"GNN embeddings shape: {gnn_embeddings.shape}")

    # Create test dataset and loader
    test_dataset = FakedditMultimodalGNNDataset(
        df, tokenizer, image_processor, Config.IMAGE_ROOT,
        Config.MAX_LEN, post_to_idx, domain_weights, mean_weight
    )
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)

    # Initialize model
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

    # Load trained weights
    print(f"Loading checkpoint: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))

    # Evaluate
    print("\n" + "=" * 60)
    print("Running evaluation on test set...")
    print("=" * 60)

    test_acc, test_loss, test_preds, test_labels = eval_model(
        model, test_loader, gnn_embeddings, Config.DEVICE, Config
    )

    # Results
    modalities_desc = []
    if Config.USE_TEXT:
        modalities_desc.append("Text")
    if Config.USE_IMAGE:
        modalities_desc.append("Image")
    if Config.USE_GRAPH:
        modalities_desc.append("Graph")
    if Config.USE_CREDIBILITY:
        modalities_desc.append("Credibility")

    print(f"\nModalities: {' + '.join(modalities_desc)}")
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=["Fake", "Real"], digits=4))

    print("Confusion Matrix:")
    cm = confusion_matrix(test_labels, test_preds)
    print(f"  {'':>10} Pred Fake  Pred Real")
    print(f"  {'Actual Fake':>10}    {cm[0][0]:>5}      {cm[0][1]:>5}")
    print(f"  {'Actual Real':>10}    {cm[1][0]:>5}      {cm[1][1]:>5}")


if __name__ == "__main__":
    main()
