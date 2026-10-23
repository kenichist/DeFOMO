"""
Standalone test script for the trainable Multimodal + GNN Fakeddit model.
"""

import os
import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from train_multimodal_gnn import (
    Config,
    load_domain_weights,
    extract_post_text_features,
    FakedditGraphBuilder,
    FakedditMultimodalGNNDataset,
    MultiModalGNNModel,
    eval_model,
)

from transformers import (
    DistilBertTokenizer,
    DistilBertModel,
    ViTModel,
    AutoImageProcessor,
)


def main():
    parser = argparse.ArgumentParser(description="Test Multimodal + GNN Model")
    parser.add_argument("--text", action="store_true", help="Enable text modality")
    parser.add_argument("--image", action="store_true", help="Enable image modality")
    parser.add_argument("--graph", action="store_true", help="Enable graph modality")
    parser.add_argument("--cred", action="store_true", help="Enable credibility modality")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to model checkpoint (auto-detected from modalities if not given)")
    parser.add_argument("--test-tsv", type=str, default=os.path.join("data", "test.tsv"),
                        help="Path to test TSV")
    args = parser.parse_args()

    if any([args.text, args.image, args.graph, args.cred]):
        Config.USE_TEXT = args.text
        Config.USE_IMAGE = args.image
        Config.USE_GRAPH = args.graph
        Config.USE_CREDIBILITY = args.cred

    Config.validate()

    modality_str = Config.get_modality_string()
    model_path = args.model_path or f"best_model_{modality_str}.pt"

    if not os.path.exists(model_path):
        print(f"ERROR: checkpoint not found: {model_path}")
        return

    print("=" * 60)
    print("MULTIMODAL MISINFORMATION DETECTION - TEST")
    print("=" * 60)
    print(f"Device: {Config.DEVICE}")
    print(f"Model:  {model_path}")
    print(f"Test:   {args.test_tsv}")
    print(f"Text:   {Config.USE_TEXT}")
    print(f"Image:  {Config.USE_IMAGE}")
    print(f"Graph:  {Config.USE_GRAPH}")
    print(f"Cred:   {Config.USE_CREDIBILITY}")
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

    if Config.USE_IMAGE:
        print("\nLoading image processor...")
        image_processor = AutoImageProcessor.from_pretrained(Config.VIT_MODEL_NAME, use_fast=True)

        print("Loading ViT...")
        vit_model = ViTModel.from_pretrained(Config.VIT_MODEL_NAME)
        for p in vit_model.parameters():
            p.requires_grad = False
        vit_model.to(Config.DEVICE)

    print("\nLoading test data...")
    df = pd.read_csv(args.test_tsv, sep="\t")
    df = df.dropna(subset=["2_way_label", "clean_title"]).copy()
    df["id"] = df["id"].astype(str)
    df["2_way_label"] = df["2_way_label"].astype(int)
    print(f"Test samples: {len(df)}")

    graph_data = None
    post_to_idx = {str(row['id']): idx for idx, row in df.iterrows()}

    if Config.USE_GRAPH:
        print("\nPreparing TEST graph...")
        post_features = extract_post_text_features(
            df, tokenizer, text_model, Config.DEVICE, batch_size=32, max_len=Config.MAX_LEN
        )
        builder = FakedditGraphBuilder(
            post_feature_dim=Config.POST_FEATURE_DIM,
            user_embed_dim=Config.USER_EMBED_DIM,
            subreddit_embed_dim=Config.SUBREDDIT_EMBED_DIM,
            domain_embed_dim=Config.DOMAIN_EMBED_DIM,
            seed=Config.SEED,
        )
        graph_data = builder.build_graph(df, post_features)
        graph_data = graph_data.to(Config.DEVICE)
        post_to_idx = builder.post_to_idx

    test_dataset = FakedditMultimodalGNNDataset(
        df,
        tokenizer,
        image_processor,
        Config.IMAGE_ROOT,
        Config.MAX_LEN,
        post_to_idx,
        domain_weights,
        mean_weight,
    )
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)

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

    print(f"Loading checkpoint: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))

    print("\n" + "=" * 60)
    print("Running evaluation on test set...")
    print("=" * 60)

    test_acc, test_loss, test_preds, test_labels = eval_model(
        model, test_loader, Config.DEVICE, Config, graph_data=graph_data
    )

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
    print(classification_report(test_labels, test_preds, target_names=["Fake", "Real"], digits=4, zero_division=0))

    print("Confusion Matrix:")
    cm = confusion_matrix(test_labels, test_preds)
    print(f"  {'':>10} Pred Fake  Pred Real")
    print(f"  {'Actual Fake':>10}    {cm[0][0]:>5}      {cm[0][1]:>5}")
    print(f"  {'Actual Real':>10}    {cm[1][0]:>5}      {cm[1][1]:>5}")


if __name__ == "__main__":
    main()