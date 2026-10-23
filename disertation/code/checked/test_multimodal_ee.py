"""
Standalone test script for the Multimodal + GNN CHECKED model.
Matches the training code where:
- Text branch: the configured Chinese transformer is trainable end-to-end
- Image branch: ViT is trainable end-to-end
- Graph branch: uses a separate frozen transformer for static post-node features
"""

import os
import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from transformers import (
    AutoTokenizer,
    AutoModel,
    ViTModel,
    AutoImageProcessor,
)

# Assumes the training module is importable
from train_multimodal_ee import (
    Config,
    load_domain_weights,
    extract_post_text_features,
    FakedditGraphBuilder,
    FakedditMultimodalGNNDataset,
    MultiModalGNNModel,
    eval_model,
    hf_load_kwargs,
    load_hf_model,
    normalize_checked_dataframe,
    read_checked_table,
)
from gnn_module import GRAPH_VARIANTS, ResponseGraphConfig

SAFE_METADATA_COLUMNS = ["text_length", "media_count", "has_image", "has_video"]
UPPER_BOUND_METADATA_COLUMNS = [
    "text_length",
    "media_count",
    "has_image",
    "has_video",
    "has_comments",
    "has_reposts",
    "repost_count",
    "comment_count",
    "like_count",
    "num_comments",
    "num_reposts",
]


def resolve_checkpoint_path(modality_str: str, gpus: int, user_path: str = None) -> str:
    """Auto-detect the best checkpoint matching the modality string."""
    if user_path is not None:
        return user_path

    candidates = [
        f"best_model_{modality_str}_{gpus}gpu.pt.best",
        f"best_model_{modality_str}_{gpus}gpu.pt",
        f"best_model_{modality_str}.pt.best",
        f"best_model_{modality_str}.pt",
    ]
    folder = "td_models"
    folder_paths = [os.path.join(folder, c) for c in candidates]
    for path in folder_paths + candidates:
        if os.path.exists(path):
            return path
    return folder_paths[0]


def main():
    parser = argparse.ArgumentParser(description="Test CHECKED Multimodal + GNN End-to-End Model")
    parser.add_argument("--text", action="store_true", help="Enable text modality")
    parser.add_argument("--image", action="store_true", help="Enable image modality")
    parser.add_argument("--graph", action="store_true", help="Enable graph modality")
    parser.add_argument("--prop", action="store_true", help="Enable tweet-ID temporal propagation features")
    parser.add_argument("--cred", action="store_true", help="Enable credibility modality (learnable alpha)")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model checkpoint. If omitted, auto-detects.",
    )
    parser.add_argument(
        "--test-csv",
        "--test-tsv",
        dest="test_csv",
        type=str,
        default=os.path.join("CHECKED", "data_checked", "processed", "test.csv"),
        help="Path to CHECKED test CSV",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=5,
        help="GPU count used during training (for checkpoint auto-detection only)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size for testing",
    )
    parser.add_argument("--text-model", default=Config.TEXT_MODEL_NAME)
    parser.add_argument("--vit-model", default=Config.VIT_MODEL_NAME)
    parser.add_argument("--max-length", type=int, default=Config.MAX_LEN)
    parser.add_argument("--use_social_graph", action="store_true")
    parser.add_argument("--disable_user_graph", action="store_true", help="Keep graph modality but remove user nodes/edges.")
    parser.add_argument("--enable_user_graph", action="store_true")
    parser.add_argument("--graph_variant", choices=sorted(GRAPH_VARIANTS), default="media_only")
    parser.add_argument("--use_response_graph", action="store_true")
    parser.add_argument("--comment-csv", default="")
    parser.add_argument("--repost-csv", default="")
    parser.add_argument("--response-dataset-root", default="CHECKED/dataset")
    parser.add_argument("--response-window-hours", type=float, default=24.0)
    parser.add_argument("--include_late_responses", action="store_true")
    parser.add_argument("--max-comments-per-post", type=int, default=8)
    parser.add_argument("--max-reposts-per-post", type=int, default=8)
    parser.add_argument("--fair_mode", action="store_true", help="Disable user identity and credibility at evaluation.")
    parser.add_argument(
        "--metadata_mode",
        choices=["legacy", "safe", "upper_bound", "none"],
        default="legacy",
        help="Must match the metadata mode used to train the checkpoint.",
    )
    args = parser.parse_args()

    Config.TEXT_MODEL_NAME = args.text_model
    Config.VIT_MODEL_NAME = args.vit_model
    Config.MAX_LEN = args.max_length
    if args.fair_mode:
        args.disable_user_graph = True
        args.cred = False
        if args.metadata_mode == "legacy":
            args.metadata_mode = "safe"
    if args.metadata_mode == "safe":
        Config.PROPAGATION_FEATURE_COLUMNS = list(SAFE_METADATA_COLUMNS)
        args.cred = False
    elif args.metadata_mode == "upper_bound":
        Config.PROPAGATION_FEATURE_COLUMNS = list(UPPER_BOUND_METADATA_COLUMNS)
    elif args.metadata_mode == "none":
        args.prop = False
        args.cred = False
    Config.PROPAGATION_FEATURE_DIM = len(Config.PROPAGATION_FEATURE_COLUMNS)
    Config.USE_SOCIAL_GRAPH = bool(args.use_social_graph)
    Config.USE_USER_GRAPH = bool(args.enable_user_graph) and not bool(args.disable_user_graph)
    if args.fair_mode:
        Config.USE_USER_GRAPH = False
    Config.GRAPH_VARIANT = args.graph_variant
    Config.METADATA_MODE = args.metadata_mode
    Config.USE_RESPONSE_GRAPH = bool(args.use_response_graph)

    # If any modality flags are given, use them; otherwise keep Config defaults.
    if any([args.text, args.image, args.graph, args.prop, args.cred]):
        Config.USE_TEXT = args.text
        Config.USE_IMAGE = args.image
        Config.USE_GRAPH = args.graph
        Config.USE_PROPAGATION = args.prop
        Config.USE_CREDIBILITY = args.cred

    Config.validate()

    if torch.cuda.is_available():
        Config.DEVICE = torch.device("cuda")
    else:
        Config.DEVICE = torch.device("cpu")

    modality_str = Config.get_modality_string()
    model_path = resolve_checkpoint_path(modality_str, args.gpus, args.model_path)

    if not os.path.exists(model_path):
        print(f"ERROR: checkpoint not found: {model_path}")
        return

    print("=" * 60)
    print("MULTIMODAL MISINFORMATION DETECTION - TEST (End-to-End ViT + text encoder)")
    print("=" * 60)
    print(f"Device: {Config.DEVICE}")
    print(f"Model:  {model_path}")
    print(f"Test:   {args.test_csv}")
    print(f"Text:   {Config.USE_TEXT} (trainable text encoder: {Config.TEXT_MODEL_NAME})")
    print(f"Image:  {Config.USE_IMAGE} (trainable ViT end-to-end)")
    graph_uses_post_text = args.graph_variant not in {"metadata_only", "response_metadata"}
    print(
        f"Graph:  {Config.USE_GRAPH} (variant={args.graph_variant}, "
        f"frozen post text={graph_uses_post_text})"
    )
    print(f"Propagation: {Config.USE_PROPAGATION} (tweet-ID temporal features)")
    print(f"Cred:   {Config.USE_CREDIBILITY}")
    print("=" * 60)

    tokenizer = None
    image_processor = None
    graph_text_model = None      # frozen text encoder for graph post features
    trainable_text_model = None  # text branch (weights loaded from checkpoint)
    vit_model = None             # image branch (weights loaded from checkpoint)

    domain_weights = {}
    mean_weight = 0.5

    if Config.USE_CREDIBILITY:
        domain_weights, mean_weight = load_domain_weights(Config.DOMAIN_WEIGHTS_JSON)

    # Load tokenizer (shared between text branch and graph post feature extraction)
    if Config.USE_TEXT or Config.USE_GRAPH:
        print("\nLoading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(Config.TEXT_MODEL_NAME, **hf_load_kwargs(Config.TEXT_MODEL_NAME))

    # Frozen text encoder for extracting static graph post-node features
    if Config.USE_GRAPH:
        print(f"Loading frozen text encoder for graph features: {Config.TEXT_MODEL_NAME}")
        graph_text_model = load_hf_model(AutoModel, Config.TEXT_MODEL_NAME)
        Config.POST_FEATURE_DIM = int(graph_text_model.config.hidden_size)
        for p in graph_text_model.parameters():
            p.requires_grad = False
        graph_text_model = graph_text_model.to(Config.DEVICE)
        graph_text_model.eval()

    # Trainable text branch: create a fresh model; weights will be overwritten by checkpoint
    if Config.USE_TEXT:
        print(f"Loading text branch encoder (will be overwritten by checkpoint): {Config.TEXT_MODEL_NAME}")
        trainable_text_model = load_hf_model(AutoModel, Config.TEXT_MODEL_NAME)
        # No freezing; the checkpoint supplies the trained weights
        trainable_text_model = trainable_text_model.to(Config.DEVICE)
        # We will call model.eval() later; no need to set requires_grad manually here

    # Trainable image branch: create a fresh ViT, load checkpoint weights later
    if Config.USE_IMAGE:
        print("\nLoading image processor...")
        image_processor = AutoImageProcessor.from_pretrained(
            Config.VIT_MODEL_NAME,
            use_fast=True,
            **hf_load_kwargs(Config.VIT_MODEL_NAME),
        )

        print("Loading ViT (end-to-end trained, will be overwritten by checkpoint)...")
        vit_model = load_hf_model(ViTModel, Config.VIT_MODEL_NAME)
        # No freezing; the checkpoint contains the fully trained ViT
        vit_model = vit_model.to(Config.DEVICE)
        # model.eval() later, after loading state dict

    # ------- Test data -------
    print("\nLoading test data...")
    df = normalize_checked_dataframe(read_checked_table(args.test_csv))
    df = df.dropna(subset=["2_way_label", "clean_title"]).copy()
    df["id"] = df["id"].astype(str)
    df["2_way_label"] = df["2_way_label"].astype(int)
    print(f"Test samples: {len(df)}")

    graph_data = None
    post_to_idx = {str(row['id']): idx for idx, row in df.iterrows()}

    if Config.USE_GRAPH:
        print("\nPreparing TEST graph...")
        post_features = extract_post_text_features(
            df=df,
            tokenizer=tokenizer,
            text_model=graph_text_model,
            device=Config.DEVICE,
            batch_size=32,
            max_len=Config.MAX_LEN,
        )
        builder = FakedditGraphBuilder(
            post_feature_dim=int(graph_text_model.config.hidden_size),
            user_embed_dim=Config.USER_EMBED_DIM,
            subreddit_embed_dim=Config.SUBREDDIT_EMBED_DIM,
            domain_embed_dim=Config.DOMAIN_EMBED_DIM,
            seed=Config.SEED,
            social_graph_root=Config.SOCIAL_GRAPH_ROOT,
            use_social_graph=Config.USE_SOCIAL_GRAPH,
            use_user_graph=Config.USE_USER_GRAPH,
            graph_variant=args.graph_variant,
            metadata_mode=args.metadata_mode,
            response_config=ResponseGraphConfig(
                enabled=bool(args.use_response_graph),
                dataset_root=args.response_dataset_root,
                comment_csv=args.comment_csv,
                repost_csv=args.repost_csv,
                response_window_hours=args.response_window_hours,
                include_late_responses=bool(args.include_late_responses),
                max_comments_per_post=args.max_comments_per_post,
                max_reposts_per_post=args.max_reposts_per_post,
                fair_mode=bool(args.fair_mode),
            ),
        )
        graph_data = builder.build_graph(df, post_features).to(Config.DEVICE)
        Config.POST_FEATURE_DIM = builder.post_feature_dim
        post_to_idx = builder.post_to_idx

    test_dataset = FakedditMultimodalGNNDataset(
        df=df,
        tokenizer=tokenizer,
        image_processor=image_processor,
        image_root=Config.IMAGE_ROOT,
        max_len=Config.MAX_LEN,
        post_to_idx=post_to_idx,
        domain_weights=domain_weights,
        default_weight=mean_weight,
    )

    if args.batch_size is not None:
        test_batch_size = args.batch_size
    else:
        # Reasonable defaults for inference, matching training's effective capacities
        if Config.USE_IMAGE:
            test_batch_size = 8
        elif Config.USE_TEXT and Config.USE_GRAPH:
            test_batch_size = 8
        elif Config.USE_TEXT:
            test_batch_size = 16
        elif Config.USE_GRAPH:
            test_batch_size = 128
        else:
            test_batch_size = 32

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # ------- Model initialisation -------
    print("\nInitializing model...")
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
    ).to(Config.DEVICE)

    print(f"Loading checkpoint: {model_path}")
    state_dict = torch.load(model_path, map_location=Config.DEVICE)
    model.load_state_dict(state_dict, strict=True)

    # After loading the trained weights, freeze everything for inference
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}  (all frozen for inference)")

    # ------- Evaluation -------
    print("\n" + "=" * 60)
    print("Running evaluation on test set...")
    print("=" * 60)

    with torch.no_grad():
        test_acc, test_loss, test_preds, test_labels = eval_model(
            model=model,
            data_loader=test_loader,
            device=Config.DEVICE,
            config=Config,
            graph_data=graph_data,
        )

    modalities_desc = []
    if Config.USE_TEXT:
        modalities_desc.append(f"Text (trainable {Config.TEXT_MODEL_NAME})")
    if Config.USE_IMAGE:
        modalities_desc.append("Image (trainable ViT)")
    if Config.USE_GRAPH:
        modalities_desc.append("Graph")
    if Config.USE_PROPAGATION:
        modalities_desc.append("Propagation")
    if Config.USE_CREDIBILITY:
        modalities_desc.append("Cred (Learnable alpha)")

    print(f"\nModalities: {' + '.join(modalities_desc)}")
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            test_labels,
            test_preds,
            labels=[0, 1],
            target_names=["Real", "Fake"],
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    cm = confusion_matrix(test_labels, test_preds, labels=[0, 1])
    print(f"  {'':>10} Pred Real  Pred Fake")
    print(f"  {'Actual Real':>10}    {cm[0][0]:>5}      {cm[0][1]:>5}")
    print(f"  {'Actual Fake':>10}    {cm[1][0]:>5}      {cm[1][1]:>5}")


if __name__ == "__main__":
    main()
