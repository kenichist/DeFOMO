import argparse
import csv
import datetime as dt
import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    ViTModel,
    get_linear_schedule_with_warmup,
)

from train_multimodal_ee import (
    Config,
    FakedditGraphBuilder,
    FakedditMultimodalGNNDataset,
    MultiModalGNNModel,
    extract_post_text_features,
    load_domain_weights,
)


ABLATIONS = {
    "text_only": {"text": True, "image": False, "graph": False, "prop": False, "cred": False},
    "image_only": {"text": False, "image": True, "graph": False, "prop": False, "cred": False},
    "graph_only": {"text": False, "image": False, "graph": True, "prop": False, "cred": False},
    "cred_only": {"text": False, "image": False, "graph": False, "prop": False, "cred": True},
    "text_image": {"text": True, "image": True, "graph": False, "prop": False, "cred": False},
    "text_graph": {"text": True, "image": False, "graph": True, "prop": False, "cred": False},
    "text_image_graph": {"text": True, "image": True, "graph": True, "prop": False, "cred": False},
    "full": {"text": True, "image": True, "graph": True, "prop": False, "cred": True},
}


METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "fake_precision",
    "fake_recall",
    "fake_f1",
    "real_precision",
    "real_recall",
    "real_f1",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Dict], fieldnames: Optional[List[str]] = None) -> None:
    rows = list(rows)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def detect_label_mapping(df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, str]]:
    if "label" not in df.columns or "2_way_label" not in df.columns:
        raise ValueError("Cannot detect labels: TSV must contain 'label' and '2_way_label' columns.")

    mapping: Dict[str, set] = {}
    for _, row in df[["label", "2_way_label"]].dropna().iterrows():
        name = str(row["label"]).strip().lower()
        idx = int(row["2_way_label"])
        mapping.setdefault(name, set()).add(idx)

    if "fake" not in mapping or "real" not in mapping:
        raise ValueError(f"Cannot detect fake/real label mapping from labels: {mapping}")
    if len(mapping["fake"]) != 1 or len(mapping["real"]) != 1:
        raise ValueError(f"Ambiguous label mapping: {mapping}")

    label_to_idx = {"fake": next(iter(mapping["fake"])), "real": next(iter(mapping["real"]))}
    if label_to_idx["fake"] == label_to_idx["real"]:
        raise ValueError(f"Invalid label mapping: {label_to_idx}")

    idx_to_label = {idx: name for name, idx in label_to_idx.items()}
    print(f"Detected label mapping: Real={label_to_idx['real']}, Fake={label_to_idx['fake']}")
    return label_to_idx, idx_to_label


def validate_propagation_columns(df: pd.DataFrame) -> None:
    missing = [column for column in Config.PROPAGATION_FEATURE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            "Propagation features requested, but TSV is missing columns "
            f"{missing}. Rebuild the dataset with FakeNewsNet\\build_training_data.cmd."
        )


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


class CredOnlyModel(nn.Module):
    """Trainable logistic baseline over the scalar domain credibility score."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.classifier = nn.Linear(1, num_classes)
        self.last_alpha = None

    def forward(self, credibility_score=None, labels=None, **kwargs):
        if credibility_score is None:
            raise ValueError("credibility_score is required for cred_only ablation")
        x = credibility_score.float().view(-1, 1)
        logits = self.classifier(x)
        return type("ModelOutput", (), {"logits": logits, "loss": None})


def class_counts(labels: Iterable[int]) -> Dict[int, int]:
    labels = list(map(int, labels))
    return {int(label): int(labels.count(label)) for label in sorted(set(labels))}


def compute_weights(labels: Iterable[int], device: torch.device) -> torch.Tensor:
    y = np.asarray(list(map(int, labels)))
    classes = np.asarray(sorted(np.unique(y)))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    full_weights = np.ones(2, dtype=np.float32)
    for cls, weight in zip(classes, weights):
        full_weights[int(cls)] = float(weight)
    return torch.tensor(full_weights, dtype=torch.float32, device=device)


def build_criterion(args, train_labels: Iterable[int], device: torch.device) -> nn.Module:
    class_weight_tensor = None
    if args.use_class_weights or args.focal_alpha == "auto":
        class_weight_tensor = compute_weights(train_labels, device)
        print(f"Computed class weights [class0, class1]: {class_weight_tensor.detach().cpu().tolist()}")

    if args.loss_type == "ce":
        return nn.CrossEntropyLoss(weight=class_weight_tensor if args.use_class_weights else None)

    if args.loss_type == "focal":
        alpha = None
        if args.focal_alpha == "auto":
            alpha = class_weight_tensor
        elif args.focal_alpha not in ("none", "", None):
            values = [float(value) for value in str(args.focal_alpha).split(",")]
            if len(values) != 2:
                raise ValueError("--focal_alpha must be 'auto', 'none', or two comma-separated class weights")
            alpha = torch.tensor(values, dtype=torch.float32, device=device)
        return FocalLoss(gamma=args.focal_gamma, alpha=alpha)

    raise ValueError(f"Unknown loss_type: {args.loss_type}")


def batch_to_kwargs(batch, modalities: Dict[str, bool], device: torch.device, graph_data=None):
    labels = batch["labels"].to(device)
    kwargs = {}
    if modalities["text"]:
        kwargs["input_ids"] = batch["input_ids"].to(device, non_blocking=True)
        kwargs["attention_mask"] = batch["attention_mask"].to(device, non_blocking=True)
    if modalities["image"]:
        kwargs["pixel_values"] = batch["pixel_values"].to(device, non_blocking=True)
        kwargs["has_image"] = batch["has_image"].to(device, non_blocking=True)
    if modalities["graph"]:
        kwargs["graph_idx"] = batch["graph_idx"].to(device, non_blocking=True)
        kwargs["graph_data"] = graph_data
    if modalities["prop"]:
        kwargs["propagation_features"] = batch["propagation_features"].to(device, non_blocking=True)
    if modalities["cred"]:
        kwargs["credibility_score"] = batch["credibility_score"].to(device, non_blocking=True)
    return kwargs, labels


def get_probs(model, loader, modalities, device, graph_data=None, criterion=None):
    model.eval()
    ids, labels, probs = [], [], []
    total_loss = 0.0
    total = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            kwargs, y = batch_to_kwargs(batch, modalities, device, graph_data)
            out = model(**kwargs)
            logits = out.logits
            if criterion is not None:
                loss = criterion(logits, y)
                total_loss += float(loss.item()) * y.size(0)
            total += y.size(0)
            probs.append(F.softmax(logits, dim=-1).detach().cpu().numpy())
            labels.extend(y.detach().cpu().tolist())
            if "id" in batch:
                ids.extend(batch["id"])

    prob_array = np.concatenate(probs, axis=0) if probs else np.empty((0, 2), dtype=np.float32)
    return ids, np.asarray(labels, dtype=np.int64), prob_array, total_loss / max(1, total)


def predict_with_threshold(probs: np.ndarray, threshold: float, target_class: int, other_class: int) -> np.ndarray:
    preds = np.full(probs.shape[0], other_class, dtype=np.int64)
    preds[probs[:, target_class] >= threshold] = target_class
    return preds


def metric_summary(y_true, y_pred, label_to_idx: Dict[str, int]) -> Dict[str, float]:
    fake_idx = label_to_idx["fake"]
    real_idx = label_to_idx["real"]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[real_idx, fake_idx],
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[0]),
        "macro_recall": float(precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[1]),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "real_precision": float(precision[0]),
        "real_recall": float(recall[0]),
        "real_f1": float(f1[0]),
        "fake_precision": float(precision[1]),
        "fake_recall": float(recall[1]),
        "fake_f1": float(f1[1]),
    }


def find_best_threshold(
    y_true,
    probs,
    target_class: int,
    other_class: int,
    label_to_idx: Dict[str, int],
    metric: str = "macro_f1",
    min_threshold: float = 0.05,
    max_threshold: float = 0.95,
    step: float = 0.01,
):
    rows = []
    thresholds = np.arange(min_threshold, max_threshold + (step / 2), step)
    for threshold in thresholds:
        preds = predict_with_threshold(probs, float(threshold), target_class, other_class)
        summary = metric_summary(y_true, preds, label_to_idx)
        rows.append({"threshold": round(float(threshold), 4), **summary})
    if metric not in rows[0]:
        raise ValueError(f"Unsupported threshold metric '{metric}'. Available: {sorted(rows[0].keys())}")
    best = max(rows, key=lambda row: row[metric])
    return best, rows


def write_evaluation_outputs(
    output_dir: Path,
    prefix: str,
    ids: List[str],
    y_true,
    probs,
    preds,
    label_to_idx: Dict[str, int],
    idx_to_label: Dict[int, str],
    loss: float,
):
    metrics = metric_summary(y_true, preds, label_to_idx)
    metrics["loss"] = float(loss)
    metrics["mode"] = prefix

    report = classification_report(
        y_true,
        preds,
        labels=[label_to_idx["real"], label_to_idx["fake"]],
        target_names=["Real", "Fake"],
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, preds, labels=[label_to_idx["real"], label_to_idx["fake"]])

    (output_dir / f"classification_report_{prefix}.txt").write_text(report, encoding="utf-8")
    write_csv(
        output_dir / f"confusion_matrix_{prefix}.csv",
        [
            {"actual": "Real", "pred_real": int(cm[0, 0]), "pred_fake": int(cm[0, 1])},
            {"actual": "Fake", "pred_real": int(cm[1, 0]), "pred_fake": int(cm[1, 1])},
        ],
        ["actual", "pred_real", "pred_fake"],
    )

    prediction_rows = []
    for i, true_label in enumerate(y_true):
        pred = int(preds[i])
        prediction_rows.append(
            {
                "sample_id": ids[i] if i < len(ids) else "",
                "true_label": idx_to_label.get(int(true_label), str(true_label)),
                "true_label_idx": int(true_label),
                "predicted_label": idx_to_label.get(pred, str(pred)),
                "predicted_label_idx": pred,
                "real_probability": float(probs[i, label_to_idx["real"]]),
                "fake_probability": float(probs[i, label_to_idx["fake"]]),
                "correct": int(pred == int(true_label)),
            }
        )
    write_csv(output_dir / f"predictions_{prefix}.csv", prediction_rows)
    return metrics, report, cm


class DatasetWithIds(FakedditMultimodalGNNDataset):
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item["id"] = str(self.df.iloc[idx]["id"])
        return item


def make_dataset(df, tokenizer, image_processor, post_to_idx, domain_weights, mean_weight):
    return DatasetWithIds(
        df,
        tokenizer,
        image_processor,
        Config.IMAGE_ROOT,
        Config.MAX_LEN,
        post_to_idx,
        domain_weights,
        mean_weight,
    )


def make_sampler(train_df: pd.DataFrame):
    labels = train_df["2_way_label"].astype(int).values
    counts = class_counts(labels)
    weights_by_class = {label: 1.0 / count for label, count in counts.items()}
    sample_weights = torch.tensor([weights_by_class[int(label)] for label in labels], dtype=torch.float32)
    print(f"Training class distribution: {counts}")
    print(f"Balanced sampler enabled; draw distribution is balanced in expectation across classes.")
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def build_domain_weights_from_train(train_df: pd.DataFrame):
    if "domain" not in train_df.columns:
        return {}, 0.5

    counts: Dict[str, Dict[int, int]] = {}
    for _, row in train_df.iterrows():
        domain = str(row.get("domain", "")).lower().strip()
        if not domain or domain == "nan":
            continue
        label = int(row["2_way_label"])
        if domain not in counts:
            counts[domain] = {0: 0, 1: 0}
        counts[domain][label] += 1

    weights = {}
    for domain, label_counts in counts.items():
        total = label_counts[0] + label_counts[1]
        if total > 0:
            weights[domain] = label_counts[1] / total

    mean_weight = float(np.mean(list(weights.values()))) if weights else 0.5
    print(f"Built train-only domain credibility weights for {len(weights)} domains.")
    return weights, mean_weight


def load_backbones(modalities, device):
    tokenizer = None
    graph_text_model = None
    trainable_text_model = None
    image_processor = None
    vit_model = None

    if modalities["text"] or modalities["graph"]:
        tokenizer = AutoTokenizer.from_pretrained(Config.TEXT_MODEL_NAME)
    if modalities["graph"]:
        graph_text_model = AutoModel.from_pretrained(Config.TEXT_MODEL_NAME).to(device)
        Config.POST_FEATURE_DIM = int(graph_text_model.config.hidden_size)
        graph_text_model.eval()
        for param in graph_text_model.parameters():
            param.requires_grad = False
    if modalities["text"]:
        trainable_text_model = AutoModel.from_pretrained(Config.TEXT_MODEL_NAME).to(device)
        try:
            trainable_text_model.gradient_checkpointing_enable()
        except Exception:
            pass
    if modalities["image"]:
        image_processor = AutoImageProcessor.from_pretrained(Config.VIT_MODEL_NAME, use_fast=True)
        vit_model = ViTModel.from_pretrained(Config.VIT_MODEL_NAME).to(device)
        try:
            vit_model.gradient_checkpointing_enable()
        except Exception:
            pass

    return tokenizer, graph_text_model, trainable_text_model, image_processor, vit_model


def build_graph_if_needed(df, tokenizer, graph_text_model, device):
    post_to_idx = {str(row["id"]): idx for idx, row in df.iterrows()}
    graph_data = None
    if graph_text_model is not None:
        features = extract_post_text_features(df, tokenizer, graph_text_model, device, batch_size=32, max_len=Config.MAX_LEN)
        builder = FakedditGraphBuilder(
            post_feature_dim=Config.POST_FEATURE_DIM,
            user_embed_dim=Config.USER_EMBED_DIM,
            subreddit_embed_dim=Config.SUBREDDIT_EMBED_DIM,
            domain_embed_dim=Config.DOMAIN_EMBED_DIM,
            seed=Config.SEED,
            social_graph_root=Config.SOCIAL_GRAPH_ROOT,
            use_social_graph=Config.USE_SOCIAL_GRAPH,
        )
        graph_data = builder.build_graph(df, features).to(device)
        post_to_idx = builder.post_to_idx
    return graph_data, post_to_idx


def make_model(modalities, trainable_text_model, vit_model, device):
    if modalities["cred"] and not (modalities["text"] or modalities["image"] or modalities["graph"] or modalities["prop"]):
        return CredOnlyModel().to(device)
    return MultiModalGNNModel(
        text_model=trainable_text_model,
        vit_model=vit_model,
        d_proj=Config.D_PROJ,
        gnn_dim=Config.GNN_HIDDEN_DIM,
        num_classes=2,
        fusion_type=Config.FUSION_TYPE,
        credibility_weight=Config.CREDIBILITY_WEIGHT,
        use_text=modalities["text"],
        use_image=modalities["image"],
        use_graph=modalities["graph"],
        use_propagation=modalities["prop"],
        use_credibility=modalities["cred"],
    ).to(device)


def train_one_seed(args, ablation: str, seed: int, run_dir: Path, train_full_df, test_df, label_to_idx, idx_to_label):
    set_seed(seed)
    modalities = dict(ABLATIONS[ablation])
    if args.use_propagation_features:
        modalities["prop"] = True
    Config.USE_TEXT = modalities["text"]
    Config.USE_IMAGE = modalities["image"]
    Config.USE_GRAPH = modalities["graph"]
    Config.USE_PROPAGATION = modalities["prop"]
    Config.USE_CREDIBILITY = modalities["cred"]

    seed_dir = ensure_dir(run_dir / f"seed_{seed}")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"\n=== Ablation={ablation} Seed={seed} Device={device} ===")

    train_df, val_df = train_test_split(
        train_full_df,
        test_size=args.val_size,
        random_state=seed,
        stratify=train_full_df["2_way_label"].astype(int).values,
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    domain_weights, mean_weight = ({}, 0.5)
    if modalities["cred"]:
        if args.domain_weights_source == "file":
            domain_weights, mean_weight = load_domain_weights(Config.DOMAIN_WEIGHTS_JSON)
            print("Using domain credibility weights from file; this may include test domains if built globally.")
        else:
            domain_weights, mean_weight = build_domain_weights_from_train(train_df)

    tokenizer, graph_text_model, trainable_text_model, image_processor, vit_model = load_backbones(modalities, device)

    train_graph, train_post_to_idx = build_graph_if_needed(train_df, tokenizer, graph_text_model, device)
    val_graph, val_post_to_idx = build_graph_if_needed(val_df, tokenizer, graph_text_model, device)
    test_graph, test_post_to_idx = build_graph_if_needed(test_df, tokenizer, graph_text_model, device)

    dataset_tokenizer = tokenizer if modalities["text"] else None
    train_ds = make_dataset(train_df, dataset_tokenizer, image_processor, train_post_to_idx, domain_weights, mean_weight)
    val_ds = make_dataset(val_df, dataset_tokenizer, image_processor, val_post_to_idx, domain_weights, mean_weight)
    test_ds = make_dataset(test_df, dataset_tokenizer, image_processor, test_post_to_idx, domain_weights, mean_weight)

    sampler = make_sampler(train_df) if args.use_balanced_sampler else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)

    model = make_model(modalities, trainable_text_model, vit_model, device)
    criterion = build_criterion(args, train_df["2_way_label"].astype(int).values, device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=max(1, len(train_loader) * args.epochs),
    )

    best_val_macro_f1 = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            kwargs, labels = batch_to_kwargs(batch, modalities, device, train_graph)
            optimizer.zero_grad(set_to_none=True)
            logits = model(**kwargs).logits
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.item()) * labels.size(0)
            total += labels.size(0)

        val_ids, val_y, val_probs, val_loss = get_probs(model, val_loader, modalities, device, val_graph, criterion)
        val_argmax = np.argmax(val_probs, axis=1)
        val_metrics = metric_summary(val_y, val_argmax, label_to_idx)
        print(
            f"Epoch {epoch + 1}: train_loss={total_loss / max(1, total):.4f} "
            f"val_loss={val_loss:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_real_recall={val_metrics['real_recall']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_macro_f1 + args.early_stopping_min_delta:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                print(f"Early stopping after epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), seed_dir / "model.pt")

    val_ids, val_y, val_probs, val_loss = get_probs(model, val_loader, modalities, device, val_graph, criterion)
    test_ids, test_y, test_probs, test_loss = get_probs(model, test_loader, modalities, device, test_graph, criterion)
    test_argmax = np.argmax(test_probs, axis=1)
    argmax_metrics, report, cm = write_evaluation_outputs(
        seed_dir, "argmax", test_ids, test_y, test_probs, test_argmax, label_to_idx, idx_to_label, test_loss
    )

    threshold_info = None
    final_metrics = dict(argmax_metrics)
    final_mode = "argmax"
    if args.use_threshold_tuning or args.fixed_threshold is not None:
        best_threshold = None
        if args.use_threshold_tuning:
            best_threshold, rows = find_best_threshold(
                val_y,
                val_probs,
                target_class=label_to_idx["fake"],
                other_class=label_to_idx["real"],
                label_to_idx=label_to_idx,
                metric=args.threshold_metric,
                min_threshold=args.threshold_search_min,
                max_threshold=args.threshold_search_max,
                step=args.threshold_search_step,
            )
            write_csv(seed_dir / "threshold_search.csv", rows)
        threshold_value = args.fixed_threshold if args.fixed_threshold is not None else best_threshold["threshold"]
        test_threshold = predict_with_threshold(test_probs, threshold_value, label_to_idx["fake"], label_to_idx["real"])
        threshold_metrics, _, _ = write_evaluation_outputs(
            seed_dir,
            "threshold",
            test_ids,
            test_y,
            test_probs,
            test_threshold,
            label_to_idx,
            idx_to_label,
            test_loss,
        )
        threshold_info = {"selected_threshold": threshold_value, "selection_metric": args.threshold_metric}
        if best_threshold:
            threshold_info.update(best_threshold)
        final_metrics = dict(threshold_metrics)
        final_mode = "threshold"

    # Required generic filenames point at the final evaluation mode. For threshold
    # runs, keep a combined report so argmax and tuned-threshold behavior are
    # visible together without opening multiple files.
    if final_mode == "threshold":
        argmax_report = (seed_dir / "classification_report_argmax.txt").read_text(encoding="utf-8")
        threshold_report = (seed_dir / "classification_report_threshold.txt").read_text(encoding="utf-8")
        generic_report = (
            "ARGMAX EVALUATION\n"
            "=================\n"
            f"{argmax_report}\n\n"
            "THRESHOLD EVALUATION\n"
            "====================\n"
            f"Selected Fake threshold: {threshold_info.get('selected_threshold')}\n\n"
            f"{threshold_report}"
        )
        print(generic_report)
    else:
        generic_report = (seed_dir / f"classification_report_{final_mode}.txt").read_text(encoding="utf-8")
    (seed_dir / "classification_report.txt").write_text(generic_report, encoding="utf-8")
    if (seed_dir / f"confusion_matrix_{final_mode}.csv").exists():
        (seed_dir / "confusion_matrix.csv").write_text(
            (seed_dir / f"confusion_matrix_{final_mode}.csv").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    if (seed_dir / f"predictions_{final_mode}.csv").exists():
        (seed_dir / "predictions.csv").write_text(
            (seed_dir / f"predictions_{final_mode}.csv").read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    metrics_payload = {
        "seed": seed,
        "ablation": ablation,
        "final_mode": final_mode,
        "argmax": argmax_metrics,
        "final": final_metrics,
        "threshold": threshold_info,
    }
    write_json(seed_dir / "metrics.json", metrics_payload)
    write_json(seed_dir / "config.json", {**vars(args), "seed": seed, "ablation": ablation})

    result_row = {"seed": seed, "ablation": ablation, "final_mode": final_mode, **final_metrics}
    if threshold_info:
        result_row["selected_threshold"] = threshold_info["selected_threshold"]
    return result_row


def aggregate_seed_results(rows: List[Dict]) -> List[Dict]:
    if not rows:
        return []
    metrics = ["accuracy", "macro_f1", "weighted_f1", "fake_f1", "real_f1", "real_recall"]
    aggregate = []
    for metric in metrics:
        values = [float(row[metric]) for row in rows if metric in row]
        aggregate.append(
            {
                "metric": metric,
                "mean": float(np.mean(values)) if values else "",
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
        )
    return aggregate


def parse_args():
    parser = argparse.ArgumentParser(description="FakeNewsNet experiment runner")
    parser.add_argument("--train-tsv", default="data/train.tsv")
    parser.add_argument("--test-tsv", default="data/test.tsv")
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--ablation", choices=sorted(ABLATIONS), default="full")
    parser.add_argument("--run_all_ablations", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("FAKENEWSNET_EPOCHS", "10")))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0 if os.name == "nt" else 4)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--text-model", default=Config.TEXT_MODEL_NAME)
    parser.add_argument("--vit-model", default=Config.VIT_MODEL_NAME)
    parser.add_argument("--max-length", type=int, default=Config.MAX_LEN)
    parser.add_argument("--use_social_graph", action="store_true")
    parser.add_argument("--use_propagation_features", action="store_true")
    parser.add_argument("--domain-weights-source", choices=["train", "file"], default="train")

    parser.add_argument("--loss_type", choices=["ce", "focal"], default="ce")
    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--focal_alpha", default="none")
    parser.add_argument("--use_balanced_sampler", action="store_true")

    parser.add_argument("--use_threshold_tuning", action="store_true")
    parser.add_argument("--threshold_metric", default="macro_f1")
    parser.add_argument("--fixed_threshold", type=float, default=None)
    parser.add_argument("--threshold_search_min", type=float, default=0.05)
    parser.add_argument("--threshold_search_max", type=float, default=0.95)
    parser.add_argument("--threshold_search_step", type=float, default=0.01)
    return parser.parse_args()


def main():
    args = parse_args()
    Config.TEXT_MODEL_NAME = args.text_model
    Config.VIT_MODEL_NAME = args.vit_model
    Config.MAX_LEN = args.max_length
    Config.USE_SOCIAL_GRAPH = bool(args.use_social_graph)
    Config.USE_PROPAGATION = bool(args.use_propagation_features)
    train_full_df = pd.read_csv(args.train_tsv, sep="\t")
    test_df = pd.read_csv(args.test_tsv, sep="\t")
    train_full_df = train_full_df.dropna(subset=["2_way_label", "clean_title"]).copy()
    test_df = test_df.dropna(subset=["2_way_label", "clean_title"]).copy()
    train_full_df["id"] = train_full_df["id"].astype(str)
    test_df["id"] = test_df["id"].astype(str)
    train_full_df["2_way_label"] = train_full_df["2_way_label"].astype(int)
    test_df["2_way_label"] = test_df["2_way_label"].astype(int)

    label_to_idx, idx_to_label = detect_label_mapping(pd.concat([train_full_df, test_df], ignore_index=True))
    print("Threshold tuning targets the Fake class probability; raising the threshold reduces Real->Fake false positives.")
    if args.use_propagation_features:
        validate_propagation_columns(train_full_df)
        validate_propagation_columns(test_df)
        print(f"Using {Config.PROPAGATION_FEATURE_DIM} offline tweet-ID temporal propagation features.")

    ablations = sorted(ABLATIONS) if args.run_all_ablations else [args.ablation]
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"fakenewsnet_{'all_ablations' if args.run_all_ablations else args.ablation}_{timestamp}"
    run_dir = ensure_dir(Path(args.output_root) / run_name)
    write_json(run_dir / "config.json", {**vars(args), "label_to_idx": label_to_idx, "ablations": ablations})

    all_rows = []
    for ablation in ablations:
        ablation_rows = []
        ablation_dir = ensure_dir(run_dir / ablation) if len(ablations) > 1 else run_dir
        for seed in args.seeds:
            row = train_one_seed(args, ablation, seed, ablation_dir, train_full_df, test_df, label_to_idx, idx_to_label)
            ablation_rows.append(row)
            all_rows.append(row)
        write_csv(ablation_dir / "per_seed_results.csv", ablation_rows)
        write_csv(ablation_dir / "aggregate_results.csv", aggregate_seed_results(ablation_rows))

    write_csv(run_dir / "per_seed_results.csv", all_rows)
    write_csv(run_dir / "aggregate_results.csv", aggregate_seed_results(all_rows))
    print(f"\nRun complete: {run_dir}")


if __name__ == "__main__":
    main()
