"""
Comprehensive Politifact trainer supporting text, multi-modal, and propagation settings.

The script expects JSON metadata files describing each split. Each record must contain:
    - id (str): unique identifier
    - label (int): 0 for real, 1 for fake
    - text (str): article or statement text
Optional keys:
    - image_path (str): path (absolute or relative to --image-root) to an image
    - graph_split (str): one of {"train", "val", "test"} when referencing UPFD graphs
    - graph_index (int): index within the chosen UPFD split

Example usage:
    python train_politifact.py --modality text \\
        --train-metadata data/politifact/train.json \\
        --val-metadata data/politifact/val.json \\
        --test-metadata data/politifact/test.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import contextlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

try:
    from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
except ImportError as exc:
    raise ImportError(
        "transformers is required; install with `pip install transformers`"
    ) from exc

try:
    from torch_geometric.data import Batch
    from torch_geometric.datasets import UPFD
    from torch_geometric.nn import SAGEConv, global_mean_pool
except ImportError as exc:
    raise ImportError(
        "torch_geometric is required for propagation mode; "
        "install with `pip install torch-geometric` (and its dependencies)."
    ) from exc


# ------------------------------- data objects -------------------------------- #


@dataclass
class SampleRecord:
    """Container describing one Politifact example."""

    id: str
    label: int
    text: str
    image_path: Optional[str] = None
    graph_split: Optional[str] = None
    graph_index: Optional[int] = None


def load_metadata(path: str) -> List[SampleRecord]:
    """Read metadata JSON (list or JSONL) and produce sample records."""

    if path is None:
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f) if path.endswith(".json") else [json.loads(line) for line in f]
    records = []
    for item in raw:
        records.append(
            SampleRecord(
                id=item["id"],
                label=int(item["label"]),
                text=item["text"],
                image_path=item.get("image_path"),
                graph_split=item.get("graph_split"),
                graph_index=item.get("graph_index"),
            )
        )
    return records


def set_seed(seed: int) -> None:
    """Seed python, numpy, and torch for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------- graph access -------------------------------- #


class UPFDGraphStore:
    """Lazy loader that maps (split, index) -> torch_geometric Data."""

    def __init__(self, root: str, feature: str = "bert", device: Optional[str] = None) -> None:
        self.feature = feature
        self.device = device
        self._datasets: Dict[str, UPFD] = {}
        self.root = root

    def _get_dataset(self, split: str) -> UPFD:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown UPFD split '{split}'")
        if split not in self._datasets:
            self._datasets[split] = UPFD(
                root=self.root, name="politifact", feature=self.feature, split=split
            )
        return self._datasets[split]

    def fetch(self, split: Optional[str], index: Optional[int]):
        if split is None or index is None:
            raise ValueError("Graph requests require both split and index.")
        data = self._get_dataset(split)[index]
        if self.device:
            return data.to(self.device)
        return data


# ------------------------------- dataset ------------------------------------- #


class PolitifactDataset(Dataset):
    """Dataset generating batches for the requested modality."""

    def __init__(
        self,
        records: Sequence[SampleRecord],
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        use_image: bool,
        image_root: Optional[str],
        image_transform: Optional[transforms.Compose],
        use_graph: bool,
        graph_store: Optional[UPFDGraphStore],
        image_size: int = 224,
        allow_image_dummy: bool = True,
    ) -> None:
        self.records = list(records)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_image = use_image
        self.image_root = image_root
        self.image_transform = image_transform
        self.use_graph = use_graph
        self.graph_store = graph_store
        self.allow_image_dummy = allow_image_dummy
        self.image_size = image_size

        if self.use_image and self.image_transform is None:
            raise ValueError("image_transform must be provided when use_image=True")
        if self.use_graph and self.graph_store is None:
            raise ValueError("graph_store must be provided when use_graph=True")
        if self.use_image and self.allow_image_dummy:
            blank = Image.new("RGB", (self.image_size, self.image_size), color=(0, 0, 0))
            self._dummy_image = self.image_transform(blank)
        else:
            self._dummy_image = None

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_image(self, path: Optional[str]) -> torch.Tensor:
        if not path:
            if not self.allow_image_dummy:
                raise FileNotFoundError("Missing image path and dummy disabled.")
            return torch.zeros(3, 224, 224)

        actual_path = path
        if self.image_root and not os.path.isabs(path):
            actual_path = os.path.join(self.image_root, path)

        if not os.path.exists(actual_path):
            if not self.allow_image_dummy or self._dummy_image is None:
                raise FileNotFoundError(f"Image not found: {actual_path}")
            return self._dummy_image.clone()

        with Image.open(actual_path) as img:
            img = img.convert("RGB")
            return self.image_transform(img)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        encoded = self.tokenizer(
            record.text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        example: Dict[str, Any] = {
            "input_ids": encoded["input_ids"][0],
            "attention_mask": encoded["attention_mask"][0],
            "label": torch.tensor(record.label, dtype=torch.long),
            "id": record.id,
        }

        if self.use_image:
            example["image"] = self._resolve_image(record.image_path)

        if self.use_graph:
            assert self.graph_store is not None
            example["graph"] = self.graph_store.fetch(record.graph_split, record.graph_index)

        return example


def multimodal_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge dataset items into a batch; handles optional images/graphs."""

    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    ids = [item["id"] for item in batch]

    collated: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "ids": ids,
    }

    if "image" in batch[0]:
        collated["image"] = torch.stack([item["image"] for item in batch])

    if "graph" in batch[0]:
        collated["graph"] = Batch.from_data_list([item["graph"] for item in batch])

    return collated


# ------------------------------- metrics ------------------------------------- #


def compute_metrics(preds: Iterable[int], labels: Iterable[int]) -> Dict[str, float]:
    preds_arr = np.array(list(preds))
    labels_arr = np.array(list(labels))
    assert preds_arr.shape == labels_arr.shape

    accuracy = float((preds_arr == labels_arr).mean())

    tp = float(((preds_arr == 1) & (labels_arr == 1)).sum())
    fp = float(((preds_arr == 1) & (labels_arr == 0)).sum())
    fn = float(((preds_arr == 0) & (labels_arr == 1)).sum())

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ------------------------------- model --------------------------------------- #


class MultiModalFakeNewsModel(nn.Module):
    """Flexible classifier covering text, text+image, and text+image+graph inputs."""

    def __init__(
        self,
        text_model_name: str,
        dropout: float,
        use_image: bool,
        use_graph: bool,
        graph_in_dim: int = 768,
        graph_hidden_dim: int = 256,
        image_out_dim: int = 256,
        freeze_text_encoder: bool = False,
        freeze_image_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.use_image = use_image
        self.use_graph = use_graph

        self.text_encoder: PreTrainedModel = AutoModel.from_pretrained(text_model_name)
        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        self.text_dropout = nn.Dropout(dropout)
        text_dim = self.text_encoder.config.hidden_size

        fused_dims = [text_dim]

        if self.use_image:
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            if freeze_image_encoder:
                for param in resnet.parameters():
                    param.requires_grad = False
            image_feat_dim = resnet.fc.in_features
            resnet.fc = nn.Identity()
            self.image_encoder = resnet
            self.image_projection = nn.Linear(image_feat_dim, image_out_dim)
            fused_dims.append(image_out_dim)
        else:
            self.image_encoder = None
            self.image_projection = None

        if self.use_graph:
            self.gnn_conv1 = SAGEConv(graph_in_dim, graph_hidden_dim)
            self.gnn_conv2 = SAGEConv(graph_hidden_dim, graph_hidden_dim)
            self.gnn_dropout = nn.Dropout(dropout)
            fused_dims.append(graph_hidden_dim)
        else:
            self.gnn_conv1 = None
            self.gnn_conv2 = None
            self.gnn_dropout = None

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(sum(fused_dims), 1),
        )

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0]
        return self.text_dropout(cls)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        assert self.image_encoder is not None and self.image_projection is not None
        feats = self.image_encoder(image)
        if feats.ndim > 2:
            feats = torch.flatten(feats, 1)
        return F.relu(self.image_projection(feats))

    def encode_graph(self, graph: Batch) -> torch.Tensor:
        assert self.gnn_conv1 is not None and self.gnn_conv2 is not None and self.gnn_dropout is not None
        x = F.relu(self.gnn_conv1(graph.x, graph.edge_index))
        x = self.gnn_dropout(x)
        x = F.relu(self.gnn_conv2(x, graph.edge_index))
        pooled = global_mean_pool(x, graph.batch)
        return pooled

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image: Optional[torch.Tensor] = None,
        graph: Optional[Batch] = None,
    ) -> torch.Tensor:
        representations = [self.encode_text(input_ids, attention_mask)]

        if self.use_image:
            if image is None:
                raise ValueError("image tensor required when use_image=True")
            representations.append(self.encode_image(image))

        if self.use_graph:
            if graph is None:
                raise ValueError("graph batch required when use_graph=True")
            representations.append(self.encode_graph(graph))

        fused = torch.cat(representations, dim=1)
        logits = self.classifier(fused).squeeze(1)
        return logits


# ------------------------------- training ------------------------------------ #


def run_epoch(
    model: MultiModalFakeNewsModel,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
) -> Tuple[float, Dict[str, float]]:
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    preds: List[int] = []
    labels: List[int] = []

    progress = tqdm(loader, desc="train" if training else "eval", leave=False)
    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label_tensor = batch["labels"].to(device, dtype=torch.float32)
        image = batch.get("image")
        graph = batch.get("graph")

        if image is not None:
            image = image.to(device)
        if graph is not None:
            graph = graph.to(device)

        autocast_ctx = torch.cuda.amp.autocast if scaler else contextlib.nullcontext
        with autocast_ctx():
            logits = model(input_ids=input_ids, attention_mask=attention_mask, image=image, graph=graph)
            loss = F.binary_cross_entropy_with_logits(logits, label_tensor)

        if training:
            optimizer.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * input_ids.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds.extend((probs >= 0.5).astype(int).tolist())
        labels.extend(label_tensor.detach().cpu().numpy().astype(int).tolist())

    metrics = compute_metrics(preds, labels)
    avg_loss = total_loss / max(len(loader.dataset), 1)
    metrics["loss"] = avg_loss
    return avg_loss, metrics


def train_model(
    model: MultiModalFakeNewsModel,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> Tuple[MultiModalFakeNewsModel, Dict[str, float]]:
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    best_state = None
    best_metric = -float("inf")
    best_metrics: Dict[str, float] = {}

    for epoch in range(1, epochs + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, optimizer, device, scaler)
        print(f"Epoch {epoch}/{epochs} - train_loss={train_loss:.4f} - train_f1={train_metrics['f1']:.4f}")

        if val_loader is not None:
            _, val_metrics = run_epoch(model, val_loader, None, device, None)
            score = val_metrics["f1"]
            print(
                f"Epoch {epoch}/{epochs} - val_loss={val_metrics['loss']:.4f} "
                f"- val_f1={score:.4f} - val_acc={val_metrics['accuracy']:.4f}"
            )
            if score > best_metric:
                best_metric = score
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                best_metrics = val_metrics
        else:
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_metrics = train_metrics

    if best_state:
        model.load_state_dict(best_state)

    return model, best_metrics


# ------------------------------- CLI ----------------------------------------- #


def build_image_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Politifact multi-modal trainer")
    parser.add_argument("--modality", choices=["text", "text_image", "text_image_graph"], required=True)
    parser.add_argument("--train-metadata", required=True, help="Path to training metadata JSON/JSONL")
    parser.add_argument("--val-metadata", help="Validation metadata JSON/JSONL")
    parser.add_argument("--test-metadata", help="Test metadata JSON/JSONL")
    parser.add_argument("--image-root", help="Root directory prepended to image_path values")
    parser.add_argument("--upfd-root", help="Cache root for UPFD graphs (required for propagation mode)")
    parser.add_argument("--upfd-feature", default="bert", help="Feature type for UPFD graphs")
    parser.add_argument("--text-model", default="distilbert-base-uncased", help="HF model name for text encoder")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--graph-hidden", type=int, default=256)
    parser.add_argument("--image-hidden", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/politifact", help="Directory to store best model")
    parser.add_argument("--no-freeze-image", action="store_true", help="Allow image encoder fine-tuning")
    parser.add_argument("--freeze-text", action="store_true", help="Freeze text encoder weights")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    use_image = args.modality in {"text_image", "text_image_graph"}
    use_graph = args.modality == "text_image_graph"

    train_records = load_metadata(args.train_metadata)
    val_records = load_metadata(args.val_metadata) if args.val_metadata else []
    test_records = load_metadata(args.test_metadata) if args.test_metadata else []

    if use_graph and not args.upfd_root:
        raise ValueError("--upfd-root is required for propagation training")

    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    image_transform = build_image_transform(args.image_size) if use_image else None
    graph_store = (
        UPFDGraphStore(args.upfd_root, feature=args.upfd_feature, device=None) if use_graph else None
    )

    train_dataset = PolitifactDataset(
        train_records,
        tokenizer=tokenizer,
        max_length=args.max_length,
        use_image=use_image,
        image_root=args.image_root,
        image_transform=image_transform,
        use_graph=use_graph,
        graph_store=graph_store,
        image_size=args.image_size,
    )
    val_dataset = (
        PolitifactDataset(
            val_records,
            tokenizer=tokenizer,
            max_length=args.max_length,
            use_image=use_image,
            image_root=args.image_root,
            image_transform=image_transform,
            use_graph=use_graph,
            graph_store=graph_store,
            image_size=args.image_size,
        )
        if val_records
        else None
    )
    test_dataset = (
        PolitifactDataset(
            test_records,
            tokenizer=tokenizer,
            max_length=args.max_length,
            use_image=use_image,
            image_root=args.image_root,
            image_transform=image_transform,
            use_graph=use_graph,
            graph_store=graph_store,
            image_size=args.image_size,
        )
        if test_records
        else None
    )

    collate = multimodal_collate
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate,
        )
        if val_dataset
        else None
    )
    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate,
        )
        if test_dataset
        else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalFakeNewsModel(
        text_model_name=args.text_model,
        dropout=args.dropout,
        use_image=use_image,
        use_graph=use_graph,
        graph_in_dim=768,
        graph_hidden_dim=args.graph_hidden,
        image_out_dim=args.image_hidden,
        freeze_text_encoder=args.freeze_text,
        freeze_image_encoder=not args.no_freeze_image,
    )
    model.to(device)

    os.makedirs(args.output_dir, exist_ok=True)

    model, best_metrics = train_model(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    torch.save(model.state_dict(), os.path.join(args.output_dir, f"best_{args.modality}.pt"))
    print(f"Best validation metrics: {best_metrics}")

    if test_loader:
        _, test_metrics = run_epoch(model, test_loader, None, device, None)
        print(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
