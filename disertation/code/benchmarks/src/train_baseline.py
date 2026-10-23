#!/usr/bin/env python3
"""Train one matched-split baseline and save reproducible test outputs.

All models consume the same standardized CSV schema and select the checkpoint
using validation macro-F1. The held-out test split is evaluated once, after
selection. Labels are standardized as 1=fake and 0=real.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm.auto import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPModel,
    CLIPProcessor,
    CLIPVisionModelWithProjection,
)


MODEL_CHOICES = [
    "distilbert_text",
    "bert_text",
    "roberta_text",
    "resnet50_image",
    "bert_resnet_concat",
    "debertnext_matched",
    "clip_linear",
    "spotfake_faknow",
    "cafe_faknow",
]


@dataclass
class RunConfig:
    dataset: str
    model: str
    seed: int
    train_csv: str
    val_csv: str
    test_csv: str
    output_dir: str
    text_model: str
    clip_model: str
    epochs: int
    batch_size: int
    learning_rate: float
    head_learning_rate: float
    weight_decay: float
    patience: int
    max_length: int
    num_workers: int
    freeze_encoders: bool
    mixed_precision: bool
    smoke_limit: Optional[int]
    validation_only: bool
    feature_cache_root: Optional[str]
    require_feature_cache: bool
    auto_resume: bool


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class StandardDataset(Dataset):
    def __init__(self, csv_path: Path, limit: Optional[int] = None):
        frame = pd.read_csv(
            csv_path,
            usecols=["item_id", "text", "image_path", "has_image", "label_fake"],
            dtype={"item_id": str, "text": str, "image_path": str},
        )
        if limit:
            # A deterministic prefix is sufficient for dependency smoke tests only.
            frame = frame.iloc[:limit].copy()
        self.ids = frame["item_id"].astype(str).tolist()
        self.texts = frame["text"].fillna("").astype(str).tolist()
        self.image_paths = frame["image_path"].fillna("").astype(str).tolist()
        self.has_images = frame["has_image"].astype(str).str.lower().isin(["true", "1"]).tolist()
        self.labels = frame["label_fake"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Dict[str, object]:
        return {
            "item_id": self.ids[index],
            "text": self.texts[index],
            "image_path": self.image_paths[index],
            "has_image": self.has_images[index],
            "label": self.labels[index],
        }


class CachedFeatureDataset(Dataset):
    """Memory-map frozen encoder outputs so all seeds share the same features."""

    def __init__(self, csv_path: Path, cache_dir: Path, limit: Optional[int] = None):
        frame = pd.read_csv(csv_path, usecols=["item_id", "has_image", "label_fake"], dtype={"item_id": str})
        if limit:
            frame = frame.iloc[:limit].copy()
        meta_path = cache_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Frozen feature cache is missing: {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        ordered_hash = hashlib.sha256("\n".join(frame["item_id"].astype(str)).encode("utf-8")).hexdigest()
        if int(meta["rows"]) < len(frame) or (not limit and meta["ordered_id_sha256"] != ordered_hash):
            raise RuntimeError(f"Frozen feature cache does not match {csv_path}")
        self.ids = frame["item_id"].astype(str).tolist()
        self.labels = frame["label_fake"].astype(int).tolist()
        self.has_images = frame["has_image"].astype(str).str.lower().isin(["true", "1"]).tolist()
        self.arrays = {
            name: np.load(cache_dir / filename, mmap_mode="r")
            for name, filename in meta["features"].items()
        }

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row: Dict[str, object] = {
            "item_id": self.ids[index],
            "label": self.labels[index],
            "has_image": self.has_images[index],
        }
        for name, array in self.arrays.items():
            row[name] = torch.tensor(np.asarray(array[index]), dtype=torch.float32)
        return row


class CachedFeatureCollator:
    def __call__(self, rows: List[Dict[str, object]]) -> Dict[str, object]:
        keys = [key for key in rows[0] if key not in {"item_id", "label", "has_image"}]
        batch: Dict[str, object] = {
            "item_ids": [str(row["item_id"]) for row in rows],
            "labels": torch.tensor([int(row["label"]) for row in rows], dtype=torch.long),
            "has_image": torch.tensor([bool(row["has_image"]) for row in rows], dtype=torch.float32),
        }
        for key in keys:
            batch[key] = torch.stack([row[key] for row in rows])
        return batch


class BatchCollator:
    def __init__(self, model_name: str, text_model: str, clip_model: str, max_length: int):
        self.model_name = model_name
        self.max_length = max_length
        self.needs_text = model_name in {
            "distilbert_text",
            "bert_text",
            "roberta_text",
            "bert_resnet_concat",
            "debertnext_matched",
            "spotfake_faknow",
            "cafe_faknow",
        }
        self.needs_image = model_name in {
            "resnet50_image",
            "bert_resnet_concat",
            "debertnext_matched",
            "spotfake_faknow",
            "cafe_faknow",
        }
        self.is_clip = model_name == "clip_linear"
        self.is_cafe = model_name == "cafe_faknow"
        self.tokenizer = AutoTokenizer.from_pretrained(text_model) if self.needs_text else None
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model) if self.is_clip else None
        self.cafe_image_processor = CLIPImageProcessor.from_pretrained(clip_model) if self.is_cafe else None
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def load_image(path: str, present: bool) -> tuple[Image.Image, bool]:
        if present and path:
            try:
                with Image.open(path) as source:
                    return source.convert("RGB"), True
            except (OSError, ValueError, Image.DecompressionBombError):
                pass
        return Image.new("RGB", (224, 224), color="black"), False

    def __call__(self, rows: List[Dict[str, object]]) -> Dict[str, object]:
        texts = [str(row["text"]) for row in rows]
        present = [bool(row["has_image"]) for row in rows]
        batch: Dict[str, object] = {
            "item_ids": [str(row["item_id"]) for row in rows],
            "labels": torch.tensor([int(row["label"]) for row in rows], dtype=torch.long),
            "has_image": torch.tensor(present, dtype=torch.float32),
        }
        if self.is_clip:
            loaded = [self.load_image(str(row["image_path"]), ok) for row, ok in zip(rows, present)]
            images = [image for image, _ in loaded]
            present = [ok for _, ok in loaded]
            batch["has_image"] = torch.tensor(present, dtype=torch.float32)
            encoded = self.clip_processor(
                text=texts,
                images=images,
                padding=True,
                truncation=True,
                max_length=min(self.max_length, 77),
                return_tensors="pt",
            )
            batch.update(encoded)
            return batch
        if self.needs_text:
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch["input_ids"] = encoded["input_ids"]
            batch["attention_mask"] = encoded["attention_mask"]
        if self.needs_image:
            loaded = [self.load_image(str(row["image_path"]), ok) for row, ok in zip(rows, present)]
            images = [image for image, _ in loaded]
            present = [ok for _, ok in loaded]
            batch["has_image"] = torch.tensor(present, dtype=torch.float32)
            if self.is_cafe:
                batch["pixel_values"] = self.cafe_image_processor(images=images, return_tensors="pt")["pixel_values"]
            else:
                batch["pixel_values"] = torch.stack([self.image_transform(image) for image in images])
        return batch


class BertTextClassifier(nn.Module):
    def __init__(self, model_name: str, freeze: bool):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        if freeze:
            self.encoder.requires_grad_(False)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(self.encoder.config.hidden_size, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 2)
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        output = self.encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        return self.classifier(output.last_hidden_state[:, 0])


def resnet_backbone(freeze: bool) -> tuple[nn.Module, int]:
    network = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    dimension = network.fc.in_features
    network.fc = nn.Identity()
    if freeze:
        network.requires_grad_(False)
    return network, dimension


class ResNetClassifier(nn.Module):
    def __init__(self, freeze: bool):
        super().__init__()
        self.encoder, dimension = resnet_backbone(freeze)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(dimension + 1, 256), nn.ReLU(), nn.Linear(256, 2))

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = self.encoder(batch["pixel_values"])
        features = torch.cat([features, batch["has_image"].unsqueeze(1)], dim=1)
        return self.classifier(features)


class BertResNetClassifier(nn.Module):
    def __init__(self, text_model: str, freeze: bool):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model)
        self.image_encoder, image_dim = resnet_backbone(freeze)
        if freeze:
            self.text_encoder.requires_grad_(False)
        total = self.text_encoder.config.hidden_size + image_dim + 1
        self.classifier = nn.Sequential(
            nn.Dropout(0.25), nn.Linear(total, 512), nn.GELU(), nn.Dropout(0.25), nn.Linear(512, 2)
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        text = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        ).last_hidden_state[:, 0]
        image = self.image_encoder(batch["pixel_values"])
        features = torch.cat([text, image, batch["has_image"].unsqueeze(1)], dim=1)
        return self.classifier(features)


def convnext_large_backbone(freeze: bool) -> tuple[nn.Module, int]:
    """Return the ImageNet ConvNeXt-L encoder used by DeBERTNeXT."""
    network = models.convnext_large(weights=models.ConvNeXt_Large_Weights.DEFAULT)
    dimension = network.classifier[-1].in_features
    network.classifier[-1] = nn.Identity()
    if freeze:
        network.requires_grad_(False)
    return network, dimension


class DeBERTNeXTMatchedClassifier(nn.Module):
    """Matched-split implementation of the DeBERTNeXT fusion architecture.

    The ICCS 2023 model concatenates a DeBERTa-v3 representation with a
    1,024-dimensional projection of ConvNeXt-L features. This implementation
    keeps that architecture but consumes the benchmark's immutable rows and
    missing-image mask. It is an independent implementation from the paper
    description; no code was copied from the unlicensed notebook repository.
    """

    def __init__(self, text_model: str, freeze: bool):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model)
        self.image_encoder, image_dimension = convnext_large_backbone(freeze)
        if freeze:
            self.text_encoder.requires_grad_(False)
        self.image_projection = nn.Sequential(
            nn.Linear(image_dimension, 1024),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        total = self.text_encoder.config.hidden_size + 1024 + 1
        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(total, 512),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(512, 2),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        text = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        ).last_hidden_state[:, 0]
        image = self.image_projection(self.image_encoder(batch["pixel_values"]))
        image = image * batch["has_image"].unsqueeze(1)
        features = torch.cat([text, image, batch["has_image"].unsqueeze(1)], dim=1)
        return self.classifier(features)


class CLIPLinearClassifier(nn.Module):
    def __init__(self, model_name: str, freeze: bool, cached: bool = False):
        super().__init__()
        self.clip = None if cached else CLIPModel.from_pretrained(model_name)
        if self.clip is not None and freeze:
            self.clip.requires_grad_(False)
        dimension = 512 if cached else self.clip.config.projection_dim
        self.classifier = nn.Sequential(
            nn.Linear(dimension * 3 + 2, 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, 2)
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "clip_features" in batch:
            return self.classifier(batch["clip_features"])
        text = nn.functional.normalize(
            self.clip.get_text_features(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]), dim=-1
        )
        image = nn.functional.normalize(self.clip.get_image_features(pixel_values=batch["pixel_values"]), dim=-1)
        similarity = (text * image).sum(dim=1, keepdim=True)
        features = torch.cat([text, image, torch.abs(text - image), similarity, batch["has_image"].unsqueeze(1)], dim=1)
        return self.classifier(features)


class SpotFakeAdapter(nn.Module):
    """Thin adapter around the FaKnow SpotFake implementation."""

    def __init__(self, text_model: str, freeze: bool, repository_root: Path, cached: bool = False):
        super().__init__()
        self.cached = cached
        if cached:
            self.text_fc1 = nn.Linear(768, 2742)
            self.text_fc2 = nn.Linear(2742, 32)
            self.image_fc1 = nn.Linear(4096, 2742)
            self.image_fc2 = nn.Linear(2742, 32)
            self.fusion = nn.Linear(64, 35)
            self.fc = nn.Linear(35, 1)
            self.dropout = nn.Dropout(0.4)
            self.spotfake = None
            return
        faknow_root = repository_root / "benchmarks" / "third_party" / "FaKnow"
        sys.path.insert(0, str(faknow_root))
        from faknow.model.content_based.multi_modal.spotfake import SpotFake

        self.spotfake = SpotFake(
            fine_tune_text_module=not freeze,
            fine_tune_vis_module=not freeze,
            pre_trained_bert_name=text_model,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.cached:
            text = self.dropout(torch.relu(self.text_fc1(batch["spotfake_text_base"])))
            text = self.dropout(torch.relu(self.text_fc2(text)))
            image = self.dropout(torch.relu(self.image_fc1(batch["spotfake_image_base"])))
            image = self.dropout(torch.relu(self.image_fc2(image)))
            fused = self.dropout(torch.relu(self.fusion(self.dropout(torch.cat([text, image], dim=1)))))
            probability = torch.sigmoid(self.fc(fused)).squeeze(-1)
        else:
            probability = self.spotfake.model(
                [batch["input_ids"], batch["attention_mask"]], batch["pixel_values"]
            )
        probability = probability.clamp(1e-6, 1 - 1e-6)
        return torch.stack([torch.log1p(-probability), torch.log(probability)], dim=1)


class CAFEAdapter(nn.Module):
    """FaKnow CAFE architecture with a fixed cross-dataset feature protocol.

    The CAFE release expects 30x200 text features and 512-d image features but
    does not publish its extractor. We use frozen BERT token representations,
    deterministically pool their hidden dimension to 200, and use frozen CLIP
    image projections. The same protocol is used on both datasets.
    """

    def __init__(self, text_model: str, clip_model: str, repository_root: Path, cached: bool = False):
        super().__init__()
        faknow_root = repository_root / "benchmarks" / "third_party" / "FaKnow"
        sys.path.insert(0, str(faknow_root))
        from faknow.model.content_based.multi_modal.cafe import CAFE

        self.text_encoder = None if cached else AutoModel.from_pretrained(text_model).requires_grad_(False)
        self.image_encoder = None if cached else CLIPVisionModelWithProjection.from_pretrained(clip_model).requires_grad_(False)
        self.cafe = CAFE()

    def features(self, batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if "cafe_text" in batch:
            return batch["cafe_text"], batch["cafe_image"]
        with torch.no_grad():
            hidden = self.text_encoder(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            ).last_hidden_state
            hidden = hidden[:, :30]
            if hidden.shape[1] < 30:
                hidden = nn.functional.pad(hidden, (0, 0, 0, 30 - hidden.shape[1]))
            text = nn.functional.adaptive_avg_pool1d(hidden, 200)
            image = self.image_encoder(pixel_values=batch["pixel_values"]).image_embeds
            image = image * batch["has_image"].unsqueeze(1)
        return text, image

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        text, image = self.features(batch)
        return self.cafe(text, image)

    def training_loss(self, batch: Dict[str, torch.Tensor], criterion: nn.Module) -> torch.Tensor:
        text, image = self.features(batch)
        detection = criterion(self.cafe(text, image), batch["labels"])
        if int((batch["labels"] == 1).sum()) < 2:
            return detection
        similarity = self.cafe.similarity_module.calculate_loss(
            {"text": text, "image": image, "label": batch["labels"]}
        )
        return detection + similarity


def build_model(config: RunConfig, repository_root: Path, cached_features: bool = False) -> nn.Module:
    if config.model in {"distilbert_text", "bert_text", "roberta_text"}:
        return BertTextClassifier(config.text_model, config.freeze_encoders)
    if config.model == "resnet50_image":
        return ResNetClassifier(config.freeze_encoders)
    if config.model == "bert_resnet_concat":
        return BertResNetClassifier(config.text_model, config.freeze_encoders)
    if config.model == "debertnext_matched":
        return DeBERTNeXTMatchedClassifier(config.text_model, config.freeze_encoders)
    if config.model == "clip_linear":
        return CLIPLinearClassifier(config.clip_model, config.freeze_encoders, cached=cached_features)
    if config.model == "spotfake_faknow":
        return SpotFakeAdapter(config.text_model, config.freeze_encoders, repository_root, cached=cached_features)
    if config.model == "cafe_faknow":
        return CAFEAdapter(config.text_model, config.clip_model, repository_root, cached=cached_features)
    raise ValueError(config.model)


def move_batch(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def metrics(labels: List[int], predictions: List[int], scores: List[float]) -> Dict[str, float]:
    result = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "fake_f1": float(f1_score(labels, predictions, pos_label=1, zero_division=0)),
        "real_f1": float(f1_score(labels, predictions, pos_label=0, zero_division=0)),
        "precision_fake": float(precision_score(labels, predictions, pos_label=1, zero_division=0)),
        "recall_fake": float(recall_score(labels, predictions, pos_label=1, zero_division=0)),
    }
    try:
        result["roc_auc"] = float(roc_auc_score(labels, scores))
    except ValueError:
        result["roc_auc"] = float("nan")
    return result


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    description: str = "Evaluating",
) -> tuple[Dict[str, float], pd.DataFrame]:
    model.eval()
    ids: List[str] = []
    labels: List[int] = []
    predictions: List[int] = []
    scores: List[float] = []
    progress = tqdm(
        loader,
        desc=description,
        unit="batch",
        dynamic_ncols=True,
        leave=True,
    )
    for batch in progress:
        ids.extend(batch["item_ids"])
        moved = move_batch(batch, device)
        logits = model(moved)
        probabilities = torch.softmax(logits, dim=1)[:, 1]
        labels.extend(moved["labels"].cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        scores.extend(probabilities.cpu().tolist())
    output = pd.DataFrame({"item_id": ids, "label_fake": labels, "prediction_fake": predictions, "probability_fake": scores})
    return metrics(labels, predictions, scores), output


def optimizer_for(model: nn.Module, config: RunConfig) -> torch.optim.Optimizer:
    encoder, head = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if any(token in name for token in ("classifier", "fc", "fusion", "cafe")):
            head.append(parameter)
        else:
            encoder.append(parameter)
    groups = []
    if encoder:
        groups.append({"params": encoder, "lr": config.learning_rate})
    if head:
        groups.append({"params": head, "lr": config.head_learning_rate})
    return torch.optim.AdamW(groups, weight_decay=config.weight_decay)


def train(config: RunConfig) -> Dict[str, object]:
    set_seed(config.seed)
    repository_root = Path(__file__).resolve().parents[2]
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")

    cached_models = {"clip_linear", "spotfake_faknow", "cafe_faknow"}
    use_cached_features = False
    cache_root = Path(config.feature_cache_root) if config.feature_cache_root else None
    if cache_root is not None and config.freeze_encoders and config.model in cached_models:
        required_splits = ("train", "val") if config.validation_only else ("train", "val", "test")
        required = [cache_root / config.dataset / config.model / split / "meta.json" for split in required_splits]
        use_cached_features = all(path.exists() for path in required)
        if config.require_feature_cache and not use_cached_features:
            missing = [str(path) for path in required if not path.exists()]
            raise FileNotFoundError(f"Required frozen feature caches are missing: {missing}")

    if use_cached_features:
        collator = CachedFeatureCollator()
        train_set = CachedFeatureDataset(Path(config.train_csv), cache_root / config.dataset / config.model / "train", config.smoke_limit)
        val_set = CachedFeatureDataset(Path(config.val_csv), cache_root / config.dataset / config.model / "val", config.smoke_limit)
        test_set = None if config.validation_only else CachedFeatureDataset(
            Path(config.test_csv), cache_root / config.dataset / config.model / "test", config.smoke_limit
        )
    else:
        collator = BatchCollator(config.model, config.text_model, config.clip_model, config.max_length)
        train_set = StandardDataset(Path(config.train_csv), config.smoke_limit)
        val_set = StandardDataset(Path(config.val_csv), config.smoke_limit)
        test_set = None if config.validation_only else StandardDataset(Path(config.test_csv), config.smoke_limit)
    generator = torch.Generator().manual_seed(config.seed)
    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collator,
        "persistent_workers": config.num_workers > 0,
    }
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = None if test_set is None else DataLoader(test_set, shuffle=False, **loader_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, repository_root, cached_features=use_cached_features).to(device)
    optimizer = optimizer_for(model, config)
    label_counts = np.bincount(np.asarray(train_set.labels), minlength=2)
    class_weights = len(train_set) / (2.0 * np.maximum(label_counts, 1))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    # SpotFake's large fully connected adapter over cached VGG features can
    # overflow even with BF16. Keep that small trainable head in FP32.
    amp_enabled = (
        config.mixed_precision
        and device.type == "cuda"
        and config.model != "spotfake_faknow"
    )
    amp_dtype = (
        torch.bfloat16
        if amp_enabled and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled and amp_dtype == torch.float16,
    )

    best_metric = -math.inf
    best_epoch = -1
    epochs_without_improvement = 0
    history: List[Dict[str, object]] = []
    started = time.time()
    checkpoint = output_dir / "best_model.pt"
    last_checkpoint = output_dir / "last_checkpoint.pt"
    start_epoch = 0
    if config.auto_resume and last_checkpoint.exists():
        resume = torch.load(last_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        if scaler.is_enabled():
            scaler.load_state_dict(resume.get("scaler", {}))
        start_epoch = int(resume["epoch"]) + 1
        best_metric = float(resume.get("best_metric", -math.inf))
        best_epoch = int(resume.get("best_epoch", -1))
        epochs_without_improvement = int(resume.get("epochs_without_improvement", 0))
        history = list(resume.get("history", []))
        torch.set_rng_state(resume["torch_rng_state"].cpu())
        np.random.set_state(resume["numpy_rng_state"])
        random.setstate(resume["python_rng_state"])
        if device.type == "cuda" and resume.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in resume["cuda_rng_state_all"]]
            )
        print(f"Resuming from epoch {start_epoch + 1}: {last_checkpoint}", flush=True)

    for epoch in range(start_epoch, config.epochs):
        model.train()
        total_loss = 0.0
        seen = 0
        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{config.epochs} - Training",
            unit="batch",
            dynamic_ncols=True,
            leave=True,
        )
        for batch_index, batch in enumerate(progress, start=1):
            moved = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                if hasattr(model, "training_loss"):
                    loss = model.training_loss(moved, criterion)
                else:
                    logits = model(moved)
                    loss = criterion(logits, moved["labels"])
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss for {config.dataset}/{config.model}, "
                    f"seed={config.seed}, epoch={epoch + 1}, batch={batch_index}"
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            batch_size = int(moved["labels"].shape[0])
            total_loss += float(loss.detach()) * batch_size
            seen += batch_size
            if batch_index == 1 or batch_index % 25 == 0:
                progress.set_postfix(
                    loss=f"{total_loss / max(seen, 1):.4f}",
                    refresh=False,
                )
        val_metrics, _ = evaluate(
            model,
            val_loader,
            device,
            description=f"Epoch {epoch + 1}/{config.epochs} - Validation",
        )
        record = {"epoch": epoch + 1, "train_loss": total_loss / max(seen, 1), "val": val_metrics}
        history.append(record)
        print(json.dumps(record), flush=True)
        should_stop = False
        if val_metrics["macro_f1"] > best_metric + 1e-4:
            best_metric = val_metrics["macro_f1"]
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                should_stop = True
        temporary = last_checkpoint.with_suffix(".pt.tmp")
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
                "history": history,
                "torch_rng_state": torch.get_rng_state(),
                "numpy_rng_state": np.random.get_state(),
                "python_rng_state": random.getstate(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            temporary,
        )
        temporary.replace(last_checkpoint)
        if should_stop:
            break

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    validation_metrics, validation_predictions = evaluate(
        model,
        val_loader,
        device,
        description="Final validation",
    )
    validation_predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    if config.validation_only:
        test_metrics = {}
    else:
        assert test_loader is not None
        test_metrics, test_predictions = evaluate(
            model,
            test_loader,
            device,
            description="Held-out test",
        )
        test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    result = {
        "dataset": config.dataset,
        "model": config.model,
        "seed": config.seed,
        "best_epoch": best_epoch,
        "selection_metric": "macro_f1",
        "validation": validation_metrics,
        "test": test_metrics,
        "split_sizes": {"train": len(train_set), "val": len(val_set), "test": None if test_set is None else len(test_set)},
        "class_weights": class_weights.tolist(),
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "elapsed_seconds": time.time() - started,
        "frozen_feature_cache_used": use_cached_features,
        "history": history,
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    if config.smoke_limit:
        checkpoint.unlink(missing_ok=True)
        last_checkpoint.unlink(missing_ok=True)
    return result


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["fakeddit", "checked"], required=True)
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--prepared-root", type=Path, default=Path("benchmarks/prepared"))
    parser.add_argument("--output-root", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--text-model", default=None)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--freeze-encoders", action="store_true")
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--smoke-limit", type=int, default=None)
    parser.add_argument("--validation-only", action="store_true", help="Do not evaluate the held-out test split")
    parser.add_argument("--feature-cache-root", type=Path, default=None)
    parser.add_argument("--require-feature-cache", action="store_true")
    parser.add_argument("--no-auto-resume", action="store_true")
    args = parser.parse_args()
    if args.text_model:
        text_model = args.text_model
    elif args.model == "distilbert_text":
        text_model = (
            "distilbert-base-uncased"
            if args.dataset == "fakeddit"
            else "distilbert-base-multilingual-cased"
        )
    elif args.model == "roberta_text":
        text_model = "roberta-base" if args.dataset == "fakeddit" else "hfl/chinese-roberta-wwm-ext"
    elif args.model == "debertnext_matched":
        text_model = "microsoft/deberta-v3-base" if args.dataset == "fakeddit" else "microsoft/mdeberta-v3-base"
    else:
        text_model = "bert-base-uncased" if args.dataset == "fakeddit" else "bert-base-chinese"
    data_root = args.prepared_root / args.dataset
    output = args.output_root / args.dataset / args.model / f"seed_{args.seed}"
    return RunConfig(
        dataset=args.dataset,
        model=args.model,
        seed=args.seed,
        train_csv=str(data_root / "train.csv"),
        val_csv=str(data_root / "val.csv"),
        test_csv=str(data_root / "test.csv"),
        output_dir=str(output),
        text_model=text_model,
        clip_model=args.clip_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        head_learning_rate=args.head_learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        max_length=args.max_length,
        num_workers=args.num_workers,
        freeze_encoders=args.freeze_encoders,
        mixed_precision=not args.no_mixed_precision,
        smoke_limit=args.smoke_limit,
        validation_only=args.validation_only,
        feature_cache_root=str(args.feature_cache_root.resolve()) if args.feature_cache_root else None,
        require_feature_cache=args.require_feature_cache,
        auto_resume=not args.no_auto_resume,
    )


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), indent=2, allow_nan=True))
