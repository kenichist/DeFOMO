import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    DistilBertTokenizer,
    DistilBertModel,
    ViTModel,
    AutoImageProcessor,
    get_linear_schedule_with_warmup,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm


# ==========================
# Config
# ==========================

class Config:
    MAX_LEN = 128
    BATCH_SIZE = 8
    EPOCHS = 3
    LEARNING_RATE = 2e-5

    TEXT_MODEL_NAME = "distilbert-base-uncased"
    VIT_MODEL_NAME = "google/vit-base-patch16-224-in21k"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42

    # paths
    TRAIN_TSV = os.path.join("data", "train.tsv")
    IMAGE_ROOT = os.path.join("data", "")                  # folder with .jpg images

    # fusion method: "concat", "gated", "hadamard", "modattn"
    FUSION_TYPE = "concat"

    # where to save best model
    SAVE_PATH = "best_multimodal_vit.pt"


torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)


# ==========================
# Dataset
# ==========================

class FakedditDataset(Dataset):
    """
    Expects a DataFrame with at least:
    - 'id'              (for image filename)
    - '2_way_label'     (0/1)
    - 'clean_title'     (string)
    - optional 'text'   (body text)
    If image file is missing / cannot be opened:
        - returns a zero image tensor
        - sets has_image = 0 (so model can fall back to text-only)
    """

    def __init__(self, df, tokenizer, image_processor, image_root, max_len):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_root = image_root
        self.max_len = max_len

        # figure out image size for zero placeholder
        size = image_processor.size
        if isinstance(size, dict):
            self.height = size.get("height", 224)
            self.width = size.get("width", 224)
        else:
            self.height = self.width = size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --- text ---
        title = str(row.get("clean_title", ""))
        body = str(row.get("text", ""))
        if body == "nan":
            body = ""
        text = (title + " " + body).strip()

        enc = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # --- image ---
        img_id = str(row["id"])
        img_path = os.path.join(self.image_root, img_id + ".jpg")

        has_image = 1.0
        try:
            image = Image.open(img_path).convert("RGB")
            img_enc = self.image_processor(image, return_tensors="pt")
            pixel_values = img_enc["pixel_values"].squeeze(0)
        except Exception:
            # image missing or unreadable → use zero tensor and mark has_image = 0
            has_image = 0.0
            pixel_values = torch.zeros(
                3, self.height, self.width, dtype=torch.float32
            )

        # --- label ---
        label = int(row["2_way_label"])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "has_image": torch.tensor(has_image, dtype=torch.float32),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def load_fakeddit_split(tsv_path, image_root):
    """
    Load TSV. DOES NOT filter out rows without images.
    Missing images are handled in the Dataset with text-only fallback.
    """
    print(f"Loading TSV: {tsv_path}")
    df = pd.read_csv(tsv_path, sep="\t")

    # drop rows without 2-way label or title
    df = df.dropna(subset=["2_way_label", "clean_title"])

    df["id"] = df["id"].astype(str)
    df["2_way_label"] = df["2_way_label"].astype(int)

    print(f"Total rows (text available): {len(df)}")
    # You can optionally print how many actually have images:
    img_paths = df["id"].apply(lambda x: os.path.join(image_root, f"{x}.jpg"))
    num_with_img = sum(os.path.exists(p) for p in img_paths)
    print(f"Rows with existing image files: {num_with_img}")

    return df


def create_data_loaders_fakeddit(train_tsv, image_root, tokenizer, image_processor):
    df = load_fakeddit_split(train_tsv, image_root)

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=Config.SEED,
        stratify=df["2_way_label"].values,
    )

    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

    train_dataset = FakedditDataset(
        train_df, tokenizer, image_processor, image_root, Config.MAX_LEN
    )
    val_dataset = FakedditDataset(
        val_df, tokenizer, image_processor, image_root, Config.MAX_LEN
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader


# ==========================
# Multimodal Model (DistilBERT + ViT)
# ==========================

class MultiModalViTModel(nn.Module):
    """
    DistilBERT (text) + ViT (image) with configurable fusion.
    fusion_type: "concat", "gated", "hadamard", "modattn"

    If has_image == 0 for a sample, we zero out its image embedding.
    For "gated" fusion we also force the gate to rely on text only.
    """

    def __init__(
        self,
        text_model,
        vit_model,
        d_proj=256,
        num_classes=2,
        fusion_type="concat",
    ):
        super().__init__()
        self.text_model = text_model
        self.vit_model = vit_model
        self.fusion_type = fusion_type
        self.d_proj = d_proj

        d_t = self.text_model.config.hidden_size
        d_v = self.vit_model.config.hidden_size

        self.proj_t = nn.Linear(d_t, d_proj)
        self.proj_v = nn.Linear(d_v, d_proj)

        if fusion_type == "gated":
            self.gate_fc = nn.Linear(2 * d_proj, 2)

        if fusion_type == "modattn":
            self.mod_attn = nn.MultiheadAttention(
                embed_dim=d_proj, num_heads=4, batch_first=True
            )

        if fusion_type == "concat":
            fusion_dim = 2 * d_proj
        elif fusion_type == "hadamard":
            fusion_dim = 3 * d_proj
        else:  # "gated" or "modattn"
            fusion_dim = d_proj

        self.fc1 = nn.Linear(fusion_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def encode_text(self, input_ids, attention_mask):
        out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        # DistilBERT pooled representation: first token
        t_cls = out.last_hidden_state[:, 0, :]  # [B, d_t]
        return t_cls

    def encode_image(self, pixel_values):
        out = self.vit_model(pixel_values=pixel_values)
        v_cls = out.last_hidden_state[:, 0, :]  # [B, d_v]
        return v_cls

    def fuse(self, t, v, has_image=None):
        """
        t, v: [B, d_proj]
        has_image: [B] or [B,1] floats in {0,1}, or None
        returns fused: [B, fusion_dim]
        """

        if self.fusion_type == "concat":
            return torch.cat([t, v], dim=-1)

        elif self.fusion_type == "gated":
            concat = torch.cat([t, v], dim=-1)  # [B, 2d]
            alpha = torch.softmax(self.gate_fc(concat), dim=-1)  # [B, 2]

            if has_image is not None:
                # if no image: force alpha = [1, 0] (text only)
                mask = has_image.view(-1, 1)  # [B,1]
                forced = torch.cat(
                    [torch.ones_like(mask), torch.zeros_like(mask)], dim=-1
                )  # [B,2] -> [1,0]
                alpha = alpha * mask + forced * (1.0 - mask)

            a_t = alpha[:, 0:1]
            a_v = alpha[:, 1:2]
            return a_t * t + a_v * v

        elif self.fusion_type == "hadamard":
            prod = t * v
            return torch.cat([t, v, prod], dim=-1)

        elif self.fusion_type == "modattn":
            M = torch.stack([t, v], dim=1)  # [B, 2, d]
            M_attn, _ = self.mod_attn(M, M, M)
            return M_attn.mean(dim=1)  # [B, d]

        else:
            raise ValueError(f"Unknown fusion_type {self.fusion_type}")

    def forward(self, input_ids, attention_mask, pixel_values, has_image=None, labels=None):
        # 1) Encode text
        t = self.encode_text(input_ids, attention_mask)  # [B, d_t]
        t = self.proj_t(t)                               # [B, d_proj]

        # 2) Encode image
        v = self.encode_image(pixel_values)              # [B, d_v]
        v = self.proj_v(v)                               # [B, d_proj]

        # 3) Zero out image features where there is no image
        if has_image is not None:
            mask = has_image.view(-1, 1).to(v.device)    # [B, 1]
            v = v * mask                                 # [B, d_proj] (0 if no image)

        # 4) Fuse
        z = self.fuse(t, v, has_image=has_image)         # [B, fusion_dim]

        # 5) Classifier head
        h = self.dropout(self.act(self.fc1(z)))
        logits = self.fc2(h)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )



# ==========================
# Training & Evaluation
# ==========================

def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    losses = []
    correct_predictions = 0
    total_predictions = 0

    progress_bar = tqdm(data_loader, desc="Training")

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        has_image = batch["has_image"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            has_image=has_image,
            labels=labels,
        )

        loss = outputs.loss
        logits = outputs.logits

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        preds = torch.argmax(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        total_predictions += labels.size(0)

        losses.append(loss.item())
        progress_bar.set_postfix({"loss": loss.item()})

    accuracy = correct_predictions.double() / total_predictions
    return accuracy, np.mean(losses)


def eval_model(model, data_loader, device):
    model.eval()
    losses = []
    correct_predictions = 0
    total_predictions = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Evaluating")

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            has_image = batch["has_image"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                has_image=has_image,
                labels=labels,
            )

            loss = outputs.loss
            logits = outputs.logits

            preds = torch.argmax(logits, dim=1)

            correct_predictions += torch.sum(preds == labels)
            total_predictions += labels.size(0)
            losses.append(loss.item())

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct_predictions.double() / total_predictions
    return accuracy, np.mean(losses), all_preds, all_labels


# ==========================
# Main
# ==========================

def main():
    print(f"Using device: {Config.DEVICE}")
    print(f"Fusion type: {Config.FUSION_TYPE}")

    print("\nLoading tokenizer and image processor...")
    tokenizer = DistilBertTokenizer.from_pretrained(Config.TEXT_MODEL_NAME)
    image_processor = AutoImageProcessor.from_pretrained(Config.VIT_MODEL_NAME)

    train_loader, val_loader = create_data_loaders_fakeddit(
        Config.TRAIN_TSV,
        Config.IMAGE_ROOT,
        tokenizer,
        image_processor,
    )

    print("\nLoading encoders...")
    text_model = DistilBertModel.from_pretrained(Config.TEXT_MODEL_NAME)
    vit_model = ViTModel.from_pretrained(Config.VIT_MODEL_NAME)

    # freeze encoders initially
    for p in text_model.parameters():
        p.requires_grad = False
    for p in vit_model.parameters():
        p.requires_grad = False

    model = MultiModalViTModel(
        text_model=text_model,
        vit_model=vit_model,
        d_proj=256,
        num_classes=2,
        fusion_type=Config.FUSION_TYPE,
    ).to(Config.DEVICE)

    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    total_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    best_accuracy = 0.0

    print("\n" + "=" * 60)
    print("Starting Multimodal Training (Text + ViT)")
    print("=" * 60)

    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")
        print("-" * 60)

        train_acc, train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, Config.DEVICE
        )
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

        val_acc, val_loss, val_preds, val_labels = eval_model(
            model, val_loader, Config.DEVICE
        )
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            print(f"Saving best multimodal model with accuracy: {best_accuracy:.4f}")
            torch.save(model.state_dict(), Config.SAVE_PATH)

    print("\nTraining complete.")
    print(f"Best Validation Accuracy: {best_accuracy:.4f}")

    # Reload best model for final report
    model.load_state_dict(torch.load(Config.SAVE_PATH, map_location=Config.DEVICE))
    _, _, val_preds, val_labels = eval_model(model, val_loader, Config.DEVICE)

    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds, target_names=["Fake", "Real"]))


if __name__ == "__main__":
    main()
