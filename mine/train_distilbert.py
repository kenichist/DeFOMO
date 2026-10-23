import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import numpy as np

class Config:
    MAX_LEN = 512
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    MODEL_NAME = "distilbert-base-uncased"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42

torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data():
    print("Loading dataset...")

    fake_df = pd.read_csv('dataset/Fake.csv')
    true_df = pd.read_csv('dataset/True.csv')

    fake_df['label'] = 0
    true_df['label'] = 1

    df = pd.concat([fake_df, true_df], ignore_index=True)

    df['combined_text'] = df['title'] + " " + df['text']

    df = df.dropna(subset=['combined_text', 'label'])

    df = df.sample(frac=1, random_state=Config.SEED).reset_index(drop=True)

    print(f"Total samples: {len(df)}")
    print(f"Fake news: {len(df[df['label'] == 0])}")
    print(f"True news: {len(df[df['label'] == 1])}")

    return df

def create_data_loaders(df, tokenizer):
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['combined_text'].values,
        df['label'].values,
        test_size=0.2,
        random_state=Config.SEED,
        stratify=df['label'].values
    )

    print(f"\nTrain size: {len(train_texts)}")
    print(f"Validation size: {len(val_texts)}")

    train_dataset = NewsDataset(train_texts, train_labels, tokenizer, Config.MAX_LEN)
    val_dataset = NewsDataset(val_texts, val_labels, tokenizer, Config.MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    return train_loader, val_loader

def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    losses = []
    correct_predictions = 0
    total_predictions = 0

    progress_bar = tqdm(data_loader, desc="Training")

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        preds = torch.argmax(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        total_predictions += labels.size(0)

        losses.append(loss.item())
        progress_bar.set_postfix({'loss': loss.item()})

    return correct_predictions.double() / total_predictions, np.mean(losses)

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
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
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

def main():
    print(f"Using device: {Config.DEVICE}")

    df = load_data()

    print("\nLoading tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained(Config.MODEL_NAME)

    train_loader, val_loader = create_data_loaders(df, tokenizer)

    print("\nLoading model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        Config.MODEL_NAME,
        num_labels=2
    )
    model = model.to(Config.DEVICE)

    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    total_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    best_accuracy = 0

    print("\n" + "="*50)
    print("Starting Training")
    print("="*50)

    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")
        print("-" * 50)

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
            print(f"Saving best model with accuracy: {best_accuracy:.4f}")
            torch.save(model.state_dict(), 'best_model.pt')
            model.save_pretrained('fine_tuned_distilbert')
            tokenizer.save_pretrained('fine_tuned_distilbert')

    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"Best Validation Accuracy: {best_accuracy:.4f}")

    model.load_state_dict(torch.load('best_model.pt'))
    val_acc, val_loss, val_preds, val_labels = eval_model(
        model, val_loader, Config.DEVICE
    )

    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds, target_names=['Fake', 'True']))

if __name__ == "__main__":
    main()
