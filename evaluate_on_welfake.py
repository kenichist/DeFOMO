import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from tqdm import tqdm
import numpy as np

class WELFakeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
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

def load_model(model_path='fine_tuned_distilbert'):
    """Load the fine-tuned model"""
    print(f"Loading model from {model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully on {device}")
    return model, tokenizer, device

def load_welfake_data(file_path='dataset/WELFake_Dataset.csv', max_samples=None):
    """Load WELFake dataset"""
    print(f"\nLoading WELFake dataset from {file_path}...")

    df = pd.read_csv(file_path)

    # Limit to first N samples if specified
    if max_samples is not None:
        df = df.head(max_samples)
        print(f"Limited to first {max_samples} samples")

    # Remove rows with missing data
    df = df.dropna(subset=['title', 'text', 'label'])

    # Combine title and text
    df['combined_text'] = df['title'].astype(str) + " " + df['text'].astype(str)

    # Note: WELFake uses 0 for real, 1 for fake
    # Our model uses 0 for fake, 1 for real
    # So we need to invert the labels
    df['label'] = 1 - df['label']

    print(f"Total samples: {len(df)}")
    print(f"Fake news (label=0): {len(df[df['label'] == 0])}")
    print(f"Real news (label=1): {len(df[df['label'] == 1])}")

    return df

def evaluate_model(model, data_loader, device):
    """Evaluate the model on the dataset"""
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    print("\nRunning predictions...")
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

def print_metrics(y_true, y_pred, y_probs):
    """Print evaluation metrics"""
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)

    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Precision, Recall, F1-Score
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary'
    )
    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Fake    Real")
    print(f"Actual Fake   {cm[0][0]:5d}   {cm[0][1]:5d}")
    print(f"       Real   {cm[1][0]:5d}   {cm[1][1]:5d}")

    # Classification Report
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Fake', 'Real']))

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

def save_results(df, predictions, probabilities, output_file='welfake_predictions.csv'):
    """Save predictions to CSV"""
    results_df = df.copy()

    # Add predictions and probabilities
    results_df['predicted_label'] = predictions
    results_df['prob_fake'] = probabilities[:, 0]
    results_df['prob_real'] = probabilities[:, 1]
    results_df['confidence'] = np.max(probabilities, axis=1)
    results_df['correct'] = (predictions == results_df['label'].values)

    # Map labels to readable format
    results_df['actual'] = results_df['label'].map({0: 'Fake', 1: 'Real'})
    results_df['predicted'] = results_df['predicted_label'].map({0: 'Fake', 1: 'Real'})

    # Reorder columns for better readability
    columns = ['title', 'text', 'actual', 'predicted', 'correct',
               'confidence', 'prob_fake', 'prob_real']
    results_df = results_df[columns]

    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

    # Show some example misclassifications
    print("\n" + "="*70)
    print("SAMPLE MISCLASSIFICATIONS")
    print("="*70)

    misclassified = results_df[~results_df['correct']].head(5)
    for idx, row in misclassified.iterrows():
        print(f"\nTitle: {row['title'][:100]}...")
        print(f"Actual: {row['actual']} | Predicted: {row['predicted']}")
        print(f"Confidence: {row['confidence']:.2%}")
        print("-" * 70)

def main():
    # Configuration
    MODEL_PATH = 'fine_tuned_distilbert'
    DATASET_PATH = 'dataset/WELFake_Dataset.csv'
    BATCH_SIZE = 32
    MAX_LEN = 512
    MAX_SAMPLES = 5000  # Test on first 5000 samples only

    # Load model
    model, tokenizer, device = load_model(MODEL_PATH)

    # Load dataset
    df = load_welfake_data(DATASET_PATH, max_samples=MAX_SAMPLES)

    # Create dataset and dataloader
    print("\nPreparing data loader...")
    dataset = WELFakeDataset(
        df['combined_text'].values,
        df['label'].values,
        tokenizer,
        max_len=MAX_LEN
    )

    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Evaluate
    predictions, true_labels, probabilities = evaluate_model(model, data_loader, device)

    # Print metrics
    metrics = print_metrics(true_labels, predictions, probabilities)

    # Save results
    save_results(df, predictions, probabilities)

    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()
