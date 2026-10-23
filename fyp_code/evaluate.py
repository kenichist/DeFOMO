"""
Evaluation script for fake news detection model.
"""

import os
import argparse
import json
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from contextlib import nullcontext

from models import (
    BERTClassifier,
    BERTTokenizerWrapper,
    HybridFakeNewsDetector,
    initialize_model,
    initialize_hybrid_model
)
from data import PolitiFactDataModule
from utils.metrics import (
    compute_metrics,
    compute_class_metrics,
    log_metrics,
    print_confusion_matrix
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate fake news detection model')

    # Model arguments
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default='hybrid',
                        choices=['bert', 'hybrid'],
                        help='Model type')
    parser.add_argument('--bert_model', type=str, default='bert-base-uncased',
                        help='Pre-trained BERT model name')
    parser.add_argument('--max_length', type=int, default=256,
                        help='Maximum sequence length (default reduced for lower VRAM usage)')

    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./dataset',
                        help='Directory containing test data')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Evaluation batch size (default reduced for lower VRAM usage)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    # GPU arguments
    parser.add_argument('--use_gpu', action='store_true', default=True,
                        help='Use GPU if available')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU device ID')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save predictions to file')
    parser.add_argument('--plot_confusion', action='store_true',
                        help='Plot confusion matrix')
    parser.add_argument('--plot_roc', action='store_true',
                        help='Plot ROC curve')

    return parser.parse_args()


def evaluate_model(model, dataloader, device, amp_enabled):
    """
    Evaluate model on test data.

    Args:
        model: Model to evaluate
        dataloader: Test dataloader
        device: Device to use

    Returns:
        predictions, labels, probabilities
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []

    def autocast_context():
        if device.type == 'cuda' and amp_enabled:
            return torch.cuda.amp.autocast()
        return nullcontext()

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Evaluating')
        for batch in pbar:
            # Move data to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            token_type_ids = batch.get('token_type_ids')
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            graph_stats = batch.get('graph_stats')
            if graph_stats is not None:
                graph_stats = graph_stats.to(device)

            # Forward pass
            with autocast_context():
                if isinstance(model, HybridFakeNewsDetector):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        graph_stats=graph_stats
                    )
                    logits = outputs['logits']
                    probs = outputs['probabilities']
                else:
                    logits, _ = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids
                    )
                    probs = torch.softmax(logits, dim=1)

            # Track metrics
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probs.float().cpu().numpy())

    return (
        np.array(all_predictions),
        np.array(all_labels),
        np.array(all_probabilities)
    )


def plot_confusion_matrix(labels, predictions, output_path):
    """
    Plot and save confusion matrix.

    Args:
        labels: True labels
        predictions: Predicted labels
        output_path: Path to save plot
    """
    cm = confusion_matrix(labels, predictions)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Real', 'Fake'],
        yticklabels=['Real', 'Fake']
    )
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Confusion matrix saved to {output_path}")


def plot_roc_curve(labels, probabilities, output_path):
    """
    Plot and save ROC curve.

    Args:
        labels: True labels
        probabilities: Prediction probabilities
        output_path: Path to save plot
    """
    # Get probabilities for positive class (fake news)
    if probabilities.ndim == 2:
        pos_probs = probabilities[:, 1]
    else:
        pos_probs = probabilities

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(labels, pos_probs)
    roc_auc = auc(fpr, tpr)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr,
        color='darkorange',
        lw=2,
        label=f'ROC curve (AUC = {roc_auc:.4f})'
    )
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"ROC curve saved to {output_path}")


def save_predictions_to_file(predictions, labels, probabilities, output_path):
    """
    Save predictions to JSON file.

    Args:
        predictions: Predicted labels
        labels: True labels
        probabilities: Prediction probabilities
        output_path: Path to save file
    """
    results = []
    for i, (pred, label, prob) in enumerate(zip(predictions, labels, probabilities)):
        results.append({
            'index': i,
            'true_label': int(label),
            'predicted_label': int(pred),
            'real_probability': float(prob[0]) if prob.ndim > 0 else float(1 - prob),
            'fake_probability': float(prob[1]) if prob.ndim > 0 else float(prob),
            'correct': bool(pred == label)
        })

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Predictions saved to {output_path}")


def main():
    """Main evaluation function."""
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup device
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"Using GPU: {torch.cuda.get_device_name(args.gpu_id)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    amp_enabled = device.type == 'cuda'
    print(f"Mixed precision (AMP): {'enabled' if amp_enabled else 'disabled'}")

    # Initialize tokenizer
    print(f"Loading tokenizer: {args.bert_model}")
    tokenizer = BERTTokenizerWrapper(args.bert_model, args.max_length)

    # Setup data
    print("Setting up test data...")
    data_module = PolitiFactDataModule(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
        include_graph=(args.model_type == 'hybrid')
    )
    data_module.setup('test')

    if data_module.test_dataset is None:
        print("Error: No test dataset found!")
        return

    test_loader = data_module.test_dataloader()
    print(f"Test samples: {len(data_module.test_dataset)}")

    # Initialize model
    print(f"Initializing {args.model_type} model...")
    if args.model_type == 'bert':
        model = initialize_model(
            device=device,
            pretrained_model=args.bert_model
        )
    else:
        model = initialize_hybrid_model(
            device=device,
            bert_model=args.bert_model
        )

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    print("Model loaded successfully")

    # Evaluate
    print("\nEvaluating model...")
    predictions, labels, probabilities = evaluate_model(model, test_loader, device, amp_enabled)

    # Compute metrics
    metrics = compute_metrics(predictions, labels, probabilities)
    class_metrics = compute_class_metrics(predictions, labels)

    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nOverall Metrics:")
    print(log_metrics(metrics))

    print(f"\nPer-Class Metrics:")
    print(f"Real News - {log_metrics(class_metrics['real_news'])}")
    print(f"Fake News - {log_metrics(class_metrics['fake_news'])}")

    print_confusion_matrix(predictions, labels)
    print("="*60)

    # Save results
    results = {
        'overall_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else int(v)
                           for k, v in metrics.items()},
        'class_metrics': class_metrics,
        'num_samples': len(labels),
        'model_type': args.model_type,
        'checkpoint': args.checkpoint_path
    }

    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Save predictions
    if args.save_predictions:
        pred_path = os.path.join(args.output_dir, 'predictions.json')
        save_predictions_to_file(predictions, labels, probabilities, pred_path)

    # Plot confusion matrix
    if args.plot_confusion:
        cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
        plot_confusion_matrix(labels, predictions, cm_path)

    # Plot ROC curve
    if args.plot_roc:
        roc_path = os.path.join(args.output_dir, 'roc_curve.png')
        plot_roc_curve(labels, probabilities, roc_path)

    print("\nEvaluation completed!")


if __name__ == '__main__':
    main()
