"""
Metrics computation for fake news detection.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, Optional


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    probabilities: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        predictions: Predicted labels (0 or 1)
        labels: True labels (0 or 1)
        probabilities: Prediction probabilities (optional, for AUC)

    Returns:
        Dictionary with metrics
    """
    metrics = {}

    # Basic metrics
    metrics['accuracy'] = accuracy_score(labels, predictions)
    metrics['precision'] = precision_score(labels, predictions, zero_division=0)
    metrics['recall'] = recall_score(labels, predictions, zero_division=0)
    metrics['f1'] = f1_score(labels, predictions, zero_division=0)

    # AUC-ROC (if probabilities provided)
    if probabilities is not None:
        try:
            # Use probability of positive class (fake news)
            if probabilities.ndim == 2:
                pos_probs = probabilities[:, 1]
            else:
                pos_probs = probabilities
            metrics['auc_roc'] = roc_auc_score(labels, pos_probs)
        except:
            metrics['auc_roc'] = 0.0

    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)

        # Specificity
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return metrics


def compute_class_metrics(
    predictions: np.ndarray,
    labels: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class metrics.

    Args:
        predictions: Predicted labels
        labels: True labels

    Returns:
        Dictionary with metrics for each class
    """
    report = classification_report(labels, predictions, output_dict=True, zero_division=0)

    class_metrics = {
        'real_news': {
            'precision': report['0']['precision'],
            'recall': report['0']['recall'],
            'f1': report['0']['f1-score'],
            'support': report['0']['support']
        },
        'fake_news': {
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1': report['1']['f1-score'],
            'support': report['1']['support']
        }
    }

    return class_metrics


def log_metrics(metrics: Dict[str, float]) -> str:
    """
    Format metrics for logging.

    Args:
        metrics: Dictionary of metrics

    Returns:
        Formatted string
    """
    lines = []
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"{key}: {value:.4f}")
        else:
            lines.append(f"{key}: {value}")

    return " | ".join(lines)


def print_confusion_matrix(predictions: np.ndarray, labels: np.ndarray):
    """
    Print confusion matrix with labels.

    Args:
        predictions: Predicted labels
        labels: True labels
    """
    cm = confusion_matrix(labels, predictions)

    print("\nConfusion Matrix:")
    print("                Predicted")
    print("              Real  Fake")
    print(f"Actual Real   {cm[0, 0]:4d}  {cm[0, 1]:4d}")
    print(f"       Fake   {cm[1, 0]:4d}  {cm[1, 1]:4d}")


def compute_detection_latency(
    start_times: np.ndarray,
    end_times: np.ndarray
) -> Dict[str, float]:
    """
    Compute detection latency statistics.

    Args:
        start_times: Start timestamps
        end_times: End timestamps

    Returns:
        Latency statistics
    """
    latencies = end_times - start_times

    return {
        'mean_latency': np.mean(latencies),
        'median_latency': np.median(latencies),
        'std_latency': np.std(latencies),
        'min_latency': np.min(latencies),
        'max_latency': np.max(latencies),
        'p95_latency': np.percentile(latencies, 95),
        'p99_latency': np.percentile(latencies, 99)
    }
