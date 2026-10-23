"""
Training utilities including checkpointing, early stopping, and seed setting.
"""

import os
import random
import torch
import numpy as np
from typing import Dict, Optional


def set_seed(seed: int):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_dir: str,
    is_best: bool = False,
    global_step: int = 0,
    best_val_acc: float = 0.0
) -> str:
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch
        metrics: Validation metrics
        checkpoint_dir: Directory to save checkpoint
        is_best: Whether this is the best model so far
        global_step: Global training step
        best_val_acc: Best validation accuracy so far

    Returns:
        Path to saved checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'best_val_acc': best_val_acc
    }

    # Save regular checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)

    # Save best model
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pt')
        torch.save(checkpoint, best_path)

    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    model,
    optimizer=None
) -> Dict:
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state into

    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"Epoch: {checkpoint['epoch']}, Metrics: {checkpoint.get('metrics', {})}")

    return checkpoint


class EarlyStopping:
    """
    Early stopping to stop training when validation loss stops improving.
    """

    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float):
        """
        Check if training should stop.

        Args:
            score: Current validation metric (loss or accuracy)
        """
        if self.best_score is None:
            self.best_score = score
            return

        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


class AverageMeter:
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def count_parameters(model) -> Dict[str, int]:
    """
    Count model parameters.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params
    }


def get_learning_rate(optimizer) -> float:
    """
    Get current learning rate from optimizer.

    Args:
        optimizer: PyTorch optimizer

    Returns:
        Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def freeze_layers(model, freeze_embeddings: bool = True, freeze_encoder_layers: int = 0):
    """
    Freeze specific layers of the model.

    Args:
        model: Model to freeze layers in
        freeze_embeddings: Whether to freeze embedding layers
        freeze_encoder_layers: Number of encoder layers to freeze (from bottom)
    """
    # Freeze embeddings
    if freeze_embeddings and hasattr(model, 'bert'):
        for param in model.bert.embeddings.parameters():
            param.requires_grad = False

    # Freeze encoder layers
    if freeze_encoder_layers > 0 and hasattr(model, 'bert'):
        for layer in model.bert.encoder.layer[:freeze_encoder_layers]:
            for param in layer.parameters():
                param.requires_grad = False


def unfreeze_layers(model):
    """
    Unfreeze all model parameters.

    Args:
        model: Model to unfreeze
    """
    for param in model.parameters():
        param.requires_grad = True


def get_gpu_memory_usage() -> Dict[str, float]:
    """
    Get current GPU memory usage.

    Returns:
        Dictionary with memory statistics in GB
    """
    if not torch.cuda.is_available():
        return {'allocated': 0.0, 'cached': 0.0, 'total': 0.0}

    return {
        'allocated': torch.cuda.memory_allocated() / 1e9,
        'cached': torch.cuda.memory_reserved() / 1e9,
        'total': torch.cuda.get_device_properties(0).total_memory / 1e9
    }


def print_gpu_memory():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        memory = get_gpu_memory_usage()
        print(f"GPU Memory - Allocated: {memory['allocated']:.2f}GB, "
              f"Cached: {memory['cached']:.2f}GB, "
              f"Total: {memory['total']:.2f}GB")
