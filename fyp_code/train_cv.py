"""
Training script with K-Fold Cross-Validation for BERT-based fake news detection.
Addresses overfitting by training and evaluating on multiple data splits.
"""

import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import KFold
from contextlib import nullcontext

from models import (
    BERTClassifier,
    BERTTokenizerWrapper,
    HybridFakeNewsDetector,
    initialize_model,
    initialize_hybrid_model
)
from data import PolitiFactDataset, create_sample_data
from utils.metrics import compute_metrics, log_metrics
from utils.training_utils import (
    set_seed,
    save_checkpoint,
    load_checkpoint,
    EarlyStopping
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train fake news detection model with K-Fold CV')

    # Model arguments
    parser.add_argument('--model_type', type=str, default='hybrid',
                        choices=['bert', 'hybrid'],
                        help='Model type to train')
    parser.add_argument('--bert_model', type=str, default='bert-base-uncased',
                        help='Pre-trained BERT model name')
    parser.add_argument('--max_length', type=int, default=256,
                        help='Maximum sequence length (default reduced for lower VRAM usage)')
    parser.add_argument('--freeze_bert', action='store_true',
                        help='Freeze BERT parameters')

    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./dataset',
                        help='Directory containing data')
    parser.add_argument('--data_file', type=str, default='train.csv',
                        help='Data file name (will combine train/val for CV)')
    parser.add_argument('--create_sample_data', action='store_true',
                        help='Create sample data for testing')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples to create if using sample data')
    parser.add_argument('--cache_dir', type=str, default='./cache',
                        help='Directory to cache preprocessed data')

    # Cross-validation arguments
    parser.add_argument('--k_folds', type=int, default=5,
                        help='Number of folds for cross-validation')
    parser.add_argument('--fold', type=int, default=None,
                        help='Train specific fold only (0 to k_folds-1). If None, train all folds')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Training batch size (default reduced for lower VRAM usage)')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs per fold')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=500,
                        help='Number of warmup steps')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                        help='Gradient clipping value')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout probability')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help='Number of steps to accumulate gradients before optimizer update')

    # GPU arguments
    parser.add_argument('--use_gpu', action='store_true', default=True,
                        help='Use GPU if available')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--no_amp', dest='mixed_precision', action='store_false',
                        help='Disable automatic mixed precision (AMP) training')
    parser.add_argument('--no_gradient_checkpointing', dest='gradient_checkpointing', action='store_false',
                        help='Disable gradient checkpointing on the BERT encoder')
    parser.set_defaults(mixed_precision=True, gradient_checkpointing=True)

    # Checkpoint arguments
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_cv',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_every', type=int, default=1,
                        help='Save checkpoint every N epochs')

    # Logging arguments
    parser.add_argument('--log_dir', type=str, default='./logs_cv',
                        help='Directory for tensorboard logs')
    parser.add_argument('--log_every', type=int, default=10,
                        help='Log metrics every N steps')

    # Early stopping
    parser.add_argument('--early_stopping', action='store_true',
                        help='Use early stopping')
    parser.add_argument('--patience', type=int, default=3,
                        help='Early stopping patience')

    # Seed
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    return parser.parse_args()


def train_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    gradient_clip,
    writer,
    global_step,
    log_every,
    gradient_accumulation_steps,
    amp_enabled,
    scaler
):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []

    optimizer.zero_grad()
    dataloader_length = len(dataloader)

    def autocast_context():
        if device.type == 'cuda' and amp_enabled:
            return torch.cuda.amp.autocast()
        return nullcontext()

    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, batch in enumerate(pbar):
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

        with autocast_context():
            if isinstance(model, HybridFakeNewsDetector):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    graph_stats=graph_stats
                )
                logits = outputs['logits']
            else:
                logits, _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )

            loss = criterion(logits, labels)

        loss_value = loss.item()
        total_loss += loss_value

        loss = loss / gradient_accumulation_steps

        scaler.scale(loss).backward()

        step_now = ((batch_idx + 1) % gradient_accumulation_steps == 0) or ((batch_idx + 1) == dataloader_length)
        if step_now:
            if gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Track metrics
        predictions = torch.argmax(logits, dim=1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Update progress bar
        pbar.set_postfix({'loss': loss_value})

        # Log to tensorboard
        if writer and (batch_idx % log_every == 0):
            writer.add_scalar('train/loss', loss_value, global_step)
            global_step += 1

    # Compute epoch metrics
    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(
        np.array(all_predictions),
        np.array(all_labels)
    )
    metrics['loss'] = avg_loss

    return metrics, global_step


def validate(model, dataloader, criterion, device, amp_enabled):
    """Validate model."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    all_probabilities = []

    def autocast_context():
        if device.type == 'cuda' and amp_enabled:
            return torch.cuda.amp.autocast()
        return nullcontext()

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
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

            # Compute loss
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # Track metrics
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probs.float().cpu().numpy())

            pbar.set_postfix({'loss': loss.item()})

    # Compute metrics
    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(
        np.array(all_predictions),
        np.array(all_labels),
        np.array(all_probabilities)
    )
    metrics['loss'] = avg_loss

    return metrics


def train_fold(
    fold_idx,
    train_dataset,
    val_dataset,
    args,
    device,
    tokenizer
):
    """Train a single fold."""
    print(f"\n{'='*80}")
    print(f"Training Fold {fold_idx + 1}/{args.k_folds}")
    print(f"{'='*80}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Initialize model for this fold
    print(f"Initializing {args.model_type} model...")
    if args.model_type == 'bert':
        model = initialize_model(
            device=device,
            pretrained_model=args.bert_model,
            dropout=args.dropout,
            freeze_bert=args.freeze_bert,
            gradient_checkpointing=args.gradient_checkpointing
        )
    else:
        model = initialize_hybrid_model(
            device=device,
            bert_model=args.bert_model,
            freeze_bert=args.freeze_bert,
            gradient_checkpointing=args.gradient_checkpointing
        )

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    num_training_steps = len(train_loader) * args.num_epochs
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.1,
        total_iters=num_training_steps
    )
    amp_enabled = getattr(args, 'amp_enabled', False)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    # Tensorboard writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, f"fold_{fold_idx}_{timestamp}")
    writer = SummaryWriter(log_dir)

    # Early stopping
    early_stopping = None
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=args.patience)

    # Training loop
    global_step = 0
    best_val_acc = 0.0
    fold_results = {
        'fold': fold_idx,
        'epochs': [],
        'best_val_acc': 0.0,
        'best_val_f1': 0.0,
        'best_epoch': 0
    }

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")

        # Train
        train_metrics, global_step = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            gradient_clip=args.gradient_clip,
            writer=writer,
            global_step=global_step,
            log_every=args.log_every,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            amp_enabled=amp_enabled,
            scaler=scaler
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, amp_enabled)

        # Log metrics
        print(f"\nTrain - {log_metrics(train_metrics)}")
        print(f"Val   - {log_metrics(val_metrics)}")

        writer.add_scalar('epoch/train_loss', train_metrics['loss'], epoch)
        writer.add_scalar('epoch/train_acc', train_metrics['accuracy'], epoch)
        writer.add_scalar('epoch/val_loss', val_metrics['loss'], epoch)
        writer.add_scalar('epoch/val_acc', val_metrics['accuracy'], epoch)
        writer.add_scalar('epoch/val_f1', val_metrics['f1'], epoch)

        # Store epoch results
        fold_results['epochs'].append({
            'epoch': epoch,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        })

        # Update best metrics
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            fold_results['best_val_acc'] = best_val_acc
            fold_results['best_val_f1'] = val_metrics['f1']
            fold_results['best_epoch'] = epoch

            # Save best model for this fold
            fold_checkpoint_dir = os.path.join(args.checkpoint_dir, f'fold_{fold_idx}')
            os.makedirs(fold_checkpoint_dir, exist_ok=True)

            checkpoint_path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                checkpoint_dir=fold_checkpoint_dir,
                is_best=True,
                global_step=global_step,
                best_val_acc=best_val_acc
            )
            print(f"Saved best model for fold {fold_idx}: {checkpoint_path}")

        # Early stopping
        if early_stopping:
            early_stopping(val_metrics['loss'])
            if early_stopping.should_stop:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        # Step scheduler
        scheduler.step()

    writer.close()

    return fold_results


def main():
    """Main training function with K-Fold cross-validation."""
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    # Create sample data if requested
    if args.create_sample_data:
        print("Creating sample data...")
        create_sample_data(args.data_dir, args.num_samples)

    # Setup device
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"Using GPU: {torch.cuda.get_device_name(args.gpu_id)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(args.gpu_id).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")

    args.amp_enabled = args.mixed_precision and device.type == 'cuda'
    print(f"Mixed precision (AMP): {'enabled' if args.amp_enabled else 'disabled'}")
    print(f"Gradient checkpointing: {'enabled' if args.gradient_checkpointing else 'disabled'}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")

    # Initialize tokenizer
    print(f"Loading tokenizer: {args.bert_model}")
    tokenizer = BERTTokenizerWrapper(args.bert_model, args.max_length)

    # Load full dataset
    print("Loading dataset...")
    try:
        from models.graph_network import GraphFeatureExtractor
    except ImportError:
        from models import GraphFeatureExtractor

    graph_extractor = GraphFeatureExtractor() if args.model_type == 'hybrid' else None

    # Try to load and combine train and val files for cross-validation
    train_file = os.path.join(args.data_dir, 'train.csv')
    val_file = os.path.join(args.data_dir, 'val.csv')

    if os.path.exists(train_file) and os.path.exists(val_file):
        # Load both files and combine
        import pandas as pd
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)
        combined_df = pd.concat([train_df, val_df], ignore_index=True)

        # Save combined file temporarily
        combined_file = os.path.join(args.cache_dir, 'combined_for_cv.csv')
        combined_df.to_csv(combined_file, index=False)

        full_dataset = PolitiFactDataset(
            combined_file,
            tokenizer,
            args.max_length,
            args.model_type == 'hybrid',
            graph_extractor,
            args.cache_dir
        )
    else:
        # Load single data file
        data_file = os.path.join(args.data_dir, args.data_file)
        full_dataset = PolitiFactDataset(
            data_file,
            tokenizer,
            args.max_length,
            args.model_type == 'hybrid',
            graph_extractor,
            args.cache_dir
        )

    print(f"Total samples: {len(full_dataset)}")

    # Initialize K-Fold
    kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)

    # Save configuration
    config = vars(args)
    config_file = os.path.join(args.checkpoint_dir, 'config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    # Track results across folds
    all_fold_results = []

    # Determine which folds to train
    if args.fold is not None:
        folds_to_train = [args.fold]
        print(f"\nTraining only fold {args.fold}")
    else:
        folds_to_train = range(args.k_folds)
        print(f"\nTraining all {args.k_folds} folds")

    # Train each fold
    dataset_indices = np.arange(len(full_dataset))

    for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(dataset_indices)):
        if fold_idx not in folds_to_train:
            continue

        # Create subsets
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)

        # Train this fold
        fold_results = train_fold(
            fold_idx=fold_idx,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            args=args,
            device=device,
            tokenizer=tokenizer
        )

        all_fold_results.append(fold_results)

    # Print summary
    print(f"\n{'='*80}")
    print("Cross-Validation Summary")
    print(f"{'='*80}")

    val_accuracies = [r['best_val_acc'] for r in all_fold_results]
    val_f1s = [r['best_val_f1'] for r in all_fold_results]

    for i, results in enumerate(all_fold_results):
        print(f"Fold {results['fold']}: Val Acc = {results['best_val_acc']:.4f}, "
              f"Val F1 = {results['best_val_f1']:.4f} (Epoch {results['best_epoch']})")

    print(f"\nMean Val Accuracy: {np.mean(val_accuracies):.4f} ± {np.std(val_accuracies):.4f}")
    print(f"Mean Val F1 Score: {np.mean(val_f1s):.4f} ± {np.std(val_f1s):.4f}")

    # Save summary
    summary = {
        'k_folds': args.k_folds,
        'fold_results': all_fold_results,
        'mean_val_acc': float(np.mean(val_accuracies)),
        'std_val_acc': float(np.std(val_accuracies)),
        'mean_val_f1': float(np.mean(val_f1s)),
        'std_val_f1': float(np.std(val_f1s))
    }

    summary_file = os.path.join(args.checkpoint_dir, 'cv_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nCross-validation completed! Summary saved to: {summary_file}")


if __name__ == '__main__':
    main()
