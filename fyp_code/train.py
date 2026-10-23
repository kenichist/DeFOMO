"""
Training script for BERT-based fake news detection with GPU support.
Supports both BERT-only and hybrid (BERT + Graph) models.
"""

import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from pathlib import Path
from datetime import datetime
from contextlib import nullcontext

from models import (
    BERTClassifier,
    BERTTokenizerWrapper,
    HybridFakeNewsDetector,
    initialize_model,
    initialize_hybrid_model
)
from data import PolitiFactDataModule, create_sample_data
from utils.metrics import compute_metrics, log_metrics
from utils.training_utils import (
    set_seed,
    save_checkpoint,
    load_checkpoint,
    EarlyStopping
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train fake news detection model')

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
    parser.add_argument('--create_sample_data', action='store_true',
                        help='Create sample data for testing')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples to create if using sample data')
    parser.add_argument('--cache_dir', type=str, default='./cache',
                        help='Directory to cache preprocessed data')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Training batch size (default reduced for lower VRAM usage)')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs')
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
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_every', type=int, default=1,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume training from checkpoint')

    # Logging arguments
    parser.add_argument('--log_dir', type=str, default='./logs',
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

        # Scale loss for gradient accumulation
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


def main():
    """Main training function."""
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

    amp_enabled = args.mixed_precision and device.type == 'cuda'
    print(f"Mixed precision (AMP): {'enabled' if amp_enabled else 'disabled'}")
    print(f"Gradient checkpointing: {'enabled' if args.gradient_checkpointing else 'disabled'}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")

    # Initialize tokenizer
    print(f"Loading tokenizer: {args.bert_model}")
    tokenizer = BERTTokenizerWrapper(args.bert_model, args.max_length)

    # Setup data
    print("Setting up data...")
    data_module = PolitiFactDataModule(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
        include_graph=(args.model_type == 'hybrid'),
        cache_dir=args.cache_dir
    )
    data_module.setup('fit')

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    print(f"Train samples: {len(data_module.train_dataset)}")
    print(f"Val samples: {len(data_module.val_dataset)}")

    # Initialize model
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
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    # Tensorboard writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, f"{args.model_type}_{timestamp}")
    writer = SummaryWriter(log_dir)

    # Early stopping
    early_stopping = None
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=args.patience)

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    best_val_acc = 0.0

    if args.resume_from:
        checkpoint = load_checkpoint(args.resume_from, model, optimizer)
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        print(f"Resumed from epoch {start_epoch}")

    # Save configuration
    config = vars(args)
    with open(os.path.join(args.checkpoint_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Training loop
    print("\nStarting training...")
    for epoch in range(start_epoch, args.num_epochs):
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

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or val_metrics['accuracy'] > best_val_acc:
            is_best = val_metrics['accuracy'] > best_val_acc
            best_val_acc = max(val_metrics['accuracy'], best_val_acc)

            checkpoint_path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                checkpoint_dir=args.checkpoint_dir,
                is_best=is_best,
                global_step=global_step,
                best_val_acc=best_val_acc
            )
            print(f"Saved checkpoint: {checkpoint_path}")

        # Early stopping
        if early_stopping:
            early_stopping(val_metrics['loss'])
            if early_stopping.should_stop:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        # Step scheduler
        scheduler.step()

    print("\nTraining completed!")
    writer.close()

    # Save final model
    final_path = os.path.join(args.checkpoint_dir, 'final_model.pt')
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to: {final_path}")


if __name__ == '__main__':
    main()
