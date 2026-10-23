"""
Configuration file for fake news detection system.
"""

import os
from pathlib import Path


class Config:
    """Base configuration."""

    # Project paths
    ROOT_DIR = Path(__file__).parent
    DATA_DIR = ROOT_DIR / 'dataset'
    CHECKPOINT_DIR = ROOT_DIR / 'checkpoints'
    LOG_DIR = ROOT_DIR / 'logs'
    CACHE_DIR = ROOT_DIR / 'cache'
    RESULTS_DIR = ROOT_DIR / 'results'

    # Model configuration
    MODEL_TYPE = 'hybrid'  # 'bert' or 'hybrid'
    BERT_MODEL = 'bert-base-uncased'
    MAX_LENGTH = 512
    DROPOUT = 0.3
    FREEZE_BERT = False

    # Training configuration
    BATCH_SIZE = 16
    NUM_EPOCHS = 10
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 500
    GRADIENT_CLIP = 1.0

    # Data configuration
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
    NUM_WORKERS = 4

    # GPU configuration
    USE_GPU = True
    GPU_ID = 0

    # Logging configuration
    LOG_EVERY = 10
    SAVE_EVERY = 1

    # Early stopping
    EARLY_STOPPING = True
    PATIENCE = 3

    # Seed for reproducibility
    SEED = 42

    @classmethod
    def create_directories(cls):
        """Create necessary directories."""
        for dir_path in [
            cls.DATA_DIR,
            cls.CHECKPOINT_DIR,
            cls.LOG_DIR,
            cls.CACHE_DIR,
            cls.RESULTS_DIR
        ]:
            os.makedirs(dir_path, exist_ok=True)

    @classmethod
    def to_dict(cls):
        """Convert configuration to dictionary."""
        return {
            key: value
            for key, value in cls.__dict__.items()
            if not key.startswith('_') and not callable(value)
        }


class TrainingConfig(Config):
    """Configuration for training."""

    # Use larger batch size if GPU memory allows
    BATCH_SIZE = 16

    # More epochs for better convergence
    NUM_EPOCHS = 15

    # Enable early stopping
    EARLY_STOPPING = True
    PATIENCE = 5


class EvaluationConfig(Config):
    """Configuration for evaluation."""

    # Larger batch size for evaluation (no gradients)
    BATCH_SIZE = 32

    # Save all outputs
    SAVE_PREDICTIONS = True
    PLOT_CONFUSION = True
    PLOT_ROC = True


class InferenceConfig(Config):
    """Configuration for inference."""

    # Single sample inference
    BATCH_SIZE = 1

    # Fast inference
    USE_GPU = True


# Default configuration
DEFAULT_CONFIG = Config
