"""
BERT-based Text Classifier for Fake News Detection
Implements a fine-tuned BERT model with GPU support for classifying news articles.
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig
from typing import Dict, Tuple, Optional


class BERTClassifier(nn.Module):
    """
    BERT-based binary classifier for fake news detection.

    Architecture:
    - Pre-trained BERT base model for text encoding
    - Dropout layer for regularization
    - Linear classifier head
    - Supports GPU acceleration
    """

    def __init__(
        self,
        pretrained_model: str = 'bert-base-uncased',
        num_classes: int = 2,
        dropout: float = 0.3,
        freeze_bert: bool = False,
        gradient_checkpointing: bool = False
    ):
        """
        Initialize BERT classifier.

        Args:
            pretrained_model: Name of pretrained BERT model
            num_classes: Number of output classes (default: 2 for fake/real)
            dropout: Dropout probability
            freeze_bert: Whether to freeze BERT parameters
            gradient_checkpointing: Enable gradient checkpointing to reduce memory usage
        """
        super(BERTClassifier, self).__init__()

        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.bert_config = self.bert.config

        if gradient_checkpointing:
            # Gradient checkpointing trades compute for reduced activation memory
            self.bert.gradient_checkpointing_enable()
            self.bert_config.gradient_checkpointing = True
            # HF models require use_cache disabled when checkpointing
            self.bert_config.use_cache = False

        # Freeze BERT parameters if specified
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Classifier head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert_config.hidden_size, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            input_ids: Token IDs from tokenizer
            attention_mask: Attention mask (1 for real tokens, 0 for padding)
            token_type_ids: Segment IDs for sentence pairs (optional)

        Returns:
            logits: Raw prediction scores
            pooled_output: [CLS] token representation from BERT
        """
        # Pass through BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Get [CLS] token representation
        pooled_output = outputs.pooler_output  # [batch_size, hidden_size]

        # Apply dropout
        pooled_output = self.dropout(pooled_output)

        # Classification
        logits = self.classifier(pooled_output)  # [batch_size, num_classes]

        return logits, pooled_output

    def get_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract text features from BERT for hybrid model.

        Args:
            input_ids: Token IDs from tokenizer
            attention_mask: Attention mask
            token_type_ids: Segment IDs (optional)

        Returns:
            Text feature embeddings
        """
        with torch.no_grad():
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            return outputs.pooler_output


class BERTTokenizerWrapper:
    """
    Wrapper for BERT tokenizer with convenience methods.
    """

    def __init__(self, pretrained_model: str = 'bert-base-uncased', max_length: int = 512):
        """
        Initialize tokenizer.

        Args:
            pretrained_model: Name of pretrained BERT model
            max_length: Maximum sequence length
        """
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        self.max_length = max_length

    def encode_text(
        self,
        text: str,
        add_special_tokens: bool = True,
        return_tensors: str = 'pt'
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize and encode text.

        Args:
            text: Input text to encode
            add_special_tokens: Whether to add [CLS] and [SEP] tokens
            return_tensors: Format of returned tensors ('pt' for PyTorch)

        Returns:
            Dictionary with input_ids, attention_mask, token_type_ids
        """
        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=add_special_tokens,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors=return_tensors
        )
        return encoded

    def batch_encode(
        self,
        texts: list,
        add_special_tokens: bool = True,
        return_tensors: str = 'pt'
    ) -> Dict[str, torch.Tensor]:
        """
        Batch encode multiple texts.

        Args:
            texts: List of texts to encode
            add_special_tokens: Whether to add [CLS] and [SEP] tokens
            return_tensors: Format of returned tensors

        Returns:
            Dictionary with batched input_ids, attention_mask, token_type_ids
        """
        encoded = self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=add_special_tokens,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors=return_tensors
        )
        return encoded


def initialize_model(
    device: torch.device,
    pretrained_model: str = 'bert-base-uncased',
    num_classes: int = 2,
    dropout: float = 0.3,
    freeze_bert: bool = False,
    gradient_checkpointing: bool = False
) -> BERTClassifier:
    """
    Initialize BERT classifier and move to device.

    Args:
        device: Device to use (cuda or cpu)
        pretrained_model: Name of pretrained BERT model
        num_classes: Number of output classes
        dropout: Dropout probability
        freeze_bert: Whether to freeze BERT parameters
        gradient_checkpointing: Enable gradient checkpointing to reduce memory usage

    Returns:
        Initialized model on specified device
    """
    model = BERTClassifier(
        pretrained_model=pretrained_model,
        num_classes=num_classes,
        dropout=dropout,
        freeze_bert=freeze_bert,
        gradient_checkpointing=gradient_checkpointing
    )
    model = model.to(device)

    print(f"Model initialized on device: {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    return model
