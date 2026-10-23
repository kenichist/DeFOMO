"""
Prediction script for fake news detection on individual articles.
"""

import argparse
import torch
import time
import json
from typing import Dict, List
from contextlib import nullcontext

from models import (
    BERTTokenizerWrapper,
    HybridFakeNewsDetector,
    initialize_model,
    initialize_hybrid_model
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Predict fake news for individual articles')

    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--text', type=str, default=None,
                        help='Text to analyze')
    parser.add_argument('--text_file', type=str, default=None,
                        help='File containing text to analyze')
    parser.add_argument('--model_type', type=str, default='hybrid',
                        choices=['bert', 'hybrid'],
                        help='Model type')
    parser.add_argument('--bert_model', type=str, default='bert-base-uncased',
                        help='Pre-trained BERT model name')
    parser.add_argument('--max_length', type=int, default=256,
                        help='Maximum sequence length (default reduced for lower VRAM usage)')
    parser.add_argument('--use_gpu', action='store_true', default=True,
                        help='Use GPU if available')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--output_json', type=str, default=None,
                        help='Save results to JSON file')

    return parser.parse_args()


def predict_article(
    text: str,
    model,
    tokenizer: BERTTokenizerWrapper,
    device: torch.device,
    model_type: str = 'hybrid',
    amp_enabled: bool = False
) -> Dict:
    """
    Predict whether an article is fake news.

    Args:
        text: Article text
        model: Trained model
        tokenizer: BERT tokenizer
        device: Device to use
        model_type: Type of model

    Returns:
        Dictionary with prediction results
    """
    model.eval()

    # Tokenize text
    encoded = tokenizer.encode_text(text)
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    token_type_ids = encoded.get('token_type_ids')
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)

    # Measure inference time
    start_time = time.time()

    def autocast_context():
        if device.type == 'cuda' and amp_enabled:
            return torch.cuda.amp.autocast()
        return nullcontext()

    with torch.no_grad():
        if isinstance(model, HybridFakeNewsDetector):
            # For hybrid model, use zero graph features if not provided
            graph_stats = torch.zeros(1, 15).to(device)
            with autocast_context():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    graph_stats=graph_stats
                )
                logits = outputs['logits']
                probs = outputs['probabilities']
        else:
            with autocast_context():
                logits, _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                probs = torch.softmax(logits, dim=1)

    inference_time = time.time() - start_time

    # Get prediction
    probs = probs.float()
    prediction = torch.argmax(probs, dim=1).item()
    confidence = probs[0][prediction].item()

    # Prepare results
    results = {
        'text': text[:200] + '...' if len(text) > 200 else text,
        'prediction': 'FAKE NEWS' if prediction == 1 else 'REAL NEWS',
        'prediction_label': int(prediction),
        'confidence': float(confidence),
        'probabilities': {
            'real': float(probs[0][0].item()),
            'fake': float(probs[0][1].item())
        },
        'inference_time_seconds': float(inference_time)
    }

    return results


def main():
    """Main prediction function."""
    args = parse_args()

    # Get text to analyze
    if args.text:
        text = args.text
    elif args.text_file:
        with open(args.text_file, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        print("Error: Please provide either --text or --text_file")
        return

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

    print("Model loaded successfully\n")

    # Make prediction
    print("Analyzing article...")
    results = predict_article(
        text,
        model,
        tokenizer,
        device,
        args.model_type,
        amp_enabled=amp_enabled
    )

    # Print results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"\nArticle Preview: {results['text']}")
    print(f"\nPrediction: {results['prediction']}")
    print(f"Confidence: {results['confidence']:.2%}")
    print(f"\nProbabilities:")
    print(f"  Real News: {results['probabilities']['real']:.2%}")
    print(f"  Fake News: {results['probabilities']['fake']:.2%}")
    print(f"\nInference Time: {results['inference_time_seconds']:.4f} seconds")
    print("="*60)

    # Save to JSON if requested
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == '__main__':
    main()
