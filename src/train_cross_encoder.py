#!/usr/bin/env python3
"""
Fine-tune a CrossEncoder model using triplets of questions, positive answers, and negative answers.
"""

import json
import argparse
import logging
from typing import List, Dict, Any, Optional
import os
import torch
from datetime import datetime
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    handlers=[LoggingHandler()]
)
logger = logging.getLogger(__name__)

def load_triplets(triplets_file: str) -> List[Dict[str, str]]:
    """
    Load triplets from a JSON file.
    
    Args:
        triplets_file: Path to the JSON file containing triplets
        
    Returns:
        List of dictionaries with 'question', 'pos', and 'neg' keys
    """
    logger.info(f"Loading triplets from {triplets_file}")
    with open(triplets_file, 'r', encoding='utf-8') as f:
        triplets = json.load(f)
    
    logger.info(f"Loaded {len(triplets)} triplets")
    return triplets

def prepare_training_data(triplets: List[Dict[str, str]]):
    """
    Prepare training data for the CrossEncoder.
    
    Args:
        triplets: List of dictionaries with 'question', 'pos', and 'neg' keys
        
    Returns:
        Tuple of (train_samples, train_labels)
    """
    train_samples = []
    train_labels = []
    
    # For each triplet, create positive and negative pairs
    for triplet in triplets:
        # Positive pair (question, positive answer) with label 1
        train_samples.append([triplet['question'], triplet['pos']])
        train_labels.append(1)
        
        # Negative pair (question, negative answer) with label 0
        train_samples.append([triplet['question'], triplet['neg']])
        train_labels.append(0)
    
    return train_samples, train_labels

def train_cross_encoder(
    triplets_file: str,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    output_dir: Optional[str] = None,
    num_epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    warmup_steps: int = 100,
    max_length: int = 512,
    use_amp: bool = True,
    verbose: bool = False
):
    """
    Train a CrossEncoder model using triplets.
    
    Args:
        triplets_file: Path to the JSON file containing triplets
        model_name: Name or path of the model to fine-tune
        output_dir: Directory to save the model (default: creates a timestamped directory)
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        warmup_steps: Number of warmup steps
        max_length: Maximum sequence length
        use_amp: Whether to use automatic mixed precision
        verbose: Whether to print verbose output
    
    Returns:
        Path to the saved model
    """
    # Set logging level based on verbosity
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load triplets
    triplets = load_triplets(triplets_file)
    
    # Create output directory if not specified
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_name_short = model_name.split('/')[-1]
        output_dir = f"models/cross-encoder-{model_name_short}-{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Model will be saved to {output_dir}")
    
    # Prepare training data
    train_samples, train_labels = prepare_training_data(triplets)
    
    # Initialize the CrossEncoder model
    logger.info(f"Initializing CrossEncoder with model: {model_name}")
    model = CrossEncoder(
        model_name, 
        num_labels=1,
        max_length=max_length,
        device=None  # Use GPU if available, otherwise CPU
    )
    
    # Check if GPU is available
    if torch.cuda.is_available():
        logger.info(f"Training on GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("Training on CPU")
    
    # Train the model
    logger.info("Starting training")
    model.fit(
        train_samples=train_samples,
        train_labels=train_labels,
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': learning_rate},
        output_path=output_dir,
        use_amp=use_amp,
        show_progress_bar=verbose
    )
    
    logger.info(f"Training completed. Model saved to {output_dir}")
    return output_dir

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a CrossEncoder model using triplets")
    parser.add_argument("--triplets", required=True, help="Path to the JSON file containing triplets")
    parser.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L-6-v2", 
                        help="Name or path of the model to fine-tune")
    parser.add_argument("--output", help="Directory to save the model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Number of warmup steps")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    
    args = parser.parse_args()
    
    train_cross_encoder(
        triplets_file=args.triplets,
        model_name=args.model,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_length=args.max_length,
        use_amp=not args.no_amp,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()
