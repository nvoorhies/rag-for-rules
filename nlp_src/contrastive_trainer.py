#!/usr/bin/env python3
"""
Contrastive learning trainer for fine-tuning embedding models for RAG.
Uses QA pairs to create positive and negative examples for training.
"""

import os
import json
import logging
import argparse
import random
import time
import datetime
import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from tqdm.auto import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("contrastive_trainer")

class ContrastiveQADataset(Dataset):
    """Dataset for contrastive learning with QA pairs."""
    
    def __init__(self, 
                 qa_pairs: List[Dict[str, Any]],
                 srd_data: Dict[str, Any],
                 is_train: bool = True):
        """
        Initialize the contrastive dataset.
        
        Args:
            qa_pairs: List of QA pairs with 'question' and 'rules' fields
            srd_data: Processed SRD data with 'rules' field
            is_train: Whether this is a training dataset (affects example generation)
        """
        self.qa_pairs = qa_pairs
        self.srd_sections = srd_data['rules']
        self.is_train = is_train
        
        # Create a mapping from rule ID to section
        self.rule_id_to_section = {}
        for section in self.srd_sections:
            section_id = ' > '.join(section['path'] + [section['title']])
            self.rule_id_to_section[section_id] = section
        
        # Generate training examples
        self.examples = self._generate_examples()
        
    def _generate_examples(self) -> List[InputExample]:
        """Generate training examples from QA pairs."""
        examples = []
        
        for qa_pair in self.qa_pairs:
            question = qa_pair.get('question', '')
            if not question:
                continue
                
            # Get the rule ID from the QA pair
            rule_ids = qa_pair.get('rules', [])
            if not rule_ids or not isinstance(rule_ids, list):
                continue
                
            rule_id = rule_ids[0]
            
            # Find the corresponding section
            if rule_id not in self.rule_id_to_section:
                continue
                
            section = self.rule_id_to_section[rule_id]
            
            # Create positive example
            section_text = self._format_section_text(section)
            examples.append(InputExample(texts=[question, section_text], label=1.0))
            
            # Create negative examples (only for training)
            if self.is_train:
                # Get random sections that are not the correct one
                negative_sections = random.sample([s for s in self.srd_sections 
                                                 if ' > '.join(s['path'] + [s['title']]) != rule_id], 
                                                 k=1)  # Just one negative for balance
                
                for neg_section in negative_sections:
                    neg_text = self._format_section_text(neg_section)
                    examples.append(InputExample(texts=[question, neg_text], label=0.0))
        
        return examples
    
    def _format_section_text(self, section: Dict[str, Any]) -> str:
        """Format section text for embedding."""
        return f"{' > '.join(section['path'] + [section['title']])}\n{section['text']}"
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> InputExample:
        return self.examples[idx]

def load_qa_pairs(qa_file: str) -> List[Dict[str, Any]]:
    """Load QA pairs from a JSON file."""
    with open(qa_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_srd_data(srd_file: str) -> Dict[str, Any]:
    """Load SRD data from a JSON file."""
    with open(srd_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def split_data(data: List[Any], test_ratio: float = 0.1) -> Tuple[List[Any], List[Any]]:
    """Split data into training and test sets."""
    random.shuffle(data)
    split_idx = int(len(data) * (1 - test_ratio))
    return data[:split_idx], data[split_idx:]

def train_model(model: SentenceTransformer,
                train_dataset: Dataset,
                eval_dataset: Dataset,
                output_dir: str,
                batch_size: int = 128,
                epochs: int = 1,
                warmup_steps: int = 100,
                checkpoint_steps: int = 10) -> SentenceTransformer:
    """
    Train the model using contrastive learning.
    
    Args:
        model: SentenceTransformer model to train
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        output_dir: Directory to save checkpoints
        batch_size: Batch size for training
        epochs: Number of epochs to train
        warmup_steps: Number of warmup steps
        checkpoint_steps: Save checkpoint every N steps
        
    Returns:
        Trained model
    """
    # Create data loaders
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    train_dataloader.collate_fn = model.smart_batching_collate

    # Create loss function
    train_loss = losses.ContrastiveLoss(model=model)
    
    # Create evaluator
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(eval_dataset)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize variables for checkpoint management
    last_checkpoint_path = None
    best_score = -1.0
    
    # Train the model
    logger.info(f"Starting training with {len(train_dataset)} examples")
    logger.info(f"Evaluation set has {len(eval_dataset)} examples")
    
    # Training loop
    global_step = 0
    for epoch in range(epochs):
        model.train()
        print(f"Training epoch {epoch+1}/{epochs}")
        train_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        print("Training iterator: ", train_iterator)
        for batch_idx, batch in enumerate(train_iterator):
            global_step += 1
            
            # Train on batch
            model.train()
            print("Batch: ", batch)
            train_loss(batch, labels=[1,0] * batch_size)
            
            # Evaluate and save checkpoint every checkpoint_steps
            if global_step % checkpoint_steps == 0:
                # Evaluate
                model.eval()
                score = evaluator(model)
                
                # Generate timestamp for checkpoint
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_path = os.path.join(output_dir, f"checkpoint_{timestamp}.ckpt")
                
                # Save model state dict
                torch.save(model.state_dict(), checkpoint_path)
                
                # Remove previous checkpoint if it exists
                if last_checkpoint_path and os.path.exists(last_checkpoint_path):
                    os.remove(last_checkpoint_path)
                
                last_checkpoint_path = checkpoint_path
                
                # Update best score
                if score > best_score:
                    best_score = score
                    best_model_path = os.path.join(output_dir, "best_model.ckpt")
                    torch.save(model.state_dict(), best_model_path)
                
                # Log progress
                logger.info(f"Step {global_step}: Evaluation score = {score:.4f}, Checkpoint saved to {checkpoint_path}")
                train_iterator.set_postfix({"eval_score": f"{score:.4f}"})
    
    # Final evaluation
    model.eval()
    final_score = evaluator(model)
    logger.info(f"Final evaluation score: {final_score:.4f}")
    
    # Save final model
    final_path = os.path.join(output_dir, "final_model.ckpt")
    torch.save(model.state_dict(), final_path)
    logger.info(f"Final model saved to {final_path}")
    
    return model

def main():
    """Main entry point for command-line execution."""
    parser = argparse.ArgumentParser(description="Contrastive Learning Trainer for RAG")
    
    parser.add_argument('--qa-pairs', '-q', required=True, 
                        help='Path to QA pairs JSON file')
    parser.add_argument('--srd', '-s', required=True, 
                        help='Path to processed SRD JSON file')
    parser.add_argument('--model', '-m', default='all-MiniLM-L6-v2', 
                        help='Base SentenceTransformer model name')
    parser.add_argument('--checkpoint', '-c', 
                        help='Path to initial model checkpoint (optional)')
    parser.add_argument('--output-dir', '-o', default='model_checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--batch-size', '-b', type=int, default=128,
                        help='Training batch size')
    parser.add_argument('--epochs', '-e', type=int, default=1,
                        help='Number of training epochs')
    parser.add_argument('--test-ratio', '-t', type=float, default=0.1,
                        help='Ratio of data to use for evaluation')
    parser.add_argument('--checkpoint-steps', type=int, default=10,
                        help='Save checkpoint every N steps')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load data
    logger.info(f"Loading QA pairs from {args.qa_pairs}")
    qa_pairs = load_qa_pairs(args.qa_pairs)
    
    logger.info(f"Loading SRD data from {args.srd}")
    srd_data = load_srd_data(args.srd)
    
    # Split data into train and test sets
    train_qa, eval_qa = split_data(qa_pairs, args.test_ratio)
    logger.info(f"Split data into {len(train_qa)} training and {len(eval_qa)} evaluation examples")
    
    # Create datasets
    train_dataset = ContrastiveQADataset(train_qa, srd_data, is_train=True)
    eval_dataset = ContrastiveQADataset(eval_qa, srd_data, is_train=False)
    
    # Initialize model
    logger.info(f"Initializing model: {args.model}")
    model = SentenceTransformer(args.model)
    
    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        state_dict = torch.load(args.checkpoint)
        model.load_state_dict(state_dict)
    
    # Train model
    logger.info("Starting training")
    train_model(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        checkpoint_steps=args.checkpoint_steps
    )
    
    logger.info("Training complete")

if __name__ == "__main__":
    main()
