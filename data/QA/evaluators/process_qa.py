#!/usr/bin/env python3
"""
Process QA pairs to create training data for contrastive learning.
Generates triplets of (question, positive example, negative example) for training.
"""

import os
import json
import random
import argparse
import logging
from typing import Dict, Any, List, Callable, Optional
import sys

# Add the nlp_src directory to the path so we can import augmentation_functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from nlp_src.augmentation_functions import (
    augment_with_path_references_scope,
    augment_with_references_scope,
    augment_with_title
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("process_qa")

def load_qa_pairs(qa_file: str) -> List[Dict[str, Any]]:
    """Load QA pairs from a JSON file."""
    with open(qa_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_srd_data(srd_file: str) -> Dict[str, Any]:
    """Load SRD data from a JSON file."""
    with open(srd_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_augmented_text_lookup(srd_data: Dict[str, Any], 
                                augmentation_func: Callable[[Dict[str, Any]], str]) -> Dict[str, str]:
    """
    Create a lookup table from rule path to augmented text.
    
    Args:
        srd_data: Processed SRD data
        augmentation_func: Function to augment section text
        
    Returns:
        Dictionary mapping rule paths to augmented text
    """
    lookup = {}
    
    for section in srd_data['rules']:
        # Create the path key
        path_key = ' > '.join(section['path'] + [section['title']])
        
        # Apply augmentation function
        augmented_text = augmentation_func(section)
        
        # Store in lookup
        lookup[path_key] = augmented_text
    
    return lookup

def generate_training_triplets(qa_pairs: List[Dict[str, Any]], 
                              text_lookup: Dict[str, str]) -> List[Dict[str, str]]:
    """
    Generate training triplets (question, positive example, negative example).
    
    Args:
        qa_pairs: List of QA pairs with 'question' and 'rules' fields
        text_lookup: Lookup table from rule path to augmented text
        
    Returns:
        List of triplets for training
    """
    triplets = []
    all_paths = list(text_lookup.keys())
    
    for qa_pair in qa_pairs:
        question = qa_pair.get('question', '')
        if not question:
            continue
            
        # Get the rule ID from the QA pair
        rule_ids = qa_pair.get('rules', [])
        if not rule_ids or not isinstance(rule_ids, list):
            continue
            
        rule_id = rule_ids[0]
        
        # Find the corresponding augmented text
        if rule_id not in text_lookup:
            logger.warning(f"Rule ID {rule_id} not found in lookup table")
            continue
            
        pos_text = text_lookup[rule_id]
        
        # Get a random negative example
        available_neg_paths = [p for p in all_paths if p != rule_id]
        if not available_neg_paths:
            logger.warning(f"No negative examples available for {rule_id}")
            continue
            
        neg_path = random.choice(available_neg_paths)
        neg_text = text_lookup[neg_path]
        
        # Create triplet
        triplet = {
            "question": question,
            "pos": pos_text,
            "neg": neg_text,
            "rule_id": rule_id,
            "neg_rule_id": neg_path
        }
        
        triplets.append(triplet)
    
    return triplets

def get_augmentation_function(name: str) -> Callable[[Dict[str, Any]], str]:
    """Get the augmentation function by name."""
    functions = {
        "path_references_scope": augment_with_path_references_scope,
        "references_scope": augment_with_references_scope,
        "title": augment_with_title,
        "none": lambda section: section['text']
    }
    
    if name not in functions:
        logger.warning(f"Unknown augmentation function: {name}, using 'none'")
        return functions["none"]
    
    return functions[name]

def main():
    """Main entry point for command-line execution."""
    parser = argparse.ArgumentParser(description="Process QA pairs for contrastive learning")
    
    parser.add_argument('--qa-pairs', '-q', required=True, 
                        help='Path to QA pairs JSON file')
    parser.add_argument('--srd', '-s', required=True, 
                        help='Path to processed SRD JSON file')
    parser.add_argument('--output', '-o', required=True,
                        help='Path to output JSON file')
    parser.add_argument('--augmentation', '-a', default='path_references_scope',
                        choices=['path_references_scope', 'references_scope', 'title', 'none'],
                        help='Augmentation function to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Load data
    logger.info(f"Loading QA pairs from {args.qa_pairs}")
    qa_pairs = load_qa_pairs(args.qa_pairs)
    
    logger.info(f"Loading SRD data from {args.srd}")
    srd_data = load_srd_data(args.srd)
    
    # Get augmentation function
    augmentation_func = get_augmentation_function(args.augmentation)
    logger.info(f"Using augmentation function: {args.augmentation}")
    
    # Create lookup table
    logger.info("Creating augmented text lookup table")
    text_lookup = create_augmented_text_lookup(srd_data, augmentation_func)
    
    # Generate training triplets
    logger.info("Generating training triplets")
    triplets = generate_training_triplets(qa_pairs, text_lookup)
    
    # Save output
    logger.info(f"Generated {len(triplets)} training triplets")
    logger.info(f"Saving to {args.output}")
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(triplets, f, indent=2)
    
    logger.info("Processing complete")

if __name__ == "__main__":
    main()
