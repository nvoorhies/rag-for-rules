#!/usr/bin/env python3
"""
Fine-tuned Hierarchical RAG that uses a checkpoint of a fine-tuned SentenceTransformer model.
"""

import os
import json
import logging
import torch
import numpy as np
from typing import Dict, Any, List, Optional
from augmented_hierarchical_rag import AugmentedHierarchicalRAG
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("finetuned_hierarchical_rag")

class FineTunedHierarchicalRAG(AugmentedHierarchicalRAG):
    """Hierarchical RAG with fine-tuned embedding model support."""
    
    def __init__(self, 
                processed_srd_path: str,
                embeddings_path: Optional[str] = None,
                model_name: str = "all-MiniLM-L6-v2",
                checkpoint_path: str = "AugmentedHierarchicalRAG.ckpt",
                cache_dir: str = "embedding_cache",
                max_seq_length: Optional[int] = None,
                verbose: bool = False):
        """
        Initialize the fine-tuned hierarchical RAG system.
        
        Args:
            processed_srd_path: Path to processed SRD JSON with hierarchical structure
            embeddings_path: Optional path to pre-computed embeddings pickle
            model_name: Name of the sentence transformer model to use
            checkpoint_path: Path to fine-tuned model checkpoint
            cache_dir: Directory for the embedding cache
            max_seq_length: Maximum sequence length for the model
            verbose: Whether to print detailed logs
        """
        self.checkpoint_path = checkpoint_path
        
        # Initialize the base class
        super().__init__(
            processed_srd_path=processed_srd_path,
            embeddings_path=embeddings_path,
            model_name=model_name,
            cache_dir=cache_dir,
            max_seq_length=max_seq_length,
            verbose=verbose
        )
        
        # Load fine-tuned model if available
        self._load_fine_tuned_model()
    
    def _load_fine_tuned_model(self):
        """Load fine-tuned model from checkpoint if available."""
        if os.path.exists(self.checkpoint_path):
            try:
                if self.verbose:
                    logger.info(f"Loading fine-tuned model from {self.checkpoint_path}")
                
                # Load the state dict
                state_dict = torch.load(self.checkpoint_path)
                
                # Apply to the model
                self.model.load_state_dict(state_dict)
                
                if self.verbose:
                    logger.info("Successfully loaded fine-tuned model")
            except Exception as e:
                logger.warning(f"Failed to load fine-tuned model: {e}")
                logger.info("Using base model instead")
        elif self.verbose:
            logger.info(f"Fine-tuned model checkpoint not found at {self.checkpoint_path}, using base model")

# For running as a script
def main():
    """Main entry point for command-line execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tuned Hierarchical RAG for RPG Rules")
    
    # Query command
    parser.add_argument('--srd', '-s', required=True, help='Processed SRD JSON file')
    parser.add_argument('--query', '-q', help='Query text')
    parser.add_argument('--queries-file', '-f', help='File containing queries (one per line or JSON)')
    parser.add_argument('--top-k', '-k', type=int, default=5, help='Number of results to return')
    parser.add_argument('--output', '-o', help='Output file for results')
    parser.add_argument('--embeddings', '-e', help='Path to save/load embeddings')
    parser.add_argument('--cache-dir', default='embedding_cache', help='Embedding cache directory')
    parser.add_argument('--model', default='all-MiniLM-L6-v2', help='Embedding model name')
    parser.add_argument('--checkpoint', default='AugmentedHierarchicalRAG.ckpt', 
                        help='Path to fine-tuned model checkpoint')
    parser.add_argument('--max-seq-length', type=int, help='Maximum sequence length')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    parser.add_argument('--stats', '-S', action='store_true', help='Show cache statistics')
    
    args = parser.parse_args()
    
    # Initialize RAG system
    rag = FineTunedHierarchicalRAG(
        processed_srd_path=args.srd,
        embeddings_path=args.embeddings,
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        cache_dir=args.cache_dir,
        max_seq_length=args.max_seq_length,
        verbose=args.verbose
    )
    
    # Show cache stats if requested
    if args.stats:
        stats = rag.get_cache_stats()
        print("\n=== Embedding Cache Statistics ===")
        print(f"Total Models: {stats['total_models']}")
        print(f"Total Embeddings: {stats['total_embeddings']}")
        print(f"Total Cache Size: {stats['cache_size']}")
        
        for model_name, model_stats in stats['models'].items():
            print(f"  - {model_name}: {model_stats['embedding_count']} embeddings, {model_stats['size']}")
    
    # Process query or queries file
    if args.query:
        # Single query
        result = rag.query(args.query, args.top_k)
        
        # Save to file or print to stdout
        if args.output:
            rag.save_results(result, args.output)
        else:
            print(json.dumps(result, indent=2))
            
    elif args.queries_file:
        # Multiple queries from file
        results = rag.process_queries_from_file(args.queries_file, args.top_k)
        
        # Save to file or print to stdout
        if args.output:
            rag.save_results(results, args.output)
        else:
            print(json.dumps(results, indent=2))
    else:
        parser.error("Either --query or --queries-file must be specified")

if __name__ == "__main__":
    main()
