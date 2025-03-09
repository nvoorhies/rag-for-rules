#!/usr/bin/env python3

"""
Robust Embedding Cache System for RPG Rules RAG

This module provides utilities for caching embeddings to disk, with support
for cache invalidation based on content hashing and embedding parameters.
"""

import os
import json
import hashlib
import time
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
from pathlib import Path
import pickle
from tqdm.auto import tqdm
import re
from transformers import AutoTokenizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("embedding_cache")

class EmbeddingCache:
    """A robust caching system for embeddings that stores data based on content hash and parameters."""
    
    def __init__(self, cache_dir: str = "embedding_cache", verbose: bool = True):
        """
        Initialize the embedding cache.
        
        Args:
            cache_dir: Directory to store cached embeddings
            verbose: Whether to print cache operations
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.cache_dir / "metadata.json"
        self.verbose = verbose
        self.tokenizer = None
        
        # Load metadata if it exists
        self.metadata = self._load_metadata()
        
        if self.verbose:
            cached_item_count = sum(len(data['embeddings'].keys()) 
                                  for data in self.metadata.values() 
                                  if 'embeddings' in data)
            logger.info(f"Initialized embedding cache at {self.cache_dir}")
            logger.info(f"Cache contains {len(self.metadata)} models with {cached_item_count} total embeddings")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata from disk."""
        if not self.metadata_path.exists():
            return {}
        
        try:
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache metadata: {e}")
            return {}
    
    def _save_metadata(self):
        """Save cache metadata to disk."""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _compute_content_hash(self, content: str) -> str:
        """Compute a hash of the content to use as part of the cache key."""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _generate_model_key(self, model_name: str, params: Dict[str, Any]) -> str:
        """Generate a unique key for an embedding model with specific parameters."""
        # Sort parameters for consistent ordering
        param_str = json.dumps(params, sort_keys=True)
        return f"{re.sub(r'/', '_', model_name)}_{hashlib.md5(param_str.encode('utf-8')).hexdigest()}"
    
    def _get_embedding_path(self, model_key: str, content_hash: str) -> Path:
        """Get the path for a cached embedding file."""
        return self.cache_dir / f"{model_key}_{content_hash}.npy"
    
    def get(self, content: str, model_name: str, params: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Get cached embedding for content with the specified model and parameters.
        
        Args:
            content: The text content to get embeddings for
            model_name: Name of the embedding model
            params: Dictionary of model parameters
            
        Returns:
            Numpy array containing the embedding, or None if not cached
        """
        model_key = self._generate_model_key(model_name, params)
        content_hash = self._compute_content_hash(content)
        
        # Check if model exists in metadata
        if model_key not in self.metadata:
            return None
        
        # Check if this content hash exists for this model
        model_data = self.metadata[model_key]
        if 'embeddings' not in model_data or content_hash not in model_data['embeddings']:
            return None
        
        # Get the embedding file path
        embedding_path = self._get_embedding_path(model_key, content_hash)
        
        # Check if embedding file exists
        if not embedding_path.exists():
            # Remove the stale metadata entry
            model_data['embeddings'].pop(content_hash, None)
            self._save_metadata()
            return None
        
        # Load the embedding
        try:
            embedding = np.load(embedding_path)
            
            if self.verbose:
                logger.debug(f"Cache hit for {model_name} - {content_hash[:8]}")
            
            return embedding
        except Exception as e:
            logger.warning(f"Failed to load cached embedding: {e}")
            return None
    
    def put(self, content: str, embedding: np.ndarray, model_name: str, params: Dict[str, Any]):
        """
        Cache an embedding for the given content.
        
        Args:
            content: The text content that was embedded
            embedding: The generated embedding (numpy array)
            model_name: Name of the embedding model
            params: Dictionary of model parameters
        """
        model_key = self._generate_model_key(model_name, params)
        content_hash = self._compute_content_hash(content)
        
        # Initialize model in metadata if needed
        if model_key not in self.metadata:
            self.metadata[model_key] = {
                'model_name': model_name,
                'params': params,
                'created_at': time.time(),
                'embeddings': {}
            }
        
        # Update or create embedding entry
        if 'embeddings' not in self.metadata[model_key]:
            self.metadata[model_key]['embeddings'] = {}
            
        self.metadata[model_key]['embeddings'][content_hash] = {
            'timestamp': time.time(),
            'size': embedding.shape[0],
            'hash': content_hash
        }
        
        # Save the embedding
        embedding_path = self._get_embedding_path(model_key, content_hash)
        np.save(embedding_path, embedding)
        
        # Update metadata
        self._save_metadata()
        
        if self.verbose:
            logger.debug(f"Cached embedding for {model_name} - {content_hash[:8]}")
    
    def bulk_get(self, contents: List[str], model_name: str, params: Dict[str, Any]) -> Tuple[List[np.ndarray], List[int]]:
        """
        Get cached embeddings for multiple content items.
        
        Args:
            contents: List of text contents to get embeddings for
            model_name: Name of the embedding model
            params: Dictionary of model parameters
            
        Returns:
            Tuple of (cached_embeddings, missing_indices)
                - cached_embeddings: List of numpy arrays for found embeddings
                - missing_indices: List of indices in the original contents that need embedding
        """
        cached_embeddings = []
        missing_indices = []
        
        for i, content in enumerate(contents):
            embedding = self.get(content, model_name, params)
            if embedding is not None:
                cached_embeddings.append(embedding)
            else:
                missing_indices.append(i)
        
        return cached_embeddings, missing_indices
    
    def bulk_put(self, contents: List[str], embeddings: List[np.ndarray], model_name: str, params: Dict[str, Any]):
        """
        Cache embeddings for multiple content items.
        
        Args:
            contents: List of text contents that were embedded
            embeddings: List of generated embeddings (numpy arrays)
            model_name: Name of the embedding model
            params: Dictionary of model parameters
        """
        if len(contents) != len(embeddings):
            raise ValueError(f"Length mismatch: {len(contents)} contents vs {len(embeddings)} embeddings")
        
        for content, embedding in zip(contents, embeddings):
            self.put(content, embedding, model_name, params)
    
    def compute_missing_embeddings(self, contents: List[str], model_name: str, params: Dict[str, Any], 
                                  embed_func) -> Dict[str, np.ndarray]:
        """
        Get all embeddings, computing and caching any missing ones.
        
        Args:
            contents: List of text contents to get embeddings for
            model_name: Name of the embedding model
            params: Dictionary of model parameters
            embed_func: Function that takes a list of texts and returns embeddings
            
        Returns:
            Dictionary mapping content hashes to embeddings
        """
        result = {}
        
        # Try to get cached embeddings
        cached_embeddings, missing_indices = self.bulk_get(contents, model_name, params)
        
        # If all embeddings were cached
        if len(missing_indices) == 0:
            if self.verbose:
                logger.info(f"All {len(contents)} embeddings found in cache")
                
            # Add all cached embeddings to result
            for i, content in enumerate(contents):
                content_hash = self._compute_content_hash(content)
                result[content_hash] = cached_embeddings[i]
            
            return result
        
        # Compute missing embeddings
        if self.verbose:
            logger.info(f"Computing {len(missing_indices)} missing embeddings with {model_name}")
            
        missing_contents = [contents[i] for i in missing_indices]
        
        # Compute embeddings for missing contents
        missing_embeddings = embed_func(missing_contents)
        
        # Ensure we got the right number of embeddings
        if len(missing_embeddings) != len(missing_indices):
            raise ValueError(f"Expected {len(missing_indices)} embeddings, got {len(missing_embeddings)}")
        
        # Cache the newly computed embeddings
        for content, embedding in zip(missing_contents, missing_embeddings):
            content_hash = self._compute_content_hash(content)
            self.put(content, embedding, model_name, params)
            result[content_hash] = embedding
        
        # Add the previously cached embeddings to result
        cached_idx = 0
        for i, content in enumerate(contents):
            if i not in missing_indices:
                content_hash = self._compute_content_hash(content)
                result[content_hash] = cached_embeddings[cached_idx]
                cached_idx += 1
        
        return result
    
    def clear_cache(self, model_name: Optional[str] = None, older_than: Optional[float] = None):
        """
        Clear the cache, optionally filtering by model name and age.
        
        Args:
            model_name: Optional model name to clear cache for
            older_than: Optional timestamp to clear entries older than
        """
        if model_name is None and older_than is None:
            # Clear entire cache
            for file in self.cache_dir.glob("*.npy"):
                file.unlink()
            self.metadata = {}
            self._save_metadata()
            
            if self.verbose:
                logger.info("Cleared entire embedding cache")
                
        elif model_name is not None and older_than is None:
            # Clear cache for specific model
            models_to_clear = []
            for key, data in self.metadata.items():
                if data.get('model_name') == model_name:
                    models_to_clear.append(key)
            
            for model_key in models_to_clear:
                model_data = self.metadata.pop(model_key, {})
                for content_hash in model_data.get('embeddings', {}).keys():
                    embedding_path = self._get_embedding_path(model_key, content_hash)
                    if embedding_path.exists():
                        embedding_path.unlink()
            
            self._save_metadata()
            
            if self.verbose:
                logger.info(f"Cleared cache for model: {model_name}")
                
        elif older_than is not None:
            # Clear cache entries older than timestamp
            for model_key, model_data in list(self.metadata.items()):
                keep_model = True
                
                if 'embeddings' in model_data:
                    for content_hash, embed_info in list(model_data['embeddings'].items()):
                        if embed_info.get('timestamp', 0) < older_than:
                            # Remove old embedding
                            embedding_path = self._get_embedding_path(model_key, content_hash)
                            if embedding_path.exists():
                                embedding_path.unlink()
                            
                            # Remove from metadata
                            model_data['embeddings'].pop(content_hash)
                    
                    # If all embeddings removed, check if we should remove model
                    if not model_data['embeddings'] and (model_name is None or model_data.get('model_name') == model_name):
                        keep_model = False
                
                if not keep_model:
                    self.metadata.pop(model_key)
            
            self._save_metadata()
            
            if self.verbose:
                if model_name:
                    logger.info(f"Cleared cache entries for {model_name} older than {time.ctime(older_than)}")
                else:
                    logger.info(f"Cleared cache entries older than {time.ctime(older_than)}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache."""
        stats = {
            'total_models': len(self.metadata),
            'total_embeddings': 0,
            'models': {},
            'cache_size_bytes': 0
        }
        
        for model_key, model_data in self.metadata.items():
            model_name = model_data.get('model_name', 'unknown')
            
            if model_name not in stats['models']:
                stats['models'][model_name] = {
                    'embedding_count': 0,
                    'size_bytes': 0
                }
            
            # Count embeddings for this model
            embedding_count = len(model_data.get('embeddings', {}))
            stats['models'][model_name]['embedding_count'] += embedding_count
            stats['total_embeddings'] += embedding_count
            
            # Calculate size on disk
            model_size = 0
            for content_hash in model_data.get('embeddings', {}):
                embedding_path = self._get_embedding_path(model_key, content_hash)
                if embedding_path.exists():
                    model_size += embedding_path.stat().st_size
            
            stats['models'][model_name]['size_bytes'] += model_size
            stats['cache_size_bytes'] += model_size
        
        # Convert to human-readable sizes
        stats['cache_size'] = self._format_size(stats['cache_size_bytes'])
        for model_name in stats['models']:
            stats['models'][model_name]['size'] = self._format_size(
                stats['models'][model_name]['size_bytes']
            )
        
        return stats
    
    def _format_size(self, size_bytes: int) -> str:
        """Format a byte size into a human-readable string."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


# Example usage
if __name__ == "__main__":
    import argparse
    from sentence_transformers import SentenceTransformer
    
    parser = argparse.ArgumentParser(description='Embedding Cache Management')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show cache status')
    status_parser.add_argument('--cache-dir', default='embedding_cache', help='Cache directory')
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear the cache')
    clear_parser.add_argument('--cache-dir', default='embedding_cache', help='Cache directory')
    clear_parser.add_argument('--model', help='Optional model name to clear')
    clear_parser.add_argument('--days', type=int, help='Clear entries older than this many days')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run a cache test')
    test_parser.add_argument('--cache-dir', default='embedding_cache', help='Cache directory')
    test_parser.add_argument('--model', default='all-MiniLM-L6-v2', help='Model to test with')
    
    args = parser.parse_args()
    
    # Initialize cache
    cache = EmbeddingCache(args.cache_dir)
    
    if args.command == 'status':
        # Show cache stats
        stats = cache.get_cache_stats()
        print("\n=== Embedding Cache Status ===")
        print(f"Cache Directory: {args.cache_dir}")
        print(f"Total Models: {stats['total_models']}")
        print(f"Total Embeddings: {stats['total_embeddings']}")
        print(f"Total Cache Size: {stats['cache_size']}")
        
        if stats['models']:
            print("\nModels:")
            for model_name, model_stats in stats['models'].items():
                print(f"  - {model_name}: {model_stats['embedding_count']} embeddings, {model_stats['size']}")
    
    elif args.command == 'clear':
        # Clear cache
        if args.days:
            older_than = time.time() - (args.days * 24 * 60 * 60)
            cache.clear_cache(args.model, older_than)
        else:
            cache.clear_cache(args.model)
        
        # Show updated stats
        stats = cache.get_cache_stats()
        print(f"\nCache cleared. New status: {stats['total_embeddings']} embeddings, {stats['cache_size']}")
    
    elif args.command == 'test':
        # Run a simple test
        print("\n=== Running Cache Test ===")
        
        # Initialize model
        print(f"Initializing {args.model}...")
        model = SentenceTransformer(args.model)
        
        # Test texts
        texts = [
            "This is a test sentence for embedding caching.",
            "Another test sentence that's slightly different.",
            "Something completely different to embed.",
            "Let's see how the cache handles this fourth text."
        ]
        
        # Define embedding function
        def embed_func(texts):
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            token_count = tokenizer([text for text in texts], return_tensors='pt', padding=True, truncation=True)
            assert all(token_count['input_ids'].shape[1] <= model.max_seq_length)
            return model.encode(texts)
        
        # Parameters
        params = {
            'max_seq_length': model.max_seq_length,
            'normalize_embeddings': True
        }
        
        # First run - should compute all
        print("\nFirst run - computing all embeddings...")
        start_time = time.time()
        cache.compute_missing_embeddings(texts, args.model, params, embed_func)
        print(f"First run took {time.time() - start_time:.3f} seconds")
        
        # Second run - should use cache
        print("\nSecond run - should use cache...")
        start_time = time.time()
        cache.compute_missing_embeddings(texts, args.model, params, embed_func)
        print(f"Second run took {time.time() - start_time:.3f} seconds")
        
        # Third run with one new text
        print("\nThird run - adding one new text...")
        texts.append("This is a new text that shouldn't be in the cache yet.")
        start_time = time.time()
        cache.compute_missing_embeddings(texts, args.model, params, embed_func)
        print(f"Third run took {time.time() - start_time:.3f} seconds")
        
        # Show final stats
        stats = cache.get_cache_stats()
        print(f"\nTest complete. Cache now has {stats['total_embeddings']} embeddings.")
