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
import re
from transformers import AutoTokenizer
import faiss

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("embedding_cache")

class EmbeddingCache:
    """A robust caching system for embeddings that stores data based on content hash and parameters."""
    
    def __init__(self, cache_dir: str = "embedding_cache", verbose: bool = True, 
                 use_faiss: bool = False, chunk_size: int = 384):
        """
        Initialize the embedding cache.
        
        Args:
            cache_dir: Directory to store cached embeddings
            verbose: Whether to print cache operations
            use_faiss: Whether to use FAISS for vector similarity search
            chunk_size: Default chunk size for splitting long texts
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.cache_dir / "metadata.json"
        self.embeddings_path = self.cache_dir / "embeddings.pkl"
        self.verbose = verbose
        self.tokenizer = None
        self.use_faiss = use_faiss
        self.chunk_size = chunk_size
        self.faiss_indices = {}
        self.embeddings_cache = {}
        self.models = {}
        
        # Load metadata if it exists
        self.metadata = self._load_metadata()
        
        # Load embeddings from disk if they exist
        self._load_embeddings_from_disk()
        
        if self.verbose:
            cached_item_count = sum(len(data['embeddings'].keys()) 
                                  for data in self.metadata.values() 
                                  if 'embeddings' in data)
            logger.info(f"Initialized embedding cache at {self.cache_dir}")
            logger.info(f"Cache contains {len(self.metadata)} models with {cached_item_count} total embeddings")
            logger.info(f"Loaded {len(self.embeddings_cache)} embeddings from disk")
            if self.use_faiss:
                logger.info(f"Using FAISS for vector similarity search")
    
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
    
    def _load_embeddings_from_disk(self):
        """Load all embeddings from disk into memory."""
        if not self.embeddings_path.exists():
            return
        
        try:
            with open(self.embeddings_path, 'rb') as f:
                self.embeddings_cache = pickle.load(f)
                
            if self.verbose:
                logger.info(f"Loaded embeddings cache from {self.embeddings_path}")
        except Exception as e:
            logger.warning(f"Failed to load embeddings cache: {e}")
            self.embeddings_cache = {}
    
    def _save_embeddings_to_disk(self):
        """Save all embeddings to disk."""
        try:
            with open(self.embeddings_path, 'wb') as f:
                pickle.dump(self.embeddings_cache, f)
                
            if self.verbose:
                logger.debug(f"Saved embeddings cache to {self.embeddings_path}")
        except Exception as e:
            logger.warning(f"Failed to save embeddings cache: {e}")
    
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
    
    def get(self, content: str, model_name: str, max_seq_length: Optional[int] = None, 
            params: Optional[Dict[str, Any]] = None) -> Optional[np.ndarray]:
        """
        Get cached embedding for content with the specified model and parameters.
        
        Args:
            content: The text content to get embeddings for
            model_name: Name of the embedding model
            max_seq_length: Maximum sequence length for the model
            params: Dictionary of model parameters
            
        Returns:
            Numpy array containing the embedding, or None if not cached
        """
        if params is None:
            params = {}
            
        # Add max_seq_length to params if provided
        if max_seq_length is not None:
            params['max_seq_length'] = max_seq_length
            
        model_key = self._generate_model_key(model_name, params)
        content_hash = self._compute_content_hash(content)
        
        # Check if model exists in metadata
        if model_key not in self.metadata:
            return None
        
        # Check if this content hash exists for this model
        model_data = self.metadata[model_key]
        if 'embeddings' not in model_data or content_hash not in model_data['embeddings']:
            return None
        
        # First check in-memory cache
        cache_key = f"{model_key}_{content_hash}"
        if cache_key in self.embeddings_cache:
            if self.verbose:
                logger.debug(f"Memory cache hit for {model_name} - {content_hash[:8]}")
            return self.embeddings_cache[cache_key]
        
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
            
            # Store in memory cache
            self.embeddings_cache[cache_key] = embedding
            
            if self.verbose:
                logger.debug(f"Disk cache hit for {model_name} - {content_hash[:8]}")
            
            return embedding
        except Exception as e:
            logger.warning(f"Failed to load cached embedding: {e}")
            return None
    def _ensure_model_loaded(self, model_name: str, max_seq_length: Optional[int] = None):
        """Ensure the model is loaded, loading it if necessary."""
        from sentence_transformers import SentenceTransformer
        
        if model_name not in self.models:
            if self.verbose:
                logger.info(f"Loading model: {model_name}")
            self.models[model_name] = SentenceTransformer(model_name)
            
        # Set max sequence length if provided
        if max_seq_length is not None:
            self.models[model_name].max_seq_length = max_seq_length
            
        return self.models[model_name]
        
    def _ensure_tokenizer_loaded(self, model_name: str, model=None):
        """Ensure the tokenizer is loaded, loading it if necessary."""
        if self.tokenizer is None:
            from transformers import AutoTokenizer
            try:
                if self.verbose:
                    logger.info(f"Loading tokenizer for: {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            except Exception as e:
                if self.verbose:
                    logger.warning(f"Failed to load tokenizer directly, falling back to model tokenizer: {e}")
                # Fall back to the model's tokenizer
                if model is None:
                    model = self._ensure_model_loaded(model_name)
                self.tokenizer = model.tokenizer
                
        return self.tokenizer
        
    def _initialize_faiss_index(self, model_key: str, embedding_dim: int):
        """Initialize a FAISS index for a model."""
        if model_key in self.faiss_indices:
            return
            
        # Create a new FAISS index
        index = faiss.IndexFlatL2(embedding_dim)
        
        # Initialize the index data structures
        self.faiss_indices[model_key] = {
            'index': index,
            'id_to_hash': [],
            'hash_to_id': {}
        }
        
        # Add existing embeddings to the index
        if model_key in self.metadata:
            for content_hash, embed_info in self.metadata[model_key].get('embeddings', {}).items():
                # Skip chunks that are part of a parent document
                if 'parent_hash' in embed_info:
                    continue
                    
                # Load the embedding
                embedding_path = self._get_embedding_path(model_key, content_hash)
                if not embedding_path.exists():
                    continue
                    
                try:
                    embedding = np.load(embedding_path)
                    
                    # Add to FAISS index
                    faiss_idx = len(self.faiss_indices[model_key]['id_to_hash'])
                    self.faiss_indices[model_key]['id_to_hash'].append(content_hash)
                    self.faiss_indices[model_key]['hash_to_id'][content_hash] = faiss_idx
                    self.faiss_indices[model_key]['index'].add(embedding.reshape(1, -1))
                except Exception as e:
                    logger.warning(f"Failed to add embedding to FAISS index: {e}")  
                
    def put(self, content: str, embedding: np.ndarray, model_name: str, params: Optional[Dict[str, Any]] = None):
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
        cache_key = f"{model_key}_{content_hash}"
        
        # Initialize model in metadata if needed
        if model_key not in self.metadata:
            self.metadata[model_key] = {
                'model_name': model_name,
                'params': params,
                'created_at': time.time(),
                'embeddings': {},
                'embedding_dim': embedding.shape[0]
            }
        
        # Update or create embedding entry
        if 'embeddings' not in self.metadata[model_key]:
            self.metadata[model_key]['embeddings'] = {}
            
        # Store additional params in the metadata
        embed_info = {
            'timestamp': time.time(),
            'size': embedding.shape[0],
            'hash': content_hash
        }
        
        # Add any additional params (like chunk info)
        for k, v in params.items():
            if k not in ['max_seq_length', 'normalize_embeddings']:
                embed_info[k] = v
        
        self.metadata[model_key]['embeddings'][content_hash] = embed_info
        
        # Save the embedding to disk
        embedding_path = self._get_embedding_path(model_key, content_hash)
        np.save(embedding_path, embedding)
        
        # Store in memory cache
        self.embeddings_cache[cache_key] = embedding
        
        # Update FAISS index if enabled
        if self.use_faiss:
            # Initialize FAISS index if needed
            if model_key not in self.faiss_indices:
                self._initialize_faiss_index(model_key, embedding.shape[0])
            
            # Add to FAISS index if it's not a chunk of a parent document
            # or if we specifically want to index chunks
            if 'parent_hash' not in params:
                faiss_idx = len(self.faiss_indices[model_key]['id_to_hash'])
                self.faiss_indices[model_key]['id_to_hash'].append(content_hash)
                self.faiss_indices[model_key]['hash_to_id'][content_hash] = faiss_idx
                self.faiss_indices[model_key]['index'].add(embedding.reshape(1, -1))
        
        # Update metadata
        self._save_metadata()
        
        # Save embeddings cache to disk
        self._save_embeddings_to_disk()
        
        if self.verbose:
            logger.debug(f"Cached embedding for {model_name} - {content_hash[:8]}")
    
    def get_embedding(self, text: str, model_name: str, max_seq_length: Optional[int] = None, 
                     params: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Get embedding for a text, computing if necessary.
        
        Args:
            text: The text to embed
            model_name: Name of the embedding model
            max_seq_length: Maximum sequence length for the model
            params: Optional dictionary of model parameters
            
        Returns:
            Tuple of (embedding, metadata)
        """
        start_time = time.time()
        
        if params is None:
            params = {}
            
        # Try to get from cache
        embedding = self.get(text, model_name, max_seq_length, params)
        
        # If not in cache, compute it
        if embedding is None:
            # Load the model - ensure we only initialize once per model_name
            self._ensure_model_loaded(model_name, max_seq_length)
            model = self.models[model_name]
            
            # Ensure tokenizer is loaded
            self._ensure_tokenizer_loaded(model_name, model)
            
            tokens = self.tokenizer.encode(text)
            
            # Metadata to return
            metadata = {
                'chunked': False,
                'num_chunks': 1,
                'compute_time': 0
            }
            
            if len(tokens) <= model.max_seq_length:
                # Text fits in one chunk
                embedding = model.encode([text])[0]
            else:
                # Text is too long, need to chunk it
                chunks = self._chunk_text(text, model.max_seq_length)
                
                # Update metadata
                metadata['chunked'] = True
                metadata['num_chunks'] = len(chunks)
                
                if self.verbose:
                    logger.info(f"Chunking text into {len(chunks)} chunks")
                
                # Embed each chunk
                chunk_embeddings = model.encode(chunks)
                
                # Average the embeddings
                embedding = np.mean(chunk_embeddings, axis=0)
                
                # Normalize the final embedding
                embedding = embedding / np.linalg.norm(embedding)
            
            # Cache it
            self.put(text, embedding, model_name, params)
        
        # Return embedding and metadata
        content_hash = self._compute_content_hash(text)
        model_key = self._generate_model_key(model_name, params)
        
        metadata = {
            'content_hash': content_hash,
            'model_key': model_key,
            'compute_time': time.time() - start_time
        }
        
        if model_key in self.metadata and 'embeddings' in self.metadata[model_key]:
            if content_hash in self.metadata[model_key]['embeddings']:
                metadata.update(self.metadata[model_key]['embeddings'][content_hash])
        
        return embedding, metadata
        
    def bulk_get(self, contents: List[str], model_name: str, max_seq_length: Optional[int] = None, 
                params: Optional[Dict[str, Any]] = None) -> Tuple[List[np.ndarray], List[int]]:
        """
        Get cached embeddings for multiple content items.
        
        Args:
            contents: List of text contents to get embeddings for
            model_name: Name of the embedding model
            max_seq_length: Maximum sequence length for the model
            params: Dictionary of model parameters
            
        Returns:
            Tuple of (cached_embeddings, missing_indices)
                - cached_embeddings: List of numpy arrays for found embeddings
                - missing_indices: List of indices in the original contents that need embedding
        """
        cached_embeddings = []
        missing_indices = []
        
        for i, content in enumerate(contents):
            embedding = self.get(content, model_name, max_seq_length, params)
            if embedding is not None:
                cached_embeddings.append(embedding)
            else:
                missing_indices.append(i)
        
        return cached_embeddings, missing_indices
    
    def batch_get_embeddings(self, texts: List[str], model_name: str, 
                           max_seq_length: Optional[int] = None, 
                           params: Optional[Dict[str, Any]] = None) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Get embeddings for a batch of texts, using cache when possible.
        
        Args:
            texts: List of texts to embed
            model_name: Name of the embedding model
            max_seq_length: Maximum sequence length for the model
            params: Optional dictionary of model parameters
            
        Returns:
            List of tuples (embedding, metadata) for each text
        """
        if params is None:
            params = {}
            
        # Add max_seq_length to params if provided
        if max_seq_length is not None:
            params['max_seq_length'] = max_seq_length
        
        # Check cache for each text
        results = []
        missing_indices = []
        missing_texts = []
        
        for i, text in enumerate(texts):
            embedding = self.get(text, model_name, params)
            if embedding is not None:
                # Cache hit
                content_hash = self._compute_content_hash(text)
                model_key = self._generate_model_key(model_name, params)
                
                metadata = {
                    'content_hash': content_hash,
                    'model_key': model_key,
                    'cached': True
                }
                
                if model_key in self.metadata and 'embeddings' in self.metadata[model_key]:
                    if content_hash in self.metadata[model_key]['embeddings']:
                        metadata.update(self.metadata[model_key]['embeddings'][content_hash])
                
                results.append((embedding, metadata))
            else:
                # Cache miss
                missing_indices.append(i)
                missing_texts.append(text)
        
        # If there are missing embeddings, compute them
        if missing_texts:
            # Load the model
            from sentence_transformers import SentenceTransformer
            if model_name not in self.models:
                self.models[model_name] = SentenceTransformer(model_name)
            model = self.models[model_name]
            
            # Set max sequence length if provided
            if max_seq_length is not None:
                model.max_seq_length = max_seq_length
            
            # Initialize tokenizer if needed
            if self.tokenizer is None:
                from transformers import AutoTokenizer
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                except:
                    # Fall back to the model's tokenizer
                    self.tokenizer = model.tokenizer
            
            # Process each missing text
            for i, text in enumerate(missing_texts):
                start_time = time.time()
                tokens = self.tokenizer.encode(text)
                
                # Metadata to return
                metadata = {
                    'chunked': False,
                    'num_chunks': 1,
                    'compute_time': 0,
                    'cached': False
                }
                
                if len(tokens) <= model.max_seq_length:
                    # Text fits in one chunk
                    embedding = model.encode([text])[0]
                else:
                    # Text is too long, need to chunk it
                    chunks = self._chunk_text(text, model.max_seq_length)
                    
                    # Update metadata
                    metadata['chunked'] = True
                    metadata['num_chunks'] = len(chunks)
                    
                    if self.verbose:
                        logger.info(f"Chunking text into {len(chunks)} chunks")
                    
                    # Embed each chunk
                    chunk_embeddings = model.encode(chunks)
                    
                    # Average the embeddings
                    embedding = np.mean(chunk_embeddings, axis=0)
                    
                    # Normalize the final embedding
                    embedding = embedding / np.linalg.norm(embedding)
                
                # Cache it
                self.put(text, embedding, model_name, params)
                
                # Add metadata
                content_hash = self._compute_content_hash(text)
                model_key = self._generate_model_key(model_name, params)
                
                metadata['content_hash'] = content_hash
                metadata['model_key'] = model_key
                metadata['compute_time'] = time.time() - start_time
                
                # Insert at the correct position
                original_idx = missing_indices[i]
                results.insert(original_idx, (embedding, metadata))
        
        return results
    
    
    def similarity_search(self, query_embedding: np.ndarray, model_name: str, 
                         params: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform similarity search using FAISS if available.
        
        Args:
            query_embedding: Query embedding vector
            model_name: Name of the embedding model
            params: Dictionary of model parameters
            top_k: Number of results to return
            
        Returns:
            List of dictionaries with content hash and similarity score
        """
        model_key = self._generate_model_key(model_name, params)
        
        # If model not in metadata, return empty results
        if model_key not in self.metadata:
            return []
        
        # Use FAISS if available
        if self.use_faiss and model_key in self.faiss_indices:
            # Ensure query embedding is normalized and reshaped for FAISS
            query_embedding_norm = query_embedding / np.linalg.norm(query_embedding)
            query_embedding_reshaped = query_embedding_norm.reshape(1, -1)
            
            # Search the index
            k = min(top_k, self.faiss_indices[model_key]['index'].ntotal)
            if k == 0:
                return []
                
            distances, indices = self.faiss_indices[model_key]['index'].search(
                query_embedding_reshaped, k
            )
            
            # Convert results to the expected format
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < 0 or idx >= len(self.faiss_indices[model_key]['id_to_hash']):
                    continue
                    
                content_hash = self.faiss_indices[model_key]['id_to_hash'][idx]
                
                # Convert L2 distance to cosine similarity
                # For normalized vectors: cosine_sim = 1 - (L2_dist^2 / 2)
                similarity = 1 - (dist / 2)
                
                results.append({
                    'content_hash': content_hash,
                    'similarity': float(similarity)
                })
            
            return results
        
        # Fallback to brute force search
        results = []
        
        for content_hash, embed_info in self.metadata[model_key].get('embeddings', {}).items():
            # Skip chunks that are part of a parent document
            if 'parent_hash' in embed_info:
                continue
                
            # Load the embedding
            embedding_path = self._get_embedding_path(model_key, content_hash)
            if not embedding_path.exists():
                continue
                
            try:
                embedding = np.load(embedding_path)
                
                # Calculate cosine similarity
                similarity = np.dot(query_embedding, embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                )
                
                results.append({
                    'content_hash': content_hash,
                    'similarity': float(similarity)
                })
            except Exception as e:
                logger.warning(f"Failed to load embedding for similarity search: {e}")
        
        # Sort by similarity (descending)
        results = sorted(results, key=lambda x: x['similarity'], reverse=True)
        
        # Return top-k results
        return results[:top_k]
    
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
        
        # Process each embedding
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
            if self.embeddings_path.exists():
                self.embeddings_path.unlink()
            self.metadata = {}
            self.embeddings_cache = {}
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
                    # Remove from disk
                    embedding_path = self._get_embedding_path(model_key, content_hash)
                    if embedding_path.exists():
                        embedding_path.unlink()
                    
                    # Remove from memory cache
                    cache_key = f"{model_key}_{content_hash}"
                    if cache_key in self.embeddings_cache:
                        del self.embeddings_cache[cache_key]
            
            self._save_metadata()
            self._save_embeddings_to_disk()
            
            if self.verbose:
                logger.info(f"Cleared cache for model: {model_name}")
                
        elif older_than is not None:
            # Clear cache entries older than timestamp
            for model_key, model_data in list(self.metadata.items()):
                keep_model = True
                
                if 'embeddings' in model_data:
                    for content_hash, embed_info in list(model_data['embeddings'].items()):
                        if embed_info.get('timestamp', 0) < older_than:
                            # Remove old embedding from disk
                            embedding_path = self._get_embedding_path(model_key, content_hash)
                            if embedding_path.exists():
                                embedding_path.unlink()
                            
                            # Remove from memory cache
                            cache_key = f"{model_key}_{content_hash}"
                            if cache_key in self.embeddings_cache:
                                del self.embeddings_cache[cache_key]
                            
                            # Remove from metadata
                            model_data['embeddings'].pop(content_hash)
                    
                    # If all embeddings removed, check if we should remove model
                    if not model_data['embeddings'] and (model_name is None or model_data.get('model_name') == model_name):
                        keep_model = False
                
                if not keep_model:
                    self.metadata.pop(model_key)
            
            self._save_metadata()
            self._save_embeddings_to_disk()
            
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
    
    def _chunk_text(self, text: str, max_seq_length: int) -> List[str]:
        """Split text into chunks that fit within max_tokens."""
        max_tokens = max_seq_length - 10  # Leave some margin
        
        # If text is short enough, return as is
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return [text]
        
        # Split into sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_tokens = self.tokenizer.encode(sentence)
            sentence_length = len(sentence_tokens)
            
            # If a single sentence is too long, split it further
            if sentence_length > max_tokens:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Split long sentence into smaller pieces
                words = sentence.split()
                sub_chunk = []
                sub_length = 0
                
                for word in words:
                    word_tokens = self.tokenizer.encode(word)
                    word_length = len(word_tokens)
                    
                    if sub_length + word_length + 1 <= max_tokens:  # +1 for space
                        sub_chunk.append(word)
                        sub_length += word_length + 1
                    else:
                        if sub_chunk:
                            chunks.append(' '.join(sub_chunk))
                        sub_chunk = [word]
                        sub_length = word_length
                
                if sub_chunk:
                    chunks.append(' '.join(sub_chunk))
            
            # If adding this sentence would exceed max_tokens, start a new chunk
            elif current_length + sentence_length + 1 > max_tokens:  # +1 for space
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            
            # Otherwise add to current chunk
            else:
                current_chunk.append(sentence)
                current_length += sentence_length + 1  # +1 for space
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
        
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
    test_parser.add_argument('--no-faiss', action='store_true', help='Disable FAISS for vector search')
    test_parser.add_argument('--chunk-size', type=int, default=384, help='Chunk size for splitting long texts')
    test_parser.add_argument('--test-long-text', action='store_true', help='Test with a long text that requires chunking')
    
    args = parser.parse_args()
    
    # Initialize cache
    cache = EmbeddingCache(
        cache_dir=args.cache_dir,
        use_faiss=not args.no_faiss,
        chunk_size=args.chunk_size
    )
    
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
        
        
        # Add a long text if requested
        if args.test_long_text:
            long_text = """
            This is a very long text that will need to be chunked into smaller pieces before embedding.
            It contains multiple sentences spanning various topics to ensure it exceeds the token limit.
            The chunking algorithm should split this text at sentence boundaries when possible.
            This helps maintain the semantic meaning of each chunk.
            When a single sentence is too long, it will be split by words instead.
            This approach ensures that we can handle texts of arbitrary length while still producing meaningful embeddings.
            The final embedding will be created by averaging the embeddings of all chunks.
            This technique allows us to represent long documents in the same vector space as shorter texts.
            We can then perform similarity search and other operations using these embeddings.
            The FAISS library provides efficient similarity search capabilities for large collections of vectors.
            It uses various indexing techniques to speed up nearest neighbor search.
            This is particularly important when dealing with large document collections.
            By combining chunking with FAISS, we can build a scalable and efficient retrieval system.
            This system can handle documents of varying lengths and provide fast query responses.
            The embedding cache further improves performance by avoiding redundant computation.
            It stores embeddings based on content hash, ensuring that identical texts are only embedded once.
            This is especially useful in scenarios where the same text appears multiple times.
            The cache also handles parameter variations, so embeddings with different settings are stored separately.
            This comprehensive approach addresses the challenges of embedding and retrieving long texts efficiently.
            """
            texts.append(long_text)
            print(f"Added a long text with {len(long_text.split())} words for chunking test")
        
        # Define embedding function
        def embed_func(texts):
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            token_count = tokenizer([text for text in texts], return_tensors='pt', padding=True, truncation=True)
            # We don't assert token count here since we're handling chunking
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
        
        # Test similarity search if FAISS is enabled
        if cache.use_faiss:
            print("\nTesting similarity search with FAISS...")
            # Get embedding for a query
            query = "Let me test the similarity search functionality."
            query_embedding, _ = cache.get_embedding(query, args.model)
            
            # Perform similarity search
            results = cache.similarity_search(query_embedding, args.model, params, top_k=3)
            
            print(f"Found {len(results)} similar texts:")
            for i, result in enumerate(results):
                content_hash = result['content_hash']
                similarity = result['similarity']
                
                # Find the original text for this hash
                original_text = None
                for text in texts:
                    if cache._compute_content_hash(text) == content_hash:
                        original_text = text
                        break
                
                # Display the result
                print(f"{i+1}. Similarity: {similarity:.4f}")
                if original_text:
                    print(f"   Text: {original_text[:100]}..." if len(original_text) > 100 else original_text)
                else:
                    print(f"   Hash: {content_hash}")
        
        # Show final stats
        stats = cache.get_cache_stats()
        print(f"\nTest complete. Cache now has {stats['total_embeddings']} embeddings.")
        if cache.use_faiss:
            print(f"FAISS indices: {len(cache.faiss_indices)}")
