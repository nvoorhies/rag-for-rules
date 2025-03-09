#!/usr/bin/env python3

"""
Cached Naive RAG for RPG Rules - Enhanced with embedding caching
and batch query processing capabilities.
"""

import json
import os
import time
import logging
import sys
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import pickle
from sentence_transformers import SentenceTransformer
from embedding_cache import EmbeddingCache


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("naive_rag")

class CachedNaiveRAG:
    """Naive RAG implementation for RPG rules with embedding caching."""
    
    def __init__(self, 
                chunks_path: str,
                embeddings_path: Optional[str] = None,
                model_name: str = "all-MiniLM-L6-v2",
                cache_dir: str = "embedding_cache",
                max_seq_length: Optional[int] = None,
                verbose: bool = False):
        """
        Initialize the naive RAG system with embedding cache.
        
        Args:
            chunks_path: Path to JSON file containing text chunks of the SRD
            embeddings_path: Optional path to pre-computed embeddings pickle
            model_name: Name of the sentence transformer model to use
            cache_dir: Directory for the embedding cache
            max_seq_length: Maximum sequence length for the model
            verbose: Whether to print detailed logs
        """
        self.verbose = verbose
        start_time = time.time()
        
        # Load the text chunks
        with open(chunks_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        
        # Initialize the embedding model
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        
        # Set max sequence length if provided
        if max_seq_length is not None:
            self.model.max_seq_length = max_seq_length
        
        # Model parameters for cache keys
        self.model_params = {
            'max_seq_length': self.model.max_seq_length,
            'normalize_embeddings': True
        }
        
        # Initialize the embedding cache
        self.cache = EmbeddingCache(cache_dir, verbose=verbose)
        
        # Initialize chunk embeddings
        self.chunk_embeddings = self._load_or_create_embeddings(embeddings_path)
        
        if self.verbose:
            logger.info(f"Initialized CachedNaiveRAG with {len(self.chunks)} chunks in {time.time() - start_time:.2f}s")
    
    def _load_or_create_embeddings(self, embeddings_path: Optional[str]) -> Dict[str, np.ndarray]:
        """Load embeddings from file or create and cache them."""
        if embeddings_path and os.path.exists(embeddings_path):
            # Try to load pre-computed embeddings
            try:
                with open(embeddings_path, 'rb') as f:
                    embeddings = pickle.load(f)
                if self.verbose:
                    logger.info(f"Loaded pre-computed embeddings from {embeddings_path}")
                return embeddings
            except Exception as e:
                logger.warning(f"Failed to load embeddings from {embeddings_path}: {e}")
                logger.info("Will generate embeddings from scratch")
        
        # Extract texts from chunks
        texts = [chunk['text'] for chunk in self.chunks]
        
        # Generate embeddings using the cache
        start_time = time.time()
        if self.verbose:
            logger.info(f"Computing embeddings for {len(texts)} chunks with caching...")
        
        # Let the cache handle missing embeddings
        content_hash_to_embedding = self.cache.compute_missing_embeddings(
            texts, 
            self.model_name, 
            self.model_params,
            self._embed_function
        )
        
        # Map chunk IDs to embeddings
        chunk_id_to_embedding = {}
        for i, chunk in enumerate(self.chunks):
            chunk_id = chunk['id']
            content_hash = self.cache._compute_content_hash(texts[i])
            if content_hash in content_hash_to_embedding:
                chunk_id_to_embedding[chunk_id] = content_hash_to_embedding[content_hash]
        
        if self.verbose:
            logger.info(f"Embedding completed in {time.time() - start_time:.2f} seconds")
        
        # Save embeddings if path provided
        if embeddings_path:
            os.makedirs(os.path.dirname(os.path.abspath(embeddings_path)), exist_ok=True)
            with open(embeddings_path, 'wb') as f:
                pickle.dump(chunk_id_to_embedding, f)
            if self.verbose:
                logger.info(f"Saved embeddings to {embeddings_path}")
        
        return chunk_id_to_embedding
    
    def _embed_function(self, texts: List[str]) -> List[np.ndarray]:
        """Embedding function for the cache."""
        return self.model.encode(
            texts, 
            normalize_embeddings=self.model_params['normalize_embeddings'],
            show_progress_bar=self.verbose and len(texts) > 10
        )
    
    def query(self, query_text: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Process a query and return relevant chunks.
        
        Args:
            query_text: The user's query text
            top_k: Maximum number of chunks to return
            
        Returns:
            Dict containing query results
        """
        start_time = time.time()
        
        # Check if query embedding exists in cache
        cached_embedding = self.cache.get(query_text, self.model_name, self.model_params)
        
        if cached_embedding is not None:
            query_embedding = cached_embedding
        else:
            # Generate and cache the query embedding
            query_embedding = self.model.encode(
                query_text, 
                normalize_embeddings=self.model_params['normalize_embeddings']
            )
            self.cache.put(query_text, query_embedding, self.model_name, self.model_params)
        
        # Calculate similarity with all chunks
        similarities = []
        for i, chunk in enumerate(self.chunks):
            chunk_id = chunk['id']
            if chunk_id in self.chunk_embeddings:
                chunk_embedding = self.chunk_embeddings[chunk_id]
                similarity = np.dot(query_embedding, chunk_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
                )
                similarities.append((i, float(similarity)))
        
        # Sort by similarity (descending)
        sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
        
        # Take top-k results
        top_indices = [idx for idx, _ in sorted_similarities[:top_k]]
        
        # Get the corresponding chunks and add similarity scores
        retrieved_chunks = []
        for i, idx in enumerate(top_indices):
            chunk = self.chunks[idx].copy()
            chunk['similarity_score'] = sorted_similarities[i][1]
            retrieved_chunks.append(chunk)
        
        # Format results
        results = {
            'query': query_text,
            'rules': retrieved_chunks,
            'rule_count': len(retrieved_chunks),
            'query_time': time.time() - start_time
        }
        
        return results
    
    def process_queries_from_file(self, query_file: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Process multiple queries from a file.
        
        Args:
            query_file: Path to file containing queries (one per line)
            top_k: Maximum number of chunks to return per query
            
        Returns:
            List of query results
        """
        # Read queries from file
        with open(query_file, 'r', encoding='utf-8') as f:
            if query_file.endswith('.json'):
                # JSON format - expect a list of query objects
                queries = json.load(f)
                query_texts = [q['question'] if isinstance(q, dict) else q for q in queries]
            else:
                # Plain text format - one query per line
                query_texts = [line.strip() for line in f if line.strip()]
        
        if self.verbose:
            logger.info(f"Processing {len(query_texts)} queries from {query_file}")
        
        # Process each query
        results = []
        for query_text in tqdm(query_texts, desc="Processing queries", disable=not self.verbose):
            try:
                result = self.query(query_text, top_k)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing query '{query_text}': {e}")
                results.append({
                    'query': query_text,
                    'error': str(e),
                    'rules': [],
                    'rule_count': 0,
                    'query_time': 0
                })
        
        return results
    
    def save_results(self, results: Union[Dict[str, Any], List[Dict[str, Any]]], output_path: str):
        """
        Save query results to a file.
        
        Args:
            results: Single query result or list of results
            output_path: Path to save results
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save results
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        if self.verbose:
            logger.info(f"Saved results to {output_path}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the embedding cache."""
        return self.cache.get_cache_stats()
    
    @staticmethod
    def preprocess_srd(input_path: str, output_path: str, chunk_size: int = 500, chunk_overlap: int = 100):
        """Preprocess SRD text into chunks for naive RAG.
        
        Args:
            input_path: Path to raw SRD markdown file
            output_path: Path to save preprocessed chunks
            chunk_size: Size of each text chunk in characters
            chunk_overlap: Overlap between consecutive chunks in characters
        """
        logger.info(f"Preprocessing SRD from {input_path} into chunks...")
        
        # Read the SRD file
        with open(input_path, 'r', encoding='utf-8') as f:
            srd_text = f.read()
        
        # Split into chunks
        chunks = []
        current_pos = 0
        chunk_id = 0
        
        while current_pos < len(srd_text):
            # Define chunk bounds
            chunk_end = min(current_pos + chunk_size, len(srd_text))
            
            # Adjust end to avoid splitting in the middle of a word or line
            if chunk_end < len(srd_text):
                # Try to find a paragraph break
                paragraph_break = srd_text.rfind("\n\n", current_pos, chunk_end)
                if paragraph_break != -1 and paragraph_break > current_pos:
                    chunk_end = paragraph_break
                else:
                    # Try to find a line break
                    line_break = srd_text.rfind("\n", current_pos, chunk_end)
                    if line_break != -1 and line_break > current_pos:
                        chunk_end = line_break
                    else:
                        # Try to find a space
                        space = srd_text.rfind(" ", current_pos, chunk_end)
                        if space != -1 and space > current_pos:
                            chunk_end = space
            
            # Extract chunk text
            chunk_text = srd_text[current_pos:chunk_end].strip()
            
            # Extract title from chunk (first line or heading)
            chunk_lines = chunk_text.split('\n')
            chunk_title = chunk_lines[0].strip().replace('#', '').strip()
            if len(chunk_title) > 100:  # If title is too long, truncate
                chunk_title = chunk_title[:97] + "..."
            
            # Create chunk object
            chunk = {
                'id': f"chunk_{chunk_id}",
                'title': chunk_title,
                'text': chunk_text,
                'position': current_pos
            }
            
            chunks.append(chunk)
            
            # Move position for next chunk
            current_pos = current_pos + chunk_size - chunk_overlap
            chunk_id += 1
        
        # Save chunks to output file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2)
        
        logger.info(f"Preprocessing complete. Created {len(chunks)} chunks, saved to {output_path}")

# For running as a script
def main():
    """Main entry point for command-line execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cached Naive RAG for RPG Rules")
    subparsers = parser.add_subparsers(dest='command')
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess SRD into chunks')
    preprocess_parser.add_argument('--input', '-i', required=True, help='Input SRD markdown file')
    preprocess_parser.add_argument('--output', '-o', required=True, help='Output chunks JSON file')
    preprocess_parser.add_argument('--chunk-size', type=int, default=500, help='Chunk size in characters')
    preprocess_parser.add_argument('--overlap', type=int, default=100, help='Chunk overlap in characters')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the naive RAG system')
    query_parser.add_argument('--chunks', '-c', required=True, help='Chunks JSON file')
    query_parser.add_argument('--query', '-q', help='Query text')
    query_parser.add_argument('--queries-file', '-f', help='File containing queries (one per line or JSON)')
    query_parser.add_argument('--top-k', '-k', type=int, default=5, help='Number of results to return')
    query_parser.add_argument('--output', '-o', help='Output file for results')
    query_parser.add_argument('--embeddings', '-e', help='Path to save/load embeddings')
    query_parser.add_argument('--cache-dir', default='embedding_cache', help='Embedding cache directory')
    query_parser.add_argument('--model', default='all-MiniLM-L6-v2', help='Embedding model name')
    query_parser.add_argument('--max-seq-length', type=int, help='Maximum sequence length')
    query_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    query_parser.add_argument('--stats', '-s', action='store_true', help='Show cache statistics')
    
    args = parser.parse_args()
    
    if args.command == 'preprocess':
        CachedNaiveRAG.preprocess_srd(args.input, args.output, args.chunk_size, args.overlap)
    
    elif args.command == 'query':
        # Initialize RAG system
        rag = CachedNaiveRAG(
            chunks_path=args.chunks,
            embeddings_path=args.embeddings,
            model_name=args.model,
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
    
    else:
        parser.print_help()

if __name__ == "__main__":
    import pickle  # Import here for pickle.dump
    main()
