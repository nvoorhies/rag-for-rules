#!/usr/bin/env python3

"""
Hierarchical Naive RAG for RPG Rules - Uses document hierarchical structure
but with naive embedding approach.
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

# Add parent directory to path for importing embedding_cache
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embedding_cache import EmbeddingCache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("hierarchical_naive_rag")

class HierarchicalNaiveRAG:
    """Naive RAG implementation using document hierarchy but simple embedding approach."""
    
    def __init__(self, 
                processed_srd_path: str,
                embeddings_path: Optional[str] = None,
                model_name: str = "all-MiniLM-L6-v2",
                cache_dir: str = "embedding_cache",
                max_seq_length: Optional[int] = None,
                verbose: bool = False):
        """
        Initialize the hierarchical naive RAG system.
        
        Args:
            processed_srd_path: Path to processed SRD JSON with hierarchical structure
            embeddings_path: Optional path to pre-computed embeddings pickle
            model_name: Name of the sentence transformer model to use
            cache_dir: Directory for the embedding cache
            max_seq_length: Maximum sequence length for the model
            verbose: Whether to print detailed logs
        """
        self.verbose = verbose
        start_time = time.time()
        self.model_name = model_name
        
        # Load the processed SRD
        with open(processed_srd_path, 'r', encoding='utf-8') as f:
            self.srd_data = json.load(f)
        
        # Extract rules (sections)
        if 'rules' in self.srd_data:
            self.sections = self.srd_data['rules']
        else:
            raise ValueError("Processed SRD does not contain 'rules' field. Make sure to use a hierarchically processed SRD.")
        
        # Initialize the embedding model
        self.model = SentenceTransformer(model_name)
        
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
        
        # Initialize section embeddings
        self.section_embeddings = self._load_or_create_embeddings(embeddings_path)
        
        if self.verbose:
            logger.info(f"Initialized HierarchicalNaiveRAG with {len(self.sections)} sections in {time.time() - start_time:.2f}s")
    
    def _augment_section_text(self, section: Dict[str, Any]) -> str:
        """Augment section text with parent titles for better context."""
        # Add parent section titles to the text
        return f"{section['title']}: {section['text']}"


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
        
        # Prepare section texts
        sections_texts = []
        for section in self.sections:
            # Simply combine title and text - no graph augmentation
            combined_text = self._augment_section_text(section)
            sections_texts.append(combined_text)
        
        # Generate embeddings using the cache
        start_time = time.time()
        if self.verbose:
            logger.info(f"Computing embeddings for {len(sections_texts)} sections with caching...")
        
        # Let the cache handle missing embeddings
        content_hash_to_embedding = self.cache.compute_missing_embeddings(
            sections_texts, 
            self.model_name, 
            self.model_params,
            self._embed_function
        )
        
        # Map section IDs to embeddings
        section_id_to_embedding = {}
        for i, section in enumerate(self.sections):
            section_id = section['id']
            content_hash = self.cache._compute_content_hash(sections_texts[i])
            if content_hash in content_hash_to_embedding:
                section_id_to_embedding[section_id] = content_hash_to_embedding[content_hash]
        
        if self.verbose:
            logger.info(f"Embedding completed in {time.time() - start_time:.2f} seconds")
        
        # Save embeddings if path provided
        if embeddings_path:
            os.makedirs(os.path.dirname(os.path.abspath(embeddings_path)), exist_ok=True)
            with open(embeddings_path, 'wb') as f:
                pickle.dump(section_id_to_embedding, f)
            if self.verbose:
                logger.info(f"Saved section embeddings to {embeddings_path}")
        
        return section_id_to_embedding
    
    def _embed_function(self, texts: List[str]) -> List[np.ndarray]:
        """Embedding function for the cache."""
        return self.model.encode(
            texts, 
            normalize_embeddings=self.model_params['normalize_embeddings'],
            show_progress_bar=self.verbose and len(texts) > 10
        )
    
    def _rerank(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank the retrieved sections.
        
        Args:
            sections: List of retrieved sections with similarity scores
            
        Returns:
            Reranked list of sections
        """
        # For now, this is just an identity function
        # Will be extended in subclasses
        return sections
    
    def query(self, query_text: str, max_rules: int = 10) -> Dict[str, Any]:
        """
        Process a query and return relevant sections.
        
        Args:
            query_text: The user's query text
            max_rules: Maximum number of sections to return
            
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
        
        # Calculate similarity with all sections
        similarities = []
        for i, section in enumerate(self.sections):
            section_id = section['id']
            if section_id in self.section_embeddings:
                section_embedding = self.section_embeddings[section_id]
                similarity = np.dot(query_embedding, section_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(section_embedding)
                )
                similarities.append((i, float(similarity)))
        
        # Sort by similarity (descending)
        sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
        
        # Take top-k results
        top_indices = [idx for idx, _ in sorted_similarities[:max_rules]]
        
        # Get the corresponding sections and add similarity scores
        retrieved_sections = []
        for i, idx in enumerate(top_indices):
            section = self.sections[idx].copy()
            section['similarity_score'] = sorted_similarities[i][1]
            retrieved_sections.append(section)
        
        # Apply reranking
        reranked_sections = self._rerank(retrieved_sections)
        
        # Format results
        results = {
            'question': query_text,
            'rules': reranked_sections,
            'rule_count': len(reranked_sections),
            'query_time': time.time() - start_time
        }
        
        return results
    
    def process_queries_from_file(self, query_file: str, max_rules: int = 10) -> List[Dict[str, Any]]:
        """
        Process multiple queries from a file.
        
        Args:
            query_file: Path to file containing queries (one per line or JSON)
            max_rules: Maximum number of sections to return per query
            
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
                result = self.query(query_text, max_rules)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing query '{query_text}': {e}")
                results.append({
                    'question': query_text,
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

# For running as a script
def main():
    """Main entry point for command-line execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hierarchical Naive RAG for RPG Rules")
    
    # Query command
    parser.add_argument('--srd', '-s', required=True, help='Processed SRD JSON file')
    parser.add_argument('--query', '-q', help='Query text')
    parser.add_argument('--queries-file', '-f', help='File containing queries (one per line or JSON)')
    parser.add_argument('--top-k', '-k', type=int, default=5, help='Number of results to return')
    parser.add_argument('--output', '-o', help='Output file for results')
    parser.add_argument('--embeddings', '-e', help='Path to save/load embeddings')
    parser.add_argument('--cache-dir', default='embedding_cache', help='Embedding cache directory')
    parser.add_argument('--model', default='all-MiniLM-L6-v2', help='Embedding model name')
    parser.add_argument('--max-seq-length', type=int, help='Maximum sequence length')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    parser.add_argument('--stats', '-S', action='store_true', help='Show cache statistics')
    
    args = parser.parse_args()
    
    # Initialize RAG system
    rag = HierarchicalNaiveRAG(
        processed_srd_path=args.srd,
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

if __name__ == "__main__":
    import pickle  # Import here for pickle.dump
    main()
