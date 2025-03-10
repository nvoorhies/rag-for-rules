#!/usr/bin/env python3

"""
Hierarchical RAG with cross-encoder reranking for improved retrieval quality.
"""

import json
import os
import time
import logging
import sys
import cProfile
import pstats
import io
from typing import List, Dict, Any, Optional, Union
import numpy as np
from tqdm.auto import tqdm
from sentence_transformers import CrossEncoder
from hierarchical_naive_rag import HierarchicalNaiveRAG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("reranker_hierarchical_rag")

class RerankerHierarchicalRAG(HierarchicalNaiveRAG):
    """Hierarchical RAG with cross-encoder reranking."""
    
    def __init__(self, 
                processed_srd_path: str,
                embeddings_path: Optional[str] = None,
                model_name: str = "all-mpnet-base-v2",
                reranker_model_name: str = "mixedbread-ai/mxbai-rerank-xsmall-v1",
                cache_dir: str = "embedding_cache",
                max_seq_length: Optional[int] = None,
                verbose: bool = False):
        """
        Initialize the reranker hierarchical RAG system.
        
        Args:
            processed_srd_path: Path to processed SRD JSON with hierarchical structure
            embeddings_path: Optional path to pre-computed embeddings pickle
            model_name: Name of the sentence transformer model to use
            reranker_model_name: Name of the cross-encoder model to use for reranking
            cache_dir: Directory for the embedding cache
            max_seq_length: Maximum sequence length for the model
            verbose: Whether to print detailed logs
        """
        # Initialize the base class
        super().__init__(
            processed_srd_path=processed_srd_path,
            embeddings_path=embeddings_path,
            model_name=model_name,
            cache_dir=cache_dir,
            max_seq_length=max_seq_length,
            verbose=verbose
        )
        
        # Initialize the reranker
        self.reranker_model_name = reranker_model_name
        if self.verbose:
            logger.info(f"Loading reranker model: {reranker_model_name}")
        
        self.reranker = CrossEncoder(reranker_model_name)
        
        if self.verbose:
            logger.info(f"Reranker model loaded")
    
    def _rerank(self, query: str, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank the retrieved sections using a cross-encoder.
        
        Args:
            query: The query text
            sections: List of retrieved sections with similarity scores
            
        Returns:
            Reranked list of sections
        """
        if not sections:
            return sections
        
        # Get the query from the first section's similarity score context
        # query = sections[0].get('_query_text', '')
        if not query:
            # If query isn't stored in the section, we can't rerank
            if self.verbose:
                logger.warning("No query text found for reranking, returning original order")
            return sections
        
        # Prepare pairs for reranking
        format_start = time.time()
        pairs = []
        for section in sections:
            # Format section text
            section_text = self._format_section_text(section)
            pairs.append([query, section_text])
        format_time = time.time() - format_start
        
        # Get scores from reranker
        predict_start = time.time()
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        predict_time = time.time() - predict_start
        
        if self.verbose:
            logger.debug(f"Reranking: format={format_time:.3f}s, predict={predict_time:.3f}s")
        
        # Add reranker scores to sections
        for i, section in enumerate(sections):
            section['reranker_score'] = float(scores[i])
            section['rerank_format_time'] = format_time
            section['rerank_predict_time'] = predict_time
        
        # Sort by reranker score (descending)
        sort_start = time.time()
        reranked_sections = sorted(sections, key=lambda x: x.get('reranker_score', 0.0), reverse=True)
        sort_time = time.time() - sort_start
        
        if self.verbose:
            logger.debug(f"Reranking sort time: {sort_time:.3f}s")
        
        return reranked_sections
    
    def _format_section_text(self, section: Dict[str, Any]) -> str:
        """Format section text for reranking."""
        return self._augment_section_text(section)
        #return f"{' > '.join(section.get('path', [])) + ' > ' + section.get('title', '')}\n{section.get('text', '')}"
        return f"""{' > '.join(section['path'] + [section['title']])}
{section['text']} 
References: {', '.join(section.get('references', []))}                                                                                                                                            
Scope: {section.get('scope', 'Unknown')}                                                                                                                                                          
"""  

    def query(self, query_text: str, max_rules: int = 10, profile: bool = False) -> Dict[str, Any]:
        """
        Process a query and return relevant sections with reranking.
        
        Args:
            query_text: The user's query text
            max_rules: Maximum number of sections to return
            profile: Whether to enable profiling
            
        Returns:
            Dict containing query results
        """
        if profile:
            # Run with profiling
            profiler = cProfile.Profile()
            profiler.enable()
            results = self._query_impl(query_text, max_rules)
            profiler.disable()
            
            # Get profiling stats
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            ps.print_stats(30)  # Print top 30 functions by cumulative time
            
            # Add profiling info to results
            results['profiling'] = s.getvalue()
            return results
        else:
            # Run without profiling
            return self._query_impl(query_text, max_rules)
    
    def _query_impl(self, query_text: str, max_rules: int = 10) -> Dict[str, Any]:
        """
        Implementation of query processing with reranking.
        
        Args:
            query_text: The user's query text
            max_rules: Maximum number of sections to return
            
        Returns:
            Dict containing query results
        """
        start_time = time.time()
        
        # Get initial results from base class
        base_start = time.time()
        results = super().query(query_text, max_rules)
        base_time = time.time() - base_start
        
        # Store query text in each section for reranking
        for section in results['rules']:
            section['_query_text'] = query_text
        
        # Apply reranking
        rerank_start = time.time()
        reranked_sections = self._rerank(query_text, results['rules'])
        rerank_time = time.time() - rerank_start
        
        # Update results
        results['rules'] = reranked_sections
        
        # Add timing information
        if 'timings' not in results:
            results['timings'] = {}
        results['timings']['base_query'] = base_time
        results['timings']['reranking'] = rerank_time
        
        return results

# For running as a script
def main():
    """Main entry point for command-line execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Reranker Hierarchical RAG for RPG Rules")
    
    # Query command
    parser.add_argument('--srd', '-s', required=True, help='Processed SRD JSON file')
    parser.add_argument('--query', '-q', help='Query text')
    parser.add_argument('--queries-file', '-f', help='File containing queries (one per line or JSON)')
    parser.add_argument('--top-k', '-k', type=int, default=5, help='Number of results to return')
    parser.add_argument('--output', '-o', help='Output file for results')
    parser.add_argument('--embeddings', '-e', help='Path to save/load embeddings')
    parser.add_argument('--cache-dir', default='embedding_cache', help='Embedding cache directory')
    parser.add_argument('--model', default='all-mpnet-base-v2', help='Embedding model name')
    parser.add_argument('--reranker', default='mixedbread-ai/mxbai-rerank-xsmall-v1', help='Reranker model name')
    parser.add_argument('--max-seq-length', type=int, help='Maximum sequence length')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    parser.add_argument('--stats', '-S', action='store_true', help='Show cache statistics')
    parser.add_argument('--profile', action='store_true', help='Enable profiling to identify performance bottlenecks')
    parser.add_argument('--parallel', '-p', type=int, default=1, help='Number of parallel processes to use')
    
    args = parser.parse_args()
    
    # Initialize RAG system
    rag = RerankerHierarchicalRAG(
        processed_srd_path=args.srd,
        embeddings_path=args.embeddings,
        model_name=args.model,
        reranker_model_name=args.reranker,
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
        result = rag.query(args.query, args.top_k, profile=args.profile)
        
        # Save to file or print to stdout
        if args.output:
            rag.save_results(result, args.output)
        else:
            print(json.dumps(result, indent=2))
            
    elif args.queries_file:
        # Multiple queries from file
        results = rag.process_queries_from_file(args.queries_file, args.top_k, profile=args.profile)
        
        # Save to file or print to stdout
        if args.output:
            rag.save_results(results, args.output)
        else:
            print(json.dumps(results, indent=2))
    else:
        parser.error("Either --query or --queries-file must be specified")

if __name__ == "__main__":
    main()
