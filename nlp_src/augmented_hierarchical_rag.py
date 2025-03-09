#!/usr/bin/env python3
"""
Augmented Hierarchical RAG that enhances text with contextual information before embedding.
"""

from typing import Dict, Any
from hierarchical_naive_rag import HierarchicalNaiveRAG
import logging
import json
import cProfile
import pstats
import io
import time
import augmentation_functions

logger = logging.getLogger("augmented_hierarchical_rag")

class AugmentedHierarchicalRAG(HierarchicalNaiveRAG):
    """Hierarchical RAG with text augmentation before embedding."""
    
    def _augment_section_text(self, section: Dict[str, Any]) -> str:
        """Augment texts with additional context before embedding."""
        return augmentation_functions.augment_with_path_references_scope(section)
    
    def query(self, query_text: str, max_rules: int = 10, profile: bool = False) -> Dict[str, Any]:
        """
        Process a query and return relevant sections.
        
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
        Implementation of query processing with detailed timing.
        
        Args:
            query_text: The user's query text
            max_rules: Maximum number of sections to return
            
        Returns:
            Dict containing query results
        """
        start_time = time.time()
        timings = {}
        
        # Generate query embedding with chunking support
        embed_start = time.time()
        query_embedding, metadata = self.embedding_cache.get_embedding(
            query_text, 
            self.model_name,
            self.max_seq_length
        )
        embed_time = time.time() - embed_start
        timings['embedding_generation'] = embed_time
        
        if self.verbose:
            if metadata.get('chunked', False):
                logger.info(f"Query was chunked into {metadata.get('num_chunks', 0)} chunks")
        
        # Use FAISS for similarity search if available
        search_start = time.time()
        if self.use_faiss and hasattr(self.embedding_cache, 'similarity_search'):
            # Perform similarity search
            search_results = self.embedding_cache.similarity_search(
                query_embedding,
                self.model_name,
                self.model_params,
                top_k=max_rules*2  # Get more results for reranking
            )
            
            # Map content hashes to sections
            similar_sections = []
            for result in search_results:
                content_hash = result['content_hash']
                similarity = result['similarity']
                
                # Find the section with this content hash
                for section in self.sections:
                    section_id = section['id']
                    if section_id in self.section_embeddings:
                        section_text = self._augment_section_text(section)
                        section_hash = self.embedding_cache._compute_content_hash(section_text)
                        
                        if section_hash == content_hash:
                            section_copy = section.copy()
                            section_copy['similarity_score'] = float(similarity)
                            similar_sections.append(section_copy)
                            break
        else:
            # Fallback to brute force search
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
            top_indices = [idx for idx, _ in sorted_similarities[:max_rules*2]]
            
            # Get the corresponding sections and add similarity scores
            similar_sections = []
            for i, idx in enumerate(top_indices):
                section = self.sections[idx].copy()
                section['similarity_score'] = sorted_similarities[i][1]
                similar_sections.append(section)
        
        search_time = time.time() - search_start
        timings['similarity_search'] = search_time
        
        # Apply reranking
        rerank_start = time.time()
        reranked_sections = self._rerank(query_text, similar_sections)
        rerank_time = time.time() - rerank_start
        timings['reranking'] = rerank_time
        
        # Limit to max_rules
        top_sections = reranked_sections[:max_rules]
        
        # Format results
        results = {
            'question': query_text,
            'rules': top_sections,
            'rule_count': len(top_sections),
            'query_time': time.time() - start_time,
            'embedding_time': metadata.get('compute_time', 0),
            'chunked': metadata.get('chunked', False),
            'num_chunks': metadata.get('num_chunks', 1),
            'timings': timings
        }
        
        return results

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
    parser.add_argument('--model', default='all-mpnet-base-v2', help='Embedding model name')
    parser.add_argument('--max-seq-length', type=int, help='Maximum sequence length')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    parser.add_argument('--stats', '-S', action='store_true', help='Show cache statistics')
    parser.add_argument('--profile', action='store_true', help='Enable profiling to identify performance bottlenecks')
    
    args = parser.parse_args()
    
    # Initialize RAG system
    rag = AugmentedHierarchicalRAG(
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
    import pickle  # Import here for pickle.dump
    main()
