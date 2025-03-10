#!/usr/bin/env python3                                                                                                                                                                            
                                                                                                                                                                                                  
"""                                                                                                                                                                                               
Augmented Hierarchical RAG with cross-encoder reranking for improved retrieval quality.                                                                                                           
Combines text augmentation with reranking for better results.                                                                                                                                     
"""                                                                                                                                                                                               
                                                                                                                                                                                                  
import json                                                                                                                                                                                       
import logging                                                                                                                                                                                    
import time                                                                                                                                                                                       
from typing import Dict, Any, List, Optional                                                                                                                                                      
                                                                                                                                                                                                  
from reranker_hierarchical_rag import RerankerHierarchicalRAG                                                                                                                                     
from augmentation_functions import *

# Configure logging                                                                                                                                                                               
logging.basicConfig(                                                                                                                                                                              
    level=logging.INFO,                                                                                                                                                                           
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'                                                                                                                                 
)                                                                                                                                                                                                 
logger = logging.getLogger("augmented_reranker_rag")                                                                                                                                              
                                                                                                                                                                                                  
class AugmentedRerankerRAG(RerankerHierarchicalRAG):                                                                                                                                              
    """                                                                                                                                                                                           
    Hierarchical RAG with both text augmentation and cross-encoder reranking.                                                                                                                     
    Combines the strengths of both approaches for improved retrieval quality.                                                                                                                     
    """                                                                                                                                                                                           
                                                                                                                                                                                                  
    def _augment_section_text(self, section: Dict[str, Any]) -> str:                                                                                                                              
        """                                                                                                                                                                                       
        Augment section text with additional context before embedding.                                                                                                                            
                                                                                                                                                                                                  
        Args:                                                                                                                                                                                     
            section: The section to augment                                                                                                                                                       
                                                                                                                                                                                                  
        Returns:                                                                                                                                                                                  
            Augmented text string                                                                                                                                                                 
        """              
        return augment_with_path_references_scope(section)                                                                                                                                                                         
                                                                                                                                                                                                  
    def _get_section_embedding(self, section: Dict[str, Any]) -> Dict[str, Any]:                                                                                                                  
        """                                                                                                                                                                                       
        Get embedding for a section with augmentation.                                                                                                                                            
        Override to use augmented text for embedding.                                                                                                                                             
                                                                                                                                                                                                  
        Args:                                                                                                                                                                                     
            section: Section to embed                                                                                                                                                             
                                                                                                                                                                                                  
        Returns:                                                                                                                                                                                  
            Dict with embedding and metadata                                                                                                                                                      
        """                                                                                                                                                                                       
        # Augment the text before embedding                                                                                                                                                       
        augmented_text = augment_with_path_references_scope(section)                                                                                                                                      
                                                                                                                                                                                                  
        # Get embedding for the augmented text                                                                                                                                                    
        start_time = time.time()                                                                                                                                                                  
        embedding, metadata = self.embedding_cache.get_embedding(                                                                                                                                 
            augmented_text,                                                                                                                                                                       
            self.model_name                                                                                                                                                                       
        )                                                                                                                                                                                         
                                                                                                                                                                                                  
        if self.verbose:                                                                                                                                                                          
            logger.debug(f"Section embedding time: {time.time() - start_time:.3f}s")                                                                                                              
                                                                                                                                                                                                  
        return {                                                                                                                                                                                  
            "embedding": embedding,                                                                                                                                                               
            "metadata": metadata                                                                                                                                                                  
        }                                                                                                                                                                                         
                                                                                                                                                                                                  
    def _format_section_text(self, section: Dict[str, Any]) -> str:                                                                                                                               
        """                                                                                                                                                                                       
        Format section text for reranking.                                                                                                                                                        
        Override to use augmented text for reranking.                                                                                                                                             
                                                                                                                                                                                                  
        Args:                                                                                                                                                                                     
            section: The section to format                                                                                                                                                        
                                                                                                                                                                                                  
        Returns:                                                                                                                                                                                  
            Formatted text string                                                                                                                                                                 
        """                                                                                                                                                                                       
        # Use the same augmentation for reranking                                                                                                                                                 
        return self._augment_section_text(section)                                                                                                                                                
                                                                                                                                                                                                  
# For running as a script                                                                                                                                                                         
def main():                                                                                                                                                                                       
    """Main entry point for command-line execution."""                                                                                                                                            
    import argparse                                                                                                                                                                               
                                                                                                                                                                                                  
    parser = argparse.ArgumentParser(description="Augmented Reranker RAG for RPG Rules")                                                                                                          
                                                                                                                                                                                                  
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
    parser.add_argument('--parallel', '-p', type=int, default=1, help='Number of parallel processes to use')
                                                                                                                                                                                                  
    args = parser.parse_args()                                                                                                                                                                    
                                                                                                                                                                                                  
    # Initialize RAG system                                                                                                                                                                       
    rag = AugmentedRerankerRAG(                                                                                                                                                                   
        processed_srd_path=args.srd,                                                                                                                                                              
        embeddings_path=args.embeddings,                                                                                                                                                          
        model_name=args.model,                                                                                                                                                                    
        reranker_model_name=args.reranker,                                                                                                                                                        
        cache_dir=args.cache_dir,                                                                                                                                                                 
        max_seq_length=args.max_seq_length,                                                                                                                                                       
        verbose=args.verbose,
        parallel=args.parallel                                                                                                                                                                      
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
