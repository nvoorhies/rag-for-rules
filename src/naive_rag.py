#!/usr/bin/env python3

"""
Naive RAG implementation for RPG rule retrieval.
This implements a simple vector-based retrieval system without 
structure awareness or rule relationship understanding.
"""

import json
import os
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer

class NaiveRAG:
    """Naive RAG implementation for RPG rules."""
    
    def __init__(self, chunks_path: str):
        """Initialize the naive RAG system with text chunks.
        
        Args:
            chunks_path: Path to JSON file containing text chunks of the SRD
        """
        # Load the text chunks
        with open(chunks_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        
        # Initialize the embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create embeddings for all chunks
        self.chunk_embeddings = self._create_chunk_embeddings()
        
        print(f"Naive RAG initialized with {len(self.chunks)} text chunks")
    
    def _create_chunk_embeddings(self) -> np.ndarray:
        """Create embeddings for all text chunks."""
        print("Creating embeddings for text chunks...")
        
        # Extract texts from chunks
        texts = [chunk['text'] for chunk in self.chunks]
        
        # Generate embeddings
        embeddings = self.model.encode(texts)
        
        return embeddings
    
    def query(self, query_text: str, top_k: int = 10) -> Dict[str, Any]:
        """Process a query and return relevant chunks.
        
        Args:
            query_text: The user's query text
            top_k: Maximum number of chunks to return
            
        Returns:
            Dict containing query results
        """
        print(f"Processing query with naive RAG: {query_text}")
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = self.model.encode(query_text)
        
        # Calculate similarity with all chunks
        similarities = []
        for i, chunk_embedding in enumerate(self.chunk_embeddings):
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
        
        # Format results to match the structure-aware system's output
        results = {
            'question': query_text,
            'rules': retrieved_chunks,
            'rule_count': len(retrieved_chunks),
            'query_time': time.time() - start_time
        }
        
        return results

    @staticmethod
    def preprocess_srd(input_path: str, output_path: str, chunk_size: int = 500, chunk_overlap: int = 100):
        """Preprocess SRD text into chunks for naive RAG.
        
        Args:
            input_path: Path to raw SRD markdown file
            output_path: Path to save preprocessed chunks
            chunk_size: Size of each text chunk in characters
            chunk_overlap: Overlap between consecutive chunks in characters
        """
        print(f"Preprocessing SRD from {input_path} into chunks...")
        
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
        
        print(f"Preprocessing complete. Created {len(chunks)} chunks, saved to {output_path}")


# Example usage
if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Naive RAG for RPG Rules")
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
    query_parser.add_argument('--query', '-q', required=True, help='Query text')
    query_parser.add_argument('--top-k', '-k', type=int, default=5, help='Number of results to return')
    
    args = parser.parse_args()
    
    if args.command == 'preprocess':
        NaiveRAG.preprocess_srd(args.input, args.output, args.chunk_size, args.overlap)
    
    elif args.command == 'query':
        rag = NaiveRAG(args.chunks)
        results = rag.query(args.query, args.top_k)
        
        print("\n=== Query Results ===")
        print(f"Query: {results['question']}")
        print(f"Found {results['rule_count']} relevant chunks in {results['query_time']:.2f} seconds:")
        
        for i, chunk in enumerate(results['rules'], 1):
            print(f"\n--- Chunk {i} ---")
            print(f"Title: {chunk['title']}")
            print(f"Similarity: {chunk['similarity_score']:.4f}")
            
            # Print truncated text
            max_text_len = 200
            text = chunk['text']
            if len(text) > max_text_len:
                text = text[:max_text_len] + "..."
            print(f"Text: {text}")
    
    else:
        parser.print_help()
