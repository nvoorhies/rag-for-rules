import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional, Union
import torch
import argparse
import os
import pickle
import json
import re
from dataclasses import dataclass
import logging
import textwrap

@dataclass
class Document:
    """Represents a document or chunk with its content and metadata."""
    content: str
    metadata: Dict = None

class TextChunker:
    """Simple text chunker that splits documents into overlapping chunks."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logging.getLogger(__name__)
    
    def chunk_text(self, text: str) -> List[Document]:
        """Split text into overlapping chunks."""
        # Clean and normalize text
        text = text.replace('\n', ' ').strip()
        
        # Generate chunks with overlap
        chunks = []
        start = 0
        while start < len(text):
            # Get chunk of specified size
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end]
            
            # If not the last chunk, try to break at sentence boundary
            if end < len(text):
                # Find last period or punctuation mark
                last_period = max(
                    chunk.rfind('.'),
                    chunk.rfind('?'),
                    chunk.rfind('!')
                )
                if last_period != -1 and last_period > 0.5 * self.chunk_size:
                    end = start + last_period + 1
                    chunk = text[start:end]
            
            # Extract metadata (look for headers or section markers)
            metadata = self._extract_metadata(chunk)
            metadata['start_char'] = start
            metadata['end_char'] = end
            
            chunks.append(Document(
                content=chunk,
                metadata=metadata
            ))
            
            # Move start position, accounting for overlap
            start = end - self.chunk_overlap
        
        self.logger.info(f"Created {len(chunks)} chunks from text")
        return chunks
    
    def chunk_markdown(self, markdown_text: str) -> List[Document]:
        """Split markdown into chunks, preserving headers and structure."""
        # Find all headers
        header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        headers = list(header_pattern.finditer(markdown_text))
        
        chunks = []
        
        # Process each section defined by headers
        for i in range(len(headers)):
            start_pos = headers[i].start()
            # Determine end position
            if i < len(headers) - 1:
                end_pos = headers[i+1].start()
            else:
                end_pos = len(markdown_text)
            
            # Extract section content
            section_content = markdown_text[start_pos:end_pos]
            
            # Get header info
            header_level = len(headers[i].group(1))
            header_text = headers[i].group(2)
            
            # Chunk the section content
            if len(section_content) > self.chunk_size:
                section_chunks = self.chunk_text(section_content)
                # Add header info to metadata
                for chunk in section_chunks:
                    if not chunk.metadata:
                        chunk.metadata = {}
                    chunk.metadata['header'] = header_text
                    chunk.metadata['level'] = header_level
                chunks.extend(section_chunks)
            else:
                # Small enough to keep as single chunk
                chunks.append(Document(
                    content=section_content,
                    metadata={
                        'header': header_text,
                        'level': header_level,
                    }
                ))
        
        self.logger.info(f"Created {len(chunks)} chunks from markdown")
        return chunks
    
    def _extract_metadata(self, text: str) -> Dict:
        """Extract metadata from chunk text."""
        metadata = {}
        
        # Try to identify headers
        header_match = re.search(r'^(#{1,6})\s+(.+)$', text, re.MULTILINE)
        if header_match:
            metadata['header'] = header_match.group(2)
            metadata['level'] = len(header_match.group(1))
        
        # Try to identify page references
        page_match = re.search(r'\(p\.\s*(\d+)\)', text)
        if page_match:
            metadata['page'] = page_match.group(1)
        
        return metadata


class VectorStore:
    """Vector store for document retrieval using embeddings."""
    
    def __init__(self, embedding_dim: int = None):
        self.documents: List[Document] = []
        self.embeddings: np.ndarray = None
        self.embedding_dim = embedding_dim
        self.logger = logging.getLogger(__name__)
    
    def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        """Add documents and their embeddings to the store."""
        self.documents.extend(documents)
        if self.embeddings is None:
            self.embeddings = embeddings
            if self.embedding_dim is None:
                self.embedding_dim = embeddings.shape[1]
        else:
            # Ensure embeddings have the same dimensionality
            if embeddings.shape[1] != self.embeddings.shape[1]:
                raise ValueError(f"Embedding dimensions don't match: {embeddings.shape[1]} vs {self.embeddings.shape[1]}")
            self.embeddings = np.vstack([self.embeddings, embeddings])
        
        self.logger.info(f"Added {len(documents)} documents to vector store. Total: {len(self.documents)}")
    
    def similarity_search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 3
    ) -> List[Tuple[Document, float]]:
        """Find k most similar documents using cosine similarity."""
        if self.embeddings is None or len(self.embeddings) == 0:
            return []
        
        # Compute cosine similarity
        similarity_scores = np.dot(self.embeddings, query_embedding.T).flatten()
        
        # Get top k indices
        top_k_indices = np.argsort(similarity_scores)[-k:][::-1]
        
        # Return documents and scores
        results = []
        for idx in top_k_indices:
            results.append((self.documents[idx], float(similarity_scores[idx])))
        
        return results
    
    def save(self, filepath: str):
        """Save the vector store to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'embeddings': self.embeddings,
                'embedding_dim': self.embedding_dim
            }, f)
        self.logger.info(f"Saved vector store to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'VectorStore':
        """Load a vector store from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        store = cls(embedding_dim=data['embedding_dim'])
        store.documents = data['documents']
        store.embeddings = data['embeddings']
        store.logger.info(f"Loaded vector store from {filepath} with {len(store.documents)} documents")
        return store


class EmbeddingModel:
    """Wrapper for sentence-transformers embedding model."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.logger = logging.getLogger(__name__)
    
    def embed_documents(self, documents: List[Document]) -> np.ndarray:
        """Generate embeddings for a list of documents."""
        texts = [doc.content for doc in documents]
        self.logger.info(f"Generating embeddings for {len(texts)} documents")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a query string."""
        return self.model.encode([query]).reshape(1, -1)


class RAGSystem:
    """RAG system for processing and querying documents."""
    
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        vector_store_path: Optional[str] = None
    ):
        # Initialize components
        self.embedding_model = EmbeddingModel(embedding_model_name)
        self.chunker = TextChunker(chunk_size, chunk_overlap)
        self.logger = logging.getLogger(__name__)
        
        # Initialize or load vector store
        if vector_store_path and os.path.exists(vector_store_path):
            self.logger.info(f"Loading existing vector store from {vector_store_path}")
            self.vector_store = VectorStore.load(vector_store_path)
        else:
            self.logger.info("Initializing new vector store")
            self.vector_store = VectorStore(self.embedding_model.embedding_dim)
        
        self.vector_store_path = vector_store_path
    
    def process_document(self, document_path: str, is_markdown: bool = True) -> VectorStore:
        """Process a document, generate chunks and embeddings, and add to vector store."""
        # Read document
        self.logger.info(f"Reading document from {document_path}")
        with open(document_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Chunk document
        if is_markdown:
            chunks = self.chunker.chunk_markdown(content)
        else:
            chunks = self.chunker.chunk_text(content)
        
        # Generate embeddings
        embeddings = self.embedding_model.embed_documents(chunks)
        
        # Add to vector store
        self.vector_store.add_documents(chunks, embeddings)
        
        # Save vector store if path is specified
        if self.vector_store_path:
            self.vector_store.save(self.vector_store_path)
        
        return self.vector_store
    
    def query(
        self, 
        query: str, 
        k: int = 3
    ) -> List[Tuple[Document, float]]:
        """Query the vector store for similar documents."""
        # Generate query embedding
        query_embedding = self.embedding_model.embed_query(query)
        
        # Retrieve similar documents
        results = self.vector_store.similarity_search(query_embedding, k=k)
        
        return results
    
    def query_text(self, query: str, k: int = 3) -> List[str]:
        """Query and return only the document contents."""
        results = self.query(query, k=k)
        return [doc.content for doc, _ in results]


def process_document(args):
    """Process a document and save embeddings."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Initialize RAG system
    rag = RAGSystem(
        embedding_model_name=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        vector_store_path=args.output
    )
    
    # Process document
    logger.info(f"Processing document: {args.document}")
    rag.process_document(args.document, is_markdown=args.markdown)
    
    logger.info(f"Document processed and saved to {args.output}")
    
    # Print stats
    logger.info(f"Total chunks: {len(rag.vector_store.documents)}")
    logger.info(f"Embedding dimensions: {rag.vector_store.embedding_dim}")


def query_document(args):
    """Query a processed document."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Check if vector store exists
    if not os.path.exists(args.vector_store):
        logger.error(f"Vector store file not found: {args.vector_store}")
        return
    
    # Initialize RAG system
    rag = RAGSystem(
        embedding_model_name=args.model,
        vector_store_path=args.vector_store
    )
    
    # Perform query
    logger.info(f"Querying: {args.query}")
    results = rag.query(args.query, k=args.k)
    
    # Print results
    print(f"\nQuery: {args.query}")
    print(f"\nTop {len(results)} relevant chunks:")
    
    for i, (doc, score) in enumerate(results):
        header = doc.metadata.get('header', '') if doc.metadata else ''
        print(f"\n--- Result {i+1} (Score: {score:.4f}) ---")
        if header:
            print(f"Section: {header}")
        print("-" * 40)
        print(textwrap.fill(doc.content, width=80))
        print("-" * 40)


def embed_and_store_documents(
    document_path: str,
    output_path: str,
    model_name: str = "all-MiniLM-L6-v2",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    is_markdown: bool = True
) -> VectorStore:
    """Process documents and store embeddings for programmatic use."""
    rag = RAGSystem(
        embedding_model_name=model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        vector_store_path=output_path
    )
    
    vector_store = rag.process_document(document_path, is_markdown=is_markdown)
    return vector_store


def query_stored_embeddings(
    query: str,
    vector_store_path: str,
    k: int = 3,
    model_name: str = "all-MiniLM-L6-v2"
) -> List[str]:
    """Query stored embeddings and return relevant text chunks."""
    rag = RAGSystem(
        embedding_model_name=model_name,
        vector_store_path=vector_store_path
    )
    
    return rag.query_text(query, k=k)


def main():
    parser = argparse.ArgumentParser(description="RAG Document Processing and Querying")
    subparsers = parser.add_subparsers(help='Command help')
    
    # Parser for the 'process' command
    process_parser = subparsers.add_parser('process', help='Process a document')
    process_parser.add_argument('--document', required=True, help='Path to document file')
    process_parser.add_argument('--output', required=True, help='Path to save vector store')
    process_parser.add_argument('--model', default="all-MiniLM-L6-v2", help='Embedding model name')
    process_parser.add_argument('--chunk-size', type=int, default=512, help='Chunk size in characters')
    process_parser.add_argument('--chunk-overlap', type=int, default=50, help='Chunk overlap in characters')
    process_parser.add_argument('--markdown', action='store_true', help='Process as markdown with structure')
    process_parser.set_defaults(func=process_document)
    
    # Parser for the 'query' command
    query_parser = subparsers.add_parser('query', help='Query a processed document')
    query_parser.add_argument('--vector-store', required=True, help='Path to vector store file')
    query_parser.add_argument('--query', required=True, help='Query string')
    query_parser.add_argument('--model', default="all-MiniLM-L6-v2", help='Embedding model name')
    query_parser.add_argument('--k', type=int, default=3, help='Number of results to return')
    query_parser.set_defaults(func=query_document)
    
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()