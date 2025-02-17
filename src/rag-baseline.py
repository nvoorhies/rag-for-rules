import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Dict
import torch
from dataclasses import dataclass
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
    
    def chunk_text(self, text: str) -> List[Document]:
        """Split text into overlapping chunks."""
        # Clean and normalize text
        text = text.replace('\n', ' ').strip()
        
        # Generate chunks with overlap
        chunks = []
        start = 0
        while start < len(text):
            # Get chunk of specified size
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # If not the last chunk, try to break at sentence boundary
            if end < len(text):
                # Find last period or punctuation mark
                last_period = max(
                    chunk.rfind('.'),
                    chunk.rfind('?'),
                    chunk.rfind('!')
                )
                if last_period != -1:
                    end = start + last_period + 1
                    chunk = text[start:end]
            
            chunks.append(Document(
                content=chunk,
                metadata={'start_char': start, 'end_char': end}
            ))
            
            # Move start position, accounting for overlap
            start = end - self.chunk_overlap
        
        return chunks

class VectorStore:
    """Simple vector store using numpy arrays and cosine similarity."""
    
    def __init__(self, embedding_dim: int):
        self.documents: List[Document] = []
        self.embeddings: np.ndarray = None
        self.embedding_dim = embedding_dim
    
    def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        """Add documents and their embeddings to the store."""
        self.documents.extend(documents)
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
    
    def similarity_search(self, query_embedding: np.ndarray, k: int = 3) -> List[Tuple[Document, float]]:
        """Find k most similar documents using cosine similarity."""
        # Compute cosine similarity
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()
        
        # Get top k indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        # Return documents and scores
        results = []
        for idx in top_k_indices:
            results.append((self.documents[idx], float(similarities[idx])))
        
        return results

class EmbeddingModel:
    """Wrapper for sentence-transformers embedding model."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def embed_documents(self, documents: List[Document]) -> np.ndarray:
        """Generate embeddings for a list of documents."""
        texts = [doc.content for doc in documents]
        return self.model.encode(texts, convert_to_tensor=True).cpu().numpy()
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a query string."""
        return self.model.encode([query], convert_to_tensor=True).cpu().numpy()

class LlamaQA:
    """Simple QA system using Llama model."""
    
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-hf"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    def generate_answer(self, query: str, context: List[Document]) -> str:
        """Generate answer based on query and retrieved context."""
        # Combine context documents
        context_text = "\n\n".join([doc.content for doc in context])
        
        # Create prompt
        prompt = f"""Please answer the question based on the following context:

Context:
{context_text}

Question: {query}

Answer:"""
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class RAGSystem:
    """Complete RAG system combining all components."""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        llm_model_name: str = "meta-llama/Llama-2-7b-hf"
    ):
        self.chunker = TextChunker(chunk_size, chunk_overlap)
        self.embedding_model = EmbeddingModel(embedding_model_name)
        self.vector_store = VectorStore(self.embedding_model.embedding_dim)
        self.qa_model = LlamaQA(llm_model_name)
    
    def add_documents(self, texts: List[str]):
        """Process and add documents to the system."""
        for text in texts:
            # Chunk document
            chunks = self.chunker.chunk_text(text)
            
            # Generate embeddings
            embeddings = self.embedding_model.embed_documents(chunks)
            
            # Add to vector store
            self.vector_store.add_documents(chunks, embeddings)
    
    def query(self, query: str, k: int = 3) -> Tuple[str, List[Document]]:
        """Process query and return answer with supporting documents."""
        # Generate query embedding
        query_embedding = self.embedding_model.embed_query(query)
        
        # Retrieve relevant documents
        retrieved_docs = self.vector_store.similarity_search(query_embedding, k=k)
        
        # Generate answer
        relevant_docs = [doc for doc, score in retrieved_docs]
        answer = self.qa_model.generate_answer(query, relevant_docs)
        
        return answer, relevant_docs

# Example usage
def main():
    # Initialize RAG system
    rag = RAGSystem()
    
    # Add example documents
    documents = [
        "The cat sat on the mat. It was a sunny day.",
        "The dog chased the ball in the park. The children were playing nearby.",
    ]
    rag.add_documents(documents)
    
    # Query the system
    query = "What was the cat doing?"
    answer, supporting_docs = rag.query(query)
    
    print(f"Query: {query}")
    print("\nAnswer:", textwrap.fill(answer, width=80))
    print("\nSupporting documents:")
    for doc in supporting_docs:
        print("-", textwrap.fill(doc.content, width=80))

if __name__ == "__main__":
    main()