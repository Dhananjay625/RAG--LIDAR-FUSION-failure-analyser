# src/embedder.py

from typing import List, Dict
import os
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings

# src/embedder.py - UPDATED IMPORT

from langchain_ollama import OllamaEmbeddings  # Changed import

def embed_chunks(chunks: List[Dict], batch_size: int = 10) -> List[Dict]:
    """
    Convert text chunks to embeddings using local Ollama (nomic-embed-text).
    """
    
    # Initialize embeddings (local, free)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    embedded_chunks = []
    
    print(f"Embedding {len(chunks)} chunks in batches of {batch_size}...")
    
    # Loop through chunks in batches
    try:
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            batch_texts = [chunk["text"] for chunk in batch_chunks]
            
            print(f"  Processing batch {i//batch_size + 1}/{(len(chunks)//batch_size) + 1}...")
            batch_embeddings = embeddings.embed_documents(batch_texts)
            
            for chunk, embedding in zip(batch_chunks, batch_embeddings):
                embedded_chunks.append({
                    "text": chunk["text"],
                    "filename": chunk["filename"],
                    "chunk_id": chunk["chunk_id"],
                    "embedding": embedding
                })
            
    except Exception as e:
        print(f"❌ Error during embedding: {e}")
        print(f"Embedded {len(embedded_chunks)} chunks before error")
        return embedded_chunks
    
    print(f"✅ Embedded all {len(embedded_chunks)} chunks")
    return embedded_chunks


if __name__ == "__main__":
    from chunker import chunk_documents
    from data_loader import load_papers_from_directory
    
    docs = load_papers_from_directory()
    chunks = chunk_documents(docs)
    print(f"Starting embedding of {len(chunks)} chunks...")
    
    embedded = embed_chunks(chunks)
    print(f"\nComplete!")
    print(f"Sample chunk: {embedded[0]}")
    print(f"Embedding vector size: {len(embedded[0]['embedding'])}")