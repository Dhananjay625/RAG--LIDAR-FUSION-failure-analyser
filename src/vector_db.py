# src/vector_store.py

from typing import List, Dict
import os
from chromadb.config import Settings
import chromadb

def store_embeddings_in_chroma(embedded_chunks: List[Dict], chroma_path: str = "chroma_db", batch_size: int = 100) -> None:
    """
    Store embedded chunks in Chroma vector database.
    """
    
    # Initialize Chroma with PERSISTENT storage (creates folder on disk)
    client = chromadb.PersistentClient(path=chroma_path)  # ← CHANGED
    collection = client.get_or_create_collection(
        name="lidar-fusion-papers",
        metadata={"hnsw:space": "cosine"}
    )
    
    print(f"Storing {len(embedded_chunks)} chunks in Chroma at {chroma_path}...")
    
    for i in range(0, len(embedded_chunks), batch_size):
        batch = embedded_chunks[i:i+batch_size]
        ids = [f"{chunk['filename']}_{chunk['chunk_id']}" for chunk in batch]
        embeddings = [chunk['embedding'] for chunk in batch]
        documents = [chunk['text'] for chunk in batch]
        metadatas = [{"filename": chunk['filename'], "chunk_id": chunk['chunk_id']} for chunk in batch]
        
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        if (i + batch_size) % 500 == 0:
            print(f"  Stored {i + batch_size}/{len(embedded_chunks)} chunks...")

    print(f"✅ Successfully stored all {len(embedded_chunks)} chunks in Chroma at {chroma_path}")

if __name__ == "__main__":
    from embedder import embed_chunks
    from chunker import chunk_documents
    from data_loader import load_papers_from_directory
    
    docs = load_papers_from_directory()
    chunks = chunk_documents(docs)
    embedded = embed_chunks(chunks)
    
    store_embeddings_in_chroma(embedded)