# src/vector_store.py

from typing import List, Dict
import os
from chromadb.config import Settings
import chromadb

def store_embeddings_in_chroma(embedded_chunks: List[Dict], chroma_path: str = "chroma_db", batch_size: int = 100) -> None:
    """
    Store embedded chunks in Chroma vector database.
    
    Args:
        embedded_chunks: List from embed_chunks()
        chroma_path: Path to store Chroma DB
    """
    
    # Initialize Chroma
    client = chromadb.Client()
    collection = client.get_or_create_collection(
        name="lidar-fusion-papers",
        metadata={"hnsw:space": "cosine"}
    )
    
    print(f"Storing {len(embedded_chunks)} chunks in Chroma (batch size: {batch_size})...")
    
    # Your code here:
    # 1. For each embedded chunk:
    #    a. Extract: id, embedding, text, metadata (filename, chunk_id)
    #    b. Add to collection using collection.add()
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
        print(f"  Stored {i + batch_size} chunks so far...")

    print(f"Successfully stored all {len(embedded_chunks)} chunks in Chroma at {chroma_path}")

    
    # 2. Persist to disk
    # 3. Print success message
    
    # HINT: collection.add(
    #     ids=[...],
    #     embeddings=[...],
    #     metadatas=[...],
    #     documents=[...]
    # )

if __name__ == "__main__":
    from embedder import embed_chunks
    from chunker import chunk_documents
    from data_loader import load_papers_from_directory
    
    docs = load_papers_from_directory()
    chunks = chunk_documents(docs)
    embedded = embed_chunks(chunks)
    
    store_embeddings_in_chroma(embedded)