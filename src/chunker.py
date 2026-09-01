# src/chunker.py

from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict

def chunk_documents(documents: List[Dict], chunk_size: int = 400, chunk_overlap: int = 50) -> List[Dict]:
    """
    Split documents into chunks.
    
    Args:
        documents: List from load_papers_from_directory()
        chunk_size: Approximate tokens per chunk (~400 tokens ≈ 300 words)
        chunk_overlap: Overlap between chunks for context
    
    Returns:
        List of dicts: [{"text": "chunk text", "filename": "paper.pdf", "chunk_id": 0}, ...]
    """
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = []
    
    # Your code here:
    # 1. Loop through each document
    for doc in documents:
        filename = doc['filename']
        text = doc['text']

    # 2. Split its text using splitter.split_text()
        chunks_text = splitter.split_text(text)

    # 3. For each chunk, store: text, filename, chunk_id (0, 1, 2, ...)
        for chunk_id, chunk_text in enumerate(chunks_text):
    # 4. Append to chunks list
            chunks.append({
                "text": chunk_text,
                "filename": filename,
                "chunk_id": chunk_id
            })
    
    # 5. Return all chunks
    
    return chunks


# Test it
if __name__ == "__main__":
    from data_loader import load_papers_from_directory
    
    # Load papers
    docs = load_papers_from_directory()
    print(f"Loaded {len(docs)} papers")
    
    # Chunk them
    chunked = chunk_documents(docs)
    print(f"Created {len(chunked)} chunks")
    
    # Show stats
    for i, chunk in enumerate(chunked[:3]):  # First 3 chunks
        print(f"\n--- Chunk {i} ---")
        print(f"From: {chunk['filename']}")
        print(f"Text: {chunk['text'][:150]}...")