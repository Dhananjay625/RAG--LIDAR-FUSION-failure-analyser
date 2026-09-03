# src/resume_embedding.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_papers_from_directory
from src.chunker import chunk_documents
from src.embedder import embed_chunks
from src.vector_db import store_embeddings_in_chroma
import chromadb

# Check how many chunks are already stored
client = chromadb.PersistentClient(path="chroma_db")
try:
    collection = client.get_collection("lidar-fusion-papers")
    already_stored = collection.count()
    print(f"Already stored in Chroma: {already_stored} chunks")
except:
    already_stored = 0
    print("Chroma collection empty - will create new")

# Load and chunk papers
print("Loading papers...")
docs = load_papers_from_directory()
print(f"Loaded {len(docs)} papers")

print("Chunking papers...")
chunks = chunk_documents(docs)
print(f"Created {len(chunks)} chunks")

# Skip already-stored chunks
if already_stored > 0:
    print(f"Skipping first {already_stored} chunks (already stored)...")
    chunks_to_process = chunks[already_stored:]
    print(f"Need to process {len(chunks_to_process)} remaining chunks")
else:
    chunks_to_process = chunks

if len(chunks_to_process) == 0:
    print("All chunks already stored! Nothing to do.")
else:
    # Embed remaining chunks
    print(f"Embedding {len(chunks_to_process)} remaining chunks...")
    embedded = embed_chunks(chunks_to_process)
    
    # Store them
    print(f"Storing {len(embedded)} embedded chunks...")
    store_embeddings_in_chroma(embedded)
    
    print(f"Success! Total chunks in Chroma: {already_stored + len(embedded)}")