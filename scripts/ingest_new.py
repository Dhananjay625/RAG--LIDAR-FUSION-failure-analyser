"""Incrementally ingest only papers that are not yet in the Chroma index.

Reuses src/ unchanged. Safe to re-run: papers already present are skipped, so
this never duplicates chunks. Run from the repo root:
    venv/bin/python scripts/ingest_new.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import chromadb
from chunker import chunk_documents
from embedder import embed_chunks
from vector_db import store_embeddings_in_chroma

PAPERS_DIR = "data/Researchpapers"
CHROMA_PATH = "chroma_db"
COLLECTION = "lidar-fusion-papers"


def indexed_filenames(chroma_path: str, collection: str) -> set:
    """Filenames that already have chunks in the index."""
    client = chromadb.PersistentClient(path=chroma_path)
    col = client.get_or_create_collection(name=collection)
    if col.count() == 0:
        return set()
    got = col.get(limit=1_000_000, include=["metadatas"])
    return {(m or {}).get("filename") for m in got["metadatas"]}


def load_pdf(path: str, filename: str) -> dict:
    """Extract text from one PDF, matching data_loader's document shape."""
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        pages = [t for t in (p.extract_text() for p in pdf.pages) if t]
        return {"filename": filename, "text": "\n".join(pages), "pages": len(pdf.pages)}


def main() -> None:
    on_disk = {f for f in os.listdir(PAPERS_DIR) if f.lower().endswith(".pdf")}
    already = indexed_filenames(CHROMA_PATH, COLLECTION)

    new_files = sorted(on_disk - already)
    stale = already - on_disk
    if stale:
        print(f"WARNING: {len(stale)} indexed papers no longer on disk: {sorted(stale)[:5]}")
    if not new_files:
        print("Index is up to date - nothing to ingest.")
        return

    print(f"{len(on_disk)} PDFs on disk, {len(already)} already indexed.")
    print(f"Ingesting {len(new_files)} new papers.\n")

    for n, filename in enumerate(new_files, 1):
        path = os.path.join(PAPERS_DIR, filename)
        print(f"[{n}/{len(new_files)}] {filename}")
        try:
            doc = load_pdf(path, filename)
        except Exception as exc:
            print(f"    SKIP - could not read PDF: {exc}")
            continue

        if not doc["text"].strip():
            print("    SKIP - no extractable text (likely a scanned PDF)")
            continue

        chunks = chunk_documents([doc])
        embedded = embed_chunks(chunks)
        store_embeddings_in_chroma(embedded, chroma_path=CHROMA_PATH)
        print(f"    +{len(chunks)} chunks\n")

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    col = client.get_collection(COLLECTION)
    print(f"Done. Index now holds {col.count()} chunks.")


if __name__ == "__main__":
    main()
