# src/retriever.py

from typing import List, Tuple, Dict
import chromadb
from langchain_ollama import OllamaEmbeddings

class LiDARRetriever:
    """
    Retrieves relevant paper chunks from Chroma based on similarity to query.
    """
    
    def __init__(self, chroma_path: str = "chroma_db"):
        """
        Initialize retriever with Chroma DB and embeddings.
        
        Args:
            chroma_path: Path to Chroma database
        """
        # 1. Initialize Ollama embeddings (same model used for embedding chunks)
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        # 2. Connect to persistent Chroma DB (on disk)
        self.client = chromadb.PersistentClient(path=chroma_path)
        
        # 3. Load the collection
        self.collection = self.client.get_or_create_collection(
            name="lidar-fusion-papers",
            metadata={"hnsw:space": "cosine"}
        )
        
        print(f"✅ Retriever initialized with {self.collection.count()} chunks")
    
    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve top-k most relevant chunks for a query.
        
        Args:
            query: Question about LiDAR sensor fusion
            k: Number of results to return
        
        Returns:
            List of tuples: [(chunk_text, similarity_score), ...]
        """
        # 1. Embed the query
        query_embedding = self.embeddings.embed_query(query)
        
        # 2. Search Chroma
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "distances"]
        )
        
        # 3. Return results with scores
        chunks = results["documents"][0]
        distances = results["distances"][0]
        
        # Convert distance to similarity (1 - distance for cosine)
        similarities = [1 - d for d in distances]
        
        return list(zip(chunks, similarities))
    
    def retrieve_with_metadata(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve with metadata (filename, chunk_id).
        
        Args:
            query: Question about LiDAR sensor fusion
            k: Number of results to return
        
        Returns:
            List of dicts: [{"text": "...", "filename": "...", "chunk_id": 0, "score": 0.95}, ...]
        """
        # 1. Embed query
        query_embedding = self.embeddings.embed_query(query)
        
        # 2. Search Chroma with metadata
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "distances", "metadatas"]
        )
        
        # 3. Format results
        chunks = results["documents"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]
        
        formatted_results = []
        for chunk, distance, metadata in zip(chunks, distances, metadatas):
            formatted_results.append({
                "text": chunk,
                "filename": metadata["filename"],
                "chunk_id": metadata["chunk_id"],
                "score": 1 - distance
            })
        
        return formatted_results


if __name__ == "__main__":
    print("Initializing LiDAR Retriever...")
    retriever = LiDARRetriever()
    
    # Test queries
    test_queries = [
        "Why does LiDAR fail at night?",
        "How does sensor fusion handle camera degradation?",
        "What causes BEV perception to fail in fog?",
        "LiDAR accuracy in rain conditions",
        "Sensor fusion robustness challenges"
    ]
    
    print("\n" + "="*80)
    print("TESTING RETRIEVER ON LiDAR QUERIES")
    print("="*80)
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        results = retriever.retrieve_with_metadata(query, k=3)
        
        for i, result in enumerate(results, 1):
            print(f"\n  Result {i}: (Score: {result['score']:.3f})")
            print(f"  📄 {result['filename']} (chunk {result['chunk_id']})")
            print(f"  Text: {result['text'][:150]}...")