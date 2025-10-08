"""ChromaDB-based vector storage for cognitive memory system."""

import os
import uuid
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

from ...config.settings import get_logger

logger = get_logger("memory.vector_store")

try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False


class VectorStore:
    """ChromaDB-based vector storage with persistence."""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", 
                 chroma_host: str = "localhost", chroma_port: int = 8000,
                 collection_name: str = "cognitive_memory"):
        
        if not HAS_CHROMADB:
            logger.error("ChromaDB not installed")
            raise ImportError("ChromaDB not installed. Run: pip install chromadb")
        
        logger.info(f"Initializing VectorStore: {chroma_host}:{chroma_port}, collection: {collection_name}")
        
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        
        # Initialize embedding model
        if HAS_EMBEDDINGS and embedding_model != "simple":
            try:
                self.encoder = SentenceTransformer(embedding_model)
                self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
                logger.info(f"Sentence transformer loaded: {embedding_model}, dim: {self.embedding_dim}")
            except Exception as e:
                logger.warning(f"Failed to load sentence transformer: {e}, using fallback")
                self.encoder = None
                self.embedding_dim = 384
        else:
            logger.info("Using simple hash-based embeddings")
            self.encoder = None
            self.embedding_dim = 384
        
        # Initialize ChromaDB client
        try:
            self.client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=Settings(allow_reset=True)
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"embedding_model": embedding_model}
            )
            
            logger.info(f"Connected to ChromaDB server: {chroma_host}:{chroma_port}")
            
        except Exception as e:
            # Fallback to in-memory if ChromaDB server not available
            logger.warning(f"ChromaDB server not available at {chroma_host}:{chroma_port}: {e}")
            logger.info("Falling back to in-memory ChromaDB")
            
            self.client = chromadb.Client()
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"embedding_model": embedding_model}
            )

    def embed(self, text: str) -> List[float]:
        """Generate embedding for text."""
        if self.encoder:
            embedding = self.encoder.encode(text)
            return embedding.tolist()
        else:
            # Fallback: simple hash-based embedding
            import hashlib
            hash_obj = hashlib.md5(text.encode())
            hash_hex = hash_obj.hexdigest()
            
            vector = [
                int(hash_hex[i:i+2], 16) / 255.0 
                for i in range(0, min(len(hash_hex), self.embedding_dim * 2), 2)
            ]
            
            if len(vector) < self.embedding_dim:
                vector.extend([0.0] * (self.embedding_dim - len(vector)))
            
            return vector[:self.embedding_dim]

    def add(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add text with metadata to vector store."""
        logger.debug(f"Adding document: {text[:50]}...")
        
        try:
            doc_id = str(uuid.uuid4())
            embedding = self.embed(text)
            
            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[metadata or {}]
            )
            
            logger.debug(f"Document added successfully: {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            raise

    def search(self, query: str, top_k: int = 5, threshold: float = 0.0) -> List[Tuple[str, float, str, Dict]]:
        """Search for similar texts."""
        logger.debug(f"Searching for: '{query[:50]}...', top_k={top_k}")
        
        try:
            query_embedding = self.embed(query)
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Convert ChromaDB results to our format
            search_results = []
            if results["ids"] and len(results["ids"]) > 0:
                for i, doc_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i]
                    similarity = 1.0 - distance  # Convert distance to similarity
                    
                    if similarity >= threshold:
                        document = results["documents"][0][i]
                        metadata = results["metadatas"][0][i] or {}
                        search_results.append((doc_id, similarity, document, metadata))
            
            logger.debug(f"Search found {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def update_metadata(self, doc_id: str, metadata: Dict[str, Any]) -> None:
        """Update metadata for existing item."""
        logger.debug(f"Updating metadata for document: {doc_id}")
        
        try:
            self.collection.update(
                ids=[doc_id],
                metadatas=[metadata]
            )
            logger.debug(f"Metadata updated successfully: {doc_id}")
        except Exception as e:
            logger.warning(f"Failed to update metadata for {doc_id}: {e}")

    def remove(self, doc_id: str) -> bool:
        """Remove item by ID."""
        logger.debug(f"Removing document: {doc_id}")
        
        try:
            self.collection.delete(ids=[doc_id])
            logger.debug(f"Document removed successfully: {doc_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to remove document {doc_id}: {e}")
            return False

    def count(self) -> int:
        """Get total number of documents."""
        try:
            count = self.collection.count()
            logger.debug(f"Collection contains {count} documents")
            return count
        except Exception as e:
            logger.error(f"Failed to get document count: {e}")
            return 0

    def reset(self) -> None:
        """Clear all documents from collection."""
        logger.warning(f"Resetting collection: {self.collection_name}")
        
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"embedding_model": self.embedding_model}
            )
            logger.info(f"Collection reset successfully: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")

    @property
    def vectors(self) -> List:
        """Compatibility property for existing code."""
        # Return empty list since ChromaDB manages vectors internally
        return [None] * self.count()

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity (kept for compatibility)."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
