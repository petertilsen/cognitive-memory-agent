"""Enhanced vector storage for cognitive memory system."""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import hashlib

try:
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False


class VectorStore:
    """Vector storage with semantic similarity search."""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model = embedding_model
        self.vectors: List[np.ndarray] = []
        self.texts: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        
        # Initialize embedding model
        if HAS_EMBEDDINGS and embedding_model != "simple":
            try:
                self.encoder = SentenceTransformer(embedding_model)
                self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
            except Exception:
                self.encoder = None
                self.embedding_dim = 384
        else:
            self.encoder = None
            self.embedding_dim = 384

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        if self.encoder:
            return self.encoder.encode(text)
        else:
            # Fallback: hash-based embedding
            return self._hash_embedding(text)

    def add(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        """Add text with metadata to vector store."""
        vector = self.embed(text)
        self.vectors.append(vector)
        self.texts.append(text)
        self.metadata.append(metadata or {})
        return len(self.vectors) - 1

    def search(self, query: str, top_k: int = 5, threshold: float = 0.0) -> List[Tuple[int, float, str, Dict]]:
        """Search for similar texts."""
        if not self.vectors:
            return []

        query_vector = self.embed(query)
        similarities = []

        for i, vector in enumerate(self.vectors):
            similarity = self._cosine_similarity(query_vector, vector)
            if similarity >= threshold:
                similarities.append((i, similarity, self.texts[i], self.metadata[i]))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def update_metadata(self, index: int, metadata: Dict[str, Any]) -> None:
        """Update metadata for existing item."""
        if 0 <= index < len(self.metadata):
            self.metadata[index].update(metadata)

    def remove(self, index: int) -> bool:
        """Remove item by index."""
        if 0 <= index < len(self.vectors):
            del self.vectors[index]
            del self.texts[index]
            del self.metadata[index]
            return True
        return False

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)

    def _hash_embedding(self, text: str) -> np.ndarray:
        """Generate hash-based embedding as fallback."""
        hash_obj = hashlib.md5(text.encode())
        hash_hex = hash_obj.hexdigest()
        
        # Convert to vector
        vector = np.array([
            int(hash_hex[i:i+2], 16) / 255.0 
            for i in range(0, min(len(hash_hex), self.embedding_dim * 2), 2)
        ])
        
        if len(vector) < self.embedding_dim:
            vector = np.pad(vector, (0, self.embedding_dim - len(vector)))
        
        return vector[:self.embedding_dim]
