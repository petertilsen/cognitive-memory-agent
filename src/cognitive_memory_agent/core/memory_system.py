"""Core cognitive memory system implementation."""

import time
from collections import deque
from typing import List, Dict, Optional
import numpy as np

from ..models.memory import MemoryItem, CognitiveState


class CognitiveMemorySystem:
    """Multi-layered cognitive memory system."""
    
    def __init__(self):
        # Layered memory buffers
        self.immediate_buffer = deque(maxlen=8)
        self.working_buffer = deque(maxlen=64)
        self.episodic_buffer = deque(maxlen=256)
        self.semantic_memory: Dict[str, MemoryItem] = {}
        
        # Cognitive state
        self.cognitive_state: Optional[CognitiveState] = None
        self.current_time = 0
        
        # Memory management parameters
        self.attention_threshold = 0.5
        self.consolidation_threshold = 0.8

    def add_memory(self, content: str, context: str = "", source: str = "user") -> str:
        """Add new memory item to the system."""
        memory_item = MemoryItem(
            content=content,
            embedding=self._generate_embedding(content),
            task_context=context,
            source=source
        )
        
        self.immediate_buffer.append(memory_item)
        self.working_buffer.append(memory_item)
        
        return f"Memory added: {content[:50]}..."

    def retrieve_relevant(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve relevant memories based on query."""
        relevant_items = []
        all_buffers = list(self.working_buffer) + list(self.episodic_buffer)
        
        query_words = set(query.lower().split())
        
        for item in all_buffers:
            content_words = set(item.content.lower().split())
            
            # Simple relevance scoring
            if query_words & content_words:
                item.boost()
                relevant_items.append((item.relevance_score, item.content))
        
        # Sort by relevance and return top_k
        relevant_items.sort(key=lambda x: x[0], reverse=True)
        return [content for _, content in relevant_items[:top_k]]

    def consolidate_memory(self) -> Dict[str, int]:
        """Consolidate and organize memory buffers."""
        self.current_time += 1
        
        # Apply forgetting curve
        for item in self.working_buffer:
            item.decay(self.current_time)
        
        # Remove low relevance items
        before_count = len(self.working_buffer)
        self.working_buffer = deque(
            [item for item in self.working_buffer 
             if item.relevance_score > self.attention_threshold],
            maxlen=self.working_buffer.maxlen
        )
        
        # Promote high-access items to episodic memory
        for item in self.working_buffer:
            if (item.access_count > 2 and 
                item.relevance_score > self.consolidation_threshold):
                if not any(id(item) == id(e) for e in self.episodic_buffer):
                    self.episodic_buffer.append(item)
        
        return {
            "removed_items": before_count - len(self.working_buffer),
            "working_buffer_size": len(self.working_buffer),
            "episodic_buffer_size": len(self.episodic_buffer)
        }

    def get_status(self) -> Dict[str, int]:
        """Get current memory system status."""
        return {
            "immediate_buffer": len(self.immediate_buffer),
            "working_buffer": len(self.working_buffer),
            "episodic_buffer": len(self.episodic_buffer),
            "semantic_memory": len(self.semantic_memory),
            "current_time": self.current_time
        }

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate simple embedding for text."""
        # Simplified hash-based embedding for now
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_hex = hash_obj.hexdigest()
        
        vector = np.array([
            int(hash_hex[i:i+2], 16) / 255.0 
            for i in range(0, min(len(hash_hex), 384*2), 2)
        ])
        
        if len(vector) < 384:
            vector = np.pad(vector, (0, 384 - len(vector)))
        
        return vector[:384]
