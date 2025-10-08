"""Core cognitive memory system implementation."""

import time
from collections import deque
from typing import List, Dict, Optional, Any, Tuple
import numpy as np

from ..models.memory import MemoryItem, CognitiveState
from ..utils.vector_store import VectorStore
from ...config.settings import get_logger

logger = get_logger("core.memory_system")


class CognitiveMemorySystem:
    """Multi-layered cognitive memory system with vector search and ReAct integration."""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        logger.info(f"Initializing CognitiveMemorySystem with embedding_model: {embedding_model}")
        
        # Layered memory buffers
        self.immediate_buffer = deque(maxlen=8)
        self.working_buffer = deque(maxlen=64)
        self.episodic_buffer = deque(maxlen=256)
        self.semantic_memory: Dict[str, MemoryItem] = {}
        
        # Vector storage for semantic search
        try:
            self.vector_store = VectorStore(embedding_model)
            logger.info(f"Vector store initialized successfully with {self.vector_store.embedding_dim} dimensions")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
        
        # Cognitive state tracking
        self.cognitive_state: Optional[CognitiveState] = None
        self.current_time = 0
        
        # Memory management parameters
        self.attention_threshold = 0.5
        self.consolidation_threshold = 0.8
        self.similarity_threshold = 0.7
        
        logger.debug("CognitiveMemorySystem initialization complete")

    def add_memory(self, content: str, context: str = "", source: str = "user", 
                   memory_type: str = "factual") -> Dict[str, Any]:
        """Add new memory item with enhanced metadata."""
        logger.debug(f"Adding memory: type={memory_type}, source={source}, content_length={len(content)}")
        
        try:
            # Generate embedding and add to vector store
            vector_id = self.vector_store.add(content, {
                "context": context,
                "source": source,
                "memory_type": memory_type,
                "timestamp": time.time()
            })
            
            # Create memory item
            memory_item = MemoryItem(
                content=content,
                embedding=self.vector_store.vectors[vector_id],
                task_context=context,
                source=source,
                confidence=1.0 if source == "user" else 0.8
            )
            
            # Add to appropriate buffers
            self.immediate_buffer.append(memory_item)
            self.working_buffer.append(memory_item)
            
            # Check for similar existing memories
            similar_memories = self._find_similar_memories(content)
            
            result = {
                "memory_id": vector_id,
                "content_preview": content[:50] + "..." if len(content) > 50 else content,
                "similar_count": len(similar_memories),
                "buffer_sizes": self.get_status()
            }
            
            logger.info(f"Memory added successfully: id={vector_id}, similar_count={len(similar_memories)}")
            
            if similar_memories:
                logger.debug(f"Found {len(similar_memories)} similar memories with similarities: {[s for s, _ in similar_memories[:3]]}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to add memory: {e}", exc_info=True)
            raise

    def retrieve_relevant(self, query: str, top_k: int = 3, 
                         include_context: bool = True) -> Dict[str, Any]:
        """Enhanced retrieval with context and reasoning."""
        logger.debug(f"Retrieving memories for query: '{query[:100]}...', top_k={top_k}")
        
        try:
            # Vector-based semantic search
            vector_results = self.vector_store.search(query, top_k * 2, self.similarity_threshold)
            logger.debug(f"Vector search found {len(vector_results)} results")
            
            # Buffer-based contextual search
            buffer_results = self._search_buffers(query)
            logger.debug(f"Buffer search found {len(buffer_results)} results")
            
            # Combine and rank results
            combined_results = self._combine_search_results(vector_results, buffer_results, top_k)
            
            # Update access patterns
            for _, _, content, _ in combined_results:
                self._update_access_pattern(content)
            
            result = {
                "query": query,
                "results": [
                    {
                        "content": content,
                        "similarity": similarity,
                        "metadata": metadata,
                        "source_buffer": self._identify_source_buffer(content)
                    }
                    for _, similarity, content, metadata in combined_results
                ],
                "total_found": len(combined_results),
                "search_strategy": "hybrid_vector_buffer"
            }
            
            logger.info(f"Retrieved {len(combined_results)} relevant memories for query")
            return result
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}", exc_info=True)
            raise

    def consolidate_memory(self) -> Dict[str, Any]:
        """Enhanced memory consolidation with forgetting curves and promotion."""
        logger.info("Starting memory consolidation")
        self.current_time += 1
        
        consolidation_stats = {
            "before_sizes": {
                "immediate": len(self.immediate_buffer),
                "working": len(self.working_buffer),
                "episodic": len(self.episodic_buffer)
            },
            "operations": []
        }
        
        try:
            # Apply forgetting curve to working buffer
            forgotten_count = 0
            for item in list(self.working_buffer):
                item.decay(self.current_time)
                if item.relevance_score < self.attention_threshold:
                    self.working_buffer.remove(item)
                    forgotten_count += 1
            
            consolidation_stats["operations"].append(f"Forgot {forgotten_count} low-relevance items")
            logger.debug(f"Applied forgetting curve: removed {forgotten_count} items")
            
            # Promote high-value items to episodic memory
            promoted_count = 0
            for item in list(self.working_buffer):
                if (item.access_count > 2 and 
                    item.relevance_score > self.consolidation_threshold):
                    if not any(id(item) == id(e) for e in self.episodic_buffer):
                        self.episodic_buffer.append(item)
                        promoted_count += 1
            
            consolidation_stats["operations"].append(f"Promoted {promoted_count} items to episodic memory")
            logger.debug(f"Promoted {promoted_count} items to episodic memory")
            
            # Semantic clustering and organization
            cluster_count = self._organize_semantic_clusters()
            consolidation_stats["operations"].append(f"Organized {cluster_count} semantic clusters")
            logger.debug(f"Organized {cluster_count} semantic clusters")
            
            consolidation_stats["after_sizes"] = {
                "immediate": len(self.immediate_buffer),
                "working": len(self.working_buffer),
                "episodic": len(self.episodic_buffer)
            }
            
            logger.info(f"Memory consolidation complete: {consolidation_stats}")
            return consolidation_stats
            
        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}", exc_info=True)
            raise

    def update_cognitive_state(self, task: str, reasoning: str = "", 
                              action: str = "", observation: str = "") -> Dict[str, Any]:
        """Update cognitive state with task information."""
        logger.info("Updating cognitive state")
        
        if not self.cognitive_state:
            self.cognitive_state = CognitiveState(
                current_task=task,
                subtasks=[],
                completed_subtasks=[],
                information_gaps=[],
                working_hypothesis="",
                confidence_score=0.0
            )
        
        # Update cognitive state
        self.cognitive_state.current_task = task
        if reasoning or action or observation:
            context_entry = f"Task: {task}"
            if reasoning:
                context_entry += f" | Reasoning: {reasoning}"
            if action:
                context_entry += f" | Action: {action}"
            if observation:
                context_entry += f" | Observation: {observation}"
            self.cognitive_state.context_history.append(context_entry)
        
        # Simple confidence based on context history length
        self.cognitive_state.confidence_score = min(1.0, len(self.cognitive_state.context_history) * 0.1)
        
        return {
            "current_task": self.cognitive_state.current_task,
            "confidence": self.cognitive_state.confidence_score,
            "context_entries": len(self.cognitive_state.context_history)
        }

    def get_status(self) -> Dict[str, Any]:
        """Enhanced status with cognitive state and ReAct metrics."""
        base_status = {
            "immediate_buffer": len(self.immediate_buffer),
            "working_buffer": len(self.working_buffer),
            "episodic_buffer": len(self.episodic_buffer),
            "semantic_memory": len(self.semantic_memory),
            "vector_store_size": len(self.vector_store.vectors),
            "current_time": self.current_time
        }
        
        if self.cognitive_state:
            base_status.update({
                "cognitive_state": {
                    "current_task": self.cognitive_state.current_task,
                    "confidence": self.cognitive_state.confidence_score,
                    "subtasks_total": len(self.cognitive_state.subtasks),
                    "subtasks_completed": len(self.cognitive_state.completed_subtasks)
                }
            })
        
        base_status.update({
            "cognitive_metrics": {
                "has_cognitive_state": self.cognitive_state is not None,
                "context_history_length": len(self.cognitive_state.context_history) if self.cognitive_state else 0
            }
        })
        
        return base_status

    def _find_similar_memories(self, content: str, threshold: float = 0.8) -> List[Tuple[float, str]]:
        """Find similar existing memories."""
        results = self.vector_store.search(content, top_k=5, threshold=threshold)
        return [(similarity, text) for _, similarity, text, _ in results]

    def _search_buffers(self, query: str) -> List[Tuple[str, float, str]]:
        """Search memory buffers for relevant content."""
        results = []
        query_words = set(query.lower().split())
        
        # Search all buffers
        all_buffers = [
            ("immediate", self.immediate_buffer),
            ("working", self.working_buffer),
            ("episodic", self.episodic_buffer)
        ]
        
        for buffer_name, buffer in all_buffers:
            for item in buffer:
                content_words = set(item.content.lower().split())
                overlap = len(query_words & content_words)
                if overlap > 0:
                    relevance = overlap / len(query_words)
                    results.append((buffer_name, relevance, item.content))
        
        return results

    def _combine_search_results(self, vector_results: List[Tuple], 
                               buffer_results: List[Tuple], top_k: int) -> List[Tuple]:
        """Combine and rank search results from different sources."""
        combined = []
        
        # Add vector results with metadata
        for idx, similarity, content, metadata in vector_results:
            combined.append((idx, similarity * 1.2, content, metadata))  # Boost vector results
        
        # Add buffer results
        for buffer_name, relevance, content in buffer_results:
            # Check if already in vector results
            if not any(content == c for _, _, c, _ in combined):
                combined.append((len(combined), relevance, content, {"source_buffer": buffer_name}))
        
        # Sort by similarity/relevance and return top_k
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:top_k]

    def _update_access_pattern(self, content: str) -> None:
        """Update access patterns for retrieved content."""
        # Find and boost corresponding memory items
        for buffer in [self.immediate_buffer, self.working_buffer, self.episodic_buffer]:
            for item in buffer:
                if item.content == content:
                    item.boost()
                    break

    def _identify_source_buffer(self, content: str) -> str:
        """Identify which buffer contains the content."""
        for buffer_name, buffer in [
            ("immediate", self.immediate_buffer),
            ("working", self.working_buffer),
            ("episodic", self.episodic_buffer)
        ]:
            if any(item.content == content for item in buffer):
                return buffer_name
        return "vector_store"

    def _organize_semantic_clusters(self) -> int:
        """Organize memories into semantic clusters."""
        # Simplified clustering based on content similarity
        clusters = {}
        cluster_count = 0
        
        for item in self.episodic_buffer:
            # Find cluster or create new one
            assigned = False
            for cluster_key, cluster_items in clusters.items():
                if len(cluster_items) > 0:
                    similarity = self.vector_store._cosine_similarity(
                        item.embedding, cluster_items[0].embedding
                    )
                    if similarity > 0.8:
                        cluster_items.append(item)
                        assigned = True
                        break
            
            if not assigned:
                clusters[f"cluster_{cluster_count}"] = [item]
                cluster_count += 1
        
        return len(clusters)
