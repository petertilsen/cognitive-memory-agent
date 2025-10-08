"""Enhanced memory management tools for Strands Agent."""

import json
from typing import Optional
from strands import tool
from ..core.memory_system import CognitiveMemorySystem

# Global memory system instance
_memory_system = CognitiveMemorySystem()


@tool
def add_to_memory(content: str, context: str = "", memory_type: str = "factual") -> str:
    """Add information to cognitive memory with context and type classification.
    
    Args:
        content: The information to store in memory
        context: Optional context or task information
        memory_type: Type of memory (factual, procedural, episodic, preference)
    
    Returns:
        JSON string with memory addition results and similar content detection
    """
    result = _memory_system.add_memory(content, context, "agent", memory_type)
    return json.dumps(result, indent=2)


@tool
def retrieve_from_memory(query: str, max_results: int = 3, include_context: bool = True) -> str:
    """Retrieve relevant information from cognitive memory using hybrid search.
    
    Args:
        query: Search query to find relevant memories
        max_results: Maximum number of results to return
        include_context: Whether to include contextual information
    
    Returns:
        JSON string with retrieved memories, similarity scores, and metadata
    """
    result = _memory_system.retrieve_relevant(query, max_results, include_context)
    return json.dumps(result, indent=2)


@tool
def consolidate_memory() -> str:
    """Consolidate and organize memory buffers using forgetting curves and promotion.
    
    Returns:
        JSON string with detailed consolidation statistics and operations performed
    """
    result = _memory_system.consolidate_memory()
    return json.dumps(result, indent=2)


@tool
def update_cognitive_state(task: str, reasoning: str = "", action: str = "", observation: str = "") -> str:
    """Update cognitive state with ReAct pattern tracking.
    
    Args:
        task: Current task or goal
        reasoning: Reasoning step in ReAct cycle
        action: Action taken in ReAct cycle
        observation: Observation from action result
    
    Returns:
        JSON string with updated cognitive state and ReAct metrics
    """
    result = _memory_system.update_cognitive_state(task, reasoning, action, observation)
    return json.dumps(result, indent=2)


@tool
def get_memory_status() -> str:
    """Get comprehensive memory system status including cognitive state and ReAct metrics.
    
    Returns:
        JSON string with detailed memory system statistics and state information
    """
    status = _memory_system.get_status()
    return json.dumps(status, indent=2)


@tool
def search_similar_memories(content: str, threshold: float = 0.7) -> str:
    """Find memories similar to given content using vector similarity.
    
    Args:
        content: Content to find similar memories for
        threshold: Similarity threshold (0.0 to 1.0)
    
    Returns:
        JSON string with similar memories and their similarity scores
    """
    similar = _memory_system._find_similar_memories(content, threshold)
    result = {
        "query_content": content[:100] + "..." if len(content) > 100 else content,
        "threshold": threshold,
        "similar_memories": [
            {"similarity": sim, "content": text[:100] + "..." if len(text) > 100 else text}
            for sim, text in similar
        ],
        "total_found": len(similar)
    }
    return json.dumps(result, indent=2)


def get_memory_system() -> CognitiveMemorySystem:
    """Get the global memory system instance for direct access."""
    return _memory_system
