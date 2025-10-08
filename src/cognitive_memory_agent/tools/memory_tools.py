"""Memory management tools for Strands Agent."""

import json
from strands import tool
from ..core.memory_system import CognitiveMemorySystem

# Global memory system instance
_memory_system = CognitiveMemorySystem()


@tool
def add_to_memory(content: str, context: str = "") -> str:
    """Add information to cognitive memory with optional context.
    
    Args:
        content: The information to store in memory
        context: Optional context or task information
    
    Returns:
        Confirmation message about the memory addition
    """
    return _memory_system.add_memory(content, context, "agent")


@tool
def retrieve_from_memory(query: str, max_results: int = 3) -> str:
    """Retrieve relevant information from cognitive memory.
    
    Args:
        query: Search query to find relevant memories
        max_results: Maximum number of results to return
    
    Returns:
        Retrieved memory content or indication if none found
    """
    relevant = _memory_system.retrieve_relevant(query, max_results)
    if relevant:
        return f"Retrieved memories: {'; '.join(relevant)}"
    return "No relevant memories found"


@tool
def consolidate_memory() -> str:
    """Consolidate and organize memory buffers using forgetting curves.
    
    Returns:
        Summary of consolidation results
    """
    results = _memory_system.consolidate_memory()
    return f"Memory consolidated: {json.dumps(results)}"


@tool
def get_memory_status() -> str:
    """Get current memory system status and buffer sizes.
    
    Returns:
        JSON string with memory system statistics
    """
    status = _memory_system.get_status()
    return json.dumps(status, indent=2)


def get_memory_system() -> CognitiveMemorySystem:
    """Get the global memory system instance."""
    return _memory_system
