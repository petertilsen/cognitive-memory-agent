"""Main Cognitive Memory Agent implementation."""

import os
from typing import Optional
from strands import Agent
from strands.models import BedrockModel

from ..tools.memory_tools import (
    add_to_memory, 
    retrieve_from_memory, 
    consolidate_memory, 
    get_memory_status,
    update_cognitive_state,
    search_similar_memories
)


class CognitiveMemoryAgent:
    """Strands Agent with enhanced cognitive memory capabilities and ReAct patterns."""
    
    def __init__(
        self, 
        model_id: Optional[str] = None,
        region: Optional[str] = None,
        system_prompt: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.model_id = model_id or os.getenv("BEDROCK_MODEL", "anthropic.claude-3-haiku-20240307-v1:0")
        self.region = region or os.getenv("AWS_REGION", "us-east-1")
        self.embedding_model = embedding_model
        
        # Configure Bedrock model
        self.model = BedrockModel(
            model_id=self.model_id,
            region=self.region
        )
        
        # Enhanced system prompt for ReAct and cognitive behavior
        self.system_prompt = system_prompt or self._get_enhanced_system_prompt()
        
        # Create Strands agent with enhanced memory tools
        self.agent = Agent(
            model=self.model,
            tools=[
                add_to_memory, 
                retrieve_from_memory, 
                consolidate_memory, 
                get_memory_status,
                update_cognitive_state,
                search_similar_memories
            ],
            system_prompt=self.system_prompt
        )
    
    def __call__(self, message: str) -> str:
        """Process message through the cognitive agent with ReAct pattern."""
        return self.agent(message)
    
    async def invoke_async(self, message: str) -> str:
        """Async processing of message."""
        return await self.agent.invoke_async(message)
    
    def _get_enhanced_system_prompt(self) -> str:
        """Get enhanced system prompt for cognitive behavior and ReAct patterns."""
        return """You are an advanced cognitive memory agent that actively manages information using ReAct patterns and multi-layered memory systems.

## Core ReAct Behavior Pattern:
1. **REASON**: Think about what you need to know or do
2. **ACT**: Use tools to gather information, store insights, or perform actions  
3. **OBSERVE**: Analyze results and update your understanding
4. **REPEAT**: Continue the cycle until the task is complete

## Memory Management Strategy:
- **Proactive Storage**: Always add important information to memory using add_to_memory
- **Contextual Retrieval**: Use retrieve_from_memory to find relevant context before responding
- **State Tracking**: Use update_cognitive_state to track your reasoning, actions, and observations
- **Similarity Detection**: Use search_similar_memories to avoid redundancy and find connections
- **Regular Consolidation**: Periodically use consolidate_memory to organize information

## Memory Types to Consider:
- **Factual**: Concrete information and data points
- **Procedural**: How-to knowledge and processes
- **Episodic**: Specific events and interactions
- **Preference**: User preferences and patterns

## Enhanced Capabilities:
- Maintain persistent memory across conversations
- Build cumulative understanding over time
- Detect patterns and connections in information
- Adapt responses based on previous interactions
- Proactively organize and consolidate knowledge

## For Each User Interaction:
1. First, retrieve relevant context from memory
2. Update cognitive state with current reasoning
3. Store new important information
4. Provide response based on both new input and memory context
5. Update cognitive state with actions taken and observations made

You are designed to be more than a simple Q&A system - you are a cognitive partner that learns, remembers, and builds understanding over time."""
