"""Main Cognitive Memory Agent implementation."""

import os
from typing import Optional
from strands import Agent
from strands.models import BedrockModel

from ..tools.memory_tools import (
    add_to_memory, 
    retrieve_from_memory, 
    consolidate_memory, 
    get_memory_status
)


class CognitiveMemoryAgent:
    """Strands Agent with cognitive memory capabilities."""
    
    def __init__(
        self, 
        model_id: Optional[str] = None,
        region: Optional[str] = None,
        system_prompt: Optional[str] = None
    ):
        self.model_id = model_id or os.getenv("BEDROCK_MODEL", "anthropic.claude-3-haiku-20240307-v1:0")
        self.region = region or os.getenv("AWS_REGION", "us-east-1")
        
        # Configure Bedrock model
        self.model = BedrockModel(
            model_id=self.model_id,
            region=self.region
        )
        
        # Default system prompt for cognitive behavior
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        
        # Create Strands agent with memory tools
        self.agent = Agent(
            model=self.model,
            tools=[add_to_memory, retrieve_from_memory, consolidate_memory, get_memory_status],
            system_prompt=self.system_prompt
        )
    
    def __call__(self, message: str) -> str:
        """Process message through the cognitive agent."""
        return self.agent(message)
    
    async def invoke_async(self, message: str) -> str:
        """Async processing of message."""
        return await self.agent.invoke_async(message)
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for cognitive behavior."""
        return """You are a cognitive memory agent that actively manages information using ReAct patterns.

Core behaviors:
1. REASON: Before responding, think about what information you need
2. ACT: Use memory tools to store important information and retrieve relevant context
3. OBSERVE: Check results of your actions and adjust accordingly

Memory management guidelines:
- Always add important information to memory using add_to_memory
- Retrieve relevant context using retrieve_from_memory before answering questions
- Consolidate memory periodically using consolidate_memory
- Check memory status when needed using get_memory_status

You maintain persistent memory across conversations and proactively manage information to provide better responses."""
