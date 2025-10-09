"""Main Cognitive Memory Agent implementation."""

import os
from typing import Optional
from strands import Agent
from strands.models import BedrockModel

from ...config.settings import get_logger

logger = get_logger("agent.agent")


class CognitiveMemoryAgent:
    """Strands Agent with cognitive memory capabilities."""
    
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
        
        logger.info(f"Initializing CognitiveMemoryAgent with model: {self.model_id}, region: {self.region}")
        
        try:
            # Configure Bedrock model
            self.model = BedrockModel(
                model_id=self.model_id,
                region=self.region
            )
            logger.info("Bedrock model configured successfully")
        except Exception as e:
            logger.error(f"Failed to configure Bedrock model: {e}")
            raise
        
        # Enhanced system prompt for ReAct and cognitive behavior
        self.system_prompt = system_prompt or self._get_enhanced_system_prompt()
        
        try:
            # Create basic Strands agent
            self.agent = Agent(
                model=self.model,
                tools=[],  # No tools for now
                system_prompt=self.system_prompt
            )
            logger.info("Strands agent created successfully")
        except Exception as e:
            logger.error(f"Failed to create Strands agent: {e}")
            raise
    
    def __call__(self, message: str) -> str:
        """Process message through the cognitive agent."""
        logger.debug(f"Processing message: {message[:100]}...")
        
        try:
            response = self.agent(message)
            logger.info(f"Message processed successfully, response length: {len(response)}")
            return response
        except Exception as e:
            logger.error(f"Failed to process message: {e}", exc_info=True)
            raise
    
    async def invoke_async(self, message: str) -> str:
        """Async processing of message."""
        logger.debug(f"Processing async message: {message[:100]}...")
        
        try:
            response = await self.agent.invoke_async(message)
            logger.info(f"Async message processed successfully, response length: {len(response)}")
            return response
        except Exception as e:
            logger.error(f"Failed to process async message: {e}", exc_info=True)
            raise
    
    def _get_enhanced_system_prompt(self) -> str:
        """Get enhanced system prompt for cognitive behavior and ReAct patterns."""
        return """You are an advanced cognitive memory agent that actively manages information using ReAct patterns and multi-layered memory systems.

## Core ReAct Behavior Pattern:
1. **REASON**: Think about what you need to know or do
2. **ACT**: Take actions to gather information or perform tasks  
3. **OBSERVE**: Analyze results and update your understanding
4. **REPEAT**: Continue the cycle until the task is complete

## Enhanced Capabilities:
- Maintain persistent memory across conversations
- Build cumulative understanding over time
- Detect patterns and connections in information
- Adapt responses based on previous interactions
- Proactively organize and consolidate knowledge

## For Each User Interaction:
1. First, reason about what information you need
2. Take appropriate actions to gather or process information
3. Observe the results and update your understanding
4. Provide response based on your reasoning and observations

You are designed to be more than a simple Q&A system - you are a cognitive partner that learns, remembers, and builds understanding over time."""
