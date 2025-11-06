"""Base Cognitive Memory Agent Template - Boilerplate for domain-specific implementations."""

import os
from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod
from strands import Agent
from strands.models import BedrockModel
from config.settings import get_logger, load_config
from ..memory.memory_system import CognitiveMemorySystem

logger = get_logger("agent.agent")
config = load_config()


class BaseCognitiveAgent(ABC):
    """Base template for cognitive memory agents with domain-specific implementations."""
    
    def __init__(
        self, 
        model_id: Optional[str] = None,
        embedding_model_id: Optional[str] = None,
        synthesis_model_id: Optional[str] = None,
        region: Optional[str] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List] = None
    ):
        """Initialize the cognitive agent with configurable parameters."""
        # Configuration from config or environment
        self.model_id = model_id or config.model.model_id
        self.embedding_model_id = embedding_model_id or config.embedding_model.model_id
        self.synthesis_model_id = synthesis_model_id or config.synthesis_model.model_id
        self.region = region or config.model.region
        
        logger.info(f"Initializing {self.__class__.__name__} with model: {self.model_id}")
        
        # Configure main Bedrock model for agent
        self.model = BedrockModel(
            model_id=self.model_id,
            max_tokens=config.model.max_tokens
        )
        
        # Get domain-specific system prompt and tools
        self.system_prompt = system_prompt or self._get_domain_system_prompt()
        self.tools = tools or self._get_domain_tools()
        
        # Create main Strands agent with domain-specific tools
        self.agent = Agent(
            model=self.model,
            tools=self.tools,
            system_prompt=self.system_prompt
        )
        
        # Initialize cognitive memory system (manages its own models)
        self.memory_system = CognitiveMemorySystem(
            embedding_model_id=self.embedding_model_id,
            synthesis_model_id=self.synthesis_model_id,
            region=self.region
        )
        
        logger.info(f"{self.__class__.__name__} initialized successfully")
    
    def process_task(self, query: str, documents: List[str] = None) -> str:
        """Main task processing method using cognitive memory system with domain tools."""
        logger.info(f"Starting task processing: {query[:100]}...")
        
        try:
            # First, check if we have existing knowledge in memory
            logger.info("Checking cognitive memory for existing knowledge...")
            memory_result = self.memory_system.process_task(query, documents or [])
            
            # If we got high-confidence results from memory, use them
            confidence = memory_result.get('metacognitive_status', {}).get('confidence_score', 0.0)
            if confidence >= 0.8:  # High confidence threshold
                logger.info(f"High confidence ({confidence:.2f}) result from memory, using cached knowledge")
                return memory_result['final_synthesis']
            
            # If no documents provided and low confidence, use domain tools to gather information
            if not documents and confidence < 0.8:
                logger.info("Low confidence from memory, using domain tools to gather new information")

                gathered_info = self._gather_domain_information(query)
                
                documents = [gathered_info] if gathered_info else []
                
                logger.info(f"Gathered {len(documents)} documents using domain tools")
                
                # Process the new information through memory system
                memory_result = self.memory_system.process_task(query, documents)
            
            # Return the final synthesis from memory processing
            return memory_result['final_synthesis']
            
        except Exception as e:
            logger.error(f"Task processing failed: {e}")
            return f"Task processing failed: {str(e)}"
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Get current memory system status."""
        return self.memory_system.get_metacognitive_status()
    
    def __call__(self, message: str) -> str:
        """Process message through the cognitive agent."""
        logger.debug(f"Processing message: {message[:100]}...")
        
        try:
            # PLACEHOLDER: Domain-specific message routing logic
            if self._should_use_cognitive_processing(message):
                return self.process_task(message)
            else:
                # Use regular agent for other queries
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
            # PLACEHOLDER: Domain-specific async message routing
            if self._should_use_cognitive_processing(message):
                return self.process_task(message)  # Note: Could be made async
            else:
                response = await self.agent.invoke_async(message)
                logger.info(f"Async message processed successfully, response length: {len(response)}")
                return response
        except Exception as e:
            logger.error(f"Failed to process async message: {e}", exc_info=True)
            raise
    
    # ABSTRACT METHODS - Must be implemented by domain-specific agents
    
    @abstractmethod
    def _get_domain_system_prompt(self) -> str:
        """Get domain-specific system prompt. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _get_domain_tools(self) -> List:
        """Get domain-specific tools. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _gather_domain_information(self, query: str) -> str:
        """Gather information using domain-specific tools. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _should_use_cognitive_processing(self, message: str) -> bool:
        """Determine if message should use cognitive processing. Must be implemented by subclasses."""
        pass
    
    # OPTIONAL METHODS - Can be overridden by domain-specific agents
    
    def _get_base_system_prompt(self) -> str:
        """Get base system prompt that can be extended by domain-specific prompts."""
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
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess query before processing. Can be overridden by subclasses."""
        return query.strip()
    
    def _postprocess_response(self, response: str) -> str:
        """Postprocess response after processing. Can be overridden by subclasses."""
        return response
    
    def _handle_error(self, error: Exception, context: str) -> str:
        """Handle errors in a domain-specific way. Can be overridden by subclasses."""
        logger.error(f"Error in {context}: {error}")
        return f"An error occurred while processing your request: {str(error)}"


class CognitiveMemoryAgent(BaseCognitiveAgent):
    """Default implementation of cognitive memory agent - minimal example."""
    
    def _get_domain_system_prompt(self) -> str:
        """Default system prompt combining base prompt with minimal domain logic."""
        base_prompt = self._get_base_system_prompt()
        return f"""{base_prompt}

## Domain: General Purpose Assistant
You are a general-purpose cognitive assistant capable of helping with various tasks while maintaining memory and learning from interactions."""
    
    def _get_domain_tools(self) -> List:
        """Default implementation with no domain-specific tools."""
        return []
    
    def _gather_domain_information(self, query: str) -> str:
        """Default implementation - use agent to process query directly."""
        try:
            response = self.agent(f"Please provide information about: {query}")
            return str(response)
        except Exception as e:
            logger.error(f"Failed to gather information: {e}")
            return ""
    
    def _should_use_cognitive_processing(self, message: str) -> bool:
        """Default implementation - use cognitive processing for research-like queries."""
        research_keywords = ["research", "find", "search", "analyze", "study", "explain", "what is", "how does"]
        return any(keyword in message.lower() for keyword in research_keywords)
