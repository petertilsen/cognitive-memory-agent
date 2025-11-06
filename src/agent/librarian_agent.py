"""Librarian Agent - Demonstrates Cognitive Memory System capabilities."""

import os
from typing import Optional, List
from strands import Agent
from strands.models import BedrockModel
from config.settings import get_logger, load_config
from ..memory.memory_system import CognitiveMemorySystem
from .tools.book_repository import fetch_book_content, search_openlibrary_books

logger = get_logger("agent.librarian_agent")
config = load_config()

class LibrarianAgent:
    """Librarian Agent that demonstrates cognitive memory capabilities for research tasks."""
    
    def __init__(
        self, 
        model_id: Optional[str] = config.model.model_id,
        embedding_model_id: Optional[str] = config.embedding_model.model_id,
        synthesis_model_id: Optional[str] = config.synthesis_model.model_id,
        region: Optional[str] = config.model.region,
        system_prompt: Optional[str] = None

    ):
        self.model_id = model_id
        self.embedding_model_id = embedding_model_id
        self.synthesis_model_id = synthesis_model_id
        self.region = region

        # Configure main Bedrock model for agent
        self.model = BedrockModel(
            model_id=self.model_id,
            max_tokens=config.model.max_tokens
        )

        # Librarian-specific system prompt
        self.system_prompt = system_prompt or self._get_librarian_system_prompt()

        # Create main Strands agent with librarian tools
        self.agent = Agent(
            model=self.model,
            tools=[fetch_book_content, search_openlibrary_books],
            system_prompt=self.system_prompt
        )

        # Initialize cognitive memory system (manages its own models)
        self.memory_system = CognitiveMemorySystem(
            embedding_model_id=embedding_model_id,
            synthesis_model_id=synthesis_model_id,
            region=region
        )

    def research(self, query: str, documents: List[str] = None) -> str:
        """Main research method using cognitive memory system with book repository tools."""
        logger.info(f"Starting research task: {query[:100]}...")
        
        try:
            # First, check if we have existing knowledge in memory
            logger.info("Checking cognitive memory for existing knowledge...")
            memory_result = self.memory_system.process_task(query, documents or [])
            
            # If we got high-confidence results from memory, use them
            confidence = memory_result.get('metacognitive_status', {}).get('confidence_score', 0.0)
            if confidence >= 0.8:  # High confidence threshold
                logger.info(f"High confidence ({confidence:.2f}) result from memory, using cached knowledge")
                return memory_result['final_synthesis']
            
            # If no documents provided and low confidence, use book repository tools to gather information
            if not documents and confidence < 0.8:
                logger.info("Low confidence from memory, using book repository tools to gather new information")
                
                # Use the agent with tools to research the topic
                research_prompt = f"""
                Research this topic using Open Library: {query}
                
                Please:
                1. Search Open Library for relevant books on this topic
                2. Fetch content from the most relevant books
                3. Extract key information to answer the question
                
                Focus on finding authoritative sources and comprehensive information.
                """
                
                # Let the agent use its tools to gather information
                response_text = str(self.agent(research_prompt))
                
                documents = [response_text] if response_text else []
                
                logger.info(f"Gathered {len(documents)} documents using book repository tools")
                
                # Process the new information through memory system
                memory_result = self.memory_system.process_task(query, documents)
            
            # Return the final synthesis from memory processing
            return memory_result['final_synthesis']
            
        except Exception as e:
            logger.error(f"Research task failed: {e}")
            return f"Research failed: {str(e)}"
    
    def get_memory_status(self) -> dict:
        """Get current memory system status."""
        return self.memory_system.get_metacognitive_status()
    
    def __call__(self, message: str) -> str:
        """Process message through the librarian agent."""
        logger.debug(f"Processing message: {message[:100]}...")
        
        try:
            # For now, route research-related queries to the memory system
            if any(keyword in message.lower() for keyword in ["research", "find", "search", "analyze", "study"]):
                return self.research(message)
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
            # For now, use synchronous research method
            if any(keyword in message.lower() for keyword in ["research", "find", "search", "analyze", "study"]):
                return self.research(message)
            else:
                response = await self.agent.invoke_async(message)
                logger.info(f"Async message processed successfully, response length: {len(response)}")
                return response
        except Exception as e:
            logger.error(f"Failed to process async message: {e}", exc_info=True)
            raise
    
    def _get_librarian_system_prompt(self) -> str:
        """Get librarian-specific system prompt."""
        return """You are an expert librarian and research assistant with advanced cognitive memory capabilities.

## Your Role:
You help users with research tasks by finding, analyzing, and synthesizing information from various sources including books, papers, articles, and other documents.

## Core Capabilities:
- **Research & Analysis**: Deep dive into topics using cognitive memory
- **Information Synthesis**: Combine information from multiple sources
- **Source Management**: Track and organize research materials
- **Knowledge Building**: Build cumulative understanding across sessions

## Cognitive Memory Features:
- **Persistent Memory**: Remember previous research and conversations
- **Active Learning**: Proactively organize and connect information
- **Context Awareness**: Understand research context and user needs
- **Progressive Reasoning**: Build insights step-by-step

## Research Process:
1. **Understand** the research question and context
2. **Analyze** available sources and documents
3. **Synthesize** findings into coherent insights
4. **Remember** key information for future reference

## Response Style:
- Provide comprehensive, well-researched answers
- Cite sources and explain reasoning
- Ask clarifying questions when needed
- Build on previous research and conversations

You are designed to be a knowledgeable research partner that learns and remembers, making each interaction more valuable than the last."""
