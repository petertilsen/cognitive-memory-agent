"""Librarian Agent - Demonstrates Cognitive Memory System capabilities."""

from typing import List
from config.settings import get_logger
from .agent import BaseCognitiveAgent
from .tools.book_repository import fetch_book_content, search_openlibrary_books

logger = get_logger("agent.librarian_agent")


class LibrarianAgent(BaseCognitiveAgent):
    """Librarian Agent that demonstrates cognitive memory capabilities for research tasks."""
    
    def _get_domain_system_prompt(self) -> str:
        """Get librarian-specific system prompt."""
        base_prompt = self._get_base_system_prompt()
        return f"""{base_prompt}

## Domain: Research & Library Science
You are an expert librarian and research assistant with advanced cognitive memory capabilities.

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
    
    def _get_domain_tools(self) -> List:
        """Get librarian-specific tools."""
        return [fetch_book_content, search_openlibrary_books]
    
    def _gather_domain_information(self, query: str) -> str:
        """Gather information using book repository tools."""
        research_prompt = f"""
        Research this topic using Open Library: {query}
        
        Please:
        1. Search Open Library for relevant books on this topic
        2. Fetch content from the most relevant books
        3. Extract key information to answer the question
        
        Focus on finding authoritative sources and comprehensive information.
        """
        
        try:
            # Let the agent use its tools to gather information
            response_text = str(self.agent(research_prompt))
            logger.info(f"Gathered information using book repository tools")
            return response_text
        except Exception as e:
            logger.error(f"Failed to gather domain information: {e}")
            return ""
    
    def _should_use_cognitive_processing(self, message: str) -> bool:
        """Determine if message should use cognitive processing for research tasks."""
        research_keywords = ["research", "find", "search", "analyze", "study", "what is", "how does", "explain", "tell me about"]
        return any(keyword in message.lower() for keyword in research_keywords)
    
    def research(self, query: str, documents: List[str] = None) -> str:
        """Main research method - wrapper around process_task for backward compatibility."""
        return self.process_task(query, documents)
