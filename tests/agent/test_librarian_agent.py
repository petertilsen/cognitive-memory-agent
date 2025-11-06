"""Unit tests for librarian agent."""

import unittest
import sys
import os
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.agent.librarian_agent import LibrarianAgent


class TestLibrarianAgent(unittest.TestCase):
    """Test cases for LibrarianAgent class."""
    
    @patch('src.agent.librarian_agent.BaseCognitiveAgent.__init__')
    def test_librarian_agent_initialization(self, mock_base_init):
        """Test LibrarianAgent initialization."""
        mock_base_init.return_value = None
        
        agent = LibrarianAgent()
        
        # Should call parent constructor
        mock_base_init.assert_called_once()
        
    @patch('src.agent.librarian_agent.BaseCognitiveAgent.__init__')
    def test_get_domain_system_prompt(self, mock_base_init):
        """Test librarian-specific system prompt."""
        mock_base_init.return_value = None
        
        agent = LibrarianAgent()
        # Mock the base prompt method
        agent._get_base_system_prompt = Mock(return_value="Base ReAct prompt")
        
        prompt = agent._get_domain_system_prompt()
        
        self.assertIn("Research & Library Science", prompt)
        self.assertIn("librarian", prompt)
        self.assertIn("research assistant", prompt)
        self.assertIn("Base ReAct prompt", prompt)
        
    @patch('src.agent.librarian_agent.BaseCognitiveAgent.__init__')
    def test_get_domain_tools(self, mock_base_init):
        """Test librarian-specific tools."""
        mock_base_init.return_value = None
        
        agent = LibrarianAgent()
        tools = agent._get_domain_tools()
        
        self.assertEqual(len(tools), 2)
        # Should include book repository tools
        tool_names = [tool.__name__ for tool in tools]
        self.assertIn('fetch_book_content', tool_names)
        self.assertIn('search_openlibrary_books', tool_names)
        
    @patch('src.agent.librarian_agent.BaseCognitiveAgent.__init__')
    def test_gather_domain_information(self, mock_base_init):
        """Test domain information gathering using book tools."""
        mock_base_init.return_value = None
        
        agent = LibrarianAgent()
        # Mock the agent attribute
        agent.agent = Mock()
        agent.agent.return_value = "Book research results"
        
        info = agent._gather_domain_information("test query")
        
        self.assertEqual(info, "Book research results")
        # Should call agent with research prompt
        agent.agent.assert_called_once()
        call_args = agent.agent.call_args[0][0]
        self.assertIn("Research this topic using Open Library", call_args)
        self.assertIn("test query", call_args)
        
    @patch('src.agent.librarian_agent.BaseCognitiveAgent.__init__')
    def test_gather_domain_information_error_handling(self, mock_base_init):
        """Test error handling in domain information gathering."""
        mock_base_init.return_value = None
        
        agent = LibrarianAgent()
        # Mock the agent to raise an exception
        agent.agent = Mock()
        agent.agent.side_effect = Exception("Network error")
        
        info = agent._gather_domain_information("test query")
        
        self.assertEqual(info, "")
        
    @patch('src.agent.librarian_agent.BaseCognitiveAgent.__init__')
    def test_should_use_cognitive_processing(self, mock_base_init):
        """Test cognitive processing decision for research tasks."""
        mock_base_init.return_value = None
        
        agent = LibrarianAgent()
        
        # Should use cognitive processing for research-related queries
        research_queries = [
            "research machine learning",
            "find information about quantum computing",
            "search for books on AI",
            "analyze this topic",
            "study neural networks",
            "what is deep learning",
            "how does NLP work",
            "explain computer vision",
            "tell me about robotics"
        ]
        
        for query in research_queries:
            with self.subTest(query=query):
                self.assertTrue(agent._should_use_cognitive_processing(query))
                
        # Should not use cognitive processing for casual queries
        casual_queries = [
            "hello",
            "how are you",
            "good morning",
            "thank you",
            "goodbye"
        ]
        
        for query in casual_queries:
            with self.subTest(query=query):
                self.assertFalse(agent._should_use_cognitive_processing(query))
                
    @patch('src.agent.librarian_agent.BaseCognitiveAgent.__init__')
    @patch('src.agent.librarian_agent.BaseCognitiveAgent.process_task')
    def test_research_method_backward_compatibility(self, mock_process_task, mock_base_init):
        """Test research method as wrapper for backward compatibility."""
        mock_base_init.return_value = None
        mock_process_task.return_value = "Research result"
        
        agent = LibrarianAgent()
        result = agent.research("test query", ["doc1", "doc2"])
        
        self.assertEqual(result, "Research result")
        mock_process_task.assert_called_once_with("test query", ["doc1", "doc2"])
        
    @patch('src.agent.librarian_agent.BaseCognitiveAgent.__init__')
    @patch('src.agent.librarian_agent.BaseCognitiveAgent.process_task')
    def test_research_method_without_documents(self, mock_process_task, mock_base_init):
        """Test research method without documents."""
        mock_base_init.return_value = None
        mock_process_task.return_value = "Research result"
        
        agent = LibrarianAgent()
        result = agent.research("test query")
        
        self.assertEqual(result, "Research result")
        mock_process_task.assert_called_once_with("test query", None)
        
    @patch('src.agent.librarian_agent.BaseCognitiveAgent.__init__')
    def test_librarian_prompt_contains_required_elements(self, mock_base_init):
        """Test that librarian prompt contains all required elements."""
        mock_base_init.return_value = None
        
        agent = LibrarianAgent()
        agent._get_base_system_prompt = Mock(return_value="Base prompt")
        
        prompt = agent._get_domain_system_prompt()
        
        required_elements = [
            "librarian",
            "research assistant",
            "Research & Analysis",
            "Information Synthesis", 
            "Source Management",
            "Knowledge Building",
            "Persistent Memory",
            "Active Learning",
            "Context Awareness",
            "Progressive Reasoning"
        ]
        
        for element in required_elements:
            with self.subTest(element=element):
                self.assertIn(element, prompt)
                
    @patch('src.agent.librarian_agent.BaseCognitiveAgent.__init__')
    def test_research_prompt_structure(self, mock_base_init):
        """Test that research prompt has correct structure."""
        mock_base_init.return_value = None
        
        agent = LibrarianAgent()
        agent.agent = Mock()
        
        agent._gather_domain_information("quantum computing")
        
        # Get the prompt that was passed to the agent
        call_args = agent.agent.call_args[0][0]
        
        # Should contain structured research instructions
        self.assertIn("Research this topic using Open Library", call_args)
        self.assertIn("quantum computing", call_args)
        self.assertIn("1. Search Open Library", call_args)
        self.assertIn("2. Fetch content", call_args)
        self.assertIn("3. Extract key information", call_args)
        self.assertIn("authoritative sources", call_args)


if __name__ == '__main__':
    unittest.main()
