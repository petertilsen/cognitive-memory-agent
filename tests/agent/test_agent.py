"""Unit tests for base agent functionality."""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from abc import ABC


from src.agent.agent import BaseCognitiveAgent, CognitiveMemoryAgent


class TestBaseCognitiveAgent(unittest.TestCase):
    """Test cases for BaseCognitiveAgent abstract class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a concrete implementation for testing
        class TestAgent(BaseCognitiveAgent):
            def _get_domain_system_prompt(self):
                return "Test domain prompt"
                
            def _get_domain_tools(self):
                return []
                
            def _gather_domain_information(self, query):
                return f"Domain info for: {query}"
                
            def _should_use_cognitive_processing(self, message):
                return "research" in message.lower()
        
        self.TestAgent = TestAgent
        
    @patch('src.agent.agent.CognitiveMemorySystem')
    @patch('src.agent.agent.Agent')
    @patch('src.agent.agent.BedrockModel')
    def test_base_agent_initialization(self, mock_bedrock, mock_agent, mock_memory_system):
        """Test BaseCognitiveAgent initialization."""
        mock_bedrock.return_value = Mock()
        mock_agent.return_value = Mock()
        mock_memory_system.return_value = Mock()
        
        agent = self.TestAgent()
        
        self.assertIsNotNone(agent.model_id)
        self.assertIsNotNone(agent.embedding_model_id)
        self.assertIsNotNone(agent.synthesis_model_id)
        self.assertIsNotNone(agent.region)
        self.assertIsNotNone(agent.model)
        self.assertIsNotNone(agent.agent)
        self.assertIsNotNone(agent.memory_system)
        
    @patch('src.agent.agent.CognitiveMemorySystem')
    @patch('src.agent.agent.Agent')
    @patch('src.agent.agent.BedrockModel')
    def test_process_task_high_confidence(self, mock_bedrock, mock_agent, mock_memory_system):
        """Test process_task with high confidence memory result."""
        mock_bedrock.return_value = Mock()
        mock_agent.return_value = Mock()
        mock_memory_system_instance = Mock()
        mock_memory_system_instance.process_task.return_value = {
            'final_synthesis': 'High confidence result',
            'metacognitive_status': {'confidence_score': 0.9}
        }
        mock_memory_system.return_value = mock_memory_system_instance
        
        agent = self.TestAgent()
        result = agent.process_task("test query")
        
        self.assertEqual(result, 'High confidence result')
        mock_memory_system_instance.process_task.assert_called_once()
        
    @patch('src.agent.agent.CognitiveMemorySystem')
    @patch('src.agent.agent.Agent')
    @patch('src.agent.agent.BedrockModel')
    def test_process_task_low_confidence(self, mock_bedrock, mock_agent, mock_memory_system):
        """Test process_task with low confidence requiring domain information gathering."""
        mock_bedrock.return_value = Mock()
        mock_agent_instance = Mock()
        mock_agent.return_value = mock_agent_instance
        
        mock_memory_system_instance = Mock()
        # First call returns low confidence, second call returns final result
        mock_memory_system_instance.process_task.side_effect = [
            {
                'final_synthesis': 'Low confidence result',
                'metacognitive_status': {'confidence_score': 0.3}
            },
            {
                'final_synthesis': 'Enhanced result with domain info',
                'metacognitive_status': {'confidence_score': 0.8}
            }
        ]
        mock_memory_system.return_value = mock_memory_system_instance
        
        agent = self.TestAgent()
        result = agent.process_task("test query")
        
        self.assertEqual(result, 'Enhanced result with domain info')
        # Should be called twice - once for initial check, once after gathering info
        self.assertEqual(mock_memory_system_instance.process_task.call_count, 2)
        
    @patch('src.agent.agent.CognitiveMemorySystem')
    @patch('src.agent.agent.Agent')
    @patch('src.agent.agent.BedrockModel')
    def test_call_method_cognitive_processing(self, mock_bedrock, mock_agent, mock_memory_system):
        """Test __call__ method with cognitive processing."""
        mock_bedrock.return_value = Mock()
        mock_agent.return_value = Mock()
        mock_memory_system_instance = Mock()
        mock_memory_system_instance.process_task.return_value = {
            'final_synthesis': 'Research result',
            'metacognitive_status': {'confidence_score': 0.9}
        }
        mock_memory_system.return_value = mock_memory_system_instance
        
        agent = self.TestAgent()
        result = agent("research this topic")
        
        self.assertEqual(result, 'Research result')
        
    @patch('src.agent.agent.CognitiveMemorySystem')
    @patch('src.agent.agent.Agent')
    @patch('src.agent.agent.BedrockModel')
    def test_call_method_regular_processing(self, mock_bedrock, mock_agent, mock_memory_system):
        """Test __call__ method with regular agent processing."""
        mock_bedrock.return_value = Mock()
        mock_agent_instance = Mock()
        mock_agent_instance.return_value = "Regular agent response"
        mock_agent.return_value = mock_agent_instance
        mock_memory_system.return_value = Mock()
        
        agent = self.TestAgent()
        result = agent("hello there")
        
        self.assertEqual(result, "Regular agent response")
        mock_agent_instance.assert_called_once_with("hello there")
        
    @patch('src.agent.agent.CognitiveMemorySystem')
    @patch('src.agent.agent.Agent')
    @patch('src.agent.agent.BedrockModel')
    def test_get_memory_status(self, mock_bedrock, mock_agent, mock_memory_system):
        """Test get_memory_status method."""
        mock_bedrock.return_value = Mock()
        mock_agent.return_value = Mock()
        mock_memory_system_instance = Mock()
        mock_memory_system_instance.get_metacognitive_status.return_value = {
            'current_task': 'test',
            'confidence_score': 0.7
        }
        mock_memory_system.return_value = mock_memory_system_instance
        
        agent = self.TestAgent()
        status = agent.get_memory_status()
        
        self.assertEqual(status['current_task'], 'test')
        self.assertEqual(status['confidence_score'], 0.7)
        
    @patch('src.agent.agent.CognitiveMemorySystem')
    @patch('src.agent.agent.Agent')
    @patch('src.agent.agent.BedrockModel')
    def test_error_handling_in_process_task(self, mock_bedrock, mock_agent, mock_memory_system):
        """Test error handling in process_task."""
        mock_bedrock.return_value = Mock()
        mock_agent.return_value = Mock()
        mock_memory_system_instance = Mock()
        mock_memory_system_instance.process_task.side_effect = Exception("Memory error")
        mock_memory_system.return_value = mock_memory_system_instance
        
        agent = self.TestAgent()
        result = agent.process_task("test query")
        
        self.assertIn("Task processing failed", result)
        self.assertIn("Memory error", result)


class TestCognitiveMemoryAgent(unittest.TestCase):
    """Test cases for CognitiveMemoryAgent default implementation."""
    
    @patch('src.agent.agent.CognitiveMemorySystem')
    @patch('src.agent.agent.Agent')
    @patch('src.agent.agent.BedrockModel')
    def test_default_implementation_initialization(self, mock_bedrock, mock_agent, mock_memory_system):
        """Test CognitiveMemoryAgent initialization."""
        mock_bedrock.return_value = Mock()
        mock_agent.return_value = Mock()
        mock_memory_system.return_value = Mock()
        
        agent = CognitiveMemoryAgent()
        
        self.assertIsInstance(agent, BaseCognitiveAgent)
        
    @patch('src.agent.agent.CognitiveMemorySystem')
    @patch('src.agent.agent.Agent')
    @patch('src.agent.agent.BedrockModel')
    def test_default_domain_system_prompt(self, mock_bedrock, mock_agent, mock_memory_system):
        """Test default domain system prompt."""
        mock_bedrock.return_value = Mock()
        mock_agent.return_value = Mock()
        mock_memory_system.return_value = Mock()
        
        agent = CognitiveMemoryAgent()
        prompt = agent._get_domain_system_prompt()
        
        self.assertIn("General Purpose Assistant", prompt)
        self.assertIn("ReAct", prompt)
        
    @patch('src.agent.agent.CognitiveMemorySystem')
    @patch('src.agent.agent.Agent')
    @patch('src.agent.agent.BedrockModel')
    def test_default_domain_tools(self, mock_bedrock, mock_agent, mock_memory_system):
        """Test default domain tools."""
        mock_bedrock.return_value = Mock()
        mock_agent.return_value = Mock()
        mock_memory_system.return_value = Mock()
        
        agent = CognitiveMemoryAgent()
        tools = agent._get_domain_tools()
        
        self.assertEqual(tools, [])
        
    @patch('src.agent.agent.CognitiveMemorySystem')
    @patch('src.agent.agent.Agent')
    @patch('src.agent.agent.BedrockModel')
    def test_default_gather_domain_information(self, mock_bedrock, mock_agent, mock_memory_system):
        """Test default domain information gathering."""
        mock_bedrock.return_value = Mock()
        mock_agent_instance = Mock()
        mock_agent_instance.return_value = "Agent response"
        mock_agent.return_value = mock_agent_instance
        mock_memory_system.return_value = Mock()
        
        agent = CognitiveMemoryAgent()
        info = agent._gather_domain_information("test query")
        
        self.assertEqual(info, "Agent response")
        
    @patch('src.agent.agent.CognitiveMemorySystem')
    @patch('src.agent.agent.Agent')
    @patch('src.agent.agent.BedrockModel')
    def test_default_should_use_cognitive_processing(self, mock_bedrock, mock_agent, mock_memory_system):
        """Test default cognitive processing decision."""
        mock_bedrock.return_value = Mock()
        mock_agent.return_value = Mock()
        mock_memory_system.return_value = Mock()
        
        agent = CognitiveMemoryAgent()
        
        # Should use cognitive processing for research queries
        self.assertTrue(agent._should_use_cognitive_processing("research this topic"))
        self.assertTrue(agent._should_use_cognitive_processing("what is machine learning"))
        self.assertTrue(agent._should_use_cognitive_processing("explain quantum computing"))
        
        # Should not use cognitive processing for casual queries
        self.assertFalse(agent._should_use_cognitive_processing("hello"))
        self.assertFalse(agent._should_use_cognitive_processing("how are you"))


if __name__ == '__main__':
    unittest.main()
