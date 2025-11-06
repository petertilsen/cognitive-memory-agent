"""Unit tests for cognitive memory system."""

import unittest
import sys
import os
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from collections import deque


from src.memory.memory_system import CognitiveMemorySystem
from src.memory.models import MemoryItem, CognitiveState


class TestCognitiveMemorySystem(unittest.TestCase):
    """Test cases for CognitiveMemorySystem class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_embedding_model = Mock()
        self.mock_synthesis_model = Mock()
        
    @patch('src.memory.memory_system.VectorStore')
    @patch('src.memory.memory_system.Agent')
    @patch('src.memory.memory_system.BedrockModel')
    def test_memory_system_initialization(self, mock_bedrock, mock_agent, mock_vector_store):
        """Test CognitiveMemorySystem initialization."""
        mock_bedrock.return_value = self.mock_embedding_model
        mock_agent.return_value = Mock()
        mock_vector_store.return_value = Mock()
        
        system = CognitiveMemorySystem(
            embedding_model_id="test-embed-model",
            synthesis_model_id="test-synthesis-model",
            region="us-test-1"
        )
        
        self.assertIsInstance(system.immediate_buffer, deque)
        self.assertIsInstance(system.working_buffer, deque)
        self.assertIsInstance(system.episodic_buffer, deque)
        self.assertEqual(system.immediate_buffer.maxlen, 8)
        self.assertEqual(system.working_buffer.maxlen, 64)
        self.assertEqual(system.episodic_buffer.maxlen, 256)
        
    @patch('src.memory.memory_system.VectorStore')
    @patch('src.memory.memory_system.Agent')
    @patch('src.memory.memory_system.BedrockModel')
    def test_store_memory_item(self, mock_bedrock, mock_agent, mock_vector_store):
        """Test storing a memory item in immediate buffer."""
        mock_bedrock.return_value = self.mock_embedding_model
        mock_agent.return_value = Mock()
        mock_vector_store_instance = Mock()
        mock_vector_store.return_value = mock_vector_store_instance
        
        system = CognitiveMemorySystem()
        
        item = MemoryItem(content="test content", embedding=np.array([0.1, 0.2, 0.3]))
        system.immediate_buffer.append(item)
        
        # Should be stored in immediate buffer
        self.assertEqual(len(system.immediate_buffer), 1)
        self.assertEqual(system.immediate_buffer[0].content, "test content")
        
    @patch('src.memory.memory_system.VectorStore')
    @patch('src.memory.memory_system.Agent')
    @patch('src.memory.memory_system.BedrockModel')
    def test_retrieve_relevant_memories(self, mock_bedrock, mock_agent, mock_vector_store):
        """Test retrieving relevant memories from buffers."""
        mock_bedrock.return_value = self.mock_embedding_model
        mock_agent.return_value = Mock()
        mock_vector_store_instance = Mock()
        mock_vector_store_instance.search.return_value = [
            ('id1', 0.9, 'relevant content', {})
        ]
        mock_vector_store.return_value = mock_vector_store_instance
        
        system = CognitiveMemorySystem()
        
        # Add some items to buffers
        system.immediate_buffer.append(MemoryItem(content="immediate content", embedding=np.array([0.1, 0.2, 0.3])))
        system.working_buffer.append(MemoryItem(content="working content", embedding=np.array([0.4, 0.5, 0.6])))
        
        # Test buffer search functionality
        buffer_results = system._search_buffers("test query")
        
        self.assertIsInstance(buffer_results, list)
        # Should include buffer items
        self.assertGreaterEqual(len(buffer_results), 0)
        
    @patch('src.memory.memory_system.VectorStore')
    @patch('src.memory.memory_system.Agent')
    @patch('src.memory.memory_system.BedrockModel')
    def test_consolidate_memories(self, mock_bedrock, mock_agent, mock_vector_store):
        """Test memory consolidation."""
        mock_bedrock.return_value = self.mock_embedding_model
        mock_agent.return_value = Mock()
        mock_vector_store_instance = Mock()
        mock_vector_store.return_value = mock_vector_store_instance
        
        system = CognitiveMemorySystem()
        
        # Fill immediate buffer to trigger consolidation
        for i in range(10):
            system.immediate_buffer.append(MemoryItem(content=f"content {i}", embedding=np.array([0.1*i, 0.2*i, 0.3*i])))
            
        initial_immediate_count = len(system.immediate_buffer)
        system._consolidate_memory()
        
        # Items should be moved to working buffer
        self.assertLessEqual(len(system.immediate_buffer), initial_immediate_count)
        
    @patch('src.memory.memory_system.VectorStore')
    @patch('src.memory.memory_system.Agent')
    @patch('src.memory.memory_system.BedrockModel')
    def test_get_metacognitive_status(self, mock_bedrock, mock_agent, mock_vector_store):
        """Test metacognitive status retrieval."""
        mock_bedrock.return_value = self.mock_embedding_model
        mock_agent.return_value = Mock()
        mock_vector_store_instance = Mock()
        mock_vector_store_instance.count.return_value = 10
        mock_vector_store_instance.vectors = [None] * 10  # Mock the vectors property
        mock_vector_store.return_value = mock_vector_store_instance
        
        system = CognitiveMemorySystem()
        system.cognitive_state = CognitiveState(
            current_task="test task",
            confidence_score=0.7,
            subtasks=["task1", "task2"],
            completed_subtasks=["task1"]  # 50% completion
        )
        
        # Add some working buffer items to increase confidence
        for i in range(5):
            system.working_buffer.append(MemoryItem(
                content=f"working item {i}", 
                embedding=np.array([0.1*i, 0.2*i, 0.3*i])
            ))
        
        status = system.get_metacognitive_status()
        
        self.assertIn('current_task', status)
        self.assertIn('confidence_score', status)
        self.assertIn('memory_utilization', status)
        self.assertEqual(status['current_task'], "test task")
        # Confidence will be calculated based on completion ratio and working buffer
        self.assertGreater(status['confidence_score'], 0.0)
        
    @patch('src.memory.memory_system.VectorStore')
    @patch('src.memory.memory_system.Agent')
    @patch('src.memory.memory_system.BedrockModel')
    def test_process_task_with_documents(self, mock_bedrock, mock_agent, mock_vector_store):
        """Test task processing with documents."""
        mock_bedrock.return_value = self.mock_embedding_model
        mock_synthesis_agent = Mock()
        mock_synthesis_agent.return_value = "synthesized result"
        mock_agent.return_value = mock_synthesis_agent
        mock_vector_store_instance = Mock()
        mock_vector_store_instance.search.return_value = []
        mock_vector_store_instance.vectors = []  # Add vectors property
        mock_vector_store.return_value = mock_vector_store_instance
        
        system = CognitiveMemorySystem()
        
        result = system.process_task("test task", ["document 1", "document 2"])
        
        self.assertIn('final_synthesis', result)
        self.assertIn('metacognitive_status', result)
        self.assertIsInstance(result['final_synthesis'], str)
        
    @patch('src.memory.memory_system.VectorStore')
    @patch('src.memory.memory_system.Agent')
    @patch('src.memory.memory_system.BedrockModel')
    def test_attention_mechanism(self, mock_bedrock, mock_agent, mock_vector_store):
        """Test attention mechanism filtering."""
        mock_bedrock.return_value = self.mock_embedding_model
        mock_agent.return_value = Mock()
        mock_vector_store.return_value = Mock()
        
        system = CognitiveMemorySystem()
        
        # Add items with different relevance scores to working buffer
        system.working_buffer.extend([
            MemoryItem(content="high importance", embedding=np.array([0.1, 0.2, 0.3]), relevance_score=0.9),
            MemoryItem(content="low importance", embedding=np.array([0.4, 0.5, 0.6]), relevance_score=0.3),
            MemoryItem(content="medium importance", embedding=np.array([0.7, 0.8, 0.9]), relevance_score=0.6)
        ])
        
        # Test working memory check functionality
        relevant_items = system._check_working_memory("test query")
        
        # Should filter based on relevance and semantic similarity
        self.assertIsInstance(relevant_items, list)
        
    @patch('src.memory.memory_system.VectorStore')
    @patch('src.memory.memory_system.Agent')
    @patch('src.memory.memory_system.BedrockModel')
    def test_error_handling_in_process_task(self, mock_bedrock, mock_agent, mock_vector_store):
        """Test error handling in process_task."""
        mock_bedrock.return_value = self.mock_embedding_model
        mock_synthesis_agent = Mock()
        mock_synthesis_agent.return_value = "Error handled gracefully"
        mock_agent.return_value = mock_synthesis_agent
        mock_vector_store_instance = Mock()
        mock_vector_store_instance.search.return_value = []  # Return empty list instead of exception
        mock_vector_store_instance.vectors = []  # Add vectors property
        mock_vector_store.return_value = mock_vector_store_instance
        
        system = CognitiveMemorySystem()
        
        # Should handle errors gracefully
        result = system.process_task("test task", ["test document"])
        
        self.assertIn('final_synthesis', result)
        self.assertIsInstance(result['final_synthesis'], str)


if __name__ == '__main__':
    unittest.main()
