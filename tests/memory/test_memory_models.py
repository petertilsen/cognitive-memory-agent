import unittest
import time
import numpy as np
from src.memory.models import MemoryItem, CognitiveState


class TestMemoryItem(unittest.TestCase):
    """Test cases for MemoryItem class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_content = "This is test content for memory item"
        self.test_embedding = np.array([0.1, 0.2, 0.3])

    def test_memory_item_creation(self):
        """Test MemoryItem creation with default values."""
        item = MemoryItem(content=self.test_content, embedding=self.test_embedding)
        
        self.assertEqual(item.content, self.test_content)
        np.testing.assert_array_equal(item.embedding, self.test_embedding)
        self.assertEqual(item.access_count, 0)
        self.assertEqual(item.relevance_score, 1.0)
        self.assertGreater(item.creation_time, 0)
        self.assertGreater(item.last_access_time, 0)

    def test_memory_item_with_metadata(self):
        """Test MemoryItem creation with additional parameters."""
        item = MemoryItem(
            content=self.test_content,
            embedding=self.test_embedding,
            relevance_score=0.8,
            task_context="test task"
        )
        
        self.assertEqual(item.content, self.test_content)
        self.assertEqual(item.relevance_score, 0.8)
        self.assertEqual(item.task_context, "test task")

    def test_memory_item_access_increment(self):
        """Test that access count can be incremented."""
        item = MemoryItem(content=self.test_content, embedding=self.test_embedding)
        initial_access_count = item.access_count
        
        # Use boost method to increment access
        item.boost()
        
        self.assertEqual(item.access_count, initial_access_count + 1)

    def test_memory_item_string_representation(self):
        """Test string representation of MemoryItem."""
        item = MemoryItem(content=self.test_content, embedding=self.test_embedding)
        str_repr = str(item)
        
        self.assertIn(self.test_content, str_repr)


class TestCognitiveState(unittest.TestCase):
    """Test cases for CognitiveState class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_task = "Analyze machine learning trends"

    def test_cognitive_state_creation(self):
        """Test CognitiveState creation with default values."""
        state = CognitiveState(current_task=self.test_task)
        
        self.assertEqual(state.current_task, self.test_task)
        self.assertEqual(state.confidence_score, 0.0)
        self.assertEqual(len(state.subtasks), 0)
        self.assertIsInstance(state.subtasks, list)

    def test_cognitive_state_with_task(self):
        """Test CognitiveState creation with task."""
        state = CognitiveState(
            current_task=self.test_task,
            confidence_score=0.7,
            subtasks=["subtask1", "subtask2"]
        )
        
        self.assertEqual(state.current_task, self.test_task)
        self.assertEqual(state.confidence_score, 0.7)
        self.assertEqual(len(state.subtasks), 2)

    def test_cognitive_state_attention_focus(self):
        """Test subtasks list functionality."""
        state = CognitiveState(current_task=self.test_task)
        
        # Add items to subtasks
        state.subtasks.append("subtask 1")
        state.subtasks.append("subtask 2")
        
        self.assertEqual(len(state.subtasks), 2)
        self.assertIn("subtask 1", state.subtasks)

    def test_cognitive_state_confidence_bounds(self):
        """Test confidence score bounds."""
        state = CognitiveState(current_task=self.test_task)
        
        # Test setting confidence within bounds
        state.confidence_score = 0.5
        self.assertEqual(state.confidence_score, 0.5)
        
        # Test boundary values
        state.confidence_score = 0.0
        self.assertEqual(state.confidence_score, 0.0)
        
        state.confidence_score = 1.0
        self.assertEqual(state.confidence_score, 1.0)

    def test_cognitive_state_processing_depth(self):
        """Test information gaps functionality."""
        state = CognitiveState(current_task=self.test_task)
        
        state.information_gaps.append("gap1")
        self.assertEqual(len(state.information_gaps), 1)
        
        # Test adding more gaps
        state.information_gaps.append("gap2")
        self.assertEqual(len(state.information_gaps), 2)
