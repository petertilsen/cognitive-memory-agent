"""Tests for cognitive memory system."""

import pytest
import sys
from pathlib import Path

# Add src to path for testing
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from src.core.memory_system import CognitiveMemorySystem


class TestCognitiveMemorySystem:
    """Test cases for cognitive memory system."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.memory_system = CognitiveMemorySystem()
    
    def test_add_memory(self):
        """Test adding memory items."""
        result = self.memory_system.add_memory("Test content", "test context")
        assert "Memory added" in result
        assert len(self.memory_system.immediate_buffer) == 1
        assert len(self.memory_system.working_buffer) == 1
    
    def test_retrieve_relevant(self):
        """Test retrieving relevant memories."""
        # Add some test memories
        self.memory_system.add_memory("Python programming", "coding")
        self.memory_system.add_memory("Machine learning", "AI")
        
        # Retrieve relevant memories
        results = self.memory_system.retrieve_relevant("Python")
        assert len(results) > 0
        assert "Python programming" in results[0]
    
    def test_consolidate_memory(self):
        """Test memory consolidation."""
        # Add memory and consolidate
        self.memory_system.add_memory("Test memory")
        results = self.memory_system.consolidate_memory()
        
        assert "removed_items" in results
        assert "working_buffer_size" in results
    
    def test_get_status(self):
        """Test getting memory status."""
        status = self.memory_system.get_status()
        
        assert "immediate_buffer" in status
        assert "working_buffer" in status
        assert "episodic_buffer" in status
        assert isinstance(status["current_time"], int)
