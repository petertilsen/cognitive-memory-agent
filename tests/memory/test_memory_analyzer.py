"""Unit tests for memory analyzer."""

import unittest
import sys
import os
import numpy as np
from unittest.mock import Mock, patch
from collections import deque

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.memory.analyzer import MemoryAnalyzer
from src.memory.models import MemoryItem


class TestMemoryAnalyzer(unittest.TestCase):
    """Test cases for MemoryAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_memory_system = Mock()
        self.mock_memory_system.immediate_buffer = deque(maxlen=8)
        self.mock_memory_system.working_buffer = deque(maxlen=64)
        self.mock_memory_system.episodic_buffer = deque(maxlen=256)
        self.mock_memory_system.vector_store = Mock()
        self.mock_memory_system.vector_store.count.return_value = 10
        self.mock_memory_system.attention_threshold = 0.5
        self.mock_memory_system.consolidation_threshold = 0.8
        self.mock_memory_system.similarity_threshold = 0.7
        self.mock_memory_system.current_time = 1672531200.0  # 2023-01-01 as timestamp
        self.mock_memory_system.cognitive_state = Mock()
        self.mock_memory_system.cognitive_state.current_task = "test task"
        self.mock_memory_system.cognitive_state.confidence_score = 0.8
        self.mock_memory_system.operation_logs = []
        
    def test_analyzer_initialization(self):
        """Test MemoryAnalyzer initialization."""
        analyzer = MemoryAnalyzer(self.mock_memory_system)
        
        self.assertEqual(analyzer.memory_system, self.mock_memory_system)
        self.assertEqual(analyzer.analysis_history, [])
        
    def test_analyze_buffer_flow(self):
        """Test buffer flow analysis."""
        # Add test items to buffers
        self.mock_memory_system.immediate_buffer.append(
            MemoryItem(content="immediate item", embedding=np.array([0.1, 0.2, 0.3]))
        )
        self.mock_memory_system.working_buffer.extend([
            MemoryItem(content="working item 1", embedding=np.array([0.4, 0.5, 0.6])),
            MemoryItem(content="working item 2", embedding=np.array([0.7, 0.8, 0.9]))
        ])
        
        analyzer = MemoryAnalyzer(self.mock_memory_system)
        flow_analysis = analyzer.analyze_buffer_flow()
        
        self.assertIn('immediate_buffer', flow_analysis)
        self.assertIn('working_buffer', flow_analysis)
        self.assertIn('episodic_buffer', flow_analysis)
        self.assertIn('vector_store', flow_analysis)
        
        # Check immediate buffer analysis
        immediate = flow_analysis['immediate_buffer']
        self.assertEqual(immediate['size'], 1)
        self.assertEqual(immediate['capacity'], 8)
        self.assertAlmostEqual(immediate['utilization'], 1/8)
        
        # Check working buffer analysis
        working = flow_analysis['working_buffer']
        self.assertEqual(working['size'], 2)
        self.assertEqual(working['capacity'], 64)
        self.assertAlmostEqual(working['utilization'], 2/64)
        
    def test_track_memory_reuse(self):
        """Test memory reuse tracking."""
        # Mock operation logs
        self.mock_memory_system.operation_logs = [
            {'type': 'task_reuse', 'content': 'reused task'},
            {'type': 'memory_reuse', 'content': 'reused memory'},
            {'type': 'new_info', 'content': 'new information'}
        ]
        
        analyzer = MemoryAnalyzer(self.mock_memory_system)
        reuse_analysis = analyzer.track_memory_reuse()
        
        self.assertIn('reuse_rate', reuse_analysis)
        self.assertIn('task_reuse_operations', reuse_analysis)
        self.assertIn('memory_reuse_operations', reuse_analysis)
        self.assertIn('new_info_operations', reuse_analysis)
        
        # Check calculations
        self.assertEqual(reuse_analysis['task_reuse_operations'], 1)
        self.assertEqual(reuse_analysis['memory_reuse_operations'], 1)
        self.assertEqual(reuse_analysis['new_info_operations'], 1)
        self.assertAlmostEqual(reuse_analysis['reuse_rate'], 2/3)
        
    def test_visualize_consolidation_patterns(self):
        """Test consolidation pattern visualization."""
        # Add items with different access patterns
        high_access_item = MemoryItem(content="popular item", embedding=np.array([0.1, 0.2, 0.3]), access_count=10)
        low_access_item = MemoryItem(content="unpopular item", embedding=np.array([0.4, 0.5, 0.6]), access_count=1)
        
        self.mock_memory_system.working_buffer.extend([
            high_access_item, low_access_item
        ])
        
        analyzer = MemoryAnalyzer(self.mock_memory_system)
        consolidation = analyzer.visualize_consolidation_patterns()
        
        self.assertIn('semantic_clusters', consolidation)
        self.assertIn('promotion_candidates', consolidation)
        self.assertIn('attention_filtered', consolidation)
        self.assertIn('decay_analysis', consolidation)  # Changed from decay_candidates
        
        # Check that high access items are promotion candidates
        promotion_contents = [item['content'] for item in consolidation['promotion_candidates']]
        self.assertTrue(any("popular item" in content for content in promotion_contents))
        
    def test_generate_memory_report(self):
        """Test comprehensive memory report generation."""
        self.mock_memory_system.operation_logs = [
            {'type': 'task_reuse', 'content': 'test'}
        ]
        
        analyzer = MemoryAnalyzer(self.mock_memory_system)
        report = analyzer.generate_memory_report()
        
        self.assertIn('timestamp', report)
        self.assertIn('cognitive_state', report)
        self.assertIn('buffer_analysis', report)
        self.assertIn('reuse_analysis', report)
        self.assertIn('consolidation_analysis', report)
        self.assertIn('system_parameters', report)
        
        # Check cognitive state
        cognitive_state = report['cognitive_state']
        self.assertEqual(cognitive_state['current_task'], 'test task')
        self.assertEqual(cognitive_state['confidence_score'], 0.8)
        
        # Check system parameters
        params = report['system_parameters']
        self.assertEqual(params['attention_threshold'], 0.5)
        self.assertEqual(params['consolidation_threshold'], 0.8)
        
    def test_compare_memory_states(self):
        """Test memory state comparison."""
        # Create two different reports
        report1 = {
            'buffer_analysis': {
                'immediate_buffer': {'size': 3},
                'working_buffer': {'size': 5},
                'episodic_buffer': {'size': 10}
            },
            'reuse_analysis': {'reuse_rate': 0.6},
            'consolidation_analysis': {'semantic_clusters': 2}
        }
        
        report2 = {
            'buffer_analysis': {
                'immediate_buffer': {'size': 4},
                'working_buffer': {'size': 8},
                'episodic_buffer': {'size': 12}
            },
            'reuse_analysis': {'reuse_rate': 0.8},
            'consolidation_analysis': {'semantic_clusters': 3}
        }
        
        analyzer = MemoryAnalyzer(self.mock_memory_system)
        comparison = analyzer.compare_memory_states(report1, report2)
        
        self.assertIn('buffer_changes', comparison)
        self.assertIn('reuse_changes', comparison)
        
        # Check buffer changes
        working_change = comparison['buffer_changes']['working_buffer']['change']
        self.assertEqual(working_change, 3)  # 8 - 5
        
        episodic_change = comparison['buffer_changes']['episodic_buffer']['change']
        self.assertEqual(episodic_change, 2)  # 12 - 10
        
        # Check reuse rate change
        reuse_change = comparison['reuse_changes']['change']  # Changed from reuse_rate_change
        self.assertAlmostEqual(reuse_change, 0.2)  # 0.8 - 0.6
        
    def test_empty_buffers_analysis(self):
        """Test analysis with empty buffers."""
        analyzer = MemoryAnalyzer(self.mock_memory_system)
        flow_analysis = analyzer.analyze_buffer_flow()
        
        # All buffers should be empty
        for buffer_name in ['immediate_buffer', 'working_buffer', 'episodic_buffer']:
            buffer_info = flow_analysis[buffer_name]
            self.assertEqual(buffer_info['size'], 0)
            self.assertEqual(buffer_info['utilization'], 0.0)
            
    def test_reuse_analysis_with_no_operations(self):
        """Test reuse analysis with no operation logs."""
        analyzer = MemoryAnalyzer(self.mock_memory_system)
        reuse_analysis = analyzer.track_memory_reuse()
        
        self.assertEqual(reuse_analysis['reuse_rate'], 0.0)
        self.assertEqual(reuse_analysis['task_reuse_operations'], 0)
        self.assertEqual(reuse_analysis['memory_reuse_operations'], 0)
        self.assertEqual(reuse_analysis['new_info_operations'], 0)
        
    def test_consolidation_with_empty_buffers(self):
        """Test consolidation analysis with empty buffers."""
        analyzer = MemoryAnalyzer(self.mock_memory_system)
        consolidation = analyzer.visualize_consolidation_patterns()
        
        self.assertEqual(len(consolidation['promotion_candidates']), 0)
        self.assertEqual(len(consolidation['attention_filtered']), 0)
        self.assertEqual(len(consolidation['decay_analysis']), 0)


if __name__ == '__main__':
    unittest.main()
