"""Memory system analyzer and visualizer for cognitive memory behavior."""

from typing import Dict, List, Any, Tuple
from .memory_system import CognitiveMemorySystem
from .models import MemoryItem
from ...config.settings import get_logger

logger = get_logger("memory.analyzer")


class MemoryAnalyzer:
    """Analyzes and visualizes cognitive memory system behavior."""
    
    def __init__(self, memory_system: CognitiveMemorySystem):
        self.memory_system = memory_system
        self.analysis_history = []
    
    def analyze_buffer_flow(self) -> Dict[str, Any]:
        """Analyze how memories flow between buffers."""
        return {
            "immediate_buffer": {
                "size": len(self.memory_system.immediate_buffer),
                "capacity": self.memory_system.immediate_buffer.maxlen,
                "utilization": len(self.memory_system.immediate_buffer) / self.memory_system.immediate_buffer.maxlen,
                "items": [item.content[:50] + "..." for item in self.memory_system.immediate_buffer]
            },
            "working_buffer": {
                "size": len(self.memory_system.working_buffer),
                "capacity": self.memory_system.working_buffer.maxlen,
                "utilization": len(self.memory_system.working_buffer) / self.memory_system.working_buffer.maxlen,
                "items": [item.content[:50] + "..." for item in self.memory_system.working_buffer]
            },
            "episodic_buffer": {
                "size": len(self.memory_system.episodic_buffer),
                "capacity": self.memory_system.episodic_buffer.maxlen,
                "utilization": len(self.memory_system.episodic_buffer) / self.memory_system.episodic_buffer.maxlen,
                "items": [item.content[:50] + "..." for item in self.memory_system.episodic_buffer]
            },
            "vector_store": {
                "size": len(self.memory_system.vector_store.vectors),
                "items": len(self.memory_system.vector_store.texts)
            }
        }
    
    def track_memory_reuse(self) -> Dict[str, Any]:
        """Track memory reuse patterns across all buffers."""
        reuse_stats = {
            "high_access_items": [],
            "total_accesses": 0,
            "unique_items": 0,
            "reuse_rate": 0.0
        }
        
        all_items = (
            list(self.memory_system.immediate_buffer) +
            list(self.memory_system.working_buffer) +
            list(self.memory_system.episodic_buffer)
        )
        
        if all_items:
            total_accesses = sum(item.access_count for item in all_items)
            unique_items = len(all_items)
            
            # Items with multiple accesses indicate reuse
            high_access_items = [
                {
                    "content": item.content[:50] + "...",
                    "access_count": item.access_count,
                    "relevance_score": item.relevance_score,
                    "source": item.source
                }
                for item in all_items if item.access_count > 1
            ]
            
            reuse_rate = len(high_access_items) / unique_items if unique_items > 0 else 0.0
            
            reuse_stats.update({
                "high_access_items": high_access_items,
                "total_accesses": total_accesses,
                "unique_items": unique_items,
                "reuse_rate": reuse_rate
            })
        
        return reuse_stats
    
    def visualize_consolidation_patterns(self) -> Dict[str, Any]:
        """Visualize memory consolidation and decay patterns."""
        consolidation_data = {
            "decay_analysis": [],
            "promotion_candidates": [],
            "attention_filtered": [],
            "semantic_clusters": 0
        }
        
        # Analyze decay patterns
        for buffer_name, buffer in [
            ("immediate", self.memory_system.immediate_buffer),
            ("working", self.memory_system.working_buffer),
            ("episodic", self.memory_system.episodic_buffer)
        ]:
            for item in buffer:
                consolidation_data["decay_analysis"].append({
                    "buffer": buffer_name,
                    "content": item.content[:30] + "...",
                    "relevance_score": item.relevance_score,
                    "access_count": item.access_count,
                    "age": self.memory_system.current_time - item.creation_time
                })
        
        # Find promotion candidates (high access, high relevance)
        for item in self.memory_system.working_buffer:
            if item.access_count > 2 and item.relevance_score > self.memory_system.consolidation_threshold:
                consolidation_data["promotion_candidates"].append({
                    "content": item.content[:50] + "...",
                    "access_count": item.access_count,
                    "relevance_score": item.relevance_score
                })
        
        # Find items that would be filtered by attention threshold
        for item in self.memory_system.working_buffer:
            if item.relevance_score <= self.memory_system.attention_threshold:
                consolidation_data["attention_filtered"].append({
                    "content": item.content[:50] + "...",
                    "relevance_score": item.relevance_score
                })
        
        # Get semantic cluster count
        consolidation_data["semantic_clusters"] = self.memory_system._organize_semantic_clusters()
        
        return consolidation_data
    
    def generate_memory_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory system report."""
        logger.info("Generating comprehensive memory report")
        
        buffer_analysis = self.analyze_buffer_flow()
        reuse_analysis = self.track_memory_reuse()
        consolidation_analysis = self.visualize_consolidation_patterns()
        
        report = {
            "timestamp": self.memory_system.current_time,
            "cognitive_state": {
                "current_task": self.memory_system.cognitive_state.current_task if self.memory_system.cognitive_state else None,
                "confidence_score": self.memory_system.cognitive_state.confidence_score if self.memory_system.cognitive_state else 0.0
            },
            "buffer_analysis": buffer_analysis,
            "reuse_analysis": reuse_analysis,
            "consolidation_analysis": consolidation_analysis,
            "system_parameters": {
                "attention_threshold": self.memory_system.attention_threshold,
                "consolidation_threshold": self.memory_system.consolidation_threshold,
                "similarity_threshold": self.memory_system.similarity_threshold
            }
        }
        
        self.analysis_history.append(report)
        logger.info(f"Memory report generated with {len(self.analysis_history)} total reports in history")
        
        return report
    
    def compare_memory_states(self, previous_report: Dict[str, Any], current_report: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two memory states to show evolution."""
        comparison = {
            "buffer_changes": {},
            "reuse_changes": {},
            "consolidation_changes": {}
        }
        
        # Buffer size changes
        for buffer_name in ["immediate_buffer", "working_buffer", "episodic_buffer"]:
            prev_size = previous_report["buffer_analysis"][buffer_name]["size"]
            curr_size = current_report["buffer_analysis"][buffer_name]["size"]
            comparison["buffer_changes"][buffer_name] = {
                "previous": prev_size,
                "current": curr_size,
                "change": curr_size - prev_size
            }
        
        # Reuse rate changes
        prev_reuse = previous_report["reuse_analysis"]["reuse_rate"]
        curr_reuse = current_report["reuse_analysis"]["reuse_rate"]
        comparison["reuse_changes"] = {
            "previous": prev_reuse,
            "current": curr_reuse,
            "change": curr_reuse - prev_reuse
        }
        
        # Consolidation changes
        prev_clusters = previous_report["consolidation_analysis"]["semantic_clusters"]
        curr_clusters = current_report["consolidation_analysis"]["semantic_clusters"]
        comparison["consolidation_changes"] = {
            "semantic_clusters": {
                "previous": prev_clusters,
                "current": curr_clusters,
                "change": curr_clusters - prev_clusters
            }
        }
        
        return comparison
