"""Memory data models."""

import time
from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class MemoryItem:
    """Individual memory item with cognitive properties."""
    
    content: str
    embedding: np.ndarray
    relevance_score: float = 1.0
    access_count: int = 0
    last_access_time: float = field(default_factory=time.time)
    creation_time: float = field(default_factory=time.time)
    task_context: str = ""
    source: str = ""

    def decay(self, current_time: float, decay_rate: float = 0.1) -> None:
        """Apply forgetting curve decay."""
        time_diff = current_time - self.last_access_time
        
        # Handle negative time differences (when current_time < last_access_time)
        if time_diff <= 0:
            # No decay if current_time hasn't advanced past last_access_time
            return
            
        # Prevent overflow by clamping the exponent
        exponent = -decay_rate * time_diff
        if exponent < -700:  # np.exp(-700) ≈ 0, prevents overflow
            self.relevance_score = 0.0
        else:
            self.relevance_score *= np.exp(exponent)

    def boost(self, amount: float = 0.2) -> None:
        """Boost memory relevance when accessed."""
        self.relevance_score = min(1.0, self.relevance_score + amount)
        self.access_count += 1
        self.last_access_time = time.time()


@dataclass
class CognitiveState:
    """Current cognitive state of the agent."""
    
    current_task: str
    subtasks: List[str] = field(default_factory=list)
    completed_subtasks: List[str] = field(default_factory=list)
    information_gaps: List[str] = field(default_factory=list)
    working_hypothesis: str = ""
    confidence_score: float = 0.0
    context_history: List[str] = field(default_factory=list)
