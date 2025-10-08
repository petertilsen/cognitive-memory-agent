"""ReAct pattern utilities and helpers."""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ReActPhase(Enum):
    """ReAct cycle phases."""
    REASON = "reason"
    ACT = "act"
    OBSERVE = "observe"


@dataclass
class ReActStep:
    """Individual ReAct step."""
    phase: ReActPhase
    content: str
    timestamp: float
    success: bool = True
    metadata: Optional[Dict] = None


class ReActTracker:
    """Track and analyze ReAct patterns."""
    
    def __init__(self):
        self.steps: List[ReActStep] = []
        self.current_cycle = 0
        self.success_rate = 0.0
    
    def add_step(self, phase: ReActPhase, content: str, success: bool = True, 
                 metadata: Optional[Dict] = None) -> None:
        """Add a ReAct step."""
        import time
        step = ReActStep(
            phase=phase,
            content=content,
            timestamp=time.time(),
            success=success,
            metadata=metadata or {}
        )
        self.steps.append(step)
        
        # Update cycle count
        if phase == ReActPhase.REASON:
            self.current_cycle += 1
        
        # Update success rate
        successful_steps = sum(1 for s in self.steps if s.success)
        self.success_rate = successful_steps / len(self.steps) if self.steps else 0.0
    
    def get_current_cycle_steps(self) -> List[ReActStep]:
        """Get steps from current ReAct cycle."""
        if not self.steps:
            return []
        
        # Find last REASON step
        last_reason_idx = None
        for i in range(len(self.steps) - 1, -1, -1):
            if self.steps[i].phase == ReActPhase.REASON:
                last_reason_idx = i
                break
        
        if last_reason_idx is None:
            return []
        
        return self.steps[last_reason_idx:]
    
    def get_cycle_summary(self) -> Dict:
        """Get summary of ReAct cycles."""
        reason_count = sum(1 for s in self.steps if s.phase == ReActPhase.REASON)
        act_count = sum(1 for s in self.steps if s.phase == ReActPhase.ACT)
        observe_count = sum(1 for s in self.steps if s.phase == ReActPhase.OBSERVE)
        
        return {
            "total_cycles": self.current_cycle,
            "total_steps": len(self.steps),
            "reason_steps": reason_count,
            "act_steps": act_count,
            "observe_steps": observe_count,
            "success_rate": self.success_rate,
            "avg_steps_per_cycle": len(self.steps) / max(1, self.current_cycle)
        }


def parse_react_response(response: str) -> Tuple[Optional[ReActPhase], str]:
    """Parse agent response to identify ReAct phase."""
    response_lower = response.lower().strip()
    
    # Look for explicit ReAct markers
    if response_lower.startswith("reason:") or "reasoning:" in response_lower:
        return ReActPhase.REASON, response
    elif response_lower.startswith("act:") or "action:" in response_lower:
        return ReActPhase.ACT, response
    elif response_lower.startswith("observe:") or "observation:" in response_lower:
        return ReActPhase.OBSERVE, response
    
    # Look for implicit patterns
    if any(word in response_lower for word in ["think", "consider", "analyze", "need to"]):
        return ReActPhase.REASON, response
    elif any(word in response_lower for word in ["using", "calling", "retrieving", "adding"]):
        return ReActPhase.ACT, response
    elif any(word in response_lower for word in ["found", "result", "shows", "indicates"]):
        return ReActPhase.OBSERVE, response
    
    return None, response


def format_react_prompt(task: str, context: str = "") -> str:
    """Format a prompt to encourage ReAct pattern."""
    prompt = f"""Task: {task}

Please approach this using the ReAct pattern:
1. REASON: Think about what you need to know or do
2. ACT: Use available tools to gather information or perform actions
3. OBSERVE: Analyze the results and determine next steps

"""
    
    if context:
        prompt += f"Context: {context}\n\n"
    
    prompt += "Begin with your reasoning:"
    
    return prompt
