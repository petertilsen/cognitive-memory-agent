"""Cognitive Memory Agent - Strands SDK Integration."""

__version__ = "0.1.0"
__author__ = "Cognitive Memory Team"

from .agent.agent import CognitiveMemoryAgent
from .memory.memory_system import CognitiveMemorySystem

__all__ = ["CognitiveMemoryAgent", "CognitiveMemorySystem"]
