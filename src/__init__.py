"""Cognitive Memory Agent - Strands SDK Integration."""

__version__ = "0.1.0"
__author__ = "Cognitive Memory Team"

from .core.agent import CognitiveMemoryAgent
from .core.memory_system import CognitiveMemorySystem

__all__ = ["CognitiveMemoryAgent", "CognitiveMemorySystem"]
