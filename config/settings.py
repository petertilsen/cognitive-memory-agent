"""Configuration settings for Cognitive Memory Agent."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class MemoryConfig:
    """Memory system configuration."""
    immediate_buffer_size: int = 8
    working_buffer_size: int = 64
    episodic_buffer_size: int = 256
    attention_threshold: float = 0.5
    consolidation_threshold: float = 0.8
    decay_rate: float = 0.1


@dataclass
class ModelConfig:
    """Model configuration."""
    model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"
    region: str = "us-east-1"
    max_tokens: int = 1000
    temperature: float = 0.7


@dataclass
class AppConfig:
    """Application configuration."""
    memory: MemoryConfig
    model: ModelConfig
    debug: bool = False
    log_level: str = "INFO"


def load_config() -> AppConfig:
    """Load configuration from environment variables."""
    memory_config = MemoryConfig()
    
    model_config = ModelConfig(
        model_id=os.getenv("BEDROCK_MODEL", "anthropic.claude-3-haiku-20240307-v1:0"),
        region=os.getenv("AWS_REGION", "us-east-1")
    )
    
    app_config = AppConfig(
        memory=memory_config,
        model=model_config,
        debug=os.getenv("DEBUG", "false").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "INFO")
    )
    
    return app_config
