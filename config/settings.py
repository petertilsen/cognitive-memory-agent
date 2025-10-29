"""Configuration settings for Cognitive Memory Agent."""

import os
import logging
import logging.config
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use system env vars only


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
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    log_dir: str = "logs"
    enable_file_logging: bool = True
    enable_console_logging: bool = True


@dataclass
class AppConfig:
    """Application configuration."""
    memory: MemoryConfig
    model: ModelConfig
    embedding_model: ModelConfig
    synthesis_model: ModelConfig
    logging: LoggingConfig
    debug: bool = False


def setup_logging(config: LoggingConfig) -> None:
    """Setup logging configuration."""
    
    # Create logs directory
    if config.enable_file_logging:
        Path(config.log_dir).mkdir(exist_ok=True)
    
    # Logging configuration
    log_config: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "simple": {
                "format": "%(levelname)s - %(name)s - %(message)s"
            },
            "json": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": "%(asctime)s %(name)s %(levelname)s %(funcName)s %(lineno)d %(message)s"
            }
        },
        "handlers": {},
        "loggers": {
            "src": {
                "level": config.level,
                "handlers": [],
                "propagate": False
            },
            "strands": {
                "level": "WARNING",
                "handlers": [],
                "propagate": False
            }
        },
        "root": {
            "level": config.level,
            "handlers": []
        }
    }
    
    # Console handler
    if config.enable_console_logging:
        log_config["handlers"]["console"] = {
            "class": "logging.StreamHandler",
            "level": config.level,
            "formatter": "simple",
            "stream": "ext://sys.stdout"
        }
        log_config["loggers"]["src"]["handlers"].append("console")
        log_config["loggers"]["strands"]["handlers"].append("console")
        log_config["root"]["handlers"].append("console")
    
    # File handlers
    if config.enable_file_logging:
        # Main application log
        log_config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": config.level,
            "formatter": "detailed",
            "filename": f"{config.log_dir}/cognitive_memory_agent.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5
        }
        
        # Error log
        log_config["handlers"]["error_file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "detailed",
            "filename": f"{config.log_dir}/errors.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 3
        }
        
        # Memory operations log
        log_config["handlers"]["memory_file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "json",
            "filename": f"{config.log_dir}/memory_operations.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5
        }
        
        log_config["loggers"]["src"]["handlers"].extend(["file", "error_file"])
        log_config["loggers"]["src.core.memory_system"] = {
            "level": "DEBUG",
            "handlers": ["memory_file"],
            "propagate": False
        }
        log_config["root"]["handlers"].extend(["file", "error_file"])
    
    # Apply configuration
    logging.config.dictConfig(log_config)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given name."""
    return logging.getLogger(f"src.{name}")


def load_config() -> AppConfig:
    """Load configuration from environment variables."""
    memory_config = MemoryConfig()
    
    model_config = ModelConfig(
        model_id=os.getenv("MODEL", "anthropic.claude-3-haiku-20240307-v1:0"),
        region=os.getenv("AWS_REGION", "us-east-1")
    )

    embedding_model_config = ModelConfig(
        model_id=os.getenv("EMBEDDING_MODEL", "amazon.titan-embed-text-v1"),
        region=os.getenv("AWS_REGION", "us-east-1")
    )

    synthesis_model_config = ModelConfig(
        model_id=os.getenv("SYNTHESIS_MODEL", "amazon.titan-embed-text-v1"),
        region=os.getenv("AWS_REGION", "us-east-1")
    )
    
    logging_config = LoggingConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        enable_file_logging=os.getenv("ENABLE_FILE_LOGGING", "true").lower() == "true",
        enable_console_logging=os.getenv("ENABLE_CONSOLE_LOGGING", "true").lower() == "true"
    )
    
    app_config = AppConfig(
        memory=memory_config,
        model=model_config,
        embedding_model=embedding_model_config,
        synthesis_model=synthesis_model_config,
        logging=logging_config,
        debug=os.getenv("DEBUG", "false").lower() == "true"
    )
    
    # Setup logging
    setup_logging(logging_config)
    
    return app_config


# Initialize configuration on import
config = load_config()
