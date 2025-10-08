"""Logging configuration for Cognitive Memory Agent."""

import logging
import logging.config
import os
from pathlib import Path
from typing import Dict, Any


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    enable_file_logging: bool = True,
    enable_console_logging: bool = True
) -> None:
    """Setup logging configuration for the application."""
    
    # Create logs directory
    if enable_file_logging:
        Path(log_dir).mkdir(exist_ok=True)
    
    # Logging configuration
    config: Dict[str, Any] = {
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
            "cognitive_memory_agent": {
                "level": log_level,
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
            "level": log_level,
            "handlers": []
        }
    }
    
    # Console handler
    if enable_console_logging:
        config["handlers"]["console"] = {
            "class": "logging.StreamHandler",
            "level": log_level,
            "formatter": "simple",
            "stream": "ext://sys.stdout"
        }
        config["loggers"]["cognitive_memory_agent"]["handlers"].append("console")
        config["loggers"]["strands"]["handlers"].append("console")
        config["root"]["handlers"].append("console")
    
    # File handlers
    if enable_file_logging:
        # Main application log
        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": log_level,
            "formatter": "detailed",
            "filename": f"{log_dir}/cognitive_memory_agent.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5
        }
        
        # Error log
        config["handlers"]["error_file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "detailed",
            "filename": f"{log_dir}/errors.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 3
        }
        
        # Memory operations log
        config["handlers"]["memory_file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "json",
            "filename": f"{log_dir}/memory_operations.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5
        }
        
        config["loggers"]["cognitive_memory_agent"]["handlers"].extend(["file", "error_file"])
        config["loggers"]["cognitive_memory_agent.core.memory_system"] = {
            "level": "DEBUG",
            "handlers": ["memory_file"],
            "propagate": False
        }
        config["root"]["handlers"].extend(["file", "error_file"])
    
    # Apply configuration
    logging.config.dictConfig(config)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given name."""
    return logging.getLogger(f"cognitive_memory_agent.{name}")


# Initialize logging on import
setup_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    enable_file_logging=os.getenv("ENABLE_FILE_LOGGING", "true").lower() == "true",
    enable_console_logging=os.getenv("ENABLE_CONSOLE_LOGGING", "true").lower() == "true"
)
