"""Logging utilities for SDXL fine-tuning project."""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        format_string: Custom format string (optional)
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Default format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file provided)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_training_logger(log_dir: str = "logs") -> logging.Logger:
    """Get logger for training with timestamped log file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"training_{timestamp}.log"
    return setup_logger("training", str(log_file))


def get_inference_logger(log_dir: str = "logs") -> logging.Logger:
    """Get logger for inference with timestamped log file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"inference_{timestamp}.log"
    return setup_logger("inference", str(log_file))


def get_api_logger(log_file: Optional[str] = None) -> logging.Logger:
    """Get logger for API server."""
    if log_file is None:
        log_file = "logs/api.log"
    return setup_logger("api", log_file)

