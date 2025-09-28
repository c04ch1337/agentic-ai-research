"""Logging configuration"""
import logging
import logging.handlers
from pathlib import Path

def setup_logging(config: dict):
    """Setup logging configuration"""
    level = getattr(logging, config.get("level", "INFO").upper())
    log_file = config.get("file", "./logs/agentic_ai.log")
    
    # Create logs directory if it doesn't exist
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10*1024*1024, backupCount=5
            ),
            logging.StreamHandler()
        ]
    )