import logging
import sys
from pathlib import Path

def get_logger(
    name: str = __name__, 
    level: int = logging.INFO,
    log_file: str = None
) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (typically __name__ of the calling module)
        level: Logging level (default: INFO)
        log_file: Optional path to log file. If provided, logs to both console and file.
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Prevent duplicate handlers if logger already exists
    if logger.hasHandlers():
        return logger
    
    logger.setLevel(level)
    
    # Format: timestamp - level - filename:line - message
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file is provided)
    if log_file:
        # Create logs directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Example usage (remove this section when using in production):
# if __name__ == "__main__":
#     # Console only logger
#     log = get_logger(__name__)
#     log.debug("This is a debug message")
#     log.info("This is an info message")
#     log.warning("This is a warning message")
#     log.error("This is an error message")
#     log.critical("This is a critical message")
    
#     print("\n" + "="*50 + "\n")
    
#     # Logger with both console and file output
#     log_file = get_logger(__name__, level=logging.DEBUG, log_file="logs/app.log")
#     log_file.info("This message goes to both console and file")