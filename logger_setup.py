import logging
import os
import sys
from logging.handlers import RotatingFileHandler


class CustomFormatter(logging.Formatter):
    """Adds color and file path information to terminal logs."""
    
    # ANSI Color Codes
    grey = "\x1b[38;20m"
    blue = "\x1b[34;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    
    # The format including the filename and line number
    format_str = "%(asctime)s - [%(levelname)s] - %(name)s - (%(filename)s:%(lineno)d) - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: blue + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def setup_global_logging():
    # Ensure logs directory exists in the root
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "vision_dev.log")

    # Configure Rotating File Handler (prevents file from getting too large)
    # Max size 5MB, keeps 5 backup files
    file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5)
    stream_handler = logging.StreamHandler(sys.stdout)

    # 2. Create formatters
    # Plain text for the file
    file_formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - %(name)s - (%(filename)s:%(lineno)d) - %(message)s")
    # Colored for the terminal
    color_formatter = CustomFormatter()

    # 3. Attach formatters to handlers
    file_handler.setFormatter(file_formatter)
    stream_handler.setFormatter(color_formatter)

    # 4. Configure the Root Logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear existing handlers to prevent duplicate logs
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    return logging.getLogger("VisionDev")

# Initialize it
logger = setup_global_logging()