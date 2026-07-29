"""
Logger utility module for Face Presentation Classifier.
"""

import logging
import os
import sys
from typing import Optional


def setup_logger(
    name: str = "FacePresentationClassifier",
    log_dir: str = "logs",
    log_level: int = logging.INFO
) -> logging.Logger:
    """
    Sets up a production-ready logger with both console and file handlers.

    Args:
        name (str): Name of the logger.
        log_dir (str): Directory where log files will be saved.
        log_level (int): Logging level (e.g. logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Avoid adding duplicate handlers if logger is re-initialized
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(filename)s:%(lineno)d]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Stream Handler (Console)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    log_file_path = os.path.join(log_dir, "app.log")
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
