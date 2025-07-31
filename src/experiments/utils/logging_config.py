"""
Logging configuration for ASR evaluation.

This module provides centralized logging setup for the ASR evaluation pipeline.
"""

import logging
import sys

from rich.logging import RichHandler


def setup_logging() -> logging.Logger:
    """
    Configure logging with Rich formatting and console output.

    Returns:
        Configured logger instance
    """
    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
        force=True,
    )

    # Ensure our logger is set to INFO level
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Add a console handler to ensure output goes to terminal
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
