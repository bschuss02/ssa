"""
Logging configuration for experiments.

This module provides centralized logging setup for the experiments pipeline.
"""

import logging
import sys
from typing import Optional

from rich.logging import RichHandler

from experiments.config.evaluation_config import EvaluationConfig


def configure_logging(config: EvaluationConfig) -> None:
    """
    Configure the global logging system based on the provided EvaluationConfig.

    This function should be called once at the start of your application.
    After calling this function, you can use `logging.getLogger(__name__)`
    anywhere in your code to get a properly configured logger.

    Args:
        config: The evaluation configuration containing logging settings
    """
    # Clear any existing handlers from root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Convert string log level to logging constant
    log_level = getattr(logging, config.logging.log_level.upper())

    # Set the root logger level
    root_logger.setLevel(log_level)

    # Create handlers list
    handlers = []

    # Console handler (only if show_terminal_logs is True)
    if config.logging.show_terminal_logs:
        if config.logging.use_rich_logging:
            try:
                console_handler = RichHandler(
                    rich_tracebacks=True, show_time=True, show_path=False
                )
                console_handler.setLevel(log_level)
                handlers.append(console_handler)
            except ImportError:
                # Fallback to standard handler if rich is not available
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(log_level)
                formatter = logging.Formatter(config.logging.log_format)
                console_handler.setFormatter(formatter)
                handlers.append(console_handler)
        else:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            formatter = logging.Formatter(config.logging.log_format)
            console_handler.setFormatter(formatter)
            handlers.append(console_handler)

    # File handler (if log_file is specified)
    if config.logging.log_file is not None:
        # Ensure log directory exists
        config.logging.log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(config.logging.log_file)
        file_handler.setLevel(log_level)
        formatter = logging.Formatter(config.logging.log_format)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Add all handlers to the root logger
    for handler in handlers:
        root_logger.addHandler(handler)

    # Configure basic logging to ensure proper propagation
    logging.basicConfig(
        level=log_level,
        format="%(message)s"
        if config.logging.use_rich_logging
        else config.logging.log_format,
        datefmt="[%X]" if config.logging.use_rich_logging else None,
        handlers=handlers,
        force=True,
    )

    # Ensure all existing loggers are properly configured
    for logger_name in logging.root.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()  # Remove any existing handlers
        logger.propagate = True  # Ensure propagation to root logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance. This is a convenience function that can be used
    when you don't have access to the config object.

    Args:
        name: Logger name. If None, uses __name__ from calling module

    Returns:
        Logger instance
    """
    if name is None:
        import inspect

        frame = inspect.currentframe()
        try:
            caller_frame = frame.f_back
            name = caller_frame.f_globals.get("__name__", __name__)
        finally:
            del frame

    return logging.getLogger(name)
