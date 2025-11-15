import sys
from pathlib import Path

from loguru import logger


def configure_logging() -> None:
    """Configure loguru logger with hardcoded settings."""
    # Remove default handler
    logger.remove()

    # Hardcoded configuration
    log_level = "INFO"
    log_file = "output/logs/experiments.log"
    use_colors = True

    # Add console handler with colorization
    logger.add(
        sys.stderr,
        colorize=use_colors,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    # Add file handler
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        str(log_path),
        rotation="20 MB",
        retention="10 days",
        compression="zip",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        enqueue=True,  # Thread-safe logging
    )

    logger.info(f"Logging configured: level={log_level}, file={log_file}")


# Export logger for easy import
__all__ = ["logger", "configure_logging"]
