from pathlib import Path

from loguru import logger
from rich.console import Console

# Create a shared console instance for better coordination with progress bars
# This console will be used by both logging and progress bars
console = Console(stderr=True, force_terminal=True)


def configure_logging() -> None:
    """Configure loguru logger with hardcoded settings."""
    # Remove default handler
    logger.remove()

    # Hardcoded configuration
    log_level = "INFO"
    log_file = "output/logs/experiments.log"

    # Custom sink that uses Rich console and properly coordinates with progress bars
    def rich_sink(message):
        """Custom sink that uses Rich console for proper progress bar coordination."""
        record = message.record

        # Format: time first, then message, then location (dimmed)
        time_str = record["time"].strftime("%H:%M:%S")
        location = f"{record['name']}:{record['function']}:{record['line']}"

        # Use Rich's markup for colors
        level_colors = {
            "DEBUG": "dim white",
            "INFO": "cyan",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold red",
        }
        level_color = level_colors.get(record["level"].name, "white")

        # Use console.print() - Rich's console automatically coordinates with progress bars
        # when using the same console instance (which we do via ProgressManager)
        # Format: time first, then message, then location
        console.print(
            f"[dim]{time_str}[/dim] | "
            f"[{level_color}]{record['message']}[/{level_color}] "
            f"[dim]| {location}[/dim]",
            markup=True,
        )

    logger.add(
        rich_sink,
        level=log_level,
        colorize=False,  # Rich handles colors
        format="{message}",  # We format manually in the sink
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
        format="{time:YYYY-MM-DD HH:mm:ss} | {message} | {level: <8} | {name}:{function}:{line}",
        enqueue=True,  # Thread-safe logging
    )

    logger.info(f"Logging configured: level={log_level}, file={log_file}")


# Export logger for easy import
__all__ = ["logger", "configure_logging"]
