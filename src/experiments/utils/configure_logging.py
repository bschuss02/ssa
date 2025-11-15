from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.text import Text

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

        # Format: time first, then message, then clickable file location
        time_str = record["time"].strftime("%H:%M:%S")

        # Create clickable file link for VSCode terminal
        # VSCode terminal recognizes file paths in format: /absolute/path:line
        file_path = Path(record["file"].path).resolve()
        line_num = record["line"]
        # Use absolute path with line number - VSCode terminal will make this clickable
        file_link = f"{file_path}:{line_num}"

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
        # Format: time first, then message, then clickable file location
        # VSCode terminal will recognize absolute paths with line numbers as clickable links
        # Use Text object for file path to prevent Rich from interpreting it as markup
        output = Text()
        output.append(time_str, style="dim")
        output.append(" | ")
        output.append(record["message"], style=level_color)
        output.append(" | ")
        output.append(str(file_link), style="dim")
        console.print(output)

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
