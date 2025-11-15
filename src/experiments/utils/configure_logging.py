from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.text import Text
from rich.traceback import Traceback

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

        # If there's an exception traceback, extract and format it
        if record["exception"] is not None:
            try:
                # Try to get exception info directly from record
                exc_info = record["exception"]

                # Try to access exception attributes
                exc_type = None
                exc_value = None
                exc_traceback = None

                if hasattr(exc_info, "type"):
                    exc_type = exc_info.type
                    exc_value = exc_info.value
                    exc_traceback = exc_info.traceback
                elif isinstance(exc_info, tuple) and len(exc_info) == 3:
                    exc_type, exc_value, exc_traceback = exc_info
                else:
                    # Try getattr as fallback
                    exc_type = getattr(exc_info, "type", None)
                    exc_value = getattr(exc_info, "value", None)
                    exc_traceback = getattr(exc_info, "traceback", None)

                # If we have exception info, create Rich traceback
                if exc_type is not None and exc_value is not None and exc_traceback is not None:
                    rich_traceback = Traceback.from_exception(
                        type=exc_type,
                        value=exc_value,
                        traceback=exc_traceback,
                        show_locals=False,  # Hide local variables for cleaner output
                        suppress=[
                            "hydra/_internal",  # Suppress hydra internal framework frames
                            "hydra/main.py",  # Suppress hydra main entry point
                            "loguru",  # Suppress loguru internal frames
                        ],
                        max_frames=8,  # Limit number of frames shown (focus on user code)
                    )
                    console.print(rich_traceback)
                else:
                    # Fallback: Use loguru's formatted exception string
                    # The format includes {exception}, so we can get it from the formatted message
                    formatted = str(message)
                    if "\n" in formatted:
                        exception_part = formatted.split("\n", 1)[1]
                        if exception_part.strip():
                            exception_text = Text()
                            exception_text.append(exception_part, style="red")
                            console.print(exception_text)
            except Exception:
                # If anything fails, try to get exception from formatted message
                try:
                    formatted = str(message)
                    if "\n" in formatted:
                        exception_part = formatted.split("\n", 1)[1]
                        if exception_part.strip():
                            exception_text = Text()
                            exception_text.append(exception_part, style="red")
                            console.print(exception_text)
                except Exception:
                    # Final fallback: print the exception representation
                    exception_text = Text()
                    exception_text.append(f"Exception: {record['exception']}", style="red")
                    console.print(exception_text)

    logger.add(
        rich_sink,
        level=log_level,
        colorize=False,  # Rich handles colors
        format="{message}\n{exception}",  # Include exception in format for parsing
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
