from pathlib import Path
from typing import Dict, Literal

from pydantic import BaseModel, Field


class LoggerConfig(BaseModel):
    """Configuration for logging settings."""

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="The logging level to use"
    )
    log_file: Path | None = Field(
        default=None, description="Path to log file. If None, logs only to console"
    )
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Format string for log messages",
    )
    use_rich_logging: bool = Field(
        default=True, description="Whether to use rich formatting for console output"
    )


class EvaluationConfig(BaseModel):
    models: Dict[str, str] = Field(
        description="A dictionary of model names and their paths"
    )
    datasets: Dict[str, str] = Field(
        description="A dictionary of dataset names and their paths"
    )
    max_samples_per_dataset: int = Field(
        description="The maximum number of audio samples to evaluate per dataset"
    )
    batch_size: int = Field(description="The batch size to use for inference")
    output_dir: Path = Field(description="The directory to save the evaluation results")

    # Logging configuration
    logging: LoggerConfig = Field(
        default_factory=LoggerConfig, description="Logging configuration settings"
    )
