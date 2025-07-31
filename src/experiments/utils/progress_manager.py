"""
Progress tracking and UI management for ASR evaluation.

This module provides a clean interface for managing progress bars
and user feedback during the evaluation process.
"""

from typing import List

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)


class ProgressManager:
    """Manages progress tracking for multi-level evaluation tasks."""

    def __init__(self):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        )
        self.model_task: TaskID | None = None
        self.dataset_task: TaskID | None = None
        self.sample_task: TaskID | None = None

    def __enter__(self):
        self.progress.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.progress.stop()

    def start_model_processing(self, total_models: int) -> None:
        """Start progress tracking for model processing."""
        self.model_task = self.progress.add_task(
            "[cyan]Processing models...", total=total_models
        )

    def start_dataset_processing(self, model_name: str, total_datasets: int) -> None:
        """Start progress tracking for dataset processing within a model."""
        self.dataset_task = self.progress.add_task(
            f"[green]Processing datasets for {model_name}...",
            total=total_datasets,
        )

    def start_sample_processing(self, dataset_name: str, total_samples: int) -> None:
        """Start progress tracking for sample processing within a dataset."""
        self.sample_task = self.progress.add_task(
            f"[yellow]Processing samples for {dataset_name}...",
            total=total_samples,
        )

    def advance_model(self) -> None:
        """Advance the model progress bar."""
        if self.model_task is not None:
            self.progress.advance(self.model_task)

    def advance_dataset(self) -> None:
        """Advance the dataset progress bar."""
        if self.dataset_task is not None:
            self.progress.advance(self.dataset_task)

    def advance_sample(self) -> None:
        """Advance the sample progress bar."""
        if self.sample_task is not None:
            self.progress.advance(self.sample_task)

    def finish_dataset_processing(self) -> None:
        """Remove the dataset progress task."""
        if self.dataset_task is not None:
            self.progress.remove_task(self.dataset_task)
            self.dataset_task = None

    def finish_sample_processing(self) -> None:
        """Remove the sample progress task."""
        if self.sample_task is not None:
            self.progress.remove_task(self.sample_task)
            self.sample_task = None
