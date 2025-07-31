from pathlib import Path
from typing import Dict

from pydantic import BaseModel, Field


class EvaluationConfig(BaseModel):
    models: Dict[str, Path] = Field(
        description="A dictionary of model names and their paths"
    )
    datasets: Dict[str, Path] = Field(
        description="A dictionary of dataset names and their paths"
    )
    max_samples_per_dataset: int = Field(
        description="The maximum number of audio samples to evaluate per dataset"
    )
    batch_size: int = Field(description="The batch size to use for inference")
    output_dir: Path = Field(description="The directory to save the evaluation results")
