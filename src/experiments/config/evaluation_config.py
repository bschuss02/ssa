from pathlib import Path
from typing import List, Literal

from pydantic import BaseModel, Field


class BbbenjiConfig(BaseModel):
    """Configuration for Bbbenji dataset."""

    subset: Literal["fluent", "stuttered", "all"] = Field(
        default="all", description="The subset of the Bbbenji dataset to use"
    )


class EvaluationPreprocessingConfig(BaseModel):
    remove_punctuation: bool = Field(
        default=False,
        description="Whether to remove punctuation from ground truth and predicted transcripts before calculating metrics",
    )
    make_lowercase: bool = Field(
        default=False,
        description="Whether to convert ground truth and predicted transcripts to lowercase before calculating metrics",
    )
    remove_spaces: bool = Field(
        default=False,
        description="Whether to remove spaces from ground truth and predicted transcripts before calculating metrics",
    )


class EvaluationConfig(BaseModel):
    models: List[str] = Field(description="A list of model names")
    datasets: List[str] = Field(description="A list of dataset names")
    max_samples_per_dataset: int = Field(
        description="The maximum number of audio samples to evaluate per dataset"
    )
    batch_size: int = Field(description="The batch size to use for inference")
    output_dir: Path = Field(description="The directory to save the evaluation results")
    results_dir: Path = Field(description="The directory to save the evaluation results")
    eval_preprocessing: EvaluationPreprocessingConfig = Field(
        default_factory=EvaluationPreprocessingConfig,
        description="Evaluation preprocessing configuration",
    )
    max_workers: int = Field(
        default=4,
        description="The maximum number of workers to use for loading audio files",
    )
    max_output_tokens: int = Field(
        default=1024,
        description="The maximum number of output tokens to generate",
    )
    bbbenji: BbbenjiConfig = Field(
        default_factory=BbbenjiConfig, description="BBBenji configuration settings"
    )
