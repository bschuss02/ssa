"""
ASR Evaluation Metrics and Data Models

This module contains the data models and metric calculation functions
for ASR evaluation results.
"""

import jiwer
from pydantic import BaseModel


class EvaluationMetrics(BaseModel):
    """Container for ASR evaluation metrics."""

    wer: float
    mer: float
    wil: float
    wip: float
    cer: float
    visualize_alignment: str


class EvaluationResultRow(BaseModel):
    """Container for a single evaluation result."""

    model_name: str
    dataset_name: str
    metrics: EvaluationMetrics


def calculate_asr_metrics(reference: str, hypothesis: str) -> EvaluationMetrics:
    """
    Calculate ASR evaluation metrics between reference and hypothesis text.

    Args:
        reference: The ground truth transcription
        hypothesis: The model's predicted transcription

    Returns:
        EvaluationMetrics object containing all calculated metrics
    """
    wer = jiwer.wer(reference, hypothesis)
    mer = jiwer.mer(reference, hypothesis)
    wil = jiwer.wil(reference, hypothesis)
    wip = jiwer.wip(reference, hypothesis)
    cer = jiwer.cer(reference, hypothesis)

    # Create alignment visualization
    try:
        visualize_alignment = jiwer.visualize_alignment(reference, hypothesis)
    except Exception as alignment_error:
        visualize_alignment = f"Alignment failed: {alignment_error}"

    return EvaluationMetrics(
        wer=wer,
        mer=mer,
        wil=wil,
        wip=wip,
        cer=cer,
        visualize_alignment=visualize_alignment,
    )
