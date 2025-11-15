from typing import Dict, List, Tuple

import polars as pl
from pydantic import BaseModel

from experiments.utils.evaluation_result import EvaluationResult


class ModelPerformanceSummary(BaseModel):
    """Summary statistics for a model on a dataset"""

    model_name: str
    dataset_name: str
    sample_count: int
    mean_wer: float
    mean_cer: float
    mean_inference_time: float


class PerformanceRanking(BaseModel):
    """Ranking of models by WER"""

    best_model: str
    model_rankings: List[Tuple[str, float]]  # [(model, wer_score)]


def analyze_results(evaluation_results: List[EvaluationResult]) -> Dict:
    """
    Analyze ASR evaluation results

    Args:
        evaluation_results: List of evaluation results

    Returns:
        Dictionary containing analysis results
    """
    if not evaluation_results:
        return {}

    # Convert to polars DataFrame
    df = _convert_to_dataframe(evaluation_results)

    # Calculate summaries
    performance_summary = _calculate_performance_summary(df)
    performance_ranking = _rank_models(df)

    return {
        "performance_summary": performance_summary,
        "performance_ranking": performance_ranking,
        "raw_data": df,
    }


def _convert_to_dataframe(evaluation_results: List[EvaluationResult]) -> pl.DataFrame:
    """Convert evaluation results to polars DataFrame"""
    data = []
    for result in evaluation_results:
        data.append(
            {
                "model_name": result.model_name,
                "dataset_name": result.dataset_name,
                "ground_truth_transcript": result.ground_truth_transcript,
                "predicted_transcript": result.predicted_transcript,
                "inference_time": result.inference_time,
                "wer": result.metrics.wer,
                "cer": result.metrics.cer,
            }
        )

    return pl.DataFrame(data)


def _calculate_performance_summary(df: pl.DataFrame) -> List[ModelPerformanceSummary]:
    """Calculate performance summary for each model-dataset combination"""
    summaries = []

    grouped = df.group_by(["model_name", "dataset_name"]).agg(
        [
            pl.count().alias("sample_count"),
            pl.col("wer").mean().alias("mean_wer"),
            pl.col("cer").mean().alias("mean_cer"),
            pl.col("inference_time").mean().alias("mean_inference_time"),
        ]
    )

    for row in grouped.iter_rows(named=True):
        summary = ModelPerformanceSummary(
            model_name=row["model_name"],
            dataset_name=row["dataset_name"],
            sample_count=row["sample_count"],
            mean_wer=row["mean_wer"],
            mean_cer=row["mean_cer"],
            mean_inference_time=row["mean_inference_time"],
        )
        summaries.append(summary)

    return summaries


def _rank_models(df: pl.DataFrame) -> PerformanceRanking:
    """Rank models by WER"""
    model_performance = df.group_by("model_name").agg([pl.col("wer").mean().alias("avg_wer")])

    # Sort by WER (lower is better)
    sorted_models = model_performance.sort("avg_wer")
    rankings = [(row["model_name"], row["avg_wer"]) for row in sorted_models.iter_rows(named=True)]

    best_model = rankings[0][0] if rankings else ""

    return PerformanceRanking(
        best_model=best_model,
        model_rankings=rankings,
    )
