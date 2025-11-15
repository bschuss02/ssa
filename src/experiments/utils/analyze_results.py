import json
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
        row = {
            "model_name": result.model_name,
            "dataset_name": result.dataset_name,
            "ground_truth_transcript": result.ground_truth_transcript,
            "predicted_transcript": result.predicted_transcript,
            "inference_time": result.inference_time,
            "wer": result.metrics.wer,
            "cer": result.metrics.cer,
        }

        # Extract audio_path from metadata if available
        if result.metadata:
            # Get audio_path from metadata (it might be stored as a Path object)
            audio_path = result.metadata.get("audio_path")
            if audio_path is not None:
                # Convert to string (handles Path objects and strings)
                row["audio_path"] = str(audio_path)

            # Flatten all other metadata fields into the row
            for key, value in result.metadata.items():
                if key != "audio_path":  # Already handled above
                    # Convert non-serializable types to strings
                    if isinstance(value, (dict, list)):
                        # For complex types, convert to JSON string
                        try:
                            row[f"metadata_{key}"] = json.dumps(value)
                        except (TypeError, ValueError):
                            row[f"metadata_{key}"] = str(value)
                    elif hasattr(value, "__str__") and not isinstance(
                        value, (str, int, float, bool, type(None))
                    ):
                        row[f"metadata_{key}"] = str(value)
                    else:
                        row[f"metadata_{key}"] = value

        data.append(row)

    df = pl.DataFrame(data)

    # Define the desired column order (priority columns first)
    priority_columns = [
        "model_name",
        "cer",
        "ground_truth_transcript",
        "predicted_transcript",
        "audio_path",
        "dataset_name",
    ]

    # Get all columns in the dataframe
    all_columns = df.columns

    # Build the final column order: priority columns first (if they exist), then the rest
    ordered_columns = []
    remaining_columns = set(all_columns)

    # Add priority columns in order
    for col in priority_columns:
        if col in remaining_columns:
            ordered_columns.append(col)
            remaining_columns.remove(col)

    # Add remaining columns in their original order
    for col in all_columns:
        if col in remaining_columns:
            ordered_columns.append(col)

    # Reorder the dataframe
    return df.select(ordered_columns)


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
