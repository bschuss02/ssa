from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from pydantic import BaseModel

from experiments.utils.evaluation_result import EvaluationMetrics, EvaluationResult


class ModelPerformanceSummary(BaseModel):
    """Summary statistics for a model on a dataset"""

    model_name: str
    dataset_name: str
    sample_count: int
    mean_wer: float
    mean_mer: float
    mean_wil: float
    mean_wip: float
    mean_cer: float
    std_wer: float
    std_mer: float
    std_wil: float
    std_wip: float
    std_cer: float
    mean_inference_time: float
    total_inference_time: float


class ErrorAnalysisResult(BaseModel):
    """Results from error analysis"""

    worst_samples: List[EvaluationResult]
    error_patterns: Dict[str, int]
    high_error_threshold: float
    samples_above_threshold: int


class PerformanceRanking(BaseModel):
    """Ranking of models by different metrics"""

    model_rankings: Dict[str, List[Tuple[str, float]]]  # metric -> [(model, score)]
    best_model_per_metric: Dict[str, str]
    best_overall_model: str


def analyze_results(evaluation_results: List[EvaluationResult]) -> Dict:
    """
    Comprehensive analysis of ASR evaluation results

    Args:
        evaluation_results: List of evaluation results

    Returns:
        Dictionary containing all analysis results
    """
    if not evaluation_results:
        return {}

    # Convert to polars DataFrame for efficient analysis
    df = _convert_to_dataframe(evaluation_results)

    # Perform all analyses
    performance_summary = _calculate_performance_summary(df)
    error_analysis = _perform_error_analysis(df, evaluation_results)
    performance_ranking = _rank_models(df)
    statistical_analysis = _perform_statistical_analysis(df)
    dataset_analysis = _analyze_dataset_characteristics(df)

    return {
        "performance_summary": performance_summary,
        "error_analysis": error_analysis,
        "performance_ranking": performance_ranking,
        "statistical_analysis": statistical_analysis,
        "dataset_analysis": dataset_analysis,
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
                "mer": result.metrics.mer,
                "wil": result.metrics.wil,
                "wip": result.metrics.wip,
                "cer": result.metrics.cer,
            }
        )

    return pl.DataFrame(data)


def _calculate_performance_summary(df: pl.DataFrame) -> List[ModelPerformanceSummary]:
    """Calculate performance summary for each model-dataset combination"""
    summaries = []

    # Group by model and dataset
    grouped = df.group_by(["model_name", "dataset_name"]).agg(
        [
            pl.count().alias("sample_count"),
            pl.col("wer").mean().alias("mean_wer"),
            pl.col("mer").mean().alias("mean_mer"),
            pl.col("wil").mean().alias("mean_wil"),
            pl.col("wip").mean().alias("mean_wip"),
            pl.col("cer").mean().alias("mean_cer"),
            pl.col("wer").std().alias("std_wer"),
            pl.col("mer").std().alias("std_mer"),
            pl.col("wil").std().alias("std_wil"),
            pl.col("wip").std().alias("std_wip"),
            pl.col("cer").std().alias("std_cer"),
            pl.col("inference_time").mean().alias("mean_inference_time"),
            pl.col("inference_time").sum().alias("total_inference_time"),
        ]
    )

    for row in grouped.iter_rows(named=True):
        summary = ModelPerformanceSummary(
            model_name=row["model_name"],
            dataset_name=row["dataset_name"],
            sample_count=row["sample_count"],
            mean_wer=row["mean_wer"],
            mean_mer=row["mean_mer"],
            mean_wil=row["mean_wil"],
            mean_wip=row["mean_wip"],
            mean_cer=row["mean_cer"],
            std_wer=row["std_wer"],
            std_mer=row["std_mer"],
            std_wil=row["std_wil"],
            std_wip=row["std_wip"],
            std_cer=row["std_cer"],
            mean_inference_time=row["mean_inference_time"],
            total_inference_time=row["total_inference_time"],
        )
        summaries.append(summary)

    return summaries


def _perform_error_analysis(
    df: pl.DataFrame, evaluation_results: List[EvaluationResult]
) -> ErrorAnalysisResult:
    """Analyze errors and identify problematic samples"""
    # Find worst performing samples (highest WER)
    worst_samples_df = df.sort("wer", descending=True).head(10)

    # Get the actual EvaluationResult objects for worst samples
    worst_samples = []
    for row in worst_samples_df.iter_rows(named=True):
        for result in evaluation_results:
            if (
                result.model_name == row["model_name"]
                and result.dataset_name == row["dataset_name"]
                and result.ground_truth_transcript == row["ground_truth_transcript"]
                and result.predicted_transcript == row["predicted_transcript"]
            ):
                worst_samples.append(result)
                break

    # Analyze error patterns
    high_error_threshold = df["wer"].quantile(0.95)  # 95th percentile
    samples_above_threshold = df.filter(pl.col("wer") > high_error_threshold).height

    # Count error patterns (e.g., by model, dataset)
    error_patterns = {
        "high_error_by_model": df.filter(pl.col("wer") > high_error_threshold)
        .group_by("model_name")
        .count()
        .to_dict(as_series=False),
        "high_error_by_dataset": df.filter(pl.col("wer") > high_error_threshold)
        .group_by("dataset_name")
        .count()
        .to_dict(as_series=False),
    }

    return ErrorAnalysisResult(
        worst_samples=worst_samples,
        error_patterns=error_patterns,
        high_error_threshold=high_error_threshold,
        samples_above_threshold=samples_above_threshold,
    )


def _rank_models(df: pl.DataFrame) -> PerformanceRanking:
    """Rank models by different metrics"""
    metrics = ["wer", "mer", "wil", "wip", "cer"]
    model_rankings = {}
    best_model_per_metric = {}

    # Calculate average performance per model across all datasets
    model_performance = df.group_by("model_name").agg(
        [
            pl.col("wer").mean().alias("avg_wer"),
            pl.col("mer").mean().alias("avg_mer"),
            pl.col("wil").mean().alias("avg_wil"),
            pl.col("wip").mean().alias("avg_wip"),
            pl.col("cer").mean().alias("avg_cer"),
            pl.col("inference_time").mean().alias("avg_inference_time"),
        ]
    )

    # Rank by each metric (lower is better for error metrics, higher is better for wip)
    for metric in metrics:
        if metric == "wip":  # Higher is better
            sorted_models = model_performance.sort(f"avg_{metric}", descending=True)
        else:  # Lower is better
            sorted_models = model_performance.sort(f"avg_{metric}")

        rankings = [
            (row["model_name"], row[f"avg_{metric}"])
            for row in sorted_models.iter_rows(named=True)
        ]
        model_rankings[metric] = rankings
        best_model_per_metric[metric] = rankings[0][0]

    # Determine best overall model (average rank across all metrics)
    model_scores = {}
    for model in model_performance["model_name"]:
        total_rank = 0
        for metric in metrics:
            rank = next(
                i for i, (m, _) in enumerate(model_rankings[metric]) if m == model
            )
            total_rank += rank
        model_scores[model] = total_rank / len(metrics)

    best_overall_model = min(model_scores, key=model_scores.get)

    return PerformanceRanking(
        model_rankings=model_rankings,
        best_model_per_metric=best_model_per_metric,
        best_overall_model=best_overall_model,
    )


def _perform_statistical_analysis(df: pl.DataFrame) -> Dict:
    """Perform statistical analysis on the results"""
    # Calculate confidence intervals
    confidence_intervals = {}
    metrics = ["wer", "mer", "wil", "wip", "cer"]

    for metric in metrics:
        values = df[metric].to_numpy()
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        n = len(values)
        ci_95 = 1.96 * std / np.sqrt(n)  # 95% confidence interval

        confidence_intervals[metric] = {
            "mean": mean,
            "std": std,
            "ci_95_lower": mean - ci_95,
            "ci_95_upper": mean + ci_95,
            "n_samples": n,
            "median": np.median(values),
            "q25": np.percentile(values, 25),
            "q75": np.percentile(values, 75),
            "min": np.min(values),
            "max": np.max(values),
        }

    # Correlation analysis between metrics
    correlation_matrix = df.select(metrics).corr()

    # Outlier detection (samples with WER > 3 standard deviations from mean)
    wer_mean = df["wer"].mean()
    wer_std = df["wer"].std()
    outlier_threshold = wer_mean + 3 * wer_std
    outliers = df.filter(pl.col("wer") > outlier_threshold)

    # Additional statistical insights
    # Performance distribution analysis
    performance_distribution = {}
    for metric in metrics:
        values = df[metric].to_numpy()
        performance_distribution[metric] = {
            "skewness": _calculate_skewness(values),
            "kurtosis": _calculate_kurtosis(values),
            "coefficient_of_variation": std / mean if mean != 0 else 0,
        }

    # Model consistency analysis
    model_consistency = {}
    for model_name in df["model_name"].unique():
        model_df = df.filter(pl.col("model_name") == model_name)
        model_consistency[model_name] = {
            "wer_cv": model_df["wer"].std() / model_df["wer"].mean()
            if model_df["wer"].mean() != 0
            else 0,
            "avg_inference_time": model_df["inference_time"].mean(),
            "total_samples": model_df.height,
        }

    return {
        "confidence_intervals": confidence_intervals,
        "correlation_matrix": correlation_matrix,
        "outlier_analysis": {
            "outlier_threshold": outlier_threshold,
            "outlier_count": outliers.height,
            "outlier_percentage": outliers.height / df.height * 100,
        },
        "performance_distribution": performance_distribution,
        "model_consistency": model_consistency,
    }


def _calculate_skewness(values: np.ndarray) -> float:
    """Calculate skewness of a distribution"""
    if len(values) < 3:
        return 0.0
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    if std == 0:
        return 0.0
    return np.mean(((values - mean) / std) ** 3)


def _calculate_kurtosis(values: np.ndarray) -> float:
    """Calculate kurtosis of a distribution"""
    if len(values) < 4:
        return 0.0
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    if std == 0:
        return 0.0
    return np.mean(((values - mean) / std) ** 4) - 3


def _analyze_dataset_characteristics(df: pl.DataFrame) -> Dict:
    """Analyze characteristics of each dataset"""
    dataset_analysis = {}

    for dataset_name in df["dataset_name"].unique():
        dataset_df = df.filter(pl.col("dataset_name") == dataset_name)

        # Calculate average performance across all models
        avg_metrics = dataset_df.group_by("dataset_name").agg(
            [
                pl.col("wer").mean().alias("avg_wer"),
                pl.col("mer").mean().alias("avg_mer"),
                pl.col("wil").mean().alias("avg_wil"),
                pl.col("wip").mean().alias("avg_wip"),
                pl.col("cer").mean().alias("avg_cer"),
                pl.col("inference_time").mean().alias("avg_inference_time"),
                pl.count().alias("total_samples"),
            ]
        )

        # Calculate difficulty score (average WER across all models)
        difficulty_score = avg_metrics["avg_wer"][0]

        # Analyze transcript characteristics
        transcript_lengths = dataset_df["ground_truth_transcript"].str.len_chars()
        avg_transcript_length = transcript_lengths.mean()

        dataset_analysis[dataset_name] = {
            "difficulty_score": difficulty_score,
            "avg_transcript_length": avg_transcript_length,
            "total_samples": avg_metrics["total_samples"][0],
            "avg_metrics": {
                "wer": avg_metrics["avg_wer"][0],
                "mer": avg_metrics["avg_mer"][0],
                "wil": avg_metrics["avg_wil"][0],
                "wip": avg_metrics["avg_wip"][0],
                "cer": avg_metrics["avg_cer"][0],
            },
        }

    return dataset_analysis


def get_model_comparison_dataframe(
    evaluation_results: List[EvaluationResult],
) -> pl.DataFrame:
    """Get a DataFrame suitable for model comparison visualizations"""
    df = _convert_to_dataframe(evaluation_results)

    # Aggregate by model and dataset
    comparison_df = df.group_by(["model_name", "dataset_name"]).agg(
        [
            pl.col("wer").mean().alias("mean_wer"),
            pl.col("mer").mean().alias("mean_mer"),
            pl.col("wil").mean().alias("mean_wil"),
            pl.col("wip").mean().alias("mean_wip"),
            pl.col("cer").mean().alias("mean_cer"),
            pl.col("inference_time").mean().alias("mean_inference_time"),
            pl.count().alias("sample_count"),
        ]
    )

    return comparison_df


def get_error_samples(
    evaluation_results: List[EvaluationResult],
    error_threshold: float = 0.5,
    max_samples: int = 20,
) -> List[EvaluationResult]:
    """Get samples with high error rates for detailed inspection"""
    high_error_samples = [
        result for result in evaluation_results if result.metrics.wer > error_threshold
    ]

    # Sort by WER (highest first) and return top samples
    high_error_samples.sort(key=lambda x: x.metrics.wer, reverse=True)
    return high_error_samples[:max_samples]
