from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from experiments.utils.analyze_results import (
    analyze_results,
    get_error_samples,
    get_model_comparison_dataframe,
)
from experiments.utils.evaluation_result import EvaluationResult


class ASRResultsVisualizer:
    """Comprehensive visualizer for ASR evaluation results"""

    def __init__(self, style: str = "whitegrid"):
        """Initialize the visualizer with a specific style"""
        sns.set_style(style)
        plt.rcParams["figure.figsize"] = (12, 8)
        plt.rcParams["font.size"] = 10

    def create_comprehensive_dashboard(
        self,
        evaluation_results: List[EvaluationResult],
        save_path: Optional[Path] = None,
    ) -> None:
        """
        Create a comprehensive dashboard with all key visualizations

        Args:
            evaluation_results: List of evaluation results
            save_path: Optional path to save the dashboard
        """
        if not evaluation_results:
            print("No evaluation results to visualize")
            return

        # Create a large figure with subplots
        fig = plt.figure(figsize=(20, 24))

        # Analysis results
        analysis_results = analyze_results(evaluation_results)

        # 1. Performance Heatmap
        plt.subplot(4, 3, 1)
        self._plot_performance_heatmap(evaluation_results, "wer")
        plt.title("WER Performance Heatmap")

        # 2. Model Comparison Box Plot
        plt.subplot(4, 3, 2)
        self._plot_model_comparison_boxplot(evaluation_results, "wer")
        plt.title("WER Distribution by Model")

        # 3. Dataset Difficulty Analysis
        plt.subplot(4, 3, 3)
        self._plot_dataset_difficulty(analysis_results["dataset_analysis"])
        plt.title("Dataset Difficulty Analysis")

        # 4. CER Performance Heatmap
        plt.subplot(4, 3, 4)
        self._plot_performance_heatmap(evaluation_results, "cer")
        plt.title("CER Performance Heatmap")

        # 5. WIP Performance Heatmap
        plt.subplot(4, 3, 5)
        self._plot_performance_heatmap(evaluation_results, "wip")
        plt.title("WIP Performance Heatmap")

        # 6. Inference Time Analysis
        plt.subplot(4, 3, 6)
        self._plot_inference_time_analysis(evaluation_results)
        plt.title("Inference Time Analysis")

        # 7. Error Distribution
        plt.subplot(4, 3, 7)
        self._plot_error_distribution(evaluation_results)
        plt.title("Error Distribution")

        # 8. Model Ranking
        plt.subplot(4, 3, 8)
        self._plot_model_ranking(analysis_results["performance_ranking"])
        plt.title("Model Performance Ranking")

        # 9. Correlation Matrix
        plt.subplot(4, 3, 9)
        self._plot_correlation_matrix(
            analysis_results["statistical_analysis"]["correlation_matrix"]
        )
        plt.title("Metric Correlation Matrix")

        # 10. Performance Summary
        plt.subplot(4, 3, 10)
        self._plot_performance_summary(analysis_results["performance_summary"])
        plt.title("Performance Summary")

        # 11. Error Analysis
        plt.subplot(4, 3, 11)
        self._plot_error_analysis(analysis_results["error_analysis"])
        plt.title("Error Analysis")

        # 12. Statistical Summary
        plt.subplot(4, 3, 12)
        self._plot_statistical_summary(analysis_results["statistical_analysis"])
        plt.title("Statistical Summary")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Dashboard saved to {save_path}")

        plt.close()  # Close the figure instead of showing it

    def _plot_performance_heatmap(
        self, evaluation_results: List[EvaluationResult], metric: str
    ) -> None:
        """Create a performance heatmap for a specific metric"""
        df = get_model_comparison_dataframe(evaluation_results)

        # Convert to pandas for easier pivot operation
        import pandas as pd

        df_pandas = df.to_pandas()

        # Pivot data for heatmap
        heatmap_data = df_pandas.pivot(
            values=f"mean_{metric}", index="dataset_name", columns="model_name"
        )

        # Handle empty or single-value cases
        if heatmap_data.empty or heatmap_data.size == 1:
            plt.text(
                0.5,
                0.5,
                f"Insufficient data for {metric.upper()} heatmap",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )
            plt.title(f"{metric.upper()} Performance Heatmap")
            return

        # Create heatmap
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn_r" if metric != "wip" else "RdYlGn",
            center=heatmap_data.mean().mean()
            if not heatmap_data.isna().all().all()
            else 0,
            cbar_kws={"label": f"Mean {metric.upper()}"},
        )

        plt.xlabel("Model")
        plt.ylabel("Dataset")

    def _plot_model_comparison_boxplot(
        self, evaluation_results: List[EvaluationResult], metric: str
    ) -> None:
        """Create a box plot comparing models for a specific metric"""
        df = get_model_comparison_dataframe(evaluation_results)

        # Check if we have data to plot
        if df.height == 0:
            plt.text(
                0.5,
                0.5,
                f"No data available for {metric.upper()} comparison",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )
            plt.title(f"{metric.upper()} Distribution by Model")
            return

        # Create box plot
        sns.boxplot(data=df, x="model_name", y=f"mean_{metric}")
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Model")
        plt.ylabel(f"Mean {metric.upper()}")

    def _plot_dataset_difficulty(self, dataset_analysis: Dict) -> None:
        """Plot dataset difficulty analysis"""
        datasets = list(dataset_analysis.keys())
        difficulty_scores = [dataset_analysis[d]["difficulty_score"] for d in datasets]
        sample_counts = [dataset_analysis[d]["total_samples"] for d in datasets]

        # Create scatter plot with size representing sample count
        plt.scatter(
            datasets, difficulty_scores, s=np.array(sample_counts) / 10, alpha=0.7
        )
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Difficulty Score (Avg WER)")
        plt.xlabel("Dataset")

        # Add trend line only if we have more than 1 dataset
        if len(datasets) > 1:
            try:
                z = np.polyfit(range(len(datasets)), difficulty_scores, 1)
                p = np.poly1d(z)
                plt.plot(datasets, p(range(len(datasets))), "r--", alpha=0.8)
            except np.linalg.LinAlgError:
                # If polyfit fails, just skip the trend line
                pass

    def _plot_inference_time_analysis(
        self, evaluation_results: List[EvaluationResult]
    ) -> None:
        """Plot inference time analysis"""
        df = get_model_comparison_dataframe(evaluation_results)

        # Create bar plot of average inference times
        avg_times = df.group_by("model_name").agg(
            [pl.col("mean_inference_time").mean().alias("avg_inference_time")]
        )

        plt.bar(avg_times["model_name"], avg_times["avg_inference_time"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Average Inference Time (seconds)")
        plt.xlabel("Model")

    def _plot_error_distribution(
        self, evaluation_results: List[EvaluationResult]
    ) -> None:
        """Plot error distribution across all samples"""
        # Extract WER values
        wer_values = [result.metrics.wer for result in evaluation_results]

        if not wer_values:
            plt.text(
                0.5,
                0.5,
                "No WER data available",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )
            plt.title("Error Distribution")
            return

        # Create histogram with adaptive binning
        n_bins = min(30, max(5, len(wer_values) // 2))  # Adaptive bin count
        plt.hist(wer_values, bins=n_bins, alpha=0.7, edgecolor="black", density=True)
        plt.xlabel("WER")
        plt.ylabel("Density")

        # Add vertical line for mean
        mean_wer = np.mean(wer_values)
        plt.axvline(
            mean_wer, color="red", linestyle="--", label=f"Mean: {mean_wer:.3f}"
        )

        # Add median line
        median_wer = np.median(wer_values)
        plt.axvline(
            median_wer, color="blue", linestyle=":", label=f"Median: {median_wer:.3f}"
        )

        plt.legend()
        plt.title("WER Distribution")

    def _plot_model_ranking(self, performance_ranking) -> None:
        """Plot model ranking visualization"""
        metrics = ["wer", "mer", "wil", "wip", "cer"]

        # Check if we have ranking data
        if (
            not performance_ranking.model_rankings
            or "wer" not in performance_ranking.model_rankings
        ):
            plt.text(
                0.5,
                0.5,
                "No ranking data available",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )
            plt.title("Model Performance Ranking")
            return

        models = list(performance_ranking.model_rankings["wer"])
        model_names = [model[0] for model in models]

        # Create ranking matrix
        ranking_matrix = []
        for metric in metrics:
            if metric in performance_ranking.model_rankings:
                rankings = performance_ranking.model_rankings[metric]
                rank_values = [rank[1] for rank in rankings]
                ranking_matrix.append(rank_values)
            else:
                # Fill with zeros if metric not available
                ranking_matrix.append([0] * len(model_names))

        # Create heatmap
        sns.heatmap(
            ranking_matrix,
            xticklabels=model_names,
            yticklabels=[m.upper() for m in metrics],
            annot=True,
            fmt=".3f",
            cmap="RdYlGn_r",
        )

        plt.xlabel("Model")
        plt.ylabel("Metric")

    def _plot_correlation_matrix(self, correlation_matrix: pl.DataFrame) -> None:
        """Plot correlation matrix between metrics"""
        # Check if correlation matrix is valid
        if correlation_matrix.height == 0 or correlation_matrix.width == 0:
            plt.text(
                0.5,
                0.5,
                "No correlation data available",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )
            plt.title("Metric Correlation Matrix")
            return

        # Convert to numpy array for seaborn
        corr_array = correlation_matrix.to_numpy()
        metric_names = correlation_matrix.columns

        # Handle NaN values
        if np.isnan(corr_array).any():
            corr_array = np.nan_to_num(corr_array, nan=0.0)

        sns.heatmap(
            corr_array,
            xticklabels=metric_names,
            yticklabels=metric_names,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
        )

    def _plot_performance_summary(self, performance_summary) -> None:
        """Plot performance summary"""
        # Create a summary table
        models = list(set([s.model_name for s in performance_summary]))
        datasets = list(set([s.dataset_name for s in performance_summary]))

        # Create average WER matrix
        avg_wer_matrix = []
        for dataset in datasets:
            row = []
            for model in models:
                summary = next(
                    s
                    for s in performance_summary
                    if s.model_name == model and s.dataset_name == dataset
                )
                row.append(summary.mean_wer)
            avg_wer_matrix.append(row)

        # Create heatmap
        sns.heatmap(
            avg_wer_matrix,
            xticklabels=models,
            yticklabels=datasets,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn_r",
        )

        plt.xlabel("Model")
        plt.ylabel("Dataset")

    def _plot_error_analysis(self, error_analysis) -> None:
        """Plot error analysis results"""
        # Create bar plot of high error samples by model
        high_error_by_model = error_analysis.error_patterns["high_error_by_model"]

        if high_error_by_model:
            models = high_error_by_model["model_name"]
            counts = high_error_by_model["count"]

            plt.bar(models, counts)
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("High Error Samples")
            plt.xlabel("Model")
            plt.title(
                f"High Error Samples (> {error_analysis.high_error_threshold:.3f} WER)"
            )

    def _plot_statistical_summary(self, statistical_analysis: Dict) -> None:
        """Plot statistical summary"""
        # Create confidence interval plot
        metrics = ["wer", "mer", "wil", "wip", "cer"]
        means = []
        ci_lower = []
        ci_upper = []

        for metric in metrics:
            ci_data = statistical_analysis["confidence_intervals"][metric]
            means.append(ci_data["mean"])
            ci_lower.append(ci_data["ci_95_lower"])
            ci_upper.append(ci_data["ci_95_upper"])

        x_pos = np.arange(len(metrics))
        plt.errorbar(
            x_pos,
            means,
            yerr=[
                np.array(means) - np.array(ci_lower),
                np.array(ci_upper) - np.array(means),
            ],
            fmt="o",
            capsize=5,
        )

        plt.xticks(x_pos, [m.upper() for m in metrics])
        plt.ylabel("Value")
        plt.xlabel("Metric")
        plt.title("95% Confidence Intervals")

    def create_error_inspection_plot(
        self,
        evaluation_results: List[EvaluationResult],
        error_threshold: float = 0.5,
        max_samples: int = 10,
        save_path: Optional[Path] = None,
    ) -> None:
        """
        Create a detailed plot for inspecting high-error samples

        Args:
            evaluation_results: List of evaluation results
            error_threshold: WER threshold for high-error samples
            max_samples: Maximum number of samples to display
            save_path: Optional path to save the plot
        """
        error_samples = get_error_samples(
            evaluation_results, error_threshold, max_samples
        )

        if not error_samples:
            print(f"No samples found with WER > {error_threshold}")
            return

        fig, axes = plt.subplots(
            len(error_samples), 1, figsize=(15, 4 * len(error_samples))
        )
        if len(error_samples) == 1:
            axes = [axes]

        for i, sample in enumerate(error_samples):
            ax = axes[i]

            # Create text comparison
            text = f"""
Model: {sample.model_name} | Dataset: {sample.dataset_name} | WER: {sample.metrics.wer:.3f}

Ground Truth: {sample.ground_truth_transcript}

Predicted:    {sample.predicted_transcript}

Metrics: WER={sample.metrics.wer:.3f}, MER={sample.metrics.mer:.3f}, 
         WIL={sample.metrics.wil:.3f}, WIP={sample.metrics.wip:.3f}, CER={sample.metrics.cer:.3f}
            """

            ax.text(
                0.05,
                0.95,
                text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
            )

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Error inspection plot saved to {save_path}")

        plt.close()  # Close the figure instead of showing it

    def create_model_comparison_plot(
        self,
        evaluation_results: List[EvaluationResult],
        save_path: Optional[Path] = None,
    ) -> None:
        """
        Create a comprehensive model comparison plot

        Args:
            evaluation_results: List of evaluation results
            save_path: Optional path to save the plot
        """
        df = get_model_comparison_dataframe(evaluation_results)
        metrics = ["wer", "mer", "wil", "wip", "cer"]

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            ax = axes[i]

            # Create violin plot
            sns.violinplot(data=df, x="model_name", y=f"mean_{metric}", ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            ax.set_title(f"{metric.upper()} Distribution")
            ax.set_ylabel(f"Mean {metric.upper()}")

        # Add inference time plot
        ax = axes[5]
        sns.barplot(data=df, x="model_name", y="mean_inference_time", ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_title("Inference Time")
        ax.set_ylabel("Mean Inference Time (seconds)")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Model comparison plot saved to {save_path}")

        plt.close()  # Close the figure instead of showing it

    def create_dataset_analysis_plot(
        self,
        evaluation_results: List[EvaluationResult],
        save_path: Optional[Path] = None,
    ) -> None:
        """
        Create a comprehensive dataset analysis plot

        Args:
            evaluation_results: List of evaluation results
            save_path: Optional path to save the plot
        """
        analysis_results = analyze_results(evaluation_results)
        dataset_analysis = analysis_results["dataset_analysis"]

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Dataset difficulty comparison
        ax1 = axes[0, 0]
        datasets = list(dataset_analysis.keys())
        difficulty_scores = [dataset_analysis[d]["difficulty_score"] for d in datasets]

        bars = ax1.bar(datasets, difficulty_scores)
        ax1.set_title("Dataset Difficulty (Average WER)")
        ax1.set_ylabel("Difficulty Score")
        ax1.tick_params(axis="x", rotation=45)

        # Color bars based on difficulty
        colors = plt.cm.RdYlGn_r(np.array(difficulty_scores) / max(difficulty_scores))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        # 2. Sample count by dataset
        ax2 = axes[0, 1]
        sample_counts = [dataset_analysis[d]["total_samples"] for d in datasets]
        ax2.bar(datasets, sample_counts)
        ax2.set_title("Sample Count by Dataset")
        ax2.set_ylabel("Number of Samples")
        ax2.tick_params(axis="x", rotation=45)

        # 3. Average transcript length
        ax3 = axes[1, 0]
        transcript_lengths = [
            dataset_analysis[d]["avg_transcript_length"] for d in datasets
        ]
        ax3.bar(datasets, transcript_lengths)
        ax3.set_title("Average Transcript Length")
        ax3.set_ylabel("Characters")
        ax3.tick_params(axis="x", rotation=45)

        # 4. Metric correlation heatmap for datasets
        ax4 = axes[1, 1]
        metric_data = []
        for dataset in datasets:
            row = [
                dataset_analysis[dataset]["avg_metrics"]["wer"],
                dataset_analysis[dataset]["avg_metrics"]["mer"],
                dataset_analysis[dataset]["avg_metrics"]["wil"],
                dataset_analysis[dataset]["avg_metrics"]["wip"],
                dataset_analysis[dataset]["avg_metrics"]["cer"],
            ]
            metric_data.append(row)

        sns.heatmap(
            metric_data,
            xticklabels=["WER", "MER", "WIL", "WIP", "CER"],
            yticklabels=datasets,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn_r",
            ax=ax4,
        )
        ax4.set_title("Average Metrics by Dataset")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Dataset analysis plot saved to {save_path}")

        plt.close()  # Close the figure instead of showing it


def create_quick_summary_plot(
    evaluation_results: List[EvaluationResult], save_path: Optional[Path] = None
) -> None:
    """
    Create a quick summary plot for rapid assessment

    Args:
        evaluation_results: List of evaluation results
        save_path: Optional path to save the plot
    """
    if not evaluation_results:
        print("No evaluation results to visualize")
        return

    visualizer = ASRResultsVisualizer()

    # Create a simple 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. WER heatmap
    ax1 = axes[0, 0]
    visualizer._plot_performance_heatmap(evaluation_results, "wer")
    ax1.set_title("WER Performance Heatmap")

    # 2. Model comparison box plot
    ax2 = axes[0, 1]
    visualizer._plot_model_comparison_boxplot(evaluation_results, "wer")
    ax2.set_title("WER Distribution by Model")

    # 3. Inference time
    ax3 = axes[1, 0]
    visualizer._plot_inference_time_analysis(evaluation_results)
    ax3.set_title("Inference Time Analysis")

    # 4. Error distribution
    ax4 = axes[1, 1]
    visualizer._plot_error_distribution(evaluation_results)
    ax4.set_title("Error Distribution")

    # Add overall statistics as text
    fig.suptitle("ASR Evaluation Quick Summary", fontsize=16, fontweight="bold")

    # Add summary statistics
    total_samples = len(evaluation_results)
    avg_wer = np.mean([r.metrics.wer for r in evaluation_results])
    avg_inference_time = np.mean([r.inference_time for r in evaluation_results])

    summary_text = f"Total Samples: {total_samples}\nAvg WER: {avg_wer:.3f}\nAvg Inference Time: {avg_inference_time:.2f}s"
    fig.text(
        0.02,
        0.02,
        summary_text,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Quick summary plot saved to {save_path}")

    plt.close()  # Close the figure instead of showing it


def create_detailed_analysis_report(
    evaluation_results: List[EvaluationResult], save_path: Optional[Path] = None
) -> None:
    """
    Create a detailed analysis report with comprehensive insights

    Args:
        evaluation_results: List of evaluation results
        save_path: Optional path to save the report
    """
    if not evaluation_results:
        print("No evaluation results to analyze")
        return

    from experiments.utils.analyze_results import analyze_results

    # Get analysis results
    analysis_results = analyze_results(evaluation_results)

    # Create a comprehensive report
    fig = plt.figure(figsize=(20, 28))

    # 1. Performance Overview (2x2 grid)
    plt.subplot(7, 3, 1)
    visualizer = ASRResultsVisualizer()
    visualizer._plot_performance_heatmap(evaluation_results, "wer")
    plt.title("WER Performance Heatmap")

    plt.subplot(7, 3, 2)
    visualizer._plot_performance_heatmap(evaluation_results, "cer")
    plt.title("CER Performance Heatmap")

    plt.subplot(7, 3, 3)
    visualizer._plot_performance_heatmap(evaluation_results, "wip")
    plt.title("WIP Performance Heatmap")

    # 2. Model Comparisons
    plt.subplot(7, 3, 4)
    visualizer._plot_model_comparison_boxplot(evaluation_results, "wer")
    plt.title("WER Distribution by Model")

    plt.subplot(7, 3, 5)
    visualizer._plot_model_comparison_boxplot(evaluation_results, "cer")
    plt.title("CER Distribution by Model")

    plt.subplot(7, 3, 6)
    visualizer._plot_inference_time_analysis(evaluation_results)
    plt.title("Inference Time Analysis")

    # 3. Error Analysis
    plt.subplot(7, 3, 7)
    visualizer._plot_error_distribution(evaluation_results)
    plt.title("WER Distribution")

    plt.subplot(7, 3, 8)
    visualizer._plot_error_analysis(analysis_results["error_analysis"])
    plt.title("High Error Samples by Model")

    plt.subplot(7, 3, 9)
    visualizer._plot_dataset_difficulty(analysis_results["dataset_analysis"])
    plt.title("Dataset Difficulty Analysis")

    # 4. Statistical Analysis
    plt.subplot(7, 3, 10)
    visualizer._plot_correlation_matrix(
        analysis_results["statistical_analysis"]["correlation_matrix"]
    )
    plt.title("Metric Correlation Matrix")

    plt.subplot(7, 3, 11)
    visualizer._plot_statistical_summary(analysis_results["statistical_analysis"])
    plt.title("95% Confidence Intervals")

    plt.subplot(7, 3, 12)
    visualizer._plot_model_ranking(analysis_results["performance_ranking"])
    plt.title("Model Performance Ranking")

    # 5. Performance Summary
    plt.subplot(7, 3, 13)
    visualizer._plot_performance_summary(analysis_results["performance_summary"])
    plt.title("Performance Summary Heatmap")

    # 6. Additional insights
    plt.subplot(7, 3, 14)
    _plot_metric_comparison(evaluation_results)
    plt.title("Metric Comparison")

    plt.subplot(7, 3, 15)
    _plot_sample_length_analysis(evaluation_results)
    plt.title("Sample Length vs Performance")

    plt.subplot(7, 3, 16)
    _plot_inference_time_distribution(evaluation_results)
    plt.title("Inference Time Distribution")

    # 7. Summary statistics
    plt.subplot(7, 3, 17)
    _plot_summary_statistics(evaluation_results)
    plt.title("Summary Statistics")

    plt.subplot(7, 3, 18)
    _plot_outlier_analysis(evaluation_results)
    plt.title("Outlier Analysis")

    # 8. Performance trends
    plt.subplot(7, 3, 19)
    _plot_performance_trends(evaluation_results)
    plt.title("Performance Trends")

    # 9. Model efficiency
    plt.subplot(7, 3, 20)
    _plot_efficiency_analysis(evaluation_results)
    plt.title("Model Efficiency (WER vs Time)")

    # 10. Dataset characteristics
    plt.subplot(7, 3, 21)
    _plot_dataset_characteristics(evaluation_results)
    plt.title("Dataset Characteristics")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Detailed analysis report saved to {save_path}")

    plt.close()


def _plot_metric_comparison(evaluation_results: List[EvaluationResult]) -> None:
    """Plot comparison of different metrics"""
    metrics = ["wer", "mer", "wil", "wip", "cer"]
    metric_values = {
        metric: [getattr(r.metrics, metric) for r in evaluation_results]
        for metric in metrics
    }

    # Create box plot
    data_to_plot = [metric_values[metric] for metric in metrics]
    plt.boxplot(data_to_plot, labels=[m.upper() for m in metrics])
    plt.ylabel("Value")
    plt.title("Metric Distribution Comparison")


def _plot_sample_length_analysis(evaluation_results: List[EvaluationResult]) -> None:
    """Plot relationship between sample length and performance"""
    lengths = [len(r.ground_truth_transcript) for r in evaluation_results]
    wers = [r.metrics.wer for r in evaluation_results]

    plt.scatter(lengths, wers, alpha=0.6)
    plt.xlabel("Transcript Length (characters)")
    plt.ylabel("WER")

    # Add trend line if enough data
    if len(lengths) > 1:
        try:
            z = np.polyfit(lengths, wers, 1)
            p = np.poly1d(z)
            plt.plot(lengths, p(lengths), "r--", alpha=0.8)
        except:
            pass


def _plot_inference_time_distribution(
    evaluation_results: List[EvaluationResult],
) -> None:
    """Plot distribution of inference times"""
    times = [r.inference_time for r in evaluation_results]

    if times:
        plt.hist(times, bins=20, alpha=0.7, edgecolor="black")
        plt.xlabel("Inference Time (seconds)")
        plt.ylabel("Frequency")

        # Add mean line
        mean_time = np.mean(times)
        plt.axvline(
            mean_time, color="red", linestyle="--", label=f"Mean: {mean_time:.2f}s"
        )
        plt.legend()


def _plot_summary_statistics(evaluation_results: List[EvaluationResult]) -> None:
    """Plot summary statistics as a table"""
    metrics = ["wer", "mer", "wil", "wip", "cer"]

    # Calculate statistics
    stats = {}
    for metric in metrics:
        values = [getattr(r.metrics, metric) for r in evaluation_results]
        stats[metric] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
        }

    # Create table
    table_data = []
    for metric in metrics:
        table_data.append(
            [
                metric.upper(),
                f"{stats[metric]['mean']:.3f}",
                f"{stats[metric]['std']:.3f}",
                f"{stats[metric]['min']:.3f}",
                f"{stats[metric]['max']:.3f}",
            ]
        )

    table = plt.table(
        cellText=table_data,
        colLabels=["Metric", "Mean", "Std", "Min", "Max"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    plt.axis("off")


def _plot_outlier_analysis(evaluation_results: List[EvaluationResult]) -> None:
    """Plot outlier analysis"""
    wers = [r.metrics.wer for r in evaluation_results]

    if not wers:
        return

    # Calculate outliers using IQR method
    q1 = np.percentile(wers, 25)
    q3 = np.percentile(wers, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = [w for w in wers if w < lower_bound or w > upper_bound]
    normal = [w for w in wers if lower_bound <= w <= upper_bound]

    plt.hist(normal, bins=20, alpha=0.7, label="Normal", color="blue")
    plt.hist(outliers, bins=10, alpha=0.7, label="Outliers", color="red")
    plt.xlabel("WER")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title(
        f"Outliers: {len(outliers)}/{len(wers)} ({len(outliers) / len(wers) * 100:.1f}%)"
    )


def _plot_performance_trends(evaluation_results: List[EvaluationResult]) -> None:
    """Plot performance trends over samples"""
    wers = [r.metrics.wer for r in evaluation_results]

    plt.plot(range(len(wers)), wers, alpha=0.7)
    plt.xlabel("Sample Index")
    plt.ylabel("WER")

    # Add moving average
    if len(wers) > 5:
        window = min(5, len(wers) // 2)
        moving_avg = np.convolve(wers, np.ones(window) / window, mode="valid")
        plt.plot(
            range(window - 1, len(wers)),
            moving_avg,
            "r--",
            alpha=0.8,
            label=f"{window}-point MA",
        )
        plt.legend()


def _plot_efficiency_analysis(evaluation_results: List[EvaluationResult]) -> None:
    """Plot efficiency analysis (WER vs inference time)"""
    wers = [r.metrics.wer for r in evaluation_results]
    times = [r.inference_time for r in evaluation_results]

    plt.scatter(times, wers, alpha=0.6)
    plt.xlabel("Inference Time (seconds)")
    plt.ylabel("WER")
    plt.title("Efficiency: WER vs Inference Time")


def _plot_dataset_characteristics(evaluation_results: List[EvaluationResult]) -> None:
    """Plot dataset characteristics"""
    from collections import defaultdict

    dataset_stats = defaultdict(lambda: {"count": 0, "avg_wer": 0, "total_wer": 0})

    for result in evaluation_results:
        dataset_stats[result.dataset_name]["count"] += 1
        dataset_stats[result.dataset_name]["total_wer"] += result.metrics.wer

    for dataset in dataset_stats:
        dataset_stats[dataset]["avg_wer"] = (
            dataset_stats[dataset]["total_wer"] / dataset_stats[dataset]["count"]
        )

    datasets = list(dataset_stats.keys())
    counts = [dataset_stats[d]["count"] for d in datasets]
    avg_wers = [dataset_stats[d]["avg_wer"] for d in datasets]

    # Create subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.bar(datasets, counts)
    ax1.set_title("Sample Count by Dataset")
    ax1.tick_params(axis="x", rotation=45)

    ax2.bar(datasets, avg_wers)
    ax2.set_title("Average WER by Dataset")
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()
