"""
Results Analyzer Module

This module handles the analysis and reporting of ASR evaluation results.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from experiments.utils.analyze_results import analyze_results
from experiments.utils.configure_logging import logger


class ResultsAnalyzer:
    """Handles analysis of ASR evaluation results"""

    def __init__(self, results_dir: str):
        """
        Initialize the ResultsAnalyzer

        Args:
            results_dir: Directory to save analysis results
        """
        self.results_dir = Path(results_dir)

    def analyze_and_visualize(self, evaluation_results: List, config: Dict = None) -> Dict:
        """
        Perform analysis of evaluation results

        Args:
            evaluation_results: List of evaluation results to analyze
            config: Configuration dictionary with experiment settings

        Returns:
            Dictionary containing analysis results
        """
        if not evaluation_results:
            logger.warning("No evaluation results to analyze")
            return {}

        logger.info(f"Analyzing {len(evaluation_results)} evaluation results")

        # Perform analysis
        analysis_results = analyze_results(evaluation_results)

        # Create output directory structure
        now = datetime.now()
        output_dir = self.results_dir / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save evaluation dataframe as parquet
        if "raw_data" in analysis_results:
            parquet_path = output_dir / "evaluation_results.parquet"
            analysis_results["raw_data"].write_parquet(parquet_path)
            logger.info(f"Evaluation dataframe saved to {parquet_path}")

        # Create metadata file
        metadata_path = output_dir / "evaluation_metadata.json"
        self._create_metadata(metadata_path, evaluation_results, config, analysis_results)

        logger.info(f"Analysis complete. Results saved to {output_dir}")

        # Print summary
        self._print_summary(analysis_results)

        return analysis_results

    def _create_metadata(
        self,
        metadata_path: Path,
        evaluation_results: List,
        config: Dict,
        analysis_results: Dict,
    ):
        """Create simplified metadata file"""
        # Calculate simple per-model metrics
        model_metrics = {}
        for result in evaluation_results:
            if result.model_name not in model_metrics:
                model_metrics[result.model_name] = {
                    "wer": [],
                    "cer": [],
                    "inference_time": [],
                }
            model_metrics[result.model_name]["wer"].append(result.metrics.wer)
            model_metrics[result.model_name]["cer"].append(result.metrics.cer)
            model_metrics[result.model_name]["inference_time"].append(result.inference_time)

        # Calculate averages
        model_summary = {}
        for model_name, metrics in model_metrics.items():
            model_summary[model_name] = {
                "mean_wer": sum(metrics["wer"]) / len(metrics["wer"]),
                "mean_cer": sum(metrics["cer"]) / len(metrics["cer"]),
                "mean_inference_time": sum(metrics["inference_time"])
                / len(metrics["inference_time"]),
                "sample_count": len(metrics["wer"]),
            }

        # Find best model (lowest WER)
        best_model = min(model_summary.items(), key=lambda x: x[1]["mean_wer"])[0]

        metadata = {
            "timestamp": datetime.now().isoformat(),
            "total_samples": len(evaluation_results),
            "models": list(set(r.model_name for r in evaluation_results)),
            "datasets": list(set(r.dataset_name for r in evaluation_results)),
            "configuration": config or {},
            "model_summary": model_summary,
            "best_model": best_model,
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Metadata saved to {metadata_path}")

    def _print_summary(self, analysis_results: Dict):
        """Print a simple summary of the analysis results"""
        if not analysis_results:
            return

        print("\n" + "=" * 60)
        print("ASR EVALUATION SUMMARY")
        print("=" * 60)

        # Best model
        if "performance_ranking" in analysis_results:
            ranking = analysis_results["performance_ranking"]
            print(f"\nBest Model: {ranking.best_model}")

        # Model performance summary
        if "performance_summary" in analysis_results:
            summaries = analysis_results["performance_summary"]
            print("\nModel Performance (WER / CER):")
            for summary in summaries:
                print(
                    f"  {summary.model_name} ({summary.dataset_name}): "
                    f"{summary.mean_wer:.3f} / {summary.mean_cer:.3f} "
                    f"({summary.sample_count} samples)"
                )

        print("=" * 60)
