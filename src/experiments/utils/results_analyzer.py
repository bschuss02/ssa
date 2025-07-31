"""
Results Analyzer Module

This module handles the analysis, visualization, and reporting of ASR evaluation results.
It provides comprehensive analysis capabilities including performance ranking, statistical analysis,
error analysis, and dataset analysis with automated visualization generation.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from experiments.utils.analyze_results import analyze_results
from experiments.utils.visualize_results import (
    ASRResultsVisualizer,
    create_detailed_analysis_report,
    create_quick_summary_plot,
)


class ResultsAnalyzer:
    """Handles comprehensive analysis and visualization of ASR evaluation results"""

    def __init__(self, results_dir: str, log: logging.Logger = None):
        """
        Initialize the ResultsAnalyzer

        Args:
            results_dir: Directory to save analysis results
            log: Logger instance for logging
        """
        self.results_dir = Path(results_dir)
        self._log = log or logging.getLogger(__name__)

    def analyze_and_visualize(
        self, evaluation_results: List, config: Dict = None
    ) -> Dict:
        """
        Perform comprehensive analysis and visualization of evaluation results

        Args:
            evaluation_results: List of evaluation results to analyze
            config: Configuration dictionary with experiment settings

        Returns:
            Dictionary containing analysis results
        """
        if not evaluation_results:
            self._log.warning("No evaluation results to analyze")
            return {}

        self._log.info(f"Analyzing {len(evaluation_results)} evaluation results")

        # Perform comprehensive analysis
        analysis_results = analyze_results(evaluation_results)

        # Create organized output directory structure for visualizations
        now = datetime.now()
        date_folder = now.strftime("%Y-%m-%d")
        time_folder = now.strftime("%H-%M-%S")

        # Create nested directory structure: results_dir/date/time/
        output_dir = self.results_dir / date_folder / time_folder
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for different types of visualizations
        charts_dir = output_dir / "charts"
        charts_dir.mkdir(exist_ok=True)

        # Create visualizations
        self._create_visualizations(evaluation_results, charts_dir)

        # Create documentation files
        self._create_documentation(
            output_dir,
            analysis_results,
            date_folder,
            time_folder,
            evaluation_results,
            config,
        )

        self._log.info(f"Analysis complete. Visualizations saved to {output_dir}")

        # Print summary statistics
        self._print_analysis_summary(analysis_results)

        return analysis_results

    def _create_visualizations(self, evaluation_results: List, charts_dir: Path):
        """Create all visualization charts"""
        visualizer = ASRResultsVisualizer()

        # Quick summary plot
        quick_summary_path = charts_dir / "01_quick_summary_overview.png"
        create_quick_summary_plot(evaluation_results, quick_summary_path)

        # Comprehensive dashboard
        dashboard_path = charts_dir / "02_comprehensive_dashboard.png"
        visualizer.create_comprehensive_dashboard(evaluation_results, dashboard_path)

        # Model comparison plot
        model_comparison_path = charts_dir / "03_model_comparison_detailed.png"
        visualizer.create_model_comparison_plot(
            evaluation_results, model_comparison_path
        )

        # Dataset analysis plot
        dataset_analysis_path = charts_dir / "04_dataset_analysis.png"
        visualizer.create_dataset_analysis_plot(
            evaluation_results, dataset_analysis_path
        )

        # Error inspection plot
        error_inspection_path = charts_dir / "05_error_inspection_samples.png"
        visualizer.create_error_inspection_plot(
            evaluation_results, error_threshold=0.3, save_path=error_inspection_path
        )

        # Detailed analysis report
        detailed_report_path = charts_dir / "06_detailed_analysis_report.png"
        create_detailed_analysis_report(evaluation_results, detailed_report_path)

    def _create_documentation(
        self,
        output_dir: Path,
        analysis_results: Dict,
        date_folder: str,
        time_folder: str,
        evaluation_results: List,
        config: Dict = None,
    ):
        """Create documentation files including README and metadata"""
        # Create a README file explaining the visualizations
        readme_path = output_dir / "README.md"
        self._create_visualization_readme(
            readme_path, analysis_results, date_folder, time_folder
        )

        # Create evaluation metadata file
        metadata_path = output_dir / "evaluation_metadata.json"
        self._create_evaluation_metadata(
            metadata_path, date_folder, time_folder, evaluation_results, config
        )

    def _print_analysis_summary(self, analysis_results: Dict):
        """Print a summary of the analysis results"""
        if not analysis_results:
            return

        print("\n" + "=" * 80)
        print("ASR EVALUATION ANALYSIS SUMMARY")
        print("=" * 80)

        # Performance ranking
        if "performance_ranking" in analysis_results:
            ranking = analysis_results["performance_ranking"]
            print(f"\n🏆 BEST OVERALL MODEL: {ranking.best_overall_model}")
            print("\n📊 Best Model per Metric:")
            for metric, model in ranking.best_model_per_metric.items():
                print(f"   {metric.upper()}: {model}")

        # Statistical summary
        if "statistical_analysis" in analysis_results:
            stats = analysis_results["statistical_analysis"]
            print(f"\n📈 STATISTICAL SUMMARY:")
            print(
                f"   Total samples analyzed: {stats['confidence_intervals']['wer']['n_samples']}"
            )
            print(
                f"   Outlier samples (>3σ): {stats['outlier_analysis']['outlier_count']} ({stats['outlier_analysis']['outlier_percentage']:.1f}%)"
            )

        # Error analysis
        if "error_analysis" in analysis_results:
            error_analysis = analysis_results["error_analysis"]
            print(f"\n⚠️  ERROR ANALYSIS:")
            print(f"   High error threshold: {error_analysis.high_error_threshold:.3f}")
            print(
                f"   Samples above threshold: {error_analysis.samples_above_threshold}"
            )

        # Dataset analysis
        if "dataset_analysis" in analysis_results:
            dataset_analysis = analysis_results["dataset_analysis"]
            print(f"\n📚 DATASET ANALYSIS:")
            for dataset_name, analysis in dataset_analysis.items():
                print(
                    f"   {dataset_name}: Difficulty={analysis['difficulty_score']:.3f}, Samples={analysis['total_samples']}"
                )

        print("\n" + "=" * 80)

    def _create_visualization_readme(
        self,
        readme_path: Path,
        analysis_results: Dict,
        date_folder: str,
        time_folder: str,
    ):
        """Create a README file explaining the visualizations"""
        readme_content = f"""# ASR Evaluation Analysis Results

This directory contains comprehensive analysis and visualization results from the ASR evaluation.

**Evaluation Date:** {date_folder}  
**Evaluation Time:** {time_folder}

## Generated Visualizations

### 📊 Charts Directory (`charts/`)

1. **01_quick_summary_overview.png** - Quick overview of key metrics and performance
2. **02_comprehensive_dashboard.png** - Complete dashboard with all analysis plots
3. **03_model_comparison_detailed.png** - Detailed model comparison across all metrics
4. **04_dataset_analysis.png** - Dataset difficulty and characteristics analysis
5. **05_error_inspection_samples.png** - High-error samples for detailed inspection
6. **06_detailed_analysis_report.png** - Comprehensive analysis with 21 detailed visualizations

## Analysis Summary

### Performance Rankings
"""

        if "performance_ranking" in analysis_results:
            ranking = analysis_results["performance_ranking"]
            readme_content += f"""
**Best Overall Model:** {ranking.best_overall_model}

**Best Model per Metric:**
"""
            for metric, model in ranking.best_model_per_metric.items():
                readme_content += f"- **{metric.upper()}:** {model}\n"

        if "statistical_analysis" in analysis_results:
            stats = analysis_results["statistical_analysis"]
            readme_content += f"""
### Statistical Summary
- **Total samples analyzed:** {stats["confidence_intervals"]["wer"]["n_samples"]}
- **Outlier samples (>3σ):** {stats["outlier_analysis"]["outlier_count"]} ({stats["outlier_analysis"]["outlier_percentage"]:.1f}%)
"""

        if "error_analysis" in analysis_results:
            error_analysis = analysis_results["error_analysis"]
            readme_content += f"""
### Error Analysis
- **High error threshold:** {error_analysis.high_error_threshold:.3f}
- **Samples above threshold:** {error_analysis.samples_above_threshold}
"""

        if "dataset_analysis" in analysis_results:
            dataset_analysis = analysis_results["dataset_analysis"]
            readme_content += f"""
### Dataset Analysis
"""
            for dataset_name, analysis in dataset_analysis.items():
                readme_content += f"- **{dataset_name}:** Difficulty={analysis['difficulty_score']:.3f}, Samples={analysis['total_samples']}\n"

        readme_content += """
## How to Interpret the Results

### Performance Heatmaps
- **Red colors** indicate higher error rates (worse performance)
- **Green colors** indicate lower error rates (better performance)
- **WIP (Word Information Preserved)** uses opposite coloring (green = better)

### Box Plots
- Show the distribution of metrics across models
- The box shows the interquartile range (25th to 75th percentile)
- The line in the box is the median
- Whiskers extend to the most extreme non-outlier points

### Error Inspection
- High-error samples are shown with ground truth vs predicted transcripts
- Use these to understand where models struggle
- Look for patterns in transcription errors

### Correlation Matrix
- Shows relationships between different metrics
- Values range from -1 (perfect negative correlation) to +1 (perfect positive correlation)
- Values close to 0 indicate no correlation

## Files Generated
- All charts are saved as high-resolution PNG files (300 DPI)
- Charts are optimized for both screen viewing and printing
- File names are prefixed with numbers for logical ordering

## Next Steps
1. Review the comprehensive dashboard for overall performance
2. Examine model comparison plots to identify strengths/weaknesses
3. Check error inspection samples for problematic cases
4. Use dataset analysis to understand difficulty variations
"""

        with open(readme_path, "w") as f:
            f.write(readme_content)

        print(f"📖 README file created: {readme_path}")

    def _create_evaluation_metadata(
        self,
        metadata_path: Path,
        date_folder: str,
        time_folder: str,
        evaluation_results: List,
        config: Dict = None,
    ):
        """Create evaluation metadata file with run information"""
        # Collect metadata
        metadata = {
            "evaluation_info": {
                "date": date_folder,
                "time": time_folder,
                "timestamp": datetime.now().isoformat(),
                "total_samples": len(evaluation_results),
            },
            "configuration": config or {},
            "results_summary": {
                "models_evaluated": list(set(r.model_name for r in evaluation_results)),
                "datasets_evaluated": list(
                    set(r.dataset_name for r in evaluation_results)
                ),
                "total_inference_time": sum(
                    r.inference_time for r in evaluation_results
                ),
                "average_inference_time": sum(
                    r.inference_time for r in evaluation_results
                )
                / len(evaluation_results)
                if evaluation_results
                else 0,
            },
        }

        # Add performance metrics summary
        if evaluation_results:
            metrics = ["wer", "mer", "wil", "wip", "cer"]
            metrics_summary = {}
            for metric in metrics:
                values = [getattr(r.metrics, metric) for r in evaluation_results]
                metrics_summary[metric] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "std": (
                        sum((x - sum(values) / len(values)) ** 2 for x in values)
                        / len(values)
                    )
                    ** 0.5,
                }
            metadata["performance_metrics"] = metrics_summary

        # Write metadata to file
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"📋 Evaluation metadata saved to: {metadata_path}")
