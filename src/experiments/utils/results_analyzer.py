"""
Results Analyzer Module

This module handles the analysis and reporting of ASR evaluation results.
"""

import difflib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from experiments.utils.analyze_results import analyze_results
from experiments.utils.configure_logging import logger
from experiments.utils.evaluation_result import EvaluationResult


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

        # Save text files for easy comparison
        text_files_dir = output_dir / "transcript_comparisons"
        self._save_text_files(evaluation_results, text_files_dir)

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

        # Find best model (lowest CER)
        best_model = min(model_summary.items(), key=lambda x: x[1]["mean_cer"])[0]

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
            print("\nModel Performance (CER):")
            for summary in summaries:
                print(
                    f"  {summary.model_name} ({summary.dataset_name}): "
                    f"{summary.mean_cer:.3f} "
                    f"({summary.sample_count} samples)"
                )

        print("=" * 60)

    def _save_text_files(self, evaluation_results: List[EvaluationResult], output_dir: Path):
        """
        Save evaluation results as text files for easy comparison

        Args:
            evaluation_results: List of evaluation results to save
            output_dir: Directory to save text files
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving {len(evaluation_results)} text comparison files to {output_dir}")

        for idx, result in enumerate(evaluation_results):
            # Generate filename
            # Try to use audio_path from metadata if available
            filename = None
            if result.metadata and "audio_path" in result.metadata:
                audio_path = result.metadata["audio_path"]
                if audio_path:
                    # Extract filename from path and sanitize
                    audio_path_obj = Path(str(audio_path))
                    filename_base = audio_path_obj.stem
                    # Sanitize filename
                    filename = "".join(
                        c if c.isalnum() or c in "._-" else "_" for c in filename_base
                    )
                    filename = f"{result.model_name}_{result.dataset_name}_{filename}.txt"

            # Fallback to index-based filename
            if not filename:
                filename = f"{result.model_name}_{result.dataset_name}_{idx:04d}.txt"

            file_path = output_dir / filename

            # Format content for easy comparison
            content = self._format_comparison_text(result)

            # Write to file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

        logger.info(f"Text comparison files saved to {output_dir}")

    def _format_comparison_text(self, result: EvaluationResult) -> str:
        """
        Format evaluation result as text for easy comparison

        Args:
            result: EvaluationResult to format

        Returns:
            Formatted text string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("TRANSCRIPT COMPARISON")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Model: {result.model_name}")
        lines.append(f"Dataset: {result.dataset_name}")
        lines.append("")

        # Audio file path - make it easily accessible
        audio_path = None
        if result.metadata and "audio_path" in result.metadata:
            audio_path = result.metadata["audio_path"]
            if audio_path:
                audio_path_str = str(audio_path)
                lines.append("AUDIO FILE:")
                lines.append(f"  {audio_path_str}")
                # Also provide absolute path if it's relative
                try:
                    audio_path_obj = Path(audio_path_str)
                    if not audio_path_obj.is_absolute():
                        # Try to resolve relative to current working directory
                        abs_path = Path(audio_path_str).resolve()
                        if abs_path.exists():
                            lines.append(f"  Absolute: {abs_path}")
                        else:
                            # Try relative to results_dir parent (project root)
                            try:
                                project_root = self.results_dir.parent.parent.parent
                                abs_path = (project_root / audio_path_str).resolve()
                                if abs_path.exists():
                                    lines.append(f"  Absolute: {abs_path}")
                            except Exception:
                                pass
                except Exception:
                    pass
                lines.append("")

        lines.append("METRICS:")
        lines.append(f"  CER (Character Error Rate): {result.metrics.cer:.4f}")
        lines.append("")

        # Chain of Thought reasoning (if available)
        cot_reasoning = self._extract_chain_of_thought(result.metadata)
        if cot_reasoning:
            lines.append("-" * 80)
            lines.append("CHAIN OF THOUGHT REASONING")
            lines.append("-" * 80)
            if cot_reasoning.get("reasoning"):
                lines.append("Reasoning:")
                lines.append(cot_reasoning["reasoning"])
                lines.append("")
            if cot_reasoning.get("stuttering_events"):
                lines.append("Stuttering Events:")
                lines.append(cot_reasoning["stuttering_events"])
                lines.append("")
            if cot_reasoning.get("analysis"):
                lines.append("Analysis:")
                lines.append(cot_reasoning["analysis"])
                lines.append("")
            if cot_reasoning.get("whisper_output"):
                lines.append("Initial Whisper Transcription:")
                lines.append(cot_reasoning["whisper_output"])
                lines.append("")

        # Annotated text (if available)
        annotated_text = None
        if result.metadata and "annotated_text" in result.metadata:
            annotated_text = result.metadata["annotated_text"]

        if annotated_text:
            lines.append("-" * 80)
            lines.append("ANNOTATED TEXT (Ground Truth with annotations)")
            lines.append("-" * 80)
            lines.append(annotated_text)
            lines.append("")

        lines.append("-" * 80)
        lines.append("UNANNOTATED TEXT (Ground Truth - used for evaluation)")
        lines.append("-" * 80)
        lines.append(result.ground_truth_transcript)
        lines.append("")
        lines.append("-" * 80)
        lines.append("PREDICTED TRANSCRIPT")
        lines.append("-" * 80)
        lines.append(result.predicted_transcript)
        lines.append("")

        # Character-by-character aligned comparison with diff
        lines.append("-" * 80)
        lines.append("CHARACTER-BY-CHARACTER COMPARISON (Aligned with Diff)")
        lines.append("-" * 80)
        lines.append("")
        aligned_gt, aligned_pred, diff_markers = self._create_aligned_comparison(
            result.ground_truth_transcript, result.predicted_transcript
        )

        # Display ground truth on top, predicted below
        lines.append("GROUND TRUTH (spaced):")
        lines.append(aligned_gt)
        lines.append("")
        lines.append("PREDICTED (spaced):")
        lines.append(aligned_pred)
        lines.append("")
        lines.append("DIFF MARKERS:")
        lines.append("  '=' = match, '-' = deleted, '+' = inserted, '^' = modified")
        lines.append(diff_markers)
        lines.append("")

        # Add initial Whisper transcription if available
        if cot_reasoning and cot_reasoning.get("whisper_output"):
            lines.append("-" * 80)
            lines.append("INITIAL WHISPER TRANSCRIPTION")
            lines.append("-" * 80)
            lines.append(cot_reasoning["whisper_output"])
            lines.append("")

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)

    def _add_spaces_between_chars(self, text: str) -> str:
        """
        Add spaces between each character for easier reading (especially Chinese)

        Args:
            text: Input text

        Returns:
            Text with spaces between characters
        """
        return " ".join(text)

    def _create_aligned_comparison(self, ground_truth: str, predicted: str) -> Tuple[str, str, str]:
        """
        Create character-by-character aligned comparison with diff markers

        Args:
            ground_truth: Ground truth transcript
            predicted: Predicted transcript

        Returns:
            Tuple of (aligned_gt, aligned_pred, diff_markers)
        """
        # Parse the diff to create aligned sequences
        gt_chars = list(ground_truth)
        pred_chars = list(predicted)

        # Use SequenceMatcher for better alignment
        matcher = difflib.SequenceMatcher(None, gt_chars, pred_chars)

        aligned_gt_list = []
        aligned_pred_list = []
        diff_markers_list = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                # Matching characters
                for k in range(i1, i2):
                    aligned_gt_list.append(gt_chars[k])
                    aligned_pred_list.append(pred_chars[j1 + (k - i1)])
                    diff_markers_list.append("=")
            elif tag == "delete":
                # Characters deleted in predicted
                for k in range(i1, i2):
                    aligned_gt_list.append(gt_chars[k])
                    aligned_pred_list.append(" ")
                    diff_markers_list.append("-")
            elif tag == "insert":
                # Characters inserted in predicted
                for k in range(j1, j2):
                    aligned_gt_list.append(" ")
                    aligned_pred_list.append(pred_chars[k])
                    diff_markers_list.append("+")
            elif tag == "replace":
                # Characters replaced
                max_len = max(i2 - i1, j2 - j1)
                for k in range(max_len):
                    if k < (i2 - i1):
                        aligned_gt_list.append(gt_chars[i1 + k])
                    else:
                        aligned_gt_list.append(" ")

                    if k < (j2 - j1):
                        aligned_pred_list.append(pred_chars[j1 + k])
                    else:
                        aligned_pred_list.append(" ")

                    diff_markers_list.append("^")

        # Add spaces between characters
        aligned_gt = " ".join(aligned_gt_list)
        aligned_pred = " ".join(aligned_pred_list)
        diff_markers = " ".join(diff_markers_list)

        return aligned_gt, aligned_pred, diff_markers

    def _extract_chain_of_thought(self, metadata: Dict) -> Dict:
        """
        Extract Chain of Thought reasoning from metadata if available

        Args:
            metadata: Result metadata dictionary

        Returns:
            Dictionary with chain of thought fields, or empty dict if not available
        """
        if not metadata:
            return {}

        cot_info = {}

        # Check for dspy output object (from staccato model)
        if "output" in metadata:
            output = metadata["output"]
            # Try to extract attributes from the output object
            if hasattr(output, "stuttering_events"):
                cot_info["stuttering_events"] = str(output.stuttering_events)
            if hasattr(output, "analysis"):
                cot_info["analysis"] = str(output.analysis)
            if hasattr(output, "reasoning"):
                cot_info["reasoning"] = str(output.reasoning)

        # Check for whisper output (initial transcription)
        if "whisper_output" in metadata:
            cot_info["whisper_output"] = str(metadata["whisper_output"])

        return cot_info
