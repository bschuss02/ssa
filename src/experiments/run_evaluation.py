#!/usr/bin/env python3
"""
Comprehensive ASR Evaluation Script using Hydra

This script evaluates the Phi-4 multimodal ASR model on the FluencyBank dataset
and provides detailed analysis including WER, CER, and other ASR quality metrics.
"""

import os
import sys
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

import hydra
from omegaconf import DictConfig, OmegaConf
import polars as pl
import numpy as np
from tqdm import tqdm

# Add the current directory to sys.path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import our custom modules
from utils.datasets.fluencybank_dataset import create_fluencybank_dataset
from utils.models.phi_4_multimodal_instruct import Phi4MultimodalASRModel

# Import evaluation libraries
try:
    import jiwer

    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False
    logging.warning("jiwer not available - install with: uv add jiwer")

# Note: We use polars instead of pandas for better performance and memory efficiency
# polars is already included in the project dependencies

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class EvaluationSample:
    """Container for a single evaluation sample with all metrics."""

    clip_id: str
    reference_text: str
    hypothesis_text: str
    audio_path: str
    inference_time: float
    wer: Optional[float] = None
    cer: Optional[float] = None
    mer: Optional[float] = None
    wil: Optional[float] = None
    wip: Optional[float] = None
    # Additional metadata
    duration: Optional[float] = None
    speaker_id: Optional[str] = None


@dataclass
class EvaluationResults:
    """Container for overall evaluation results."""

    # Overall metrics
    mean_wer: float
    mean_cer: float
    mean_mer: float
    mean_wil: float
    mean_wip: float

    # Statistics
    total_samples: int
    total_duration: float
    total_inference_time: float
    samples_per_second: float
    real_time_factor: float

    # Per-sample results
    samples: List[EvaluationSample]

    # Error analysis
    substitutions_count: int
    deletions_count: int
    insertions_count: int
    hits_count: int


class ASRMetricsCalculator:
    """Calculate comprehensive ASR evaluation metrics."""

    def __init__(self):
        if not JIWER_AVAILABLE:
            raise ImportError("jiwer package required for ASR metrics. Install with: uv add jiwer")

    def normalize_text(self, text: str) -> str:
        """
        Normalize text for ASR evaluation by:
        - Converting to lowercase
        - Removing punctuation
        - Normalizing whitespace
        - Handling common ASR variations
        """
        import re
        import string

        # Convert to lowercase
        text = text.lower()

        # Handle common ASR number/word variations
        text = re.sub(r"\bfour\b", "4", text)
        text = re.sub(r"\btwo\b", "2", text)
        text = re.sub(r"\bone\b", "1", text)
        text = re.sub(r"\bthree\b", "3", text)

        # Handle common contractions and variations
        text = re.sub(r"\bokay\b", "ok", text)
        text = re.sub(r"\byeah\b", "yes", text)
        text = re.sub(r"\byep\b", "yes", text)
        text = re.sub(r"\buh\b", "", text)  # Remove filler words
        text = re.sub(r"\bum\b", "", text)  # Remove filler words

        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        return text

    def calculate_metrics(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """
        Calculate comprehensive ASR metrics for a single sample.

        Args:
            reference: Ground truth text
            hypothesis: Predicted text

        Returns:
            Dictionary with metric names and values
        """
        try:
            # Normalize both texts for fair comparison
            norm_reference = self.normalize_text(reference)
            norm_hypothesis = self.normalize_text(hypothesis)

            # Log the normalized versions for debugging
            logger.debug(f"Original ref: '{reference}'")
            logger.debug(f"Normalized ref: '{norm_reference}'")
            logger.debug(f"Original hyp: '{hypothesis}'")
            logger.debug(f"Normalized hyp: '{norm_hypothesis}'")

            # Calculate word-level metrics on normalized text using process_words
            word_output = jiwer.process_words(norm_reference, norm_hypothesis)

            # Calculate character-level metrics on normalized text
            cer = jiwer.cer(norm_reference, norm_hypothesis)

            return {
                "wer": word_output.wer,
                "mer": word_output.mer,
                "wil": word_output.wil,
                "wip": word_output.wip,
                "cer": cer,
                "substitutions": word_output.substitutions,
                "deletions": word_output.deletions,
                "insertions": word_output.insertions,
                "hits": word_output.hits,
            }

        except Exception as e:
            logger.warning(f"Error calculating metrics for ref='{reference}', hyp='{hypothesis}': {e}")
            return {
                "wer": 1.0,
                "mer": 1.0,
                "wil": 1.0,
                "wip": 0.0,
                "cer": 1.0,
                "substitutions": 0,
                "deletions": 0,
                "insertions": 0,
                "hits": 0,
            }


class Phi4ASREvaluator:
    """Main evaluator class for Phi-4 multimodal ASR model."""

    def __init__(self, config: DictConfig):
        self.config = config
        self.model = None
        self.metrics_calculator = ASRMetricsCalculator()
        self.results: List[EvaluationSample] = []

    def setup_model(self) -> None:
        """Initialize and load the Phi-4 multimodal ASR model."""
        logger.info("Initializing Phi-4 multimodal ASR model...")

        self.model = Phi4MultimodalASRModel(
            model_cache_dir=self.config.model.cache_dir,
            force_cpu=self.config.model.force_cpu,
            device=self.config.model.device,
        )

        logger.info(f"Loading model: {self.config.model.name}")
        self.model.load_model(model_name=self.config.model.name, use_4bit=self.config.model.use_4bit)

        logger.info("Model loaded successfully")

    def load_dataset(self):
        """Load the FluencyBank dataset."""
        logger.info(f"Loading FluencyBank dataset from: {self.config.dataset.parquet_path}")

        dataset = create_fluencybank_dataset(
            parquet_path=self.config.dataset.parquet_path,
            audio_sample_rate=self.config.dataset.audio_sample_rate,
            max_audio_length=self.config.dataset.max_audio_length,
            text_column=self.config.dataset.text_column,
            include_timing=self.config.dataset.include_timing,
            include_speaker_info=self.config.dataset.include_speaker_info,
        )

        # Limit dataset size if specified
        if self.config.evaluation.max_samples > 0:
            dataset = dataset.select(range(min(self.config.evaluation.max_samples, len(dataset))))
            logger.info(f"Limited dataset to {len(dataset)} samples")

        return dataset

    def create_transcription_messages(self) -> List[Dict[str, str]]:
        """Create the messages for transcription."""
        return [
            {"role": "system", "content": self.config.model.system_message},
            {"role": "user", "content": self.config.model.user_prompt},
        ]

    def evaluate_sample(self, sample: Dict[str, Any]) -> EvaluationSample:
        """Evaluate a single sample from the dataset."""
        clip_id = sample["clip_id"]
        reference_text = sample["text"]
        audio_data = sample["audio"]

        # Save audio to temporary file for inference
        import tempfile
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            sf.write(temp_audio.name, audio_data["array"], audio_data["sampling_rate"])
            temp_audio_path = temp_audio.name

        try:
            # Prepare messages for transcription
            messages = self.create_transcription_messages()

            # Run inference with timing
            start_time = time.time()
            hypothesis_text = self.model.transcribe(
                audio_path=temp_audio_path,
                messages=messages,
                max_new_tokens=self.config.model.max_new_tokens,
                temperature=self.config.model.temperature,
                do_sample=self.config.model.do_sample,
            )
            inference_time = time.time() - start_time

            # Calculate ASR metrics
            metrics = self.metrics_calculator.calculate_metrics(reference_text, hypothesis_text)

            # Create evaluation sample
            eval_sample = EvaluationSample(
                clip_id=clip_id,
                reference_text=reference_text,
                hypothesis_text=hypothesis_text,
                audio_path=temp_audio_path,
                inference_time=inference_time,
                wer=metrics["wer"],
                cer=metrics["cer"],
                mer=metrics["mer"],
                wil=metrics["wil"],
                wip=metrics["wip"],
                duration=sample.get("duration"),
                speaker_id=sample.get("speaker_id"),
            )

            return eval_sample

        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_audio_path)
            except:
                pass

    def run_evaluation(self) -> EvaluationResults:
        """Run the complete evaluation pipeline."""
        logger.info("Starting evaluation pipeline...")

        # Setup model and dataset
        self.setup_model()
        dataset = self.load_dataset()

        logger.info(f"Evaluating {len(dataset)} samples...")

        # Evaluate each sample
        total_substitutions = 0
        total_deletions = 0
        total_insertions = 0
        total_hits = 0
        total_duration = 0.0
        total_inference_time = 0.0

        for i, sample in enumerate(tqdm(dataset, desc="Evaluating samples")):
            try:
                eval_sample = self.evaluate_sample(sample)
                self.results.append(eval_sample)

                # Accumulate statistics
                total_inference_time += eval_sample.inference_time
                if eval_sample.duration:
                    total_duration += eval_sample.duration

                # Get detailed metrics for error analysis
                metrics = self.metrics_calculator.calculate_metrics(
                    eval_sample.reference_text, eval_sample.hypothesis_text
                )
                total_substitutions += metrics["substitutions"]
                total_deletions += metrics["deletions"]
                total_insertions += metrics["insertions"]
                total_hits += metrics["hits"]

                # Log progress every N samples
                if (i + 1) % self.config.evaluation.log_interval == 0:
                    current_wer = np.mean([s.wer for s in self.results if s.wer is not None])
                    logger.info(f"Processed {i+1}/{len(dataset)} samples. Current WER: {current_wer:.3f}")

            except Exception as e:
                logger.error(f"Error evaluating sample {i}: {e}")
                continue

        # Calculate overall metrics
        valid_samples = [s for s in self.results if s.wer is not None]

        if not valid_samples:
            raise RuntimeError("No valid samples evaluated")

        mean_wer = np.mean([s.wer for s in valid_samples])
        mean_cer = np.mean([s.cer for s in valid_samples])
        mean_mer = np.mean([s.mer for s in valid_samples])
        mean_wil = np.mean([s.wil for s in valid_samples])
        mean_wip = np.mean([s.wip for s in valid_samples])

        # Calculate performance metrics
        samples_per_second = len(valid_samples) / total_inference_time if total_inference_time > 0 else 0
        real_time_factor = total_inference_time / total_duration if total_duration > 0 else 0

        results = EvaluationResults(
            mean_wer=mean_wer,
            mean_cer=mean_cer,
            mean_mer=mean_mer,
            mean_wil=mean_wil,
            mean_wip=mean_wip,
            total_samples=len(valid_samples),
            total_duration=total_duration,
            total_inference_time=total_inference_time,
            samples_per_second=samples_per_second,
            real_time_factor=real_time_factor,
            samples=valid_samples,
            substitutions_count=total_substitutions,
            deletions_count=total_deletions,
            insertions_count=total_insertions,
            hits_count=total_hits,
        )

        return results

    def save_results(self, results: EvaluationResults) -> None:
        """Save evaluation results to files."""
        output_dir = Path(self.config.output.dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed results as JSON
        results_dict = asdict(results)
        results_path = output_dir / "evaluation_results.json"
        with open(results_path, "w") as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved detailed results to: {results_path}")

        # Save sample-by-sample results as CSV
        samples_data = []
        for sample in results.samples:
            samples_data.append(asdict(sample))

        df = pl.DataFrame(samples_data)
        csv_path = output_dir / "evaluation_samples.csv"
        df.write_csv(csv_path)
        logger.info(f"Saved sample results to: {csv_path}")

        # Save summary report
        self.generate_summary_report(results, output_dir / "evaluation_summary.txt")

        # Save transcription examples
        self.save_transcription_examples(results, output_dir / "transcription_examples.txt")

    def generate_summary_report(self, results: EvaluationResults, output_path: Path) -> None:
        """Generate a human-readable summary report."""
        with open(output_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("ASR EVALUATION SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write("CONFIGURATION:\n")
            f.write(f"Model: {self.config.model.name}\n")
            f.write(f"Dataset: {self.config.dataset.parquet_path}\n")
            f.write(f"Text Column: {self.config.dataset.text_column}\n")
            f.write(f"Max Samples: {self.config.evaluation.max_samples}\n\n")

            f.write("OVERALL METRICS:\n")
            f.write(f"Word Error Rate (WER):     {results.mean_wer:.3f} ({results.mean_wer*100:.1f}%)\n")
            f.write(f"Character Error Rate (CER): {results.mean_cer:.3f} ({results.mean_cer*100:.1f}%)\n")
            f.write(f"Match Error Rate (MER):    {results.mean_mer:.3f} ({results.mean_mer*100:.1f}%)\n")
            f.write(f"Word Info Lost (WIL):      {results.mean_wil:.3f} ({results.mean_wil*100:.1f}%)\n")
            f.write(f"Word Info Preserved (WIP): {results.mean_wip:.3f} ({results.mean_wip*100:.1f}%)\n\n")

            f.write("DATASET STATISTICS:\n")
            f.write(f"Total Samples:     {results.total_samples}\n")
            f.write(
                f"Total Duration:    {results.total_duration:.2f} seconds ({results.total_duration/60:.1f} minutes)\n"
            )
            f.write(f"Avg Sample Length: {results.total_duration/results.total_samples:.2f} seconds\n\n")

            f.write("PERFORMANCE STATISTICS:\n")
            f.write(
                f"Total Inference Time: {results.total_inference_time:.2f} seconds ({results.total_inference_time/60:.1f} minutes)\n"
            )
            f.write(f"Samples per Second:   {results.samples_per_second:.2f}\n")
            f.write(f"Real-time Factor:     {results.real_time_factor:.2f}x\n\n")

            f.write("ERROR ANALYSIS:\n")
            total_errors = results.substitutions_count + results.deletions_count + results.insertions_count
            total_words = total_errors + results.hits_count
            f.write(f"Total Words:    {total_words}\n")

            if total_words > 0:
                f.write(f"Correct (Hits): {results.hits_count} ({results.hits_count/total_words*100:.1f}%)\n")
                f.write(
                    f"Substitutions:  {results.substitutions_count} ({results.substitutions_count/total_words*100:.1f}%)\n"
                )
                f.write(
                    f"Deletions:      {results.deletions_count} ({results.deletions_count/total_words*100:.1f}%)\n"
                )
                f.write(
                    f"Insertions:     {results.insertions_count} ({results.insertions_count/total_words*100:.1f}%)\n\n"
                )
            else:
                f.write("No word-level statistics available (no valid metrics calculated)\n\n")

            # WER distribution
            wer_values = [s.wer for s in results.samples if s.wer is not None]
            f.write("WER DISTRIBUTION:\n")
            f.write(f"Min WER:    {min(wer_values):.3f}\n")
            f.write(f"Max WER:    {max(wer_values):.3f}\n")
            f.write(f"Median WER: {np.median(wer_values):.3f}\n")
            f.write(f"Std Dev:    {np.std(wer_values):.3f}\n")

        logger.info(f"Saved summary report to: {output_path}")

    def save_transcription_examples(self, results: EvaluationResults, output_path: Path) -> None:
        """Save example transcriptions for qualitative analysis."""
        # Sort samples by WER for analysis
        sorted_samples = sorted(results.samples, key=lambda x: x.wer if x.wer else 1.0)

        with open(output_path, "w") as f:
            f.write("TRANSCRIPTION EXAMPLES\n")
            f.write("=" * 80 + "\n\n")

            # Best examples (lowest WER)
            f.write("BEST TRANSCRIPTIONS (Lowest WER):\n")
            f.write("-" * 50 + "\n")
            for i, sample in enumerate(sorted_samples[:10]):
                f.write(f"\nExample {i+1} (WER: {sample.wer:.3f}):\n")
                f.write(f"Clip ID: {sample.clip_id}\n")
                f.write(f"Reference: {sample.reference_text}\n")
                f.write(f"Hypothesis: {sample.hypothesis_text}\n")
                if sample.speaker_id:
                    f.write(f"Speaker: {sample.speaker_id}\n")

            # Worst examples (highest WER)
            f.write(f"\n\nWORST TRANSCRIPTIONS (Highest WER):\n")
            f.write("-" * 50 + "\n")
            for i, sample in enumerate(sorted_samples[-10:]):
                f.write(f"\nExample {i+1} (WER: {sample.wer:.3f}):\n")
                f.write(f"Clip ID: {sample.clip_id}\n")
                f.write(f"Reference: {sample.reference_text}\n")
                f.write(f"Hypothesis: {sample.hypothesis_text}\n")
                if sample.speaker_id:
                    f.write(f"Speaker: {sample.speaker_id}\n")

        logger.info(f"Saved transcription examples to: {output_path}")


@hydra.main(version_base=None, config_path="config", config_name="phi_4_multimodal_eval")
def main(config: DictConfig) -> None:
    """Main evaluation function."""
    logger.info("Starting Phi-4 Multimodal ASR Evaluation")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")

    # Validate configuration
    if not JIWER_AVAILABLE:
        logger.error("jiwer package is required for evaluation. Install with: uv add jiwer")
        return

    # Create evaluator and run evaluation
    evaluator = Phi4ASREvaluator(config)

    try:
        results = evaluator.run_evaluation()
        evaluator.save_results(results)

        # Print summary to console
        logger.info("=" * 60)
        logger.info("EVALUATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Overall WER: {results.mean_wer:.3f} ({results.mean_wer*100:.1f}%)")
        logger.info(f"Overall CER: {results.mean_cer:.3f} ({results.mean_cer*100:.1f}%)")
        logger.info(f"Samples evaluated: {results.total_samples}")
        logger.info(f"Real-time factor: {results.real_time_factor:.2f}x")
        logger.info(f"Results saved to: {config.output.dir}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
