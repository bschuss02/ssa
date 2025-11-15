import concurrent
import time
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
from datasets import Dataset

from experiments.config.evaluation_config import EvaluationConfig
from experiments.datasets.dataset_registry import dataset_registry
from experiments.inference_models.asr_model_base import (
    ASRModelBase,
    TranscriptionInput,
    TranscriptionOutput,
)
from experiments.inference_models.model_registry import model_registry
from experiments.utils.calculate_metrics import calculate_metrics
from experiments.utils.configure_logging import logger
from experiments.utils.evaluation_result import EvaluationResult
from experiments.utils.progress_manager import ProgressManager

AUDIO_FILE_COLUMN = "audio_path"
GROUND_TRUTH_TRANSCRIPT_COLUMN = "transcript"


class Evaluator:
    evaluation_results: List[EvaluationResult]
    active_model_name: str
    active_dataset_name: str

    def __init__(self, cfg: EvaluationConfig):
        self.cfg = cfg
        self.evaluation_results = []
        self.active_model_name = None
        self.active_dataset_name = None

    def evaluate(self):
        """Entrypoint for the evaluation process"""
        with ProgressManager() as progress:
            progress.start_model_processing(len(self.cfg.models))
            for model_name in self.cfg.models:
                self.active_model_name = model_name
                model = self._load_model(model_name)
                self._evaluate_model(model, progress)
                progress.advance_model()

        # Save and analyze results
        logger.info("Evaluation complete. Starting analysis...")
        self._analyze_results(self.evaluation_results)

    def _evaluate_model(self, model: ASRModelBase, progress: ProgressManager):
        """Iterate over all datasets and evaluate the model on each"""
        for dataset_name in self.cfg.datasets:
            progress.start_dataset_processing(model.model_name, len(self.cfg.datasets))
            self.active_dataset_name = dataset_name
            dataset = self._load_dataset(dataset_name)
            self._evaluate_dataset(model, dataset, progress)
            progress.advance_dataset()
            progress.finish_dataset_processing()

    def _evaluate_dataset(
        self,
        model: ASRModelBase,
        dataset: Dataset,
        progress: ProgressManager,
    ):
        """Evaluate a single dataset. Called within _evaluate_model"""
        progress.start_sample_processing(self.active_dataset_name, len(dataset))

        for batch in dataset.iter(batch_size=self.cfg.batch_size):
            try:
                results = self._evaluate_batch(model, batch)
                self.evaluation_results.extend(results)
                progress.advance_sample_by(len(batch))
            except Exception as e:
                logger.exception("Error evaluating batch", e)
                logger.error(f"Batch: {batch}")
                # Skip this batch and continue with the next one
                progress.advance_sample_by(len(batch))
                continue

        progress.finish_sample_processing()

    def _evaluate_batch(self, model: ASRModelBase, batch: Dict) -> List[EvaluationResult]:
        start_time = time.time()
        ground_truth_transcriptions = batch[GROUND_TRUTH_TRANSCRIPT_COLUMN]

        # Convert batch dict (column->list) to list of row dicts
        batch_size = len(ground_truth_transcriptions)
        batch_rows = [{key: batch[key][i] for key in batch.keys()} for i in range(batch_size)]

        # Prepare transcription inputs based on model type
        if model.audio_array_or_path == "audio_array":
            audio_arrays, sampling_rates = self._load_audio_files(batch[AUDIO_FILE_COLUMN])
            transcription_inputs = [
                TranscriptionInput(
                    audio_array=audio_array, sample_rate=sampling_rate, metadata=dataset_row
                )
                for audio_array, sampling_rate, dataset_row in zip(
                    audio_arrays, sampling_rates, batch_rows
                )
            ]
        elif model.audio_array_or_path == "audio_path":
            transcription_inputs = [
                TranscriptionInput(
                    audio_path=Path(dataset_row[AUDIO_FILE_COLUMN]), metadata=dataset_row
                )
                for dataset_row in batch_rows
            ]
        else:
            raise ValueError(f"Unknown audio_array_or_path: {model.audio_array_or_path}")

        # Run transcription
        transcription_outputs: List[TranscriptionOutput] = model.transcribe(transcription_inputs)
        inference_time = time.time() - start_time
        for input, output in zip(transcription_inputs, transcription_outputs):
            output.metadata = {
                **input.metadata,
                **output.metadata,
                "inference_time": inference_time,
            }

        predicted_transcriptions = [output.transcription for output in transcription_outputs]

        metrics_batch = calculate_metrics(
            predicted_transcriptions,
            ground_truth_transcriptions,
            self.cfg.remove_punctuation,
            self.cfg.make_lowercase,
        )

        evaluation_results = []
        for ground_truth_transcription, transcription_output, metrics in zip(
            ground_truth_transcriptions,
            transcription_outputs,
            metrics_batch,
        ):
            evaluation_result = EvaluationResult(
                model_name=model.model_name,
                dataset_name=self.active_dataset_name,
                ground_truth_transcript=ground_truth_transcription,
                predicted_transcript=transcription_output.transcription,
                metrics=metrics,
                metadata=transcription_output.metadata,
                inference_time=inference_time,
            )
            evaluation_results.append(evaluation_result)

        logger.info(f"Metrics batch: {metrics_batch}")

        return evaluation_results

    def _load_audio_files(self, audio_paths: List[str]) -> Tuple[List[np.ndarray], List[int]]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.cfg.max_workers) as executor:
            futures = [executor.submit(librosa.load, path) for path in audio_paths]
            results = [future.result() for future in futures]
            audio_arrays = [result[0] for result in results]
            sampling_rates = [result[1] for result in results]
        return audio_arrays, sampling_rates

    def _load_model(self, model_name: str) -> ASRModelBase:
        model_class = model_registry[model_name]
        model = model_class(model_name, self.cfg)
        model.load_model()
        return model

    def _load_dataset(self, dataset_name: str) -> Dataset:
        dataset_class = dataset_registry[dataset_name]
        dataset = dataset_class(self.cfg, dataset_name)
        return dataset.load_dataset()

    def _analyze_results(self, evaluation_results: List[EvaluationResult]):
        """Analyze evaluation results"""
        from experiments.utils.results_analyzer import ResultsAnalyzer

        # Create configuration dictionary for metadata
        config = {
            "models": self.cfg.models,
            "datasets": self.cfg.datasets,
            "max_samples_per_dataset": self.cfg.max_samples_per_dataset,
            "batch_size": self.cfg.batch_size,
            "output_dir": str(self.cfg.output_dir),
            "results_dir": str(self.cfg.results_dir),
        }

        # Use the ResultsAnalyzer to handle all analysis and visualization
        analyzer = ResultsAnalyzer(self.cfg.results_dir)
        return analyzer.analyze_and_visualize(evaluation_results, config)
