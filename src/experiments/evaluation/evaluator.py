import concurrent
import time
from logging import getLogger
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
from datasets import Dataset

from experiments.config.evaluation_config import EvaluationConfig
from experiments.datasets.dataset_registry import dataset_registry
from experiments.inference_models.asr_model_base import ASRModelBase
from experiments.inference_models.model_registry import model_registry
from experiments.utils.asr_cache import ASRCache
from experiments.utils.calculate_metrics import calculate_metrics
from experiments.utils.evaluation_result import EvaluationResult
from experiments.utils.progress_manager import ProgressManager


class Evaluator:
    evaluation_results: List[EvaluationResult]
    active_model_name: str
    active_dataset_name: str

    def __init__(self, cfg: EvaluationConfig):
        self.cfg = cfg
        self._log = getLogger(__name__)
        self.evaluation_results = []
        self.active_model_name = None
        self.active_dataset_name = None

        # Initialize ASR cache if enabled
        if self.cfg.use_asr_cache:
            self.asr_cache = ASRCache(self.cfg.asr_cache_dir)
            self._log.info(f"Initialized ASR cache at {self.cfg.asr_cache_dir}")
        else:
            self.asr_cache = None
            self._log.info("ASR cache disabled")

    def evaluate(self):
        """Entrypoint for the evaluation process"""
        with ProgressManager() as progress:
            progress.start_model_processing(len(self.cfg.models))
            for model_name, model_path in self.cfg.models.items():
                self.active_model_name = model_name
                model = self.load_model(model_name, model_path)
                self._evaluate_model(model, progress)
                progress.advance_model()

        # Log cache statistics if cache is enabled
        if self.asr_cache is not None:
            cache_stats = self.asr_cache.get_stats()
            self._log.info(f"ASR Cache Statistics: {cache_stats}")
            self.asr_cache.close()

        # Analyze and visualize results
        self._log.info("Evaluation complete. Starting analysis and visualization...")
        self._analyze_results(self.evaluation_results)

    def _evaluate_model(self, model: ASRModelBase, progress: ProgressManager):
        """Iterate over all datasets and evaluate the model on each"""
        for dataset_name, dataset_path in self.cfg.datasets.items():
            progress.start_dataset_processing(model.model_name, len(self.cfg.datasets))
            self.active_dataset_name = dataset_name
            dataset = self._load_dataset(dataset_name, dataset_path)
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
                progress.advance_sample_by(len(batch["clip_audio_file"]))
            except Exception as e:
                self._log.error(f"Error evaluating batch: {e}")
                self._log.error(f"Batch: {batch}")
                raise e

        progress.finish_sample_processing()

    def _evaluate_batch(
        self, model: ASRModelBase, batch: Dict
    ) -> List[EvaluationResult]:
        start_time = time.time()
        audio_arrays, sampling_rates = self._load_audio_files(batch["clip_audio_file"])
        ground_truth_transcriptions = batch["unannotated_text"]

        # Check cache for existing transcriptions if cache is enabled
        cached_transcriptions = None
        if self.asr_cache is not None:
            cached_transcriptions = self.asr_cache.get(
                model.model_name, audio_arrays, sampling_rates
            )

        if cached_transcriptions is not None:
            self._log.info(f"Cache hit for {len(audio_arrays)} audio samples")
            predicted_transcriptions = cached_transcriptions
        else:
            self._log.info(
                f"Cache miss for {len(audio_arrays)} audio samples, running inference"
            )
            predicted_transcriptions = model.transcribe(audio_arrays, sampling_rates[0])
            # Cache the results for future use if cache is enabled
            if self.asr_cache is not None:
                self.asr_cache.set(
                    model.model_name,
                    audio_arrays,
                    sampling_rates[0],
                    predicted_transcriptions,
                )

        self._log.info(predicted_transcriptions)
        metrics_batch = calculate_metrics(
            predicted_transcriptions,
            ground_truth_transcriptions,
            self.cfg.remove_punctuation,
            self.cfg.make_lowercase,
        )
        inference_time = time.time() - start_time
        self._log.info(metrics_batch)
        evaluation_results = []
        for ground_truth_transcription, predicted_transcription, metrics in zip(
            ground_truth_transcriptions,
            predicted_transcriptions,
            metrics_batch,
        ):
            evaluation_result = EvaluationResult(
                model_name=model.model_name,
                dataset_name=self.active_dataset_name,
                ground_truth_transcript=ground_truth_transcription,
                predicted_transcript=predicted_transcription,
                metrics=metrics,
                inference_time=inference_time,
            )
            evaluation_results.append(evaluation_result)

        return evaluation_results

    def _load_audio_files(
        self, audio_paths: List[str]
    ) -> Tuple[List[np.ndarray], List[int]]:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.cfg.max_workers
        ) as executor:
            futures = [executor.submit(librosa.load, path) for path in audio_paths]
            results = [future.result() for future in futures]
            audio_arrays = [result[0] for result in results]
            sampling_rates = [result[1] for result in results]
        return audio_arrays, sampling_rates

    def clear_cache(self):
        """Clear the ASR cache."""
        if self.asr_cache is not None:
            self.asr_cache.clear()
            self._log.info("ASR cache cleared")
        else:
            self._log.warning("ASR cache is not enabled")

    def get_cache_stats(self) -> Dict:
        """Get cache statistics.

        Returns:
            Dictionary containing cache statistics
        """
        if self.asr_cache is not None:
            return self.asr_cache.get_stats()
        else:
            return {"error": "ASR cache is not enabled"}

    def load_model(self, model_name: str, model_path: Path) -> ASRModelBase:
        model_class = model_registry[model_name]
        model = model_class(model_name, model_path)
        model.load_model()
        return model

    def _load_dataset(self, dataset_name: str, dataset_path: Path) -> Dataset:
        self._log.info(f"Loading dataset {dataset_name} from {dataset_path}")

        if dataset_name not in dataset_registry:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. Available datasets: {list(dataset_registry.keys())}"
            )

        dataset_class = dataset_registry[dataset_name]
        dataset = dataset_class(self.cfg, dataset_name, dataset_path)

        self._log.info(f"Starting loading for dataset {dataset_name}")
        dataset.load_dataset()

        return dataset._dataset

    def _analyze_results(self, evaluation_results: List[EvaluationResult]):
        """Analyze and visualize evaluation results"""
        from experiments.utils.results_analyzer import ResultsAnalyzer

        # Create configuration dictionary for metadata
        config = {
            "models": list(self.cfg.models.keys()),
            "datasets": list(self.cfg.datasets.keys()),
            "max_samples_per_dataset": self.cfg.max_samples_per_dataset,
            "batch_size": self.cfg.batch_size,
            "output_dir": str(self.cfg.output_dir),
            "results_dir": str(self.cfg.results_dir),
            "dataset_cache_dir": str(self.cfg.dataset_cache_dir),
        }

        # Use the ResultsAnalyzer to handle all analysis and visualization
        analyzer = ResultsAnalyzer(self.cfg.results_dir, self._log)
        return analyzer.analyze_and_visualize(evaluation_results, config)
