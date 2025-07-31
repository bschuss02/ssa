import time
from logging import getLogger
from pathlib import Path
from typing import Dict, List

import numpy as np
from datasets import Dataset

from experiments.config.EvaluationConfig import EvaluationConfig
from experiments.datasets.dataset_registry import dataset_registry
from experiments.inference_models.asr_model_base import ASRModelBase
from experiments.inference_models.model_registry import model_registry
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

    def evaluate(self):
        """Entrypoint for the evaluation process"""
        with ProgressManager() as progress:
            progress.start_model_processing(len(self.cfg.models))
            for model_name, model_path in self.cfg.models.items():
                self.active_model_name = model_name
                model = self.load_model(model_name, model_path)
                self._evaluate_model(model, progress)
                progress.advance_model()

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
            results = self._evaluate_batch(model, batch)
            self.evaluation_results.extend(results)
            progress.advance_sample_by(len(batch["audio"]))

        progress.finish_sample_processing()

    def _evaluate_batch(
        self, model: ASRModelBase, batch: Dict
    ) -> List[EvaluationResult]:
        start_time = time.time()
        audio_arrays = [np.array(audio_dict["array"]) for audio_dict in batch["audio"]]
        sampling_rate = batch["audio"][0]["sampling_rate"]
        ground_truth_transcriptions = batch["unannotated_text"]
        predicted_transcriptions = model.transcribe(audio_arrays, sampling_rate)
        self._log.info(predicted_transcriptions)
        metrics_batch = calculate_metrics(
            predicted_transcriptions, ground_truth_transcriptions
        )
        inference_time = time.time() - start_time
        self._log.info(metrics_batch)
        evaluation_results = []
        for ground_truth_transcriptions, predicted_transcriptions, metrics in zip(
            ground_truth_transcriptions,
            predicted_transcriptions,
            metrics_batch,
        ):
            evaluation_result = EvaluationResult(
                model_name=model.model_name,
                dataset_name=self.active_dataset_name,
                ground_truth_transcript=ground_truth_transcriptions,
                predicted_transcript=predicted_transcriptions,
                metrics=metrics,
                inference_time=inference_time,
            )
            evaluation_results.append(evaluation_result)

        return evaluation_results

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
        pass
