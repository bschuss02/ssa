from logging import getLogger
from pathlib import Path
from typing import List

from datasets import Dataset

from experiments.config.EvaluationConfig import EvaluationConfig
from experiments.datasets.dataset_registry import dataset_registry
from experiments.inference_models.asr_model_base import ASRModelBase
from experiments.inference_models.model_registry import model_registry
from experiments.utils.evaluation_result import EvaluationResult
from experiments.utils.progress_manager import ProgressManager


class Evaluator:
    evaluation_results: List[EvaluationResult]

    def __init__(self, cfg: EvaluationConfig):
        self.cfg = cfg
        self._log = getLogger(__name__)
        self.evaluation_results = []

    def evaluate(self):
        """Entrypoint for the evaluation process"""
        with ProgressManager() as progress:
            progress.start_model_processing(len(self.cfg.models))
            for model_name, model_path in self.cfg.models.items():
                model = self.load_model(model_name, model_path)
                self._evaluate_model(model, progress)
                progress.advance_model()

    def _evaluate_model(self, model: ASRModelBase, progress: ProgressManager):
        """Iterate over all datasets and evaluate the model on each"""
        for dataset_name, dataset_path in self.cfg.datasets.items():
            progress.start_dataset_processing(model.model_name, len(self.cfg.datasets))
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
