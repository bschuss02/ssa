from logging import getLogger
from typing import List
from zipfile import Path

from datasets import Dataset

from experiments.config.EvaluationConfig import EvaluationConfig
from experiments.inference_models.asr_model_base import ASRModelBase
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
            progress.end_model_processing()

    def _evaluate_model(self, model: ASRModelBase, progress: ProgressManager):
        """Iterate over all datasets and evaluate the model on each"""
        for dataset_name, dataset_path in self.cfg.datasets.items():
            progress.start_dataset_processing(model.model_name, len(self.cfg.datasets))
            dataset = self._load_dataset(dataset_name, dataset_path)
            self._evaluate_dataset(model, dataset, progress)
            progress.advance_dataset()
            progress.end_dataset_processing()

    def _evaluate_dataset(
        self,
        model: ASRModelBase,
        dataset: Dataset,
        progress: ProgressManager,
    ):
        """Evaluate a single dataset. Called within _evaluate_model"""

    def load_model(self, model_name: str, model_path: Path) -> ASRModelBase:
        pass

    def _load_dataset(self, dataset_name: str, dataset_path: Path):
        self._log.info(f"Loading dataset {dataset_name} from {dataset_path}")

    def _analyze_results(self, evaluation_results: List[EvaluationResult]):
        pass
