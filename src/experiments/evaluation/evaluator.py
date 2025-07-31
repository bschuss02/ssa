from logging import getLogger
from typing import List
from zipfile import Path

from experiments.config.EvaluationConfig import EvaluationConfig
from experiments.utils.evaluation_result_row import EvaluationResultRow
from experiments.utils.progress_manager import ProgressManager


class Evaluator:
    evaluation_results: List[EvaluationResultRow]

    def __init__(self, cfg: EvaluationConfig):
        self.cfg = cfg
        self._log = getLogger(__name__)
        self.evaluation_results = []

    def evaluate(self):
        """Entrypoint for the evaluation process"""
        with ProgressManager() as progress:
            for model_name, model_path in self.cfg.models.items():
                self._evaluate_model(model_name, model_path, progress)
                progress.advance_model()

    def _evaluate_model(self, progress: ProgressManager):
        """Iterate over all datasets and evaluate the model on each"""
        for dataset_name, dataset_path in self.cfg.datasets.items():
            progress.start_dataset_processing(model_name, len(self.cfg.datasets))
            self._evaluate_dataset(dataset_name, dataset_path, progress)
            progress.advance_dataset()

    def _evaluate_dataset(self, progress: ProgressManager):
        """Evaluate a single dataset. Called within _evaluate_model"""

    def _load_dataset(self, dataset_name: str, dataset_path: Path):
        pass

    def _analyze_results(self, evaluation_results: List[EvaluationResultRow]):
        pass
