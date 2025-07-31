from logging import getLogger
from typing import List

from experiments.config.EvaluationConfig import EvaluationConfig
from experiments.utils.evaluation_result_row import EvaluationResultRow


class Evaluator:
    evaluation_results: List[EvaluationResultRow]

    def __init__(self, cfg: EvaluationConfig):
        self.cfg = cfg
        self._log = getLogger(__name__)
        self.evaluation_results = []

    def evaluate(self):
        """Entrypoint for the evaluation process"""
        for model_name, model_path in self.cfg.models.items():
            self._evaluate_model(model_name, model_path)

    def _evaluate_model(self):
        """Iterate over all datasets and evaluate the model on each"""

    def _evaluate_dataset(self):
        """Evaluate a single dataset. Called within _evaluate_model"""

    def _load_dataset(self):
        pass

    def _analyze_results(self):
        pass
