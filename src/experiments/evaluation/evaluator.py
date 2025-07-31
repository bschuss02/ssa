from logging import getLogger

from experiments.config.EvaluationConfig import EvaluationConfig
from experiments.utils.configure_logging import configure_logging

_log = getLogger(__name__)


class Evaluator:
    def __init__(self, cfg: EvaluationConfig):
        self.cfg = cfg
        configure_logging(self.cfg)

    def evaluate(self):
        pass

    def _evaluate_model(self):
        """Iterate over all datasets and evaluate the model on each"""

    def _evaluate_dataset(self):
        pass

    def _load_dataset(self):
        pass
