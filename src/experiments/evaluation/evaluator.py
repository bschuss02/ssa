from experiments.config.EvaluationConfig import EvaluationConfig


class Evaluator:
    def __init__(self, cfg: EvaluationConfig):
        self.cfg = cfg

    def evaluate(self):
        pass

    def _evaluate_model(self):
        """Iterate over all datasets and evaluate the model on each"""

    def _evaluate_dataset(self):
        pass

    def _load_dataset(self):
        pass
