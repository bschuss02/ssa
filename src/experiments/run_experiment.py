import hydra
from omegaconf import DictConfig

from experiments.config.EvaluationConfig import EvaluationConfig
from experiments.evaluation.evaluator import Evaluator


@hydra.main(config_path="config", config_name="config.yaml", version_base=None)
def run_experiment(cfg: DictConfig):
    cfg = EvaluationConfig(**cfg)
    evaluator = Evaluator(cfg)
    evaluator.evaluate()


if __name__ == "__main__":
    run_experiment()
