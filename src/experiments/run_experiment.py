import hydra
from omegaconf import DictConfig

from experiments.config.evaluation_config import EvaluationConfig
from experiments.evaluation.evaluator import Evaluator
from experiments.utils.configure_logging import configure_logging


@hydra.main(config_path="config", config_name="config.yaml", version_base=None)
def run_experiment(cfg: DictConfig):
    evaluation_cfg = EvaluationConfig(**cfg)
    configure_logging(evaluation_cfg)
    evaluator = Evaluator(evaluation_cfg)
    evaluator.evaluate()


if __name__ == "__main__":
    run_experiment()
