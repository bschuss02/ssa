import hydra
from omegaconf import DictConfig

from experiments.config.evaluation_config import EvaluationConfig
from experiments.evaluation.evaluator import Evaluator
from experiments.utils.configure_logging import configure_logging, logger


@hydra.main(config_path="config", config_name="config.yaml", version_base=None)
def run_experiment(cfg: DictConfig):
    configure_logging()
    evaluation_cfg = EvaluationConfig(**cfg)
    logger.info("Starting experiment")
    evaluator = Evaluator(evaluation_cfg)
    evaluator.evaluate()


if __name__ == "__main__":
    run_experiment()
