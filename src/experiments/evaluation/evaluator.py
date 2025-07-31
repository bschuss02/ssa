from logging import getLogger

from experiments.config.EvaluationConfig import EvaluationConfig
from experiments.utils.configure_logging import get_logger


class Evaluator:
    def __init__(self, cfg: EvaluationConfig):
        self.cfg = cfg
        # Get logger after configuration is set up
        self._log = getLogger(__name__)

    def evaluate(self):
        self._log.info("Starting evaluation process")
        self._log.debug("Configuration loaded successfully")

        if self.cfg.logging.show_terminal_logs:
            self._log.info(
                "Terminal logging is enabled - you should see this message in the console"
            )
        else:
            self._log.info(
                "Terminal logging is disabled - this message will only appear in the log file"
            )

        self._log.info("Evaluating models")

    def _evaluate_model(self):
        """Iterate over all datasets and evaluate the model on each"""

    def _evaluate_dataset(self):
        pass

    def _load_dataset(self):
        pass
