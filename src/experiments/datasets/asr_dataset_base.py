from abc import ABC, abstractmethod

from datasets import Dataset

from experiments.config.evaluation_config import EvaluationConfig
from experiments.utils.configure_logging import logger


class ASRDatasetBase(ABC):
    @abstractmethod
    def __init__(self, cfg: EvaluationConfig, dataset_name: str):
        self.cfg = cfg
        self.dataset_name = dataset_name
        logger.info(f"Initialized dataset {dataset_name}")

    @abstractmethod
    def load_dataset(self) -> Dataset:
        """Outputs dataset where audio_file is a path to the audio file relative to the root directory of the repository"""
        pass
