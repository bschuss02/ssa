from abc import ABC, abstractmethod
from pathlib import Path

from datasets import Dataset

from experiments.config.evaluation_config import EvaluationConfig
from experiments.utils.configure_logging import logger


class ASRDatasetBase(ABC):
    @abstractmethod
    def __init__(self, cfg: EvaluationConfig, dataset_name: str, dataset_path: Path):
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        logger.info(f"Initialized dataset {dataset_name} with path {dataset_path}")

    @abstractmethod
    def load_dataset(self) -> Dataset:
        """Outputs dataset where audio_file is a path to the audio file relative to the root directory of the repository"""
        pass
