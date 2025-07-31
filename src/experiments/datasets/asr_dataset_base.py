from abc import ABC, abstractmethod
from pathlib import Path

from datasets import Dataset

from experiments.config.EvaluationConfig import EvaluationConfig
from experiments.utils.configure_logging import get_logger


class ASRDatasetBase(ABC):
    @abstractmethod
    def __init__(self, cfg: EvaluationConfig, dataset_name: str, dataset_path: Path):
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self._log = get_logger(__name__)

    @abstractmethod
    def load_dataset(self, load_from_cache: bool = True) -> Dataset:
        pass
