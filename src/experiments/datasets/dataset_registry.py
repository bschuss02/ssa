from typing import Dict, Type

from experiments.datasets.as70_dataset import AS70Dataset
from experiments.datasets.asr_dataset_base import ASRDatasetBase
from experiments.datasets.bbbenji_dataset import BbbenjiDataset
from experiments.datasets.fluencybank_dataset import FluencybankDataset

# Registry mapping dataset names to their corresponding dataset classes
dataset_registry: Dict[str, Type[ASRDatasetBase]] = {
    "fluencybank": FluencybankDataset,
    "bbbenji": BbbenjiDataset,
    "as70": AS70Dataset,
}
