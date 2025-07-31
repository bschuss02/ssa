from pathlib import Path
from typing import Any, Dict

import librosa
import polars as pl
from datasets import Dataset

from experiments.config.evaluation_config import EvaluationConfig
from experiments.datasets.asr_dataset_base import ASRDatasetBase


class FluencybankDataset(ASRDatasetBase):
    def __init__(self, cfg: EvaluationConfig, dataset_name: str, dataset_path: Path):
        super().__init__(cfg, dataset_name, dataset_path)

    def load_dataset(self) -> Dataset:
        df_path = self.dataset_path / "fluencybank_segments.parquet"
        df = pl.read_parquet(df_path)

        # If max_samples_per_dataset is 0, use the entire dataset
        if self.cfg.max_samples_per_dataset > 0:
            df = df.head(self.cfg.max_samples_per_dataset)

        arrow_table = df.to_arrow()
        self._dataset = Dataset(arrow_table)
        self._log.info(f"Successfully loaded dataset with {len(self._dataset)} samples")
        return self._dataset
