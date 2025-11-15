from logging import getLogger
from pathlib import Path

import polars as pl
from datasets import Dataset

from experiments.config.evaluation_config import EvaluationConfig
from experiments.datasets.asr_dataset_base import ASRDatasetBase

_log = getLogger(__name__)


class BbbenjiDataset(ASRDatasetBase):
    def __init__(self, cfg: EvaluationConfig, dataset_name: str, dataset_path: Path):
        super().__init__(cfg, dataset_name, dataset_path)

    def load_dataset(self) -> Dataset:
        df_path = self.dataset_path / "bbbenji.parquet"
        df = pl.read_parquet(df_path)
        original_length = len(df)

        if self.cfg.bbbenji.subset == "fluent":
            df = df.filter(pl.col("speechPatterns") == "fluent")
        elif self.cfg.bbbenji.subset == "stuttered":
            df = df.filter(pl.col("speechPatterns") == "stuttered")
        elif self.cfg.bbbenji.subset == "all":
            pass
        _log.info(
            f"Filtered Bbbenji dataset to {len(df)}/{original_length} samples for subset '{self.cfg.bbbenji.subset}'"
        )

        arrow_table = df.to_arrow()
        self._dataset = Dataset(arrow_table)
        _log.info(f"Successfully loaded dataset with {len(self._dataset)} samples")
        return self._dataset
