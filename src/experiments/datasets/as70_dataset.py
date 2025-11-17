from pathlib import Path

import polars as pl
from datasets import Dataset

from experiments.config.evaluation_config import EvaluationConfig
from experiments.datasets.asr_dataset_base import ASRDatasetBase
from experiments.utils.configure_logging import logger


class AS70Dataset(ASRDatasetBase):
    def __init__(self, cfg: EvaluationConfig, dataset_name: str):
        super().__init__(cfg, dataset_name)

    def load_dataset(self) -> Dataset:
        df_path = Path("data/as70/as70.parquet")
        df = pl.read_parquet(df_path)

        df = df.rename({"unannotated_text": "transcript"})

        if self.cfg.max_samples_per_dataset > 0:
            df = df.head(self.cfg.max_samples_per_dataset)
            logger.info(
                f"Filtered Bbbenji dataset to {len(df)} samples for max_samples_per_dataset: {self.cfg.max_samples_per_dataset}"
            )

        arrow_table = df.to_arrow()
        self._dataset = Dataset(arrow_table)
        logger.info(f"Successfully loaded dataset with {len(self._dataset)} samples")
        return self._dataset
