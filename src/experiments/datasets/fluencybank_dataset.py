from pathlib import Path
from typing import Any, Dict

import librosa
import polars as pl
from datasets import Dataset

from experiments.config.EvaluationConfig import EvaluationConfig
from experiments.datasets.asr_dataset_base import ASRDatasetBase


class FluencybankDataset(ASRDatasetBase):
    def __init__(self, cfg: EvaluationConfig, dataset_name: str, dataset_path: Path):
        super().__init__(cfg, dataset_name, dataset_path)

    def load_dataset(self, load_from_cache: bool = True) -> Dataset:
        df = self._load_dataset_df()

        return_dtype = self._get_return_dtype(df)

        processed_df = df.with_columns(
            [
                pl.struct(df.columns)
                .map_elements(
                    self.process_audio_row,
                    strategy="threading",
                    return_dtype=return_dtype,
                )
                .alias("processed_row")
            ]
        )

        metadata_list = processed_df.select("processed_row").to_dicts()
        metadata_list = [row["processed_row"] for row in metadata_list]

        self._dataset = Dataset.from_list(metadata_list)
        self._log.info(f"Successfully loaded dataset with {len(metadata_list)} samples")
        return self._dataset

    def _load_dataset_df(self):
        df_path = self.dataset_path / "fluencybank_segments.parquet"
        df = pl.read_parquet(df_path)
        df = df.head(self.cfg.max_samples_per_dataset)
        return df

    def _get_return_dtype(self, df: pl.DataFrame) -> pl.Struct:
        audio_struct = pl.Struct(
            [
                pl.Field("array", pl.List(pl.Float64)),
                pl.Field("sampling_rate", pl.Int64),
                pl.Field("path", pl.Utf8),
            ]
        )

        return_fields = []
        for col in df.columns:
            if col in ["start_time", "end_time"]:
                return_fields.append(pl.Field(col, pl.Float64))
            else:
                return_fields.append(pl.Field(col, pl.Utf8))
        return_fields.append(pl.Field("audio", audio_struct))

        return pl.Struct(return_fields)

    def process_audio_row(self, row_dict: Dict[str, Any]) -> Dict[str, Any]:
        audio_path = self.dataset_path / row_dict["clip_audio_file"]
        if audio_path.exists():
            try:
                audio_array, sr = librosa.load(str(audio_path), sr=16000)
                row_dict["audio"] = {
                    "array": audio_array.tolist(),
                    "sampling_rate": sr,
                    "path": str(audio_path),
                }
            except Exception as e:
                self._log.warning(f"Error loading audio file {audio_path}: {e}")
                row_dict["audio"] = None
        else:
            self._log.warning(f"Audio file not found: {audio_path}")
            row_dict["audio"] = None
        return row_dict
