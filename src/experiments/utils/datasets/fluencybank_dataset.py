import os
import re
from typing import Optional, Dict, Any, List
import polars as pl
import soundfile as sf
import numpy as np
from datasets import Dataset, Features, Audio, Value, Sequence
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FluencyBankDataset:
    """
    Custom Hugging Face dataset for FluencyBank audio-text data.

    Loads data from a Parquet file containing text annotations and audio file paths,
    and creates a HuggingFace Dataset with both text and audio features.
    """

    def __init__(
        self,
        parquet_path: str,
        audio_sample_rate: int = 16000,
        max_audio_length: Optional[float] = None,
        include_timing: bool = True,
        include_speaker_info: bool = True,
        text_column: str = "unannotated_text",  # Use unannotated by default
        clean_text: bool = True,  # Whether to clean special characters from text
    ):
        """
        Initialize the FluencyBank dataset.

        Args:
                parquet_path: Path to the parquet file containing the data
                audio_sample_rate: Target sample rate for audio (default: 16kHz)
                max_audio_length: Maximum audio length in seconds (None for no limit)
                include_timing: Whether to include start/end time information
                include_speaker_info: Whether to include speaker ID information
                text_column: Which text column to use ('annotated_text' or 'unannotated_text')
                clean_text: Whether to clean special characters from text (default: True)
        """
        self.parquet_path = Path(parquet_path)
        self.audio_sample_rate = audio_sample_rate
        self.max_audio_length = max_audio_length
        self.include_timing = include_timing
        self.include_speaker_info = include_speaker_info
        self.text_column = text_column
        self.clean_text = clean_text

        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        logger.info(f"Loading FluencyBank dataset from {parquet_path}")
        self._load_data()

    def _load_data(self) -> None:
        """Load and validate the parquet data."""
        try:
            self.df = pl.read_parquet(self.parquet_path)
            logger.info(f"Loaded {len(self.df)} samples from parquet file")

            # Validate required columns
            required_cols = [self.text_column, "clip_audio_file", "clip_id"]
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Check for missing audio files
            self._validate_audio_files()

        except Exception as e:
            logger.error(f"Error loading parquet file: {e}")
            raise

    def _validate_audio_files(self) -> None:
        """Validate that audio files exist and are accessible."""
        audio_paths = self.df.select("clip_audio_file").to_series().to_list()
        missing_files = []

        for audio_path in audio_paths:
            if not os.path.exists(audio_path):
                missing_files.append(audio_path)

        if missing_files:
            logger.warning(f"Found {len(missing_files)} missing audio files out of {len(audio_paths)}")
            # Remove rows with missing audio files
            valid_paths = [path for path in audio_paths if os.path.exists(path)]
            self.df = self.df.filter(pl.col("clip_audio_file").is_in(valid_paths))
            logger.info(f"Filtered to {len(self.df)} samples with valid audio files")

    def _load_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        Load and preprocess an audio file.

        Args:
                audio_path: Path to the audio file

        Returns:
                Dictionary with 'array' and 'sampling_rate' keys
        """
        try:
            audio_data, original_sr = sf.read(audio_path)

            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)

            # Resample if needed
            if original_sr != self.audio_sample_rate:
                import librosa

                audio_data = librosa.resample(
                    audio_data, orig_sr=original_sr, target_sr=self.audio_sample_rate
                )

            # Truncate if max_audio_length is specified
            if self.max_audio_length is not None:
                max_samples = int(self.max_audio_length * self.audio_sample_rate)
                if len(audio_data) > max_samples:
                    audio_data = audio_data[:max_samples]

            return {"array": audio_data.astype(np.float32), "sampling_rate": self.audio_sample_rate}

        except Exception as e:
            logger.error(f"Error loading audio file {audio_path}: {e}")
            # Return empty audio array as fallback
            return {"array": np.array([], dtype=np.float32), "sampling_rate": self.audio_sample_rate}

    def _prepare_features(self) -> Features:
        """Define the features schema for the HuggingFace dataset."""
        features = {
            "text": Value("string"),
            "audio": Audio(sampling_rate=self.audio_sample_rate),
            "clip_id": Value("string"),
        }

        # Add optional features
        if "annotated_text" in self.df.columns and self.text_column != "annotated_text":
            features["annotated_text"] = Value("string")

        if self.include_timing and "start_time" in self.df.columns:
            features["start_time"] = Value("float32")
            features["end_time"] = Value("float32")
            features["duration"] = Value("float32")

        if self.include_speaker_info and "speaker_id" in self.df.columns:
            features["speaker_id"] = Value("string")

        return Features(features)

    def _prepare_sample(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare a single data sample."""
        # Load audio
        audio = self._load_audio(row["clip_audio_file"])

        # Prepare the sample
        sample = {
            "text": row[self.text_column],
            "audio": audio,
            "clip_id": row["clip_id"],
        }

        # Add optional fields
        if "annotated_text" in row and self.text_column != "annotated_text":
            sample["annotated_text"] = row["annotated_text"]

        if self.include_timing and "start_time" in row:
            sample["start_time"] = float(row["start_time"])
            sample["end_time"] = float(row["end_time"])
            sample["duration"] = float(row["end_time"] - row["start_time"])

        if self.include_speaker_info and "speaker_id" in row:
            sample["speaker_id"] = row["speaker_id"]

        return sample

    def to_hf_dataset(self) -> Dataset:
        """
        Convert the data to a HuggingFace Dataset.

        Returns:
                HuggingFace Dataset object
        """
        logger.info("Converting to HuggingFace Dataset...")

        # Convert polars dataframe to list of dictionaries
        data_dicts = self.df.to_dicts()

        # Prepare all samples
        samples = []
        for i, row in enumerate(data_dicts):
            if i % 100 == 0:
                logger.info(f"Processing sample {i+1}/{len(data_dicts)}")

            sample = self._prepare_sample(row)
            samples.append(sample)

        # Create HuggingFace dataset
        features = self._prepare_features()
        dataset = Dataset.from_list(samples, features=features)

        logger.info(f"Created HuggingFace dataset with {len(dataset)} samples")
        return dataset

    def get_stats(self) -> Dict[str, Any]:
        """Get basic statistics about the dataset."""
        stats = {
            "num_samples": len(self.df),
            "columns": self.df.columns,
            "unique_speakers": (
                self.df.select("speaker_id").n_unique() if "speaker_id" in self.df.columns else None
            ),
            "unique_clips": self.df.select("clip_id").n_unique(),
        }

        if "start_time" in self.df.columns and "end_time" in self.df.columns:
            durations = self.df.select((pl.col("end_time") - pl.col("start_time")).alias("duration"))
            stats.update(
                {
                    "total_duration_hours": durations.sum().item() / 3600,
                    "avg_duration_seconds": durations.mean().item(),
                    "min_duration_seconds": durations.min().item(),
                    "max_duration_seconds": durations.max().item(),
                }
            )

        return stats


def create_fluencybank_dataset(
    parquet_path: str = "/Users/Benjamin/dev/ssa/data/fluencybank/processed/fluencybank_segments.parquet",
    **kwargs,
) -> Dataset:
    """
    Convenience function to create a FluencyBank HuggingFace dataset.

    Args:
            parquet_path: Path to the parquet file
            **kwargs: Additional arguments for FluencyBankDataset

    Returns:
            HuggingFace Dataset object
    """
    fb_dataset = FluencyBankDataset(parquet_path, **kwargs)

    # Print dataset statistics
    stats = fb_dataset.get_stats()
    print("Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    return fb_dataset.to_hf_dataset()


# Example usage
if __name__ == "__main__":
    # Create dataset with default settings
    dataset = create_fluencybank_dataset()

    # Print first sample
    print("\nFirst sample:")
    print(dataset[0])

    # Create dataset with custom settings
    custom_dataset = create_fluencybank_dataset(
        audio_sample_rate=22050,
        max_audio_length=30.0,  # 30 seconds max
        text_column="annotated_text",  # Use annotated text instead
        include_timing=True,
        include_speaker_info=True,
    )
