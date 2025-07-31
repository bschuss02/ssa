from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import polars as pl
import sounddevice as sd
from datasets import Dataset


class FluencyBankDataset:
    """Lazy-loading wrapper for FluencyBank dataset."""

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self._dataset: Optional[Dataset] = None
        self._metadata_df = None

    def _load_metadata(self):
        """Load metadata without loading audio files."""
        if self._metadata_df is None:
            metadata_path = self.dataset_path / "fluencybank_segments.parquet"
            self._metadata_df = pl.read_parquet(metadata_path)

    def _load_dataset(self):
        """Load the full dataset with audio files (expensive operation)."""
        if self._dataset is None:
            self._load_metadata()
            metadata_list = self._metadata_df.to_dicts()

            # Load audio files using librosa
            for item in metadata_list:
                audio_path = self.dataset_path / item["clip_audio_file"]
                if audio_path.exists():
                    # Load audio with librosa
                    audio_array, sr = librosa.load(str(audio_path), sr=16000)
                    item["audio"] = {
                        "array": audio_array.tolist(),  # Convert to list for dataset storage
                        "sampling_rate": sr,
                        "path": str(audio_path),
                    }
                else:
                    print(f"Warning: Audio file not found: {audio_path}")
                    item["audio"] = None

            self._dataset = Dataset.from_list(metadata_list)

    def __len__(self) -> int:
        """Get the number of samples without loading audio files."""
        self._load_metadata()
        return len(self._metadata_df)

    def __getitem__(self, idx):
        """Get a sample, loading the dataset if necessary."""
        self._load_dataset()
        return self._dataset[idx]

    def select(self, indices):
        """Select specific indices, loading the dataset if necessary."""
        self._load_dataset()
        return self._dataset.select(indices)

    def __iter__(self):
        """Iterate over the dataset, loading it if necessary."""
        self._load_dataset()
        return iter(self._dataset)


def create_fluencybank_dataset(dataset_path: str):
    """Legacy function for backward compatibility."""
    return FluencyBankDataset(dataset_path)


if __name__ == "__main__":
    # Test the lazy loading
    print("Creating dataset instance...")
    fluencybank_dataset = FluencyBankDataset(
        "/Users/Benjamin/dev/ssa/data/fluencybank/processed"
    )
    print(f"Dataset instance created. Length: {len(fluencybank_dataset)}")

    print("Loading first sample...")
    first_sample = fluencybank_dataset[0]
    print(f"First sample loaded: {first_sample.keys()}")

    # play the first audio
    audio = first_sample["audio"]
    if audio is not None:
        audio_array = np.array(audio["array"])  # Convert back to numpy array
        print(
            f"Audio shape: {audio_array.shape}, Sample rate: {audio['sampling_rate']}"
        )
        # play the audio
        sd.play(audio_array, audio["sampling_rate"])
        sd.wait()  # Wait for audio to finish playing
    else:
        print("No audio available for the first item")
