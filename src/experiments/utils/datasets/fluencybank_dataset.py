from pathlib import Path

import librosa
import numpy as np
import polars as pl
import sounddevice as sd
from datasets import Dataset


def create_fluencybank_dataset(dataset_path: str):
    """Create a custom huggingface audio dataset from the FluencyBank dataset."""
    dataset_path = Path(dataset_path)
    metadata_df = pl.read_parquet(dataset_path / "fluencybank_segments.parquet")
    metadata_list = metadata_df.to_dicts()

    # Load audio files using librosa instead of torchcodec
    for item in metadata_list:
        audio_path = dataset_path / item["clip_audio_file"]
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

    fluencybank_dataset = Dataset.from_list(metadata_list)
    return fluencybank_dataset


if __name__ == "__main__":
    fluencybank_dataset = create_fluencybank_dataset(
        "/Users/Benjamin/dev/ssa/data/fluencybank/processed"
    )
    print(fluencybank_dataset)
    # play the first audio
    audio = fluencybank_dataset[0]["audio"]
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
