from pathlib import Path

import polars as pl
import sounddevice as sd
from datasets import Audio, Dataset


def create_fluencybank_dataset(dataset_path: str):
    """Create a custom huggingface audio dataset from the FluencyBank dataset."""
    dataset_path = Path(dataset_path)
    metadata_df = pl.read_parquet(dataset_path / "fluencybank_segments.parquet")
    metadata_list = metadata_df.to_dicts()
    fluencybank_dataset = Dataset.from_list(metadata_list).cast_column(
        "clip_audio_file", Audio(sampling_rate=16000)
    )
    return fluencybank_dataset


if __name__ == "__main__":
    fluencybank_dataset = create_fluencybank_dataset(
        "/Users/Benjamin/dev/ssa/data/fluencybank/processed"
    )
    print(fluencybank_dataset)
    # play the first audio
    audio = fluencybank_dataset[0]["clip_audio_file"]
    print(audio)
    # play the audio
    sd.play(audio["array"], audio["sampling_rate"])
