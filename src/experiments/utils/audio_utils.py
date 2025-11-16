import concurrent.futures
from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import numpy as np


def load_audio_files(
    audio_paths: List[Path], max_workers: int = 4, sr: Optional[int] = None
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Load multiple audio files concurrently using librosa.

    Args:
        audio_paths: List of paths to audio files
        max_workers: Maximum number of worker threads for concurrent loading
        sr: Target sample rate. If None, uses the original sample rate of each file.

    Returns:
        Tuple of (audio_arrays, sampling_rates) where:
        - audio_arrays: List of numpy arrays containing audio data
        - sampling_rates: List of sampling rates for each audio file
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(librosa.load, str(audio_path), sr=sr) for audio_path in audio_paths
        ]
        results = [future.result() for future in futures]
        audio_arrays = [result[0] for result in results]
        sampling_rates = [result[1] for result in results]
    return audio_arrays, sampling_rates
