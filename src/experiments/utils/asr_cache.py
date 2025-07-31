import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from diskcache import Cache


class ASRCache:
    """Cache for ASR model transcription results to avoid redundant computations."""

    def __init__(self, cache_dir: Path):
        """Initialize the ASR cache.

        Args:
            cache_dir: Directory to store the cache files
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache = Cache(str(cache_dir))

    def _generate_cache_key(
        self, model_name: str, audio_arrays: List[np.ndarray], sampling_rate: int
    ) -> str:
        """Generate a unique cache key for the given model and audio inputs.

        Args:
            model_name: Name of the ASR model
            audio_arrays: List of audio arrays to transcribe
            sampling_rate: Sampling rate of the audio

        Returns:
            A unique string key for caching
        """
        # Create a hash of the model name, audio data, and sampling rate
        key_data = {
            "model_name": model_name,
            "sampling_rate": sampling_rate,
            "num_audio_samples": len(audio_arrays),
            "audio_hashes": [],
        }

        # Hash each audio array to create a fingerprint
        for audio_array in audio_arrays:
            # Convert to bytes and hash
            audio_bytes = audio_array.tobytes()
            audio_hash = hashlib.sha256(audio_bytes).hexdigest()
            key_data["audio_hashes"].append(audio_hash)

        # Create final hash from the key data
        key_json = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_json.encode()).hexdigest()

    def get(
        self, model_name: str, audio_arrays: List[np.ndarray], sampling_rate: int
    ) -> Optional[List[str]]:
        """Get cached transcription results if available.

        Args:
            model_name: Name of the ASR model
            audio_arrays: List of audio arrays to transcribe
            sampling_rate: Sampling rate of the audio

        Returns:
            List of transcription results if cached, None otherwise
        """
        cache_key = self._generate_cache_key(model_name, audio_arrays, sampling_rate)
        return self._cache.get(cache_key)

    def set(
        self,
        model_name: str,
        audio_arrays: List[np.ndarray],
        sampling_rate: int,
        transcriptions: List[str],
    ) -> None:
        """Cache transcription results for future use.

        Args:
            model_name: Name of the ASR model
            audio_arrays: List of audio arrays that were transcribed
            sampling_rate: Sampling rate of the audio
            transcriptions: List of transcription results to cache
        """
        cache_key = self._generate_cache_key(model_name, audio_arrays, sampling_rate)
        self._cache.set(cache_key, transcriptions)

    def clear(self) -> None:
        """Clear all cached results."""
        self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary containing cache statistics
        """
        return {
            "cache_dir": str(self.cache_dir),
            "size": len(self._cache),
            "volume": self._cache.volume(),
        }

    def close(self) -> None:
        """Close the cache and free resources."""
        self._cache.close()
