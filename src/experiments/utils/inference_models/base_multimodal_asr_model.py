#!/usr/bin/env python3
"""
Abstract Base Class for Multimodal ASR Models

This module defines the minimal interface for ASR models that can transcribe
audio with optional chat completion messages for context.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class BaseMultimodalASRModel(ABC):
    """
    Abstract base class for ASR models supporting transcription with optional chat context.
    """

    def __init__(
        self,
        *,
        model_name: str,
        model_path: str,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.model_path = Path(model_path)
        self.device = device

    @abstractmethod
    def load_model(self, **kwargs) -> None:
        """
        Load the ASR model and processor.

        Args:
            model_name: Name or path of the model to load
            **kwargs: Additional model-specific arguments
        """
        pass

    @abstractmethod
    def transcribe(
        self,
        *,
        audio_path: Optional[str] = None,
        audio_array: Optional[np.ndarray] = None,
        sample_rate: Optional[int] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> str:
        """
        Transcribe a single audio sample.

        Args:
            audio_path: Path to the audio file
            audio_array: Audio data as numpy array
            sample_rate: Sample rate of the audio
            messages: Optional chat messages for context
            **kwargs: Additional arguments

        Returns:
            Transcribed text
        """
        pass

    def transcribe_batch(
        self,
        *,
        audio_arrays: List[np.ndarray],
        sample_rates: List[int],
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> List[str]:
        """
        Transcribe multiple audio samples in batch.

        Args:
            audio_arrays: List of audio data as numpy arrays
            sample_rates: List of sample rates corresponding to each audio array
            messages: Optional chat messages for context
            **kwargs: Additional arguments

        Returns:
            List of transcribed texts
        """
        # Default implementation: process one by one
        # Subclasses can override for true batch processing
        results = []
        for audio_array, sample_rate in zip(audio_arrays, sample_rates):
            result = self.transcribe(
                audio_array=audio_array,
                sample_rate=sample_rate,
                messages=messages,
                **kwargs,
            )
            results.append(result)
        return results
