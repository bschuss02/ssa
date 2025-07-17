#!/usr/bin/env python3
"""
Abstract Base Class for Multimodal ASR Models

This module defines the minimal interface for ASR models that can transcribe
audio with optional chat completion messages for context.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional


class BaseMultimodalASRModel(ABC):
    """
    Abstract base class for ASR models supporting transcription with optional chat context.
    """

    def __init__(
        self,
        *,
        model_path: str,
        device: Optional[str] = None,
    ):
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
        audio_path: str,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> str:
        """
        Transcribe audio with optional chat context.

        Args:
            audio_path: Path to the audio file to transcribe
            messages: Optional list of chat messages with 'role' and 'content' keys
                     e.g., [{"role": "system", "content": "You are a medical transcriptionist"},
                           {"role": "user", "content": "Please transcribe this audio"}]
            **kwargs: Additional transcription parameters

        Returns:
            Transcribed text
        """
        pass
