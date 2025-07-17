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
        pass
