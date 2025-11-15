from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict

from experiments.config.evaluation_config import EvaluationConfig
from experiments.utils.configure_logging import logger


class TranscriptionInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    audio_array: Optional[np.ndarray] = None
    sample_rate: Optional[int] = None
    audio_path: Optional[Path] = None
    metadata: Optional[Dict[str, Any]] = None


class TranscriptionOutput(BaseModel):
    transcription: str
    metadata: Optional[Dict[str, Any]] = None


class ASRModelBase(ABC):
    model_name: str
    model_dir: Path
    device: str

    @abstractmethod
    def __init__(self, model_name: Path, model_dir: Path, cfg: EvaluationConfig):
        self.cfg = cfg
        self.model_name = model_name
        self.model_dir = model_dir
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        logger.info(f"Using device: {self.device}")

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def transcribe(
        self,
        inputs: List[TranscriptionInput],
    ) -> List[TranscriptionOutput]:
        pass
