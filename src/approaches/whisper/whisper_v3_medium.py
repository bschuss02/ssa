from typing import List

import whisper

from experiments.config.evaluation_config import EvaluationConfig
from experiments.inference_models.asr_model_base import (
    ASRModelBase,
    TranscriptionInput,
    TranscriptionOutput,
)
from experiments.utils.configure_logging import logger


class WhisperV3Medium(ASRModelBase):
    def __init__(self, model_name: str, cfg: EvaluationConfig):
        super().__init__(model_name, cfg)
        self.audio_array_or_path = "audio_path"

    def load_model(self):
        logger.info(f"Loading model {self.model_name}")
        self.model = whisper.load_model("medium.en", device=self.device)
        logger.info(f"Model {self.model_name} loaded successfully")

    def transcribe(
        self, transcription_inputs: List[TranscriptionInput]
    ) -> List[TranscriptionOutput]:
        audio_paths = [input.audio_path for input in transcription_inputs]
        if not all(audio_paths):
            raise ValueError("All transcription inputs must have an audio path")
        results = self.model.transcribe(audio_paths, verbose=True)
        return [
            TranscriptionOutput(transcription=result["text"], metadata=result) for result in results
        ]
