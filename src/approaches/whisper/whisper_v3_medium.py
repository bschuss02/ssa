from typing import List

from transformers import pipeline

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
        self.pipe = pipeline("automatic-speech-recognition", model="openai/whisper-medium.en")

    def transcribe(
        self, transcription_inputs: List[TranscriptionInput]
    ) -> List[TranscriptionOutput]:
        audio_paths = [str(ti.audio_path) for ti in transcription_inputs]
        if not all(audio_paths):
            raise ValueError("All transcription inputs must have an audio path")
        results = self.pipe(audio_paths)
        return [
            TranscriptionOutput(transcription=result["text"], metadata=result) for result in results
        ]
