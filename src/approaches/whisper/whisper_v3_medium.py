from pathlib import Path
from typing import List

import whisper

from experiments.config.evaluation_config import EvaluationConfig
from experiments.inference_models.asr_model_base import (
    ASRModelBase,
    TranscriptionInput,
    TranscriptionOutput,
)


class WhisperV3Medium(ASRModelBase):
    def __init__(self, model_name: Path, model_dir: Path, cfg: EvaluationConfig):
        super().__init__(model_name, model_dir)

    def load_model(self):
        self.model = whisper.load_model("medium.en", device=self.device)

    def transcribe(
        self, transcription_inputs: List[TranscriptionInput]
    ) -> List[TranscriptionOutput]:
        audio_paths = [input.audio_path for input in transcription_inputs]
        if not all(audio_paths):
            raise ValueError("All transcription inputs must have an audio path")
        results = self.model.transcribe(audio_paths, verbose=True)
        return [
            TranscriptionOutput(transcription=result["text"], metadata=result)
            for result in results
            if result is not None
        ]
