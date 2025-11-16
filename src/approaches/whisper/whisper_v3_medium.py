from typing import List

import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

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
        self.model = None
        self.processor = None

    def load_model(self):
        logger.info(f"Loading model {self.model_name}")
        model_id = "openai/whisper-medium.en"
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id)
        self.model.to(self.device)

    def transcribe(
        self, transcription_inputs: List[TranscriptionInput]
    ) -> List[TranscriptionOutput]:
        audio_paths = [str(ti.audio_path) for ti in transcription_inputs]
        if not all(audio_paths):
            raise ValueError("All transcription inputs must have an audio path")

        # Load all audio files first
        audio_arrays = []
        for audio_path in audio_paths:
            audio, sr = librosa.load(audio_path, sr=16000)
            audio_arrays.append(audio)

        # Process all audio arrays in batch
        audio_inputs = self.processor(
            audio_arrays, return_tensors="pt", sampling_rate=16000, padding=True
        )
        audio_inputs = {k: v.to(self.device) for k, v in audio_inputs.items()}

        # Generate transcriptions for entire batch at once
        with torch.no_grad():
            generated_ids = self.model.generate(input_features=audio_inputs["input_features"])

        # Decode all transcriptions at once
        transcriptions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        results = [{"text": transcription} for transcription in transcriptions]

        return [
            TranscriptionOutput(transcription=result["text"], metadata=result) for result in results
        ]
