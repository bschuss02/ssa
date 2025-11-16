from typing import List, Optional

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from experiments.config.evaluation_config import EvaluationConfig
from experiments.inference_models.asr_model_base import (
    ASRModelBase,
    TranscriptionInput,
    TranscriptionOutput,
)
from experiments.utils.audio_utils import load_audio_files
from experiments.utils.configure_logging import logger


class WhisperV3Medium(ASRModelBase):
    def __init__(self, model_name: str, cfg: EvaluationConfig, prompt: Optional[str] = None):
        super().__init__(model_name, cfg)
        self.model = None
        self.processor = None
        self.prompt = prompt

    def load_model(self):
        logger.info(f"Loading model {self.model_name}")
        model_id = "openai/whisper-medium.en"
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id)
        self.model.to(self.device)

    def transcribe(
        self, transcription_inputs: List[TranscriptionInput]
    ) -> List[TranscriptionOutput]:
        audio_paths = [ti.audio_path for ti in transcription_inputs]
        if not all(audio_paths):
            raise ValueError("All transcription inputs must have an audio path")

        # Load all audio files concurrently
        audio_arrays, _ = load_audio_files(audio_paths, max_workers=self.cfg.max_workers, sr=16000)

        # Process all audio arrays in batch
        audio_inputs = self.processor(
            audio_arrays, return_tensors="pt", sampling_rate=16000, padding=True
        )
        audio_inputs = {k: v.to(self.device) for k, v in audio_inputs.items()}

        # Prepare generate kwargs with prompt if provided
        generate_kwargs = {}
        if self.prompt is not None:
            # Convert prompt text to token IDs
            # Whisper expects the prompt tokens to come after the initial special tokens
            # Format: [[position, token_id], ...] where position is the decoder position
            prompt_token_ids = self.processor.tokenizer.encode(
                self.prompt, add_special_tokens=False
            )
            # The decoder starts with special tokens at position 0, 1, etc.
            # For prompts, we typically start after the language/task tokens (around position 1-2)
            # We'll use position starting from 1 to place prompt tokens
            forced_decoder_ids = [
                [i + 1, prompt_token_ids[i]] for i in range(len(prompt_token_ids))
            ]
            generate_kwargs["forced_decoder_ids"] = forced_decoder_ids

        # Generate transcriptions for entire batch at once
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_features=audio_inputs["input_features"], **generate_kwargs
            )

        # Decode all transcriptions at once
        transcriptions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        results = [{"text": transcription} for transcription in transcriptions]

        return [
            TranscriptionOutput(transcription=result["text"], metadata=result) for result in results
        ]
