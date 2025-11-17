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


class WhisperV3MediumEnglish(ASRModelBase):
    def __init__(
        self,
        model_name: str,
        cfg: EvaluationConfig,
        model_id: str = "openai/whisper-medium.en",
        prompt: Optional[str] = None,
        language: Optional[str] = "en",
    ):
        super().__init__(model_name, cfg)
        self.model = None
        self.processor = None
        self.model_id = model_id
        self.prompt = prompt
        self.language = language

    def load_model(self):
        logger.info(f"Loading model {self.model_name}")
        self.processor = WhisperProcessor.from_pretrained(self.model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_id)
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

        # Whisper v3 multilingual requires mel input features to be exactly 3000 frames long
        # Pad or truncate input_features to ensure they are exactly 3000 frames
        input_features = audio_inputs["input_features"]
        target_length = 3000
        current_length = input_features.shape[-1]

        if current_length < target_length:
            # Pad with the minimum value of the input features (typical for mel spectrograms)
            padding_length = target_length - current_length
            # input_features shape is [batch_size, n_mels, sequence_length]
            # We need to pad along the last dimension
            padding_value = input_features.min().item()
            padding = torch.full(
                (input_features.shape[0], input_features.shape[1], padding_length),
                padding_value,
                device=input_features.device,
                dtype=input_features.dtype,
            )
            input_features = torch.cat([input_features, padding], dim=-1)
        elif current_length > target_length:
            # Truncate if longer than 3000
            input_features = input_features[:, :, :target_length]

        audio_inputs["input_features"] = input_features

        # Prepare generate kwargs with prompt and language if provided
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
        if self.language is not None:
            generate_kwargs["language"] = self.language

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
