from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

from experiments.config.evaluation_config import EvaluationConfig
from experiments.inference_models.asr_model_base import (
    ASRModelBase,
    TranscriptionInput,
    TranscriptionOutput,
)
from experiments.utils.audio_utils import load_audio_files
from experiments.utils.configure_logging import logger


class Phi4MultimodalInstruct(ASRModelBase):
    model: Any
    processor: Any

    def __init__(self, model_name: str, cfg: EvaluationConfig):
        super().__init__(model_name, cfg)
        self.model = None
        self.processor = None
        # model_name is expected to be a path to the model directory
        self.model_dir = Path(model_name)
        self.prompt_messages = [
            {
                "role": "system",
                "content": "You are an expert audio transcriptionist.",
            },
            {
                "role": "user",
                "content": "You are tasked with transcribing the speech from this audio recording. Transcribe the speech from this audio recording exactly as it is spoken. <|audio_1|>",
            },
        ]

    def load_model(self):
        logger.info(f"Loading model {self.model_name} from {self.model_dir}")
        self.processor = AutoProcessor.from_pretrained(self.model_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map=self.device,
            _attn_implementation="eager",  # Disable Flash Attention for Mac M4 chip
        )

    def transcribe(
        self,
        transcription_inputs: List[TranscriptionInput],
    ) -> List[TranscriptionOutput]:
        # Load audio files from paths concurrently
        audio_paths = [ti.audio_path for ti in transcription_inputs]
        for audio_path in audio_paths:
            if not audio_path.exists():
                raise ValueError(f"Audio file does not exist: {audio_path}")
        
        # Resample all files to 16kHz for consistency (common sample rate for speech models)
        # This handles varying input sample rates gracefully
        target_sample_rate = 16000
        audio_arrays, sample_rates = load_audio_files(
            audio_paths, max_workers=self.cfg.max_workers, sr=target_sample_rate
        )
        sample_rate = target_sample_rate
        
        prompt_string = self._build_prompt_string_from_messages(self.prompt_messages)
        logger.info(f"Prompt: {prompt_string}")
        inputs = self._prepare_inputs(prompt_string, audio_arrays, sample_rate)
        with torch.no_grad():
            transcriptions = self._generate_outputs(inputs)
        
        # Convert transcriptions to TranscriptionOutput objects with metadata
        return [
            TranscriptionOutput(
                transcription=transcription,
                metadata=input.metadata if input.metadata else {},
            )
            for transcription, input in zip(transcriptions, transcription_inputs)
        ]

    def _build_prompt_string_from_messages(self, prompt_messages: List[Dict[str, str]]) -> str:
        user_prompt = "<|user|>"
        assistant_prompt = "<|assistant|>"
        prompt_suffix = "<|end|>"
        prompt = ""

        for msg in prompt_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                # System messages are prepended to the prompt
                prompt = content + "\n" + prompt
            elif role == "user":
                prompt += user_prompt + content + prompt_suffix
            elif role == "assistant":
                prompt += assistant_prompt + content + prompt_suffix

        # Ensure the prompt ends with an assistant token for generation
        if not prompt.strip().endswith(assistant_prompt):
            prompt += assistant_prompt

        return prompt

    def _prepare_inputs(
        self, prompt_string: str, audio_arrays: List[np.ndarray], sample_rate: int
    ) -> Dict[str, Any]:
        batch_size = len(audio_arrays)

        # Ensure audio arrays are properly shaped (mono -> 2D if needed)
        processed_audio_arrays = []
        for i, audio_data in enumerate(audio_arrays):
            # Check for empty or problematic audio files and pad them
            if audio_data.size == 0 or audio_data.shape[0] < 10:
                logger.warning(
                    f"Problematic audio file detected at index {i} (shape: {audio_data.shape}), padding with zeros"
                )
                # Create a minimal valid audio array with at least 100 samples
                if audio_data.ndim == 1:
                    processed_audio = np.zeros((100, 2), dtype=audio_data.dtype)
                else:
                    processed_audio = np.zeros((100, audio_data.shape[1]), dtype=audio_data.dtype)
            # If audio is 1D (mono), convert to stereo by duplicating the channel
            elif audio_data.ndim == 1:
                # Convert mono to stereo by duplicating the channel
                processed_audio = np.stack([audio_data, audio_data], axis=1)
                logger.debug(
                    f"Converted mono audio {i} to stereo: {audio_data.shape} -> {processed_audio.shape}"
                )
            else:
                processed_audio = audio_data
                logger.debug(f"Audio {i} already stereo: {audio_data.shape}")
            processed_audio_arrays.append(processed_audio)

        audio_tuples = [
            (audio_data, sample_rate)
            for audio_data, sample_rate in zip(processed_audio_arrays, [sample_rate] * batch_size)
        ]

        try:
            return self.processor(
                text=[prompt_string] * batch_size,
                audios=audio_tuples,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
        except Exception as e:
            logger.error(f"Error processing audio batch: {e}")
            logger.error(f"Audio shapes: {[arr.shape for arr in processed_audio_arrays]}")
            raise

    def _generate_outputs(self, inputs: Dict[str, Any]) -> List[str]:
        generation_config = GenerationConfig.from_pretrained(self.model_dir)
        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            generation_config=generation_config,
            num_logits_to_keep=1,
        )

        # Remove prompt tokens from output for each sample in the batch
        batch_responses = []
        for i in range(generate_ids.shape[0]):
            output_ids = generate_ids[i : i + 1, inputs["input_ids"].shape[1] :]
            response = self.processor.batch_decode(
                output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            batch_responses.append(response)

        return batch_responses
