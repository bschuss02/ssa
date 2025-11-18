import random
from pathlib import Path
from typing import List, Optional

import diskcache
import numpy as np
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
    def __init__(
        self,
        model_name: str,
        cfg: EvaluationConfig,
        model_id: str = "openai/whisper-medium",
        prompt: Optional[str] = None,
        language: Optional[str] = None,
    ):
        super().__init__(model_name, cfg)
        self.model = None
        self.processor = None
        self.model_id = model_id
        self.prompt = prompt
        self.language = language
        self._initialize_random_seed(42)

        # Initialize cache if enabled
        self.cache = None
        self.model_version = cfg.cache.model_version if cfg.cache.enabled else None
        if cfg.cache.enabled:
            cache_dir = Path(cfg.cache.cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache = diskcache.Cache(str(cache_dir))
            logger.info(f"Cache enabled at {cache_dir} with model_version={self.model_version}")

    def _initialize_random_seed(self, random_seed: int):
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
        if torch.backends.mps.is_available():
            torch.backends.mps.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

    def _generate_cache_key(
        self, audio_paths: List[Path], model_id: str, prompt: Optional[str], language: Optional[str]
    ) -> tuple:
        """Generate a cache key from audio paths, model_id, prompt, language, and model_version.

        Returns a tuple that diskcache will automatically hash.
        """
        # diskcache automatically hashes tuples/keys, so we can use a simple tuple
        # Include model_version in the key so changing it invalidates old cache entries
        return (
            tuple(sorted(str(p) for p in audio_paths)),
            model_id,
            prompt,
            language,
            self.model_version,
        )

    def _transcribe_impl(
        self, transcription_inputs: List[TranscriptionInput]
    ) -> List[TranscriptionOutput]:
        """Internal implementation of transcription without caching."""
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
        # Note: English-only models (with .en suffix) don't accept language parameter
        is_english_only = ".en" in self.model_id
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
        if self.language is not None and not is_english_only:
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

    def load_model(self):
        logger.info(f"Loading model {self.model_name}")
        self.processor = WhisperProcessor.from_pretrained(self.model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_id)
        self.model.to(self.device)

    def transcribe(
        self, transcription_inputs: List[TranscriptionInput]
    ) -> List[TranscriptionOutput]:
        """Transcribe audio files with optional caching."""
        audio_paths = [ti.audio_path for ti in transcription_inputs]

        # Check cache if enabled
        if self.cache is not None:
            cache_key = self._generate_cache_key(
                audio_paths, self.model_id, self.prompt, self.language
            )

            # Try to get from cache using diskcache's built-in get() method
            cached_value = self.cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit for key {str(cache_key)[:50]}...")
                # Handle both new format (list) and legacy format (dict with results key)
                if isinstance(cached_value, list):
                    cached_results = cached_value
                elif isinstance(cached_value, dict) and "results" in cached_value:
                    cached_results = cached_value["results"]
                else:
                    cached_results = []
                # Reconstruct TranscriptionOutput objects from cached data
                return [TranscriptionOutput(**item) for item in cached_results]

            logger.debug(f"Cache miss for key {str(cache_key)[:50]}...")

        # Perform transcription
        results = self._transcribe_impl(transcription_inputs)

        # Store in cache if enabled
        if self.cache is not None:
            cache_key = self._generate_cache_key(
                audio_paths, self.model_id, self.prompt, self.language
            )
            # Serialize results for caching
            # Model version is already part of the cache key, so no need for timestamps
            cache_value = [result.model_dump() for result in results]
            # Use diskcache's built-in set() method
            self.cache.set(cache_key, cache_value)
            logger.debug(f"Cached results for key {str(cache_key)[:50]}...")

        return results
