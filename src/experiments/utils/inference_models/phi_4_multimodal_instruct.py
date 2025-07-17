#!/usr/bin/env python3
"""
Phi-4 Multimodal ASR Model Implementation

This module provides a concrete implementation of the BaseMultimodalASRModel
using Microsoft's Phi-4 multimodal instruct model for audio transcription
and multimodal inference.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Disable flash attention manually for Mac M4 compatibility
os.environ["FLASH_ATTENTION_DISABLE"] = "1"
os.environ["ATTN_IMPLEMENTATION"] = "eager"
os.environ["TRANSFORMERS_USE_FLASH_ATTENTION_2"] = "false"
os.environ["DISABLE_FLASH_ATTN"] = "1"

from .base_multimodal_asr_model import BaseMultimodalASRModel

# Set up logging
logger = logging.getLogger(__name__)


class Phi4MultimodalASRModel(BaseMultimodalASRModel):
    """
    Phi-4 Multimodal ASR Model implementation.

    This class implements the BaseMultimodalASRModel interface using
    Microsoft's Phi-4 multimodal instruct model for audio transcription
    and multimodal understanding.
    """

    def __init__(
        self,
        model_cache_dir: str = "./models",
        force_cpu: bool = False,
        device: Optional[str] = None,
    ):
        """
        Initialize the Phi-4 multimodal ASR model.

        Args:
            model_cache_dir: Directory to cache downloaded models
            force_cpu: Force CPU usage even if GPU is available
            device: Specific device to use (overrides auto-detection)
        """
        super().__init__(model_cache_dir, force_cpu, device)
        self._phi4_inference = None
        self._model_loaded = False

    def load_model(
        self, model_name: str = "microsoft/Phi-4-multimodal-instruct", **kwargs
    ) -> None:
        """
        Load the Phi-4 multimodal model and processor.

        Args:
            model_name: Name or path of the Phi-4 model to load
            **kwargs: Additional model-specific arguments:
                - use_4bit: Whether to use 4-bit quantization (default: True)
        """
        try:
            # Import the Phi4MultimodalInference class
            # We need to import it here to avoid circular imports and handle the sys.path correctly
            import sys
            from pathlib import Path

            # Add the parent directory to sys.path to import the inference module
            inference_dir = Path(__file__).parent.parent
            if str(inference_dir) not in sys.path:
                sys.path.insert(0, str(inference_dir))

            from inference_phi_4_multimodal import Phi4MultimodalInference

            logger.info(
                f"Initializing Phi-4 multimodal inference with model: {model_name}"
            )

            # Initialize the Phi4MultimodalInference with our parameters
            self._phi4_inference = Phi4MultimodalInference(
                model_cache_dir=str(self.model_cache_dir), force_cpu=self.force_cpu
            )

            # Load the model
            use_4bit = kwargs.get("use_4bit", True)
            self._phi4_inference.load_model(model_name=model_name, use_4bit=use_4bit)

            self._model_loaded = True
            logger.info("Phi-4 multimodal model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Phi-4 multimodal model: {e}")
            raise

    def transcribe(
        self, audio_path: str, messages: Optional[List[Dict[str, str]]] = None, **kwargs
    ) -> str:
        """
        Transcribe audio with optional chat context using Phi-4 multimodal model.

        Args:
            audio_path: Path to the audio file to transcribe
            messages: Optional list of chat messages with 'role' and 'content' keys
                     If None, will create a default transcription request
            **kwargs: Additional transcription parameters:
                - max_new_tokens: Maximum number of tokens to generate (default: 500)
                - temperature: Sampling temperature (default: 0.7)
                - do_sample: Whether to use sampling (default: True)
                - system_message: Optional system message to set behavior

        Returns:
            Transcribed text from the audio
        """
        if not self._model_loaded or self._phi4_inference is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            # Prepare messages for transcription
            if messages is None:
                # Create default transcription request
                system_message = kwargs.get(
                    "system_message",
                    "You are an expert audio transcriptionist. Transcribe the provided audio accurately.",
                )
                messages = [
                    {"role": "system", "content": system_message},
                    {
                        "role": "user",
                        "content": "Please transcribe this audio file accurately.",
                    },
                ]
            else:
                # Use provided messages - ensure they're properly formatted
                formatted_messages = []
                for msg in messages:
                    if (
                        not isinstance(msg, dict)
                        or "role" not in msg
                        or "content" not in msg
                    ):
                        raise ValueError(
                            "Each message must be a dict with 'role' and 'content' keys"
                        )
                    formatted_messages.append(
                        {"role": msg["role"], "content": msg["content"]}
                    )
                messages = formatted_messages

            # Extract generation parameters
            max_new_tokens = kwargs.get("max_new_tokens", 500)
            temperature = kwargs.get("temperature", 0.7)
            do_sample = kwargs.get("do_sample", True)

            logger.info(f"Transcribing audio: {audio_path}")

            # Run inference using the Phi-4 multimodal model
            response = self._phi4_inference.inference(
                messages=messages,
                audio_path=audio_path,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
            )

            logger.info("Audio transcription completed successfully")
            return response

        except Exception as e:
            logger.error(f"Failed to transcribe audio {audio_path}: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        if not self.is_loaded:
            return {"loaded": False}

        return {
            "loaded": True,
            "model_cache_dir": str(self.model_cache_dir),
            "device": self._phi4_inference.device
            if self._phi4_inference
            else "unknown",
            "force_cpu": self.force_cpu,
        }
