#!/usr/bin/env python3
"""
Phi-4 Multimodal ASR Model Implementation

This module provides a concrete implementation of the BaseMultimodalASRModel
using Microsoft's Phi-4 multimodal instruct model for audio transcription
and multimodal inference.
"""

import logging
import os
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

# Disable flash attention manually for Mac M4 compatibility
os.environ["FLASH_ATTENTION_DISABLE"] = "1"
os.environ["ATTN_IMPLEMENTATION"] = "eager"
os.environ["TRANSFORMERS_USE_FLASH_ATTENTION_2"] = "false"
os.environ["DISABLE_FLASH_ATTN"] = "1"

from .base_multimodal_asr_model import BaseMultimodalASRModel

logger = logging.getLogger(__name__)


class Phi4MultimodalASRModel(BaseMultimodalASRModel):
    def __init__(
        self,
        model_cache_dir: str = "/Users/Benjamin/dev/ssa/models",
        device: Optional[str] = None,
    ):
        super().__init__(model_cache_dir=model_cache_dir, device=device)

    def load_model(
        self, model_name: str = "microsoft/Phi-4-multimodal-instruct", **kwargs
    ) -> None:
        self.processor = AutoProcessor.from_pretrained(
            model_name, cache_dir=str(self.model_cache_dir), trust_remote_code=True
        )
        config = AutoConfig.from_pretrained(
            model_name, cache_dir=str(self.model_cache_dir), trust_remote_code=True
        )

        # Force disable flash attention
        if hasattr(config, "use_flash_attention_2"):
            config.use_flash_attention_2 = False
        if hasattr(config, "_flash_attn_2_enabled"):
            config._flash_attn_2_enabled = False
        if hasattr(config, "attn_implementation"):
            config.attn_implementation = "eager"
        model_kwargs = {
            "config": config,
            "cache_dir": str(self.model_cache_dir),
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
            "attn_implementation": "eager",  # Explicitly disable flash attention
            "low_cpu_mem_usage": True,
        }
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        def prepare_inputs_for_generation(self, input_ids, **kwargs):
            return {"input_ids": input_ids, **kwargs}

        self.model.model.prepare_inputs_for_generation = types.MethodType(
            prepare_inputs_for_generation, self.model.model
        )

    def transcribe(
        self,
        *,
        audio_path: str,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> str:
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
