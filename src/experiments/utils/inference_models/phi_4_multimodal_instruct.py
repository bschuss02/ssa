#!/usr/bin/env python3
"""
Minimal Phi-4 Multimodal ASR Model Implementation
"""

import os
import re
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
)
from transformers.generation.configuration_utils import GenerationConfig

from .base_multimodal_asr_model import BaseMultimodalASRModel

# Try to import librosa
try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class Phi4MultimodalASRModel(BaseMultimodalASRModel):
    def __init__(
        self,
        model_cache_dir: str = "/Users/Benjamin/dev/ssa/models",
        device: Optional[str] = None,
    ):
        super().__init__(model_cache_dir=model_cache_dir, device=device)
        self.model = None
        self.processor = None

    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, float]:
        """Load and preprocess audio file"""
        if not LIBROSA_AVAILABLE:
            raise ImportError(
                "librosa is required for audio processing. Install with: pip install librosa"
            )

        try:
            # Load audio using librosa for better compatibility
            audio, sr = librosa.load(
                audio_path, sr=16000
            )  # Phi-4 typically expects 16kHz

            # Return as numpy array and sample rate tuple for processor
            return audio, sr

        except Exception as e:
            raise RuntimeError(f"Error loading audio from {audio_path}: {e}")

    def load_model(
        self, model_name: str = "microsoft/Phi-4-multimodal-instruct", **kwargs
    ) -> None:
        # Create the missing method for PEFT compatibility
        def prepare_inputs_for_generation(self, input_ids, **kwargs):
            """Method required by PEFT for compatibility with generation"""
            # For most models, this just returns the input_ids and kwargs
            # The actual preparation is handled by the main model's prepare_inputs_for_generation
            model_kwargs = kwargs.copy()
            model_kwargs["input_ids"] = input_ids
            return model_kwargs

        self.processor = AutoProcessor.from_pretrained(
            model_name, cache_dir=str(self.model_cache_dir), trust_remote_code=True
        )
        config = AutoConfig.from_pretrained(
            model_name, cache_dir=str(self.model_cache_dir), trust_remote_code=True
        )
        # Force disable flash attention for Mac compatibility
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
            "attn_implementation": "eager",
            "low_cpu_mem_usage": True,
        }

        # Monkey patch to fix PEFT compatibility issue using import hook
        import builtins
        import importlib.util

        # Simple and direct approach: patch using monkey patching during import
        original_import = __import__

        def patching_import(*args, **kwargs):
            module = original_import(*args, **kwargs)

            # Check all newly imported modules
            for module_name in list(sys.modules.keys()):
                if "phi4mm" in module_name and "modeling" in module_name:
                    module_obj = sys.modules[module_name]
                    if hasattr(module_obj, "Phi4MMModel"):
                        phi4mm_class = getattr(module_obj, "Phi4MMModel")
                        if not hasattr(phi4mm_class, "prepare_inputs_for_generation"):
                            phi4mm_class.prepare_inputs_for_generation = (
                                prepare_inputs_for_generation
                            )

            return module

        # Temporarily replace __import__
        builtins.__import__ = patching_import

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, **model_kwargs
            )
        except Exception as e:
            # Before re-raising, try one more direct patch attempt
            for module_name in list(sys.modules.keys()):
                if "phi4mm" in module_name and "modeling" in module_name:
                    module_obj = sys.modules[module_name]
                    if hasattr(module_obj, "Phi4MMModel"):
                        phi4mm_class = getattr(module_obj, "Phi4MMModel")
                        if not hasattr(phi4mm_class, "prepare_inputs_for_generation"):
                            phi4mm_class.prepare_inputs_for_generation = (
                                prepare_inputs_for_generation
                            )

                            # Try loading again
                            try:
                                self.model = AutoModelForCausalLM.from_pretrained(
                                    model_name, **model_kwargs
                                )
                                break
                            except Exception as e2:
                                continue
            else:
                # Re-raise original exception if patching didn't help
                raise e
        finally:
            # Restore original import
            builtins.__import__ = original_import

        # Post-load patch as backup
        if hasattr(self.model, "model") and not hasattr(
            self.model.model, "prepare_inputs_for_generation"
        ):

            def prepare_inputs_for_generation_fallback(self, input_ids, **kwargs):
                """Fallback method for PEFT compatibility"""
                return {"input_ids": input_ids, **kwargs}

            self.model.model.prepare_inputs_for_generation = types.MethodType(
                prepare_inputs_for_generation_fallback, self.model.model
            )

        if self.device:
            self.model = self.model.to(self.device)
        else:
            self.model = self.model.to("cuda" if torch.cuda.is_available() else "cpu")

    def transcribe(
        self,
        *,
        audio_path: str,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> str:
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if messages is None:
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert audio transcriptionist.",
                },
                {
                    "role": "user",
                    "content": "Please transcribe this audio file accurately.",
                },
            ]

        max_new_tokens = kwargs.get("max_new_tokens", 500)
        temperature = kwargs.get("temperature", 0.7)
        do_sample = kwargs.get("do_sample", True)

        # Format messages using the processor's chat template
        try:
            # Convert messages to text using chat template
            if hasattr(self.processor, "tokenizer") and hasattr(
                self.processor.tokenizer, "apply_chat_template"
            ):
                text_prompt = self.processor.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                # Fallback: manually format messages
                formatted_messages = []
                for message in messages:
                    role = message.get("role", "user")
                    content = message.get("content", "")
                    if role == "system":
                        formatted_messages.append(f"System: {content}")
                    elif role == "user":
                        formatted_messages.append(f"User: {content}")
                    elif role == "assistant":
                        formatted_messages.append(f"Assistant: {content}")
                    else:
                        formatted_messages.append(f"{role}: {content}")

                text_prompt = "\n".join(formatted_messages)
                if not text_prompt.endswith("Assistant:"):
                    text_prompt += "\nAssistant:"

            # Add audio placeholder token if audio is provided
            if "<|user|>" in text_prompt and "<|end|>" in text_prompt:
                # Find the user section and add audio token after <|user|>
                user_start = text_prompt.find("<|user|>")
                user_end = text_prompt.find("<|end|>", user_start)
                if user_start != -1 and user_end != -1:
                    user_content = text_prompt[
                        user_start + 8 : user_end
                    ]  # 8 is len("<|user|>")
                    # Add audio token at the beginning of user content
                    new_user_content = "<|audio_1|>" + user_content
                    text_prompt = (
                        text_prompt[: user_start + 8]
                        + new_user_content
                        + text_prompt[user_end:]
                    )

        except Exception as e:
            # Simple fallback formatting
            text_prompt = ""
            for message in messages:
                role = message.get("role", "user")
                content = message.get("content", "")
                if role == "user":
                    # Add audio token for user message when audio is provided
                    text_prompt += f"{role.capitalize()}: <|audio_1|>{content}\n"
                else:
                    text_prompt += f"{role.capitalize()}: {content}\n"
            text_prompt += "Assistant:"

        # Prepare inputs
        inputs = {"text": text_prompt}

        # Add audio - use correct format: audios parameter with list of (audio, samplerate) tuples
        audio, samplerate = self.load_audio(audio_path)
        inputs["audios"] = [(audio, samplerate)]

        # Process inputs
        processed_inputs = self.processor(**inputs, return_tensors="pt", padding=True)

        # Move inputs to the same device as the model
        device = next(self.model.parameters()).device
        processed_inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in processed_inputs.items()
        }

        # Add input_mode for multimodal model
        processed_inputs["input_mode"] = 1  # SPEECH mode
        processed_inputs["num_logits_to_keep"] = 0

        with torch.no_grad():
            generation_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
            )

            outputs = self.model.generate(
                **processed_inputs, generation_config=generation_config
            )

        # Decode response
        response = self.processor.decode(outputs[0], skip_special_tokens=True)

        # Remove the input prompt from the response
        if text_prompt in response:
            response = response.replace(text_prompt, "").strip()

        # More aggressive cleaning for chat templates
        # Remove common chat template artifacts
        chat_prefixes = [
            "Assistant:",
            "<|assistant|>",
            "System:",
            "<|system|>",
            "User:",
            "<|user|>",
            "<|end|>",
        ]

        for prefix in chat_prefixes:
            if response.startswith(prefix):
                response = response[len(prefix) :].strip()

        # Remove any remaining template tokens
        response = re.sub(r"<\|[^|]*\|>", "", response).strip()

        # Clean up duplicate system/user content that might leak through
        for message in messages:
            content = message.get("content", "")
            if content and content in response:
                response = response.replace(content, "").strip()

        return response.strip()
