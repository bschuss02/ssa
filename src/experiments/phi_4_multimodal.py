#!/usr/bin/env python3
"""
Phi-4 Multimodal Inference Script with Audio Support and Chat Completion
Supports chat completion messages + audio prompts with optional audio recording
Optimized for Mac M4 with manual flash attention disabling

Usage Examples:

1. Simple text input (backward compatibility):
   python phi_4_multimodal.py input.text="Hello, how are you?"

2. Chat completion messages:
   python phi_4_multimodal.py --config-name=chat_example

3. Multi-turn conversation:
   python phi_4_multimodal.py --config-name=conversation_example

4. Command line override with messages:
   python phi_4_multimodal.py input.messages='[{role: user, content: "Explain quantum computing"}]'

5. With system message:
   python phi_4_multimodal.py input.text="What is AI?" input.system_message="You are an expert in artificial intelligence."

6. With audio recording:
   python phi_4_multimodal.py recording.enabled=true recording.duration=5.0 input.text="Describe this audio"

Chat Message Format:
- Each message should have 'role' and 'content' fields
- Supported roles: 'system', 'user', 'assistant'
- System messages set behavior/context
- User messages are queries/inputs
- Assistant messages show previous responses in multi-turn conversations
"""

import os
import json
import time
import wave
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import logging

import hydra
from omegaconf import DictConfig, OmegaConf

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable flash attention manually for Mac M4 compatibility
os.environ["FLASH_ATTENTION_DISABLE"] = "1"
os.environ["ATTN_IMPLEMENTATION"] = "eager"
os.environ["TRANSFORMERS_USE_FLASH_ATTENTION_2"] = "false"
os.environ["DISABLE_FLASH_ATTN"] = "1"

try:
    import torch
    import torchaudio
    import sounddevice as sd
    import librosa
    from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
    from PIL import Image
    import scipy.io.wavfile as wavfile
except ImportError as e:
    logger.error(f"Missing required dependency: {e}")
    logger.error(
        "Please run: uv add torch torchvision torchaudio transformers sounddevice librosa pillow scipy accelerate numpy"
    )
    exit(1)

# Try to import bitsandbytes (optional for Mac M4)
try:
    from transformers import BitsAndBytesConfig

    BITSANDBYTES_AVAILABLE = True
except ImportError:
    logger.warning("bitsandbytes not available - quantization will be disabled (normal for Mac M4)")
    BITSANDBYTES_AVAILABLE = False
    BitsAndBytesConfig = None


class AudioRecorder:
    """Handle audio recording functionality"""

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels

    def record_audio(self, duration: float, output_path: Optional[str] = None) -> str:
        """Record audio for specified duration and save to file"""
        logger.info(f"Recording audio for {duration} seconds...")
        logger.info("Recording will start in 3 seconds...")
        time.sleep(3)

        # Record audio
        audio_data = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32,
        )

        logger.info("Recording started!")
        sd.wait()  # Wait until recording is finished
        logger.info("Recording finished!")

        # Save to file
        if output_path is None:
            timestamp = int(time.time())
            output_path = f"recorded_audio_{timestamp}.wav"

        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        # Convert to 16-bit integer and save
        audio_int16 = (audio_data * 32767).astype(np.int16)
        wavfile.write(output_path, self.sample_rate, audio_int16)

        logger.info(f"Audio saved to: {output_path}")
        return output_path


class Phi4MultimodalInference:
    """Handle Phi-4 multimodal model loading and inference"""

    def __init__(self, model_cache_dir: str = "/Users/Benjamin/dev/ssa/models", force_cpu: bool = False):
        self.model_cache_dir = Path(model_cache_dir)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        self.force_cpu = force_cpu

        self.model = None
        self.processor = None
        self.device = self._get_device()

    def _get_device(self) -> str:
        """Determine the best device to use"""
        if self.force_cpu:
            logger.info("Forcing CPU device usage (force_cpu=True)")
            return "cpu"

        try:
            if torch.backends.mps.is_available():
                # Test MPS functionality with a simple tensor operation
                test_tensor = torch.ones(1).to("mps")
                _ = test_tensor + 1
                logger.info("MPS device available and functional")
                return "mps"
        except Exception as e:
            logger.warning(f"MPS device available but not functional: {e}")

        if torch.cuda.is_available():
            return "cuda"
        else:
            logger.info("Using CPU device")
            return "cpu"

    def load_model(self, model_name: str = "microsoft/Phi-4-multimodal-instruct", use_4bit: bool = True):
        """Load the Phi-4 multimodal model with optimizations for Mac M4"""
        logger.info(f"Loading model: {model_name}")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Model cache directory: {self.model_cache_dir}")

        try:
            # Configure quantization for memory efficiency
            quantization_config = None
            if (
                use_4bit
                and self.device != "mps"
                and self.device != "cpu"  # Disable quantization for CPU
                and BITSANDBYTES_AVAILABLE
            ):  # 4-bit quantization doesn't work well with MPS or CPU
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            elif use_4bit and (not BITSANDBYTES_AVAILABLE or self.device in ["mps", "cpu"]):
                logger.warning(
                    f"4-bit quantization requested but not supported on {self.device} device or bitsandbytes not available - using full precision"
                )

            # Load processor
            logger.info("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                model_name, cache_dir=str(self.model_cache_dir), trust_remote_code=True
            )

            # Load config first and modify to disable flash attention
            logger.info("Loading model config...")
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(
                model_name, cache_dir=str(self.model_cache_dir), trust_remote_code=True
            )

            # Force disable flash attention in config
            if hasattr(config, "use_flash_attention_2"):
                config.use_flash_attention_2 = False
            if hasattr(config, "_flash_attn_2_enabled"):
                config._flash_attn_2_enabled = False
            if hasattr(config, "attn_implementation"):
                config.attn_implementation = "eager"

            # Load model with explicit flash attention disabling
            logger.info("Loading model...")
            model_kwargs = {
                "config": config,
                "cache_dir": str(self.model_cache_dir),
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
                "attn_implementation": "eager",  # Explicitly disable flash attention
                "low_cpu_mem_usage": True,
            }

            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config

            # Monkey patch to fix PEFT compatibility issue using import hook
            import types
            import sys
            import importlib.util

            # Create the missing method for PEFT compatibility
            def prepare_inputs_for_generation(self, input_ids, **kwargs):
                """Method required by PEFT for compatibility with generation"""
                # For most models, this just returns the input_ids and kwargs
                # The actual preparation is handled by the main model's prepare_inputs_for_generation
                model_kwargs = kwargs.copy()
                model_kwargs["input_ids"] = input_ids
                return model_kwargs

            logger.info("Attempting to patch Phi4MMModel for PEFT compatibility...")

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
                                phi4mm_class.prepare_inputs_for_generation = prepare_inputs_for_generation
                                logger.info(f"Successfully patched Phi4MMModel in module {module_name}")

                return module

            # Temporarily replace __import__
            import builtins

            builtins.__import__ = patching_import

            try:
                self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            except Exception as e:
                # Before re-raising, try one more direct patch attempt
                logger.warning(f"Model loading failed with: {e}")
                logger.info("Attempting direct module patching as last resort...")

                for module_name in list(sys.modules.keys()):
                    if "phi4mm" in module_name and "modeling" in module_name:
                        module_obj = sys.modules[module_name]
                        if hasattr(module_obj, "Phi4MMModel"):
                            phi4mm_class = getattr(module_obj, "Phi4MMModel")
                            if not hasattr(phi4mm_class, "prepare_inputs_for_generation"):
                                phi4mm_class.prepare_inputs_for_generation = prepare_inputs_for_generation
                                logger.info(f"Emergency patched Phi4MMModel in module {module_name}")

                                # Try loading again
                                try:
                                    self.model = AutoModelForCausalLM.from_pretrained(
                                        model_name, **model_kwargs
                                    )
                                    break
                                except Exception as e2:
                                    logger.error(f"Even after emergency patching, failed with: {e2}")
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
                logger.info("Adding missing prepare_inputs_for_generation method post-load...")

                def prepare_inputs_for_generation(self, input_ids, **kwargs):
                    """Fallback method for PEFT compatibility"""
                    return {"input_ids": input_ids, **kwargs}

                self.model.model.prepare_inputs_for_generation = types.MethodType(
                    prepare_inputs_for_generation, self.model.model
                )

            # Move to device with better handling for multimodal models
            try:
                # Always try to move to the target device for non-quantized models or MPS
                if self.device != "cpu":
                    logger.info(f"Moving model to {self.device} device...")
                    self.model = self.model.to(self.device)
                else:
                    logger.info("Model remaining on CPU device")

                # Verify model device
                if hasattr(self.model, "device"):
                    actual_device = str(self.model.device)
                elif hasattr(self.model, "parameters"):
                    actual_device = str(next(self.model.parameters()).device)
                else:
                    actual_device = "unknown"
                logger.info(f"Model device after loading: {actual_device}")

            except Exception as device_error:
                logger.warning(f"Failed to move model to {self.device}: {device_error}")
                logger.info("Falling back to CPU device")
                self.device = "cpu"
                self.model = self.model.to(self.device)

            # Set model to evaluation mode
            self.model.eval()

            # Disable flash attention in model config if it exists
            if hasattr(self.model.config, "use_flash_attention_2"):
                self.model.config.use_flash_attention_2 = False
            if hasattr(self.model.config, "_flash_attn_2_enabled"):
                self.model.config._flash_attn_2_enabled = False

            logger.info("Model loaded successfully!")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load and preprocess audio file"""
        try:
            # Load audio using librosa for better compatibility
            audio, sr = librosa.load(audio_path, sr=16000)  # Phi-4 typically expects 16kHz

            # Return as numpy array and sample rate tuple for processor
            return audio, sr

        except Exception as e:
            logger.error(f"Error loading audio from {audio_path}: {e}")
            raise

    def inference(
        self,
        messages: list[dict[str, str]],
        audio_path: Optional[str] = None,
        max_new_tokens: int = 500,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> str:
        """Run inference with chat completion messages and optional audio

        Args:
            messages: List of chat messages with 'role' and 'content' keys
                     e.g., [{"role": "user", "content": "Hello, how are you?"}]
            audio_path: Optional path to audio file
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling

        Returns:
            Generated response text
        """

        if self.model is None or self.processor is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        logger.info("Preparing inputs...")

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
            if audio_path:
                # Insert audio token in the appropriate place in the prompt
                # For Phi-4 multimodal, audio token should be in the user message
                if "<|user|>" in text_prompt and "<|end|>" in text_prompt:
                    # Find the user section and add audio token after <|user|>
                    user_start = text_prompt.find("<|user|>")
                    user_end = text_prompt.find("<|end|>", user_start)
                    if user_start != -1 and user_end != -1:
                        user_content = text_prompt[user_start + 8 : user_end]  # 8 is len("<|user|>")
                        # Add audio token at the beginning of user content
                        new_user_content = "<|audio_1|>" + user_content
                        text_prompt = (
                            text_prompt[: user_start + 8] + new_user_content + text_prompt[user_end:]
                        )

            logger.info(f"Formatted prompt: {text_prompt[:200]}...")

        except Exception as e:
            logger.warning(f"Error formatting chat template, using fallback: {e}")
            # Simple fallback formatting
            text_prompt = ""
            for message in messages:
                role = message.get("role", "user")
                content = message.get("content", "")
                if role == "user" and audio_path:
                    # Add audio token for user message when audio is provided
                    text_prompt += f"{role.capitalize()}: <|audio_1|>{content}\n"
                else:
                    text_prompt += f"{role.capitalize()}: {content}\n"
            text_prompt += "Assistant:"

        # Prepare inputs
        inputs = {"text": text_prompt}

        # Add audio if provided - use correct format: audios parameter with list of (audio, samplerate) tuples
        if audio_path:
            logger.info(f"Loading audio from: {audio_path}")
            audio, samplerate = self.load_audio(audio_path)
            inputs["audios"] = [(audio, samplerate)]  # Note: 'audios' plural and list of tuples

        # Process inputs
        try:
            processed_inputs = self.processor(**inputs, return_tensors="pt", padding=True)

            # Verify model device before moving inputs
            if hasattr(self.model, "device"):
                model_device = str(self.model.device)
            elif hasattr(self.model, "parameters"):
                model_device = str(next(self.model.parameters()).device)
            else:
                model_device = self.device

            logger.info(f"Model device: {model_device}, Target device: {self.device}")

            # Move inputs to the same device as the model
            target_device = model_device if model_device != "unknown" else self.device
            processed_inputs = {
                k: v.to(target_device) if isinstance(v, torch.Tensor) else v
                for k, v in processed_inputs.items()
            }

            logger.info("Running inference...")

            # Generate response
            with torch.no_grad():
                generation_config = GenerationConfig(
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,
                )

                # Add input_mode for multimodal model
                if audio_path:
                    processed_inputs["input_mode"] = 1  # SPEECH mode
                else:
                    processed_inputs["input_mode"] = 3  # LANGUAGE mode

                # Explicitly set num_logits_to_keep to avoid None error
                processed_inputs["num_logits_to_keep"] = 0

                outputs = self.model.generate(**processed_inputs, generation_config=generation_config)

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
            import re

            response = re.sub(r"<\|[^|]*\|>", "", response).strip()

            # Clean up duplicate system/user content that might leak through
            for message in messages:
                content = message.get("content", "")
                if content and content in response:
                    response = response.replace(content, "").strip()

            return response

        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise


@hydra.main(version_base=None, config_path="config", config_name="chat_example")
def main(cfg: DictConfig) -> None:
    """Main function for Phi-4 multimodal inference using Hydra configuration"""

    # Set logging level from config
    logging.getLogger().setLevel(getattr(logging, cfg.logging.level.upper()))

    # Validate required inputs
    text_input = cfg.input.get("text")
    messages_input = cfg.input.get("messages")

    if text_input is None and messages_input is None:
        raise ValueError(
            "Either text input or messages are required. "
            "Set input.text='your text' or input.messages=[{role: 'user', content: 'your text'}]"
        )

    try:
        # Print configuration
        logger.info("Configuration:")
        logger.info(OmegaConf.to_yaml(cfg))

        # Initialize components
        logger.info("Initializing Phi-4 Multimodal Inference...")
        phi4 = Phi4MultimodalInference(
            model_cache_dir=cfg.model.cache_dir, force_cpu=cfg.device.get("force_cpu", False)
        )

        # Load model
        phi4.load_model(cfg.model.name, use_4bit=cfg.model.use_4bit)

        # Handle audio
        audio_path = cfg.input.audio_path
        if cfg.recording.enabled:
            logger.info("Audio recording mode enabled")
            recorder = AudioRecorder(sample_rate=cfg.recording.sample_rate, channels=cfg.recording.channels)
            audio_path = recorder.record_audio(
                duration=cfg.recording.duration, output_path=cfg.recording.output_path
            )

            # Prepare messages for inference
        messages = []
        if messages_input:
            # Use provided messages directly
            messages = messages_input
            logger.info(f"Using provided messages: {len(messages)} message(s)")
        elif text_input:
            # Convert text to messages format
            messages = [{"role": "user", "content": text_input}]
            logger.info(f"Converted text to user message: {text_input[:100]}...")

        # Add system message if provided
        system_message = cfg.input.get("system_message")
        if system_message:
            messages.insert(0, {"role": "system", "content": system_message})
            logger.info("Added system message")

        # Run inference
        logger.info("Starting inference...")
        response = phi4.inference(
            messages=messages,
            audio_path=audio_path,
            max_new_tokens=cfg.generation.max_new_tokens,
            temperature=cfg.generation.temperature,
            do_sample=cfg.generation.do_sample,
        )

        # Display results
        print("\n" + "=" * 50)
        print("INFERENCE RESULTS")
        print("=" * 50)

        # Display the conversation
        print("Messages:")
        for i, message in enumerate(messages):
            role = message.get("role", "unknown")
            content = message.get("content", "")
            print(f"  {i+1}. {role.capitalize()}: {content}")

        if audio_path:
            print(f"Audio File: {audio_path}")

        print(f"\nModel Response:\n{response}")
        print("=" * 50)

        # Save results to file
        if cfg.logging.save_results:
            # Convert OmegaConf objects to regular Python objects for JSON serialization
            messages_dict = []
            for msg in messages:
                if hasattr(msg, "keys"):  # OmegaConf DictConfig
                    messages_dict.append(dict(msg))
                else:
                    messages_dict.append(msg)

            results = {
                "messages": messages_dict,
                "audio_path": audio_path,
                "model_response": response,
                "timestamp": time.time(),
                "model": cfg.model.name,
                "config": OmegaConf.to_container(cfg, resolve=True),
            }

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
