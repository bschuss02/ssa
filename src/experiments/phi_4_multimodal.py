#!/usr/bin/env python3
"""
Phi-4 Multimodal Inference Script with Audio Support
Supports text + audio prompts with optional audio recording
Optimized for Mac M4 with manual flash attention disabling
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

            # Monkey patch to fix PEFT compatibility issue
            from transformers.generation.utils import GenerationMixin
            import types

            # Patch the Phi4MMModel class before loading
            try:
                # Import the model module
                import importlib.util
                import sys
                from transformers.models.auto.modeling_auto import _get_model_class

                # Create a dummy prepare_inputs_for_generation method
                def dummy_prepare_inputs_for_generation(self, input_ids, **kwargs):
                    """Dummy method for PEFT compatibility"""
                    return {"input_ids": input_ids, **kwargs}

                # Try to patch the model class before instantiation
                logger.info("Patching model class for PEFT compatibility...")

                # Pre-emptively patch any model classes that might be loaded
                try:
                    # Load the model without instantiating to get the class
                    from transformers import AutoConfig

                    temp_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

                    # Get the model class
                    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING

                    model_class = AutoModelForCausalLM._get_model_class(
                        temp_config, MODEL_FOR_CAUSAL_LM_MAPPING
                    )

                    # Patch any nested model classes
                    if hasattr(model_class, "__module__") and "phi4mm" in model_class.__module__:
                        logger.info(f"Patching {model_class.__name__}...")

                        # Try to get the nested model class
                        module = importlib.import_module(model_class.__module__)
                        if hasattr(module, "Phi4MMModel"):
                            phi4mm_class = getattr(module, "Phi4MMModel")
                            if not hasattr(phi4mm_class, "prepare_inputs_for_generation"):
                                phi4mm_class.prepare_inputs_for_generation = (
                                    dummy_prepare_inputs_for_generation
                                )
                                logger.info("Successfully patched Phi4MMModel class")

                except Exception as patch_error:
                    logger.warning(f"Pre-patch failed, will try post-load patch: {patch_error}")

            except Exception as e:
                logger.warning(f"Could not pre-patch model class: {e}")

            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

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
                if use_4bit and self.device == "mps" and not BITSANDBYTES_AVAILABLE:
                    # For MPS without bitsandbytes, we need to be more careful
                    logger.info("Moving model to MPS device...")
                    self.model = self.model.to(self.device)
                elif not use_4bit:
                    # For non-quantized models, move normally
                    logger.info(f"Moving model to {self.device} device...")
                    self.model = self.model.to(self.device)
                else:
                    # For other cases (CUDA with quantization), the model is already on the right device
                    logger.info(f"Model will remain on current device (quantization enabled)")

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

    def load_audio(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio file"""
        try:
            # Load audio using librosa for better compatibility
            audio, sr = librosa.load(audio_path, sr=16000)  # Phi-4 typically expects 16kHz

            # Convert to tensor
            audio_tensor = torch.from_numpy(audio).float()

            # Ensure it's the right shape (add batch dimension if needed)
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            return audio_tensor

        except Exception as e:
            logger.error(f"Error loading audio from {audio_path}: {e}")
            raise

    def inference(
        self,
        text_prompt: str,
        audio_path: Optional[str] = None,
        max_new_tokens: int = 500,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> str:
        """Run inference with text and optional audio"""

        if self.model is None or self.processor is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        logger.info("Preparing inputs...")

        # Prepare inputs
        inputs = {"text": text_prompt}

        # Add audio if provided
        if audio_path:
            logger.info(f"Loading audio from: {audio_path}")
            audio_tensor = self.load_audio(audio_path)
            inputs["audio"] = audio_tensor

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

            return response

        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise


@hydra.main(version_base=None, config_path="config", config_name="phi4_config")
def main(cfg: DictConfig) -> None:
    """Main function for Phi-4 multimodal inference using Hydra configuration"""

    # Set logging level from config
    logging.getLogger().setLevel(getattr(logging, cfg.logging.level.upper()))

    # Validate required inputs
    if cfg.input.text is None:
        raise ValueError(
            "Text input is required. Set input.text in config or override with input.text='your text'"
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

        # Run inference
        logger.info("Starting inference...")
        response = phi4.inference(
            text_prompt=cfg.input.text,
            audio_path=audio_path,
            max_new_tokens=cfg.generation.max_new_tokens,
            temperature=cfg.generation.temperature,
            do_sample=cfg.generation.do_sample,
        )

        # Display results
        print("\n" + "=" * 50)
        print("INFERENCE RESULTS")
        print("=" * 50)
        print(f"Text Prompt: {cfg.input.text}")
        if audio_path:
            print(f"Audio File: {audio_path}")
        print(f"Model Response:\n{response}")
        print("=" * 50)

        # Save results to file
        if cfg.logging.save_results:
            results = {
                "text_prompt": cfg.input.text,
                "audio_path": audio_path,
                "model_response": response,
                "timestamp": time.time(),
                "model": cfg.model.name,
                "config": OmegaConf.to_container(cfg, resolve=True),
            }

            output_file = cfg.logging.results_file or f"inference_results_{int(time.time())}.json"
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)

            logger.info(f"Results saved to: {output_file}")

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
