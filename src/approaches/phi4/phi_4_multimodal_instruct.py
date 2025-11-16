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
        self.local_model_path = (
            "/home/benji/dev/ssa/data/downloaded_models/Phi-4-multimodal-instruct"
        )
        self.prompt_messages = [
            {
                "role": "system",
                "content": "You are an expert audio transcriptionist.",
            },
            {
                "role": "user",
                "content": "Transcribe the speech from this audio recording. The language is English. <|audio_1|>",
            },
        ]

    def load_model(self):
        logger.info(f"Loading model {self.model_name} to {self.device}")
        logger.info(f"Local model path: {self.local_model_path}")
        self.processor = AutoProcessor.from_pretrained(
            self.local_model_path, trust_remote_code=True
        )

        # Use mixed precision (float16) for memory efficiency and speed
        # bfloat16 is also an option if your GPU supports it (better numerical stability)
        if self.device == "cuda":
            # Check if bfloat16 is supported, otherwise use float16
            if torch.cuda.is_bf16_supported():
                torch_dtype = torch.bfloat16
                logger.info("Using bfloat16 precision (better numerical stability)")
            else:
                torch_dtype = torch.float16
                logger.info("Using float16 precision (memory efficient)")
        else:
            torch_dtype = torch.float32
            logger.info("Using float32 precision (CPU/MPS)")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.local_model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map="auto",  # Let transformers optimize device placement
            low_cpu_mem_usage=True,  # Reduce peak memory during loading
        )

        # Enable torch.compile for faster inference (PyTorch 2.0+)
        # This can significantly speed up inference
        try:
            if hasattr(torch, "compile") and self.device == "cuda":
                logger.info("Compiling model with torch.compile for faster inference")
                self.model = torch.compile(self.model, mode="reduce-overhead")
        except Exception as e:
            logger.warning(f"Could not compile model: {e}. Continuing without compilation.")

        self.model.eval()  # Set to evaluation mode

    def transcribe(
        self,
        transcription_inputs: List[TranscriptionInput],
    ) -> List[TranscriptionOutput]:
        audio_paths = [ti.audio_path for ti in transcription_inputs]
        missing = [p for p in audio_paths if not p.exists()]
        if missing:
            raise ValueError(f"Audio file(s) do not exist: {missing}")

        target_sample_rate = 16000
        audio_arrays, _ = load_audio_files(
            audio_paths, max_workers=self.cfg.max_workers, sr=target_sample_rate
        )

        prompt_string = self._build_prompt_string_from_messages(self.prompt_messages)
        inputs = self._prepare_inputs(prompt_string, audio_arrays, target_sample_rate)

        # Use inference_mode() instead of no_grad() for better performance
        with torch.inference_mode():
            transcriptions = self._generate_outputs(inputs)

        return [
            TranscriptionOutput(
                transcription=transcription,
                metadata=input.metadata or {},
            )
            for transcription, input in zip(transcriptions, transcription_inputs)
        ]

    def _build_prompt_string_from_messages(self, prompt_messages: List[Dict[str, str]]) -> str:
        system_parts = []
        user_parts = []
        assistant_parts = []

        for msg in prompt_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_parts.append(content)
            elif role == "user":
                user_parts.append(f"<|user|>{content}<|end|>")
            elif role == "assistant":
                assistant_parts.append(f"<|assistant|>{content}<|end|>")

        prompt = "\n".join(system_parts) + "".join(user_parts) + "".join(assistant_parts)
        if not prompt.strip().endswith("<|assistant|>"):
            prompt += "<|assistant|>"

        return prompt

    def _prepare_inputs(
        self, prompt_string: str, audio_arrays: List[np.ndarray], sample_rate: int
    ) -> Dict[str, Any]:
        processed_audio_arrays = []
        for i, audio_data in enumerate(audio_arrays):
            if audio_data.size == 0 or audio_data.shape[0] < 10:
                logger.warning(
                    f"Problematic audio at index {i} (shape: {audio_data.shape}), padding"
                )
                channels = 2 if audio_data.ndim == 1 else audio_data.shape[1]
                processed_audio = np.zeros((100, channels), dtype=audio_data.dtype)
            elif audio_data.ndim == 1:
                processed_audio = np.stack([audio_data, audio_data], axis=1)
                logger.debug(
                    f"Converted mono to stereo: {audio_data.shape} -> {processed_audio.shape}"
                )
            else:
                processed_audio = audio_data
            processed_audio_arrays.append(processed_audio)

        audio_tuples = [(arr, sample_rate) for arr in processed_audio_arrays]

        try:
            return self.processor(
                text=[prompt_string] * len(audio_arrays),
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
        generation_config = GenerationConfig.from_pretrained(self.local_model_path)

        # Optimize generation settings for speed and memory
        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.cfg.max_output_tokens,
            generation_config=generation_config,
            num_logits_to_keep=1,
            use_cache=True,  # Enable KV cache for faster generation
            do_sample=False,  # Use greedy decoding (faster than sampling)
            pad_token_id=self.processor.tokenizer.pad_token_id
            or self.processor.tokenizer.eos_token_id,
        )

        prompt_length = inputs["input_ids"].shape[1]
        output_ids = generate_ids[:, prompt_length:]
        return self.processor.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
