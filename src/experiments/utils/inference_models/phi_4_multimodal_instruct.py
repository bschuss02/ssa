from typing import Dict, List, Optional

import soundfile as sf
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.generation.configuration_utils import GenerationConfig

from .base_multimodal_asr_model import BaseMultimodalASRModel


class Phi4MultimodalASRModel(BaseMultimodalASRModel):
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
    ):
        super().__init__(model_path=model_path, device=device)
        self.model = None
        self.processor = None
        self.device = device

    def load_model(
        self,
        **kwargs,
    ) -> None:
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="mps",
            _attn_implementation="eager",  # Disable Flash Attention for Mac M4 chip
        )

    def transcribe(
        self,
        *,
        audio_path: str,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> str:
        if messages is None:
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert audio transcriptionist.",
                },
            ]
        audio_data, sample_rate = sf.read(audio_path)

        user_prompt = "<|user|>"
        assistant_prompt = "<|assistant|>"
        prompt_suffix = "<|end|>"

        speech_prompt = "Based on the attached audio, transcribe the spoken content."
        prompt = (
            f"{user_prompt}<|audio_1|>{speech_prompt}{prompt_suffix}{assistant_prompt}"
        )

        # Prepare input
        if self.processor is None:
            raise Exception("Processor not loaded")
        inputs = self.processor(
            text=prompt, audios=[(audio_data, sample_rate)], return_tensors="pt"
        ).to("mps")

        # Run generation
        if self.model is None:
            raise Exception("Model not loaded")
        generation_config = GenerationConfig.from_pretrained(self.model_path)
        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=1200,
            generation_config=generation_config,
            num_logits_to_keep=1,  # Fix for NoneType error
        )
        # Remove prompt tokens from output
        output_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
        response = self.processor.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        print(response)
        return response
