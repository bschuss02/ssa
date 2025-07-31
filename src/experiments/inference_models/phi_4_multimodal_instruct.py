from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

from experiments.inference_models.asr_model_base import ASRModelBase
from experiments.utils.evaluation_result import EvaluationResult


class Phi4MultimodalInstruct(ASRModelBase):
    model: Any
    processor: Any

    def __init__(self, model_name: Path, model_dir: Path):
        super().__init__(model_name, model_dir)
        self.model = None
        self.processor = None
        self._log = getLogger(__name__)

    def load_model(self):
        self._log.info(f"Loading model {self.model_name} from {self.model_dir}")
        self.processor = AutoProcessor.from_pretrained(
            self.model_dir, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map=self.device,
            _attn_implementation="eager",  # Disable Flash Attention for Mac M4 chip
        )

    def transcribe(
        self,
        audio_arrays: List[np.ndarray],
        sample_rate: int,
        prompt_messages: List[Dict[str, str]],
    ) -> List[str]:
        prompt_string = self._build_prompt_string_from_messages(prompt_messages)
        inputs = self._prepare_inputs(prompt_string, audio_arrays, sample_rate)
        outputs = self._generate_outputs(inputs)
        return outputs

    def _build_prompt_string_from_messages(
        self, prompt_messages: List[Dict[str, str]]
    ) -> str:
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
        audio_tuples = [
            (audio_data, sample_rate)
            for audio_data, sample_rate in zip(audio_arrays, [sample_rate] * batch_size)
        ]
        return self.processor(
            text=[prompt_string] * batch_size,
            audios=audio_tuples,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

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
