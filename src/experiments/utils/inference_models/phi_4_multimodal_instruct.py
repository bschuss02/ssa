from typing import Dict, List, Optional, Tuple

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
        """
        Initialize the Phi4MultimodalASRModel.
        Args:
            model_path (str): Path to the pretrained model.
            device (Optional[str]): Device to load the model on.
        """
        super().__init__(model_path=model_path, device=device)
        self.model = None
        self.processor = None
        self.device = device

    def load_model(self, **kwargs) -> None:
        """
        Load the processor and model from the specified model path.
        """
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
        """
        Transcribe the given audio file using the model and a prompt built from messages.
        Args:
            audio_path (str): Path to the audio file.
            messages (Optional[List[Dict[str, str]]]): List of message dicts to build the prompt.
        Returns:
            str: The transcription result.
        """
        if messages is None:
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert audio transcriptionist.",
                },
            ]
        audio_data, sample_rate = self._load_audio(audio_path)
        prompt = self._build_prompt_from_messages(messages)
        print(f"Prompt: {prompt}")
        inputs = self._prepare_inputs(prompt, audio_data, sample_rate)
        response = self._generate_transcription(inputs)
        print(response)
        return response

    def _build_prompt_from_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Build a prompt string from a list of message dicts.
        Args:
            messages (List[Dict[str, str]]): List of messages with 'role' and 'content'.
                Users can include '<|audio_1|>' directly in message content for audio placement.
        Returns:
            str: The constructed prompt string.
        """
        user_prompt = "<|user|>"
        assistant_prompt = "<|assistant|>"
        prompt_suffix = "<|end|>"
        prompt = ""

        for msg in messages:
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

    def _load_audio(self, audio_path: str) -> Tuple:
        """
        Load audio data and sample rate from a file.
        Args:
            audio_path (str): Path to the audio file.
        Returns:
            Tuple: (audio_data, sample_rate)
        """
        return sf.read(audio_path)

    def _prepare_inputs(self, prompt: str, audio_data, sample_rate) -> Dict:
        """
        Prepare model inputs from prompt and audio data.
        Args:
            prompt (str): The prompt string.
            audio_data: The audio data array.
            sample_rate: The sample rate of the audio.
        Returns:
            Dict: Model-ready input tensors.
        """
        if self.processor is None:
            raise Exception("Processor not loaded")
        return self.processor(
            text=prompt, audios=[(audio_data, sample_rate)], return_tensors="pt"
        ).to("mps")

    def _generate_transcription(self, inputs: Dict) -> str:
        """
        Generate transcription from model inputs.
        Args:
            inputs (Dict): Model-ready input tensors.
        Returns:
            str: The transcription result.
        """
        if self.model is None:
            raise Exception("Model not loaded")
        if self.processor is None:
            raise Exception("Processor not loaded")
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
        return response
