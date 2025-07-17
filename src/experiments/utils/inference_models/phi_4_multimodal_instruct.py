from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from transformers import AutoModelForCausalLM, AutoProcessor

from .base_multimodal_asr_model import BaseMultimodalASRModel


class Phi4MultimodalASRModel(BaseMultimodalASRModel):
    def __init__(
        self,
        model_cache_dir: str = "/Users/Benjamin/dev/ssa/models",
        device: Optional[str] = None,
    ):
        super().__init__(model_cache_dir=model_cache_dir, device=device)
        self.model = None
        self.processor = None

    def load_model(
        self,
        model_path: str = "/Users/Benjamin/dev/ssa/models/Phi-4-multimodal-instruct",
        **kwargs,
    ) -> None:
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="mps",
            _attn_implementation="eager",  # Disable Flash Attention
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
