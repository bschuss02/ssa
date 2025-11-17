from typing import Optional

from approaches.whisper.whisper_v3_medium import WhisperV3Medium
from experiments.config.evaluation_config import EvaluationConfig


class WhisperV3MediumEnglish(WhisperV3Medium):
    def __init__(
        self,
        model_name: str,
        cfg: EvaluationConfig,
        model_id: str = "openai/whisper-medium.en",
        prompt: Optional[str] = None,
    ):
        super().__init__(
            model_name=model_name,
            cfg=cfg,
            model_id=model_id,
            prompt=prompt,
            language="en",
        )
