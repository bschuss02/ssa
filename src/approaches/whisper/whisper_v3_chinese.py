from typing import Optional

from approaches.whisper.whisper_v3_medium_english import WhisperV3MediumEnglish
from experiments.config.evaluation_config import EvaluationConfig


class WhisperV3MediumChinese(WhisperV3MediumEnglish):
    def __init__(
        self,
        model_name: str,
        cfg: EvaluationConfig,
        model_id: str = "openai/whisper-medium",
        prompt: Optional[str] = None,
    ):
        super().__init__(
            model_name=model_name,
            cfg=cfg,
            model_id=model_id,
            prompt=prompt,
            language="zh",
        )
