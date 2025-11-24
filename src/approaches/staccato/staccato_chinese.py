from approaches.staccato.staccato import Staccato
from experiments.config.evaluation_config import EvaluationConfig


class StaccatoChinese(Staccato):
    def __init__(
        self,
        model_name: str,
        cfg: EvaluationConfig,
        whisper_model_id: str = "openai/whisper-medium",
        lm: str = "gpt-4o-mini-audio-preview-2024-12-17",
    ):
        super().__init__(
            model_name=model_name,
            cfg=cfg,
            language="zh",
            whisper_model_id=whisper_model_id,
            lm=lm,
        )
