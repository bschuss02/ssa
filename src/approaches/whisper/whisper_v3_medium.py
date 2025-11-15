from pathlib import Path

import whisper

from experiments.config.evaluation_config import EvaluationConfig
from experiments.inference_models.asr_model_base import ASRModelBase


class WhisperV3Medium(ASRModelBase):
    def __init__(self, model_name: Path, model_dir: Path, cfg: EvaluationConfig):
        super().__init__(model_name, model_dir)
        self.model = whisper.load_model("medium.en")
