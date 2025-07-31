from typing import Dict, Type

from experiments.inference_models.asr_model_base import ASRModelBase
from experiments.inference_models.phi_4_multimodal_instruct import (
    Phi4MultimodalInstruct,
)
from experiments.inference_models.phi_4_with_stutter_prompt import Phi4WithStutterPrompt

model_registry: Dict[str, Type[ASRModelBase]] = {
    "phi_4_multimodal_instruct": Phi4MultimodalInstruct,
    "phi_4_with_stutter_prompt": Phi4WithStutterPrompt,
}
