from typing import Dict, Type

from approaches.phi4.phi_4_multimodal_instruct import Phi4MultimodalInstruct
from approaches.phi4.phi_4_with_stutter_prompt import Phi4WithStutterPrompt
from approaches.staccato.staccato import Staccato
from approaches.whisper.whisper_v3_medium import WhisperV3Medium
from approaches.whisper.whisper_v3_multilingual import WhisperV3MediumMultilingual
from experiments.inference_models.asr_model_base import ASRModelBase

model_registry: Dict[str, Type[ASRModelBase]] = {
    "phi_4_multimodal_instruct": Phi4MultimodalInstruct,
    "phi_4_with_stutter_prompt": Phi4WithStutterPrompt,
    "whisper_v3_medium": WhisperV3Medium,
    "staccato": Staccato,
    "whisper_v3_medium_multilingual": WhisperV3MediumMultilingual,
}
