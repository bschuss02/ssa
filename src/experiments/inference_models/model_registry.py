from typing import Dict, Type

from approaches.phi4.phi_4_multimodal_instruct import Phi4MultimodalInstruct
from approaches.phi4.phi_4_with_stutter_prompt import Phi4WithStutterPrompt
from approaches.staccato.staccato import Staccato
from approaches.whisper.whisper_v3_chinese import WhisperV3MediumChinese
from approaches.whisper.whisper_v3_medium_english import WhisperV3MediumEnglish
from experiments.inference_models.asr_model_base import ASRModelBase

model_registry: Dict[str, Type[ASRModelBase]] = {
    "phi_4_multimodal_instruct": Phi4MultimodalInstruct,
    "phi_4_with_stutter_prompt": Phi4WithStutterPrompt,
    "whisper_v3_medium_english": WhisperV3MediumEnglish,
    "staccato": Staccato,
    "whisper_v3_chinese": WhisperV3MediumChinese,
}
