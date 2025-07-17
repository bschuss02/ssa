from experiments.utils.inference_models.phi_4_multimodal_instruct import (
    Phi4MultimodalASRModel,
)

phi_4_multimodal_instruct = Phi4MultimodalASRModel(
    model_name="Phi-4-Multimodal-Instruct",
    model_path="/Users/Benjamin/dev/ssa/models/phi-4-multimodal-instruct",
)

inference_model_registry = {
    "phi_4_multimodal_instruct": phi_4_multimodal_instruct,
}
