import logging

from experiments.utils.inference_models.phi_4_multimodal_instruct import (
    Phi4MultimodalASRModel,
)

logger = logging.getLogger(__name__)

logger.info("Initializing inference model registry...")

# Create model instances without loading them yet (lazy loading)
logger.info("Creating model instances (lazy loading)...")
phi_4_multimodal_instruct = Phi4MultimodalASRModel(
    model_name="Phi-4-Multimodal-Instruct",
    model_path="/Users/Benjamin/dev/ssa/models/Phi-4-multimodal-instruct",
)
logger.info("✓ Phi-4 Multimodal Instruct model instance created (not loaded yet)")

inference_model_registry = {
    "phi_4_multimodal_instruct": phi_4_multimodal_instruct,
}
logger.info(
    f"✓ Inference model registry initialized with {len(inference_model_registry)} models: {list(inference_model_registry.keys())}"
)
