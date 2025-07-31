from experiments.inference_models.phi_4_multimodal_instruct import (
    Phi4MultimodalInstruct,
)


class Phi4WithStutterPrompt(Phi4MultimodalInstruct):
    def __init__(self, model_name: str, model_path: str):
        super().__init__(model_name, model_path)
        self.prompt_messages = [
            {
                "role": "system",
                "content": "You are an expert audio transcriptionist.",
            },
            {
                "role": "user",
                "content": "The person speaking has a stutter. Transcribe the speech from this audio recording as the speaker would intend their words to be spoken. Do not include any stutter markers. <|audio_1|>",
            },
        ]
