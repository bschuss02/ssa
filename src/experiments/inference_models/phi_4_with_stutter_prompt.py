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
                "content": "You are tasked with transcribing the speech from this audio recording. The person speaking has a stutter. There may be repetitions of sounds, pauses between words, and other stutter-like behaviors. Transcribe the speech as accurately as possible. <|audio_1|>",
            },
        ]
