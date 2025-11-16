from approaches.phi4.phi_4_multimodal_instruct import (
    Phi4MultimodalInstruct,
)


class Phi4WithStutterPrompt(Phi4MultimodalInstruct):
    def __init__(self, model_name: str, cfg):
        super().__init__(model_name, cfg)
        self.prompt_messages = [
            {
                "role": "system",
                "content": "You are an expert audio transcriptionist.",
            },
            {
                "role": "user",
                "content": "You are tasked with transcribing the speech from this audio recording. The person speaking has a stutter. There may be repetitions of sounds, pauses between words, prolongations of sounds, blocks, and other stutter-like sounds. Ignore these disfluencies and transcribe the speech accurately. <|audio_1|>",
            },
        ]
