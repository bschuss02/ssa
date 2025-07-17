from experiments.utils.inference_models.phi_4_multimodal_instruct import (
    Phi4MultimodalASRModel,
)


def main():
    model = Phi4MultimodalASRModel()

    model.load_model(model_name="microsoft/Phi-4-multimodal-instruct")

    messages = [
        {"role": "system", "content": "You are an expert audio transcriptionist."},
        {"role": "user", "content": "Transcribe this audio file accurately."},
    ]

    # transcription = model.transcribe(
    #     audio_path="/Users/Benjamin/dev/ssa/data/fluencybank/processed/wav_clips/20f_000_000.wav",
    #     messages=messages,
    # )


if __name__ == "__main__":
    main()
