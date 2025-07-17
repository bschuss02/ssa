from experiments.utils.inference_models.phi_4_multimodal_instruct import (
    Phi4MultimodalASRModel,
)


def main():
    model = Phi4MultimodalASRModel(
        model_path="/Users/Benjamin/dev/ssa/models/Phi-4-multimodal-instruct"
    )

    model.load_model()

    messages = [
        {"role": "system", "content": "You are an expert audio transcriptionist."},
        {"role": "user", "content": "Transcribe this audio file accurately."},
    ]

    print("Starting transcription...")
    transcription = model.transcribe(
        audio_path="/Users/Benjamin/dev/ssa/data/fluencybank/processed/wav_clips/20f_000_000.wav",
        messages=messages,
    )

    print(f"Transcription: {transcription}")


if __name__ == "__main__":
    main()
