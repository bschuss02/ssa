#!/usr/bin/env python3
"""
Example usage of the Phi-4 Multimodal ASR Model

This script demonstrates how to use the Phi4MultimodalASRModel
for audio transcription with optional chat context.
"""

import logging
from pathlib import Path
from phi_4_multimodal_instruct import Phi4MultimodalASRModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate basic usage of the Phi-4 multimodal ASR model."""

    # Initialize the model
    model = Phi4MultimodalASRModel(
        model_cache_dir="/Users/Benjamin/dev/ssa/models",
        force_cpu=False,  # Set to True if you want to force CPU usage
    )

    # Load the model
    logger.info("Loading Phi-4 multimodal model...")
    model.load_model()

    # Check if model is loaded
    if model.is_loaded:
        logger.info("Model loaded successfully!")
        logger.info(f"Model info: {model.get_model_info()}")
    else:
        logger.error("Failed to load model")
        return

    # Example 1: Basic transcription with default system message
    audio_path = "path/to/your/audio/file.wav"  # Replace with actual audio file path

    if Path(audio_path).exists():
        logger.info("Example 1: Basic transcription")
        try:
            transcription = model.transcribe(audio_path)
            print(f"Transcription: {transcription}")
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
    else:
        logger.warning(f"Audio file not found: {audio_path}")

    # Example 2: Transcription with custom messages
    if Path(audio_path).exists():
        logger.info("Example 2: Transcription with custom context")

        custom_messages = [
            {
                "role": "system",
                "content": "You are a medical transcriptionist specializing in patient interviews. "
                "Transcribe the audio accurately and note any medical terminology.",
            },
            {"role": "user", "content": "Please transcribe this medical consultation audio recording."},
        ]

        try:
            transcription = model.transcribe(
                audio_path=audio_path,
                messages=custom_messages,
                max_new_tokens=800,
                temperature=0.3,  # Lower temperature for more deterministic output
            )
            print(f"Medical transcription: {transcription}")
        except Exception as e:
            logger.error(f"Medical transcription failed: {e}")

    # Example 3: Multi-turn conversation with audio
    if Path(audio_path).exists():
        logger.info("Example 3: Multi-turn conversation")

        conversation_messages = [
            {"role": "system", "content": "You are an AI assistant helping with audio analysis."},
            {"role": "user", "content": "Can you transcribe this audio and summarize the main points?"},
            {"role": "assistant", "content": "I'll transcribe the audio and provide a summary for you."},
            {"role": "user", "content": "Please proceed with the transcription and summary."},
        ]

        try:
            result = model.transcribe(
                audio_path=audio_path, messages=conversation_messages, max_new_tokens=1000
            )
            print(f"Transcription and summary: {result}")
        except Exception as e:
            logger.error(f"Conversation transcription failed: {e}")


if __name__ == "__main__":
    main()
