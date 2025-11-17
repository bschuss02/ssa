from typing import List

import dspy
import numpy as np
from dotenv import load_dotenv

from approaches.whisper.whisper_v3_medium import WhisperV3Medium
from experiments.config.evaluation_config import EvaluationConfig
from experiments.inference_models.asr_model_base import (
    ASRModelBase,
    TranscriptionInput,
    TranscriptionOutput,
)
from experiments.utils.audio_utils import load_audio_files


def get_signature_description(language: str) -> str:
    """Get language-specific signature description."""
    language_name = "English" if language == "en" else "Chinese" if language == "zh" else language
    return f"You are an expert speech therapist with 20 years of experience helping people who stutter.  People who stutter speak with involuntary sound repetitions, word repetitions, prolongations, and blocks. Your task is to transcribe a recording of a person who stutters speaking {language_name}. You must transcribe the words that the speaker INTENDED to say, excluding involuntary disfluencies. You are also given an initial transcription of the recording that was produced by Whisper, an automatic speech recognition model. There may be errors in this transcription because Whisper was not trained on speech data of people who stutter and is known to have poor accuracy on stuttered speech. Your job is to correct the errors in the Whisper transcription and provide a final transcription of the recording."


class TranscribeStutteredSpeechModule(dspy.Module):
    def __init__(self, language: str):
        description = get_signature_description(language)

        class TranscribeStutteredSpeechSignature(dspy.Signature):
            __doc__ = description

            stuttered_speech_audio: dspy.Audio = dspy.InputField()
            initial_transcription: str = dspy.InputField()
            revised_transcription: str = dspy.OutputField()

        self.cot = dspy.ChainOfThought(TranscribeStutteredSpeechSignature)

    def forward(self, audio_array: np.ndarray, sample_rate: int, initial_transcription: str) -> str:
        dspy_audio = dspy.Audio.from_array(audio_array, sample_rate)
        output = self.cot(
            stuttered_speech_audio=dspy_audio, initial_transcription=initial_transcription
        )
        return output


class Staccato(ASRModelBase):
    def __init__(
        self,
        model_name: str,
        cfg: EvaluationConfig,
        language: str,
        whisper_model_id: str = "openai/whisper-medium",
        lm: str = "gpt-4o-mini-audio-preview-2024-12-17",
    ):
        super().__init__(model_name, cfg)
        self.language = language
        self.whisper_model_id = whisper_model_id
        self.lm = lm
        load_dotenv()

    def load_model(self):
        self.whisper_model = WhisperV3Medium(
            f"whisper_v3_medium_{self.language}",
            self.cfg,
            model_id=self.whisper_model_id,
            language=self.language,
        )
        self.whisper_model.load_model()

        dspy.configure(lm=dspy.LM(self.lm))
        self.transcribe_stuttered_speech_module = TranscribeStutteredSpeechModule(self.language)

    def transcribe(
        self, transcription_inputs: List[TranscriptionInput]
    ) -> List[TranscriptionOutput]:
        audio_paths = [ti.audio_path for ti in transcription_inputs]
        if not all(audio_paths):
            raise ValueError("All transcription inputs must have an audio path")

        audio_arrays, _ = load_audio_files(audio_paths, max_workers=self.cfg.max_workers, sr=16000)
        initial_transcription_outputs = self.whisper_model.transcribe(transcription_inputs)

        sample_rate = 16000
        examples = [
            dspy.Example(
                audio_array=audio_array,
                sample_rate=sample_rate,
                initial_transcription=initial_transcription_output.transcription,
            ).with_inputs("audio_array", "sample_rate", "initial_transcription")
            for audio_array, initial_transcription_output in zip(
                audio_arrays, initial_transcription_outputs
            )
        ]
        outputs = self.transcribe_stuttered_speech_module.batch(examples=examples)
        return [
            TranscriptionOutput(
                transcription=output.revised_transcription, metadata={"reasoning": output.reasoning}
            )
            for output in outputs
        ]
