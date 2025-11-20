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
    return f"You are an expert speech therapist with 20 years of experience helping people who stutter.  People who stutter speak with involuntary sound repetitions, word repetitions, prolongations, and blocks. Your task is to transcribe a recording of a person who stutters speaking {language_name}. You must transcribe the words that the speaker INTENDED to say, excluding involuntary disfluencies and correcting general transcription errors and omissions. You are also given an initial transcription of the recording that was produced by Whisper, an automatic speech recognition model. There may be errors in this transcription because Whisper was not trained on speech data of people who stutter and is known to have poor accuracy on stuttered speech. The errors could be the addition of words that are actually not spoken in the recording, or the omission of words that actually are spoken in the recording, or the mistranscription of words. Your job is to correct the errors in the Whisper transcription and provide a final transcription of the recording. In order to gain an understanding of the recording, first describe where stuttering occurs in the recording. Reference specific characters in the initial transcript and specific phonemes in the recording, and how this may have influenced the transcription or introduced errors. DO NOT be vague about how stuttering may have affected the transcription or introduced errors. Only talk about specific stuttering events, where they occur in the recording, what type of stuttering they are, etc. If you make a mistake while transcribing, you will be fired from your job."
    # maybe prompt to not forget Na characters
    # return "您是一位拥有20年经验的语音治疗专家，长期致力于帮助口吃患者。口吃患者在说话时会出现非自愿的声音重复、词语重复、延长发音及卡顿现象。您的任务是转录一位口吃人士说中文的录音，必须转录说话者本意要表达的词语，排除非自愿的口吃现象。您还将获得由自动语音识别模型Whisper生成的初始转录文本。该转录可能存在错误，因为Whisper未接受过口吃人群语音数据的训练，且在处理口吃语音时准确率较低。你的工作是修正Whisper转录中的错误，并提供最终的录音转录文本。为理解录音内容，请先标注口吃现象出现的具体位置，同时参照原始转录文本的对应段落。需特别关注重复或延长发音的具体声段，以及这些现象如何影响转录结果或导致错误。若转录过程中出现失误，你将被解雇。"


class TranscribeStutteredSpeechModule(dspy.Module):
    def __init__(self, language: str):
        description = get_signature_description(language)

        class TranscribeStutteredSpeechSignature(dspy.Signature):
            __doc__ = description

            stuttered_speech_audio: dspy.Audio = dspy.InputField()
            initial_transcription: str = dspy.InputField()

            stuttering_events: str = dspy.OutputField(
                description="Describe where stuttering occurs in the recording. 1-5 sentences."
            )
            analysis: str = dspy.OutputField(
                description="How the stuttering events may have influenced the transcription or introduced errors. 1-5 sentences."
            )
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
                transcription=output.revised_transcription,
                metadata={
                    "output": output,
                    "whisper_output": initial_transcription_output.transcription,
                },
            )
            for output, initial_transcription_output in zip(outputs, initial_transcription_outputs)
        ]
