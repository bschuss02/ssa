import re
from typing import List

import jiwer

from experiments.utils.evaluation_result import EvaluationMetrics


def remove_punctuation_from_text(text: str) -> str:
    """Remove punctuation from text while preserving spaces."""
    # Remove punctuation marks but preserve spaces
    return re.sub(r"[^\w\s]", "", text)


def preprocess_text(
    text: str, remove_punctuation: bool = False, make_lowercase: bool = False
) -> str:
    """Preprocess text by removing punctuation and/or converting to lowercase."""
    if make_lowercase:
        text = text.lower()
    if remove_punctuation:
        text = remove_punctuation_from_text(text)
    return text


def calculate_metrics(
    predicted_transcriptions: List[str],
    ground_truth_transcriptions: List[str],
    remove_punctuation: bool = False,
    make_lowercase: bool = False,
) -> List[EvaluationMetrics]:
    metrics_batch = []
    for predicted_transcription, ground_truth_transcription in zip(
        predicted_transcriptions, ground_truth_transcriptions
    ):
        # Apply text preprocessing if requested
        if remove_punctuation or make_lowercase:
            predicted_transcription = preprocess_text(
                predicted_transcription, remove_punctuation, make_lowercase
            )
            ground_truth_transcription = preprocess_text(
                ground_truth_transcription, remove_punctuation, make_lowercase
            )

        wer = jiwer.wer(predicted_transcription, ground_truth_transcription)
        mer = jiwer.mer(predicted_transcription, ground_truth_transcription)
        wil = jiwer.wil(predicted_transcription, ground_truth_transcription)
        wip = jiwer.wip(predicted_transcription, ground_truth_transcription)
        cer = jiwer.cer(predicted_transcription, ground_truth_transcription)
        metrics = EvaluationMetrics(
            wer=wer,
            mer=mer,
            wil=wil,
            wip=wip,
            cer=cer,
        )
        metrics_batch.append(metrics)
    return metrics_batch
