from typing import List

import jiwer

from experiments.utils.evaluation_result import EvaluationMetrics


def calculate_metrics(
    predicted_transcriptions: List[str], ground_truth_transcriptions: List[str]
) -> EvaluationMetrics:
    metrics_batch = []
    for predicted_transcription, ground_truth_transcription in zip(
        predicted_transcriptions, ground_truth_transcriptions
    ):
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
