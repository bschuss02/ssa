from pydantic import BaseModel


class EvaluationMetrics(BaseModel):
    wer: float
    mer: float
    wil: float
    wip: float
    cer: float


class EvaluationResult(BaseModel):
    model_name: str
    dataset_name: str
    ground_truth_transcript: str
    predicted_transcript: str
    inference_time: float
    metrics: EvaluationMetrics
