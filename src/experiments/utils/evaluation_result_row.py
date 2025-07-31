from pydantic import BaseModel


class EvaluationMetrics(BaseModel):
    wer: float
    mer: float
    wil: float
    wip: float
    cer: float
    visualize_alignment: str


class EvaluationResultRow(BaseModel):
    model_name: str
    dataset_name: str
    ground_truth_transcript: str
    predicted_transcript: str
    inference_time: float
    metrics: EvaluationMetrics
