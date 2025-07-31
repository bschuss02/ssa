from typing import List, Optional

from pydantic import BaseModel


class ASREvaluatorConfig(BaseModel):
    models_to_evaluate: List[str]
    datasets_to_evaluate: List[str]
    max_samples_per_dataset: Optional[int]
    save_samples: bool
    output_dir: str
    batch_size: Optional[int] = (
        None  # None means process one by one, >1 enables batch processing
    )
