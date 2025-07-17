#!/usr/bin/env python3
"""
Comprehensive ASR Evaluation Script using Hydra

This script evaluates the Phi-4 multimodal ASR model on the FluencyBank dataset
and provides detailed analysis including WER, CER, and other ASR quality metrics.
"""

import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import jiwer
import numpy as np
import polars as pl
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel
from tqdm import tqdm

from experiments.config.models.asr_evaluator_config import ASREvaluatorConfig
from experiments.utils.datasets.dataset_registry import dataset_registry
from experiments.utils.datasets.fluencybank_dataset import create_fluencybank_dataset
from experiments.utils.inference_models.base_multimodal_asr_model import (
    BaseMultimodalASRModel,
)
from experiments.utils.inference_models.inference_model_registry import (
    inference_model_registry,
)


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
    metrics: EvaluationMetrics


class ASREvaluator:
    def __init__(self, cfg: ASREvaluatorConfig):
        self.cfg = cfg
        self.models: List[BaseMultimodalASRModel] = [
            inference_model_registry[model_name]
            for model_name in cfg.models_to_evaluate
        ]
        self.datasets = [
            dataset_registry[dataset_name] for dataset_name in cfg.datasets_to_evaluate
        ]

    def inference_and_record(self):
        results: List[EvaluationResultRow] = []
        for model in self.models:
            model.load_model()
            for dataset in self.datasets:
                if self.cfg.max_samples_per_dataset is not None:
                    dataset = dataset.select(range(self.cfg.max_samples_per_dataset))
                for item in dataset:
                    audio_path = item["audio"]["path"]
                    prediction = model.transcribe(audio_path)
                    transcription = item["unannotated_text"]
                    wer = jiwer.wer(transcription, prediction)
                    mer = jiwer.mer(transcription, prediction)
                    wil = jiwer.wil(transcription, prediction)
                    wip = jiwer.wip(transcription, prediction)
                    cer = jiwer.cer(transcription, prediction)
                    visualize_alignment = jiwer.visualize_alignment(
                        transcription, prediction
                    )
                    metrics = EvaluationMetrics(
                        wer=wer,
                        mer=mer,
                        wil=wil,
                        wip=wip,
                        cer=cer,
                        visualize_alignment=visualize_alignment,
                    )
                    results.append(
                        EvaluationResultRow(
                            model_name=model.model_name,
                            dataset_name=dataset.name,
                            metrics=metrics,
                        )
                    )
                    break
                break
        return results

    def evaluate(self):
        results = self.inference_and_record()


@hydra.main(config_path="config", config_name="asr_evaluator_config.yaml")
def main(cfg: DictConfig):
    cfg = ASREvaluatorConfig(**cfg)
    evaluator = ASREvaluator(cfg)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
