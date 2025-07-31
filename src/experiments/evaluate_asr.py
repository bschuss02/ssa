#!/usr/bin/env python3
"""
Comprehensive ASR Evaluation Script using Hydra

This script evaluates the Phi-4 multimodal ASR model on the FluencyBank dataset
and provides detailed analysis including WER, CER, and other ASR quality metrics.
"""

import logging
import sys
from typing import List

import hydra
import jiwer
import numpy as np
from omegaconf import DictConfig
from pydantic import BaseModel
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from experiments.config.models.asr_evaluator_config import ASREvaluatorConfig
from experiments.utils.datasets.dataset_registry import dataset_registry
from experiments.utils.inference_models.base_multimodal_asr_model import (
    BaseMultimodalASRModel,
)
from experiments.utils.inference_models.inference_model_registry import (
    inference_model_registry,
)


# Configure logging with more explicit setup
def setup_logging():
    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
        force=True,  # Force reconfiguration
    )

    # Ensure our logger is set to INFO level
    _log = logging.getLogger(__name__)
    _log.setLevel(logging.INFO)

    # Add a console handler to ensure output goes to terminal
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    _log.addHandler(console_handler)

    return _log


_log = setup_logging()


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
        self.dataset_names = cfg.datasets_to_evaluate
        self.datasets = [
            dataset_registry[dataset_name] for dataset_name in self.dataset_names
        ]
        _log.info(
            f"Initialized ASR Evaluator with {len(self.models)} models and {len(self.datasets)} datasets"
        )

    def inference_and_record(self):
        results: List[EvaluationResultRow] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            # Main progress bar for models
            model_task = progress.add_task(
                "[cyan]Processing models...", total=len(self.models)
            )

            for model_idx, model in enumerate(self.models):
                _log.info(f"Loading model: {model.model_name}")
                model.load_model()

                # Progress bar for datasets within each model
                dataset_task = progress.add_task(
                    f"[green]Processing datasets for {model.model_name}...",
                    total=len(self.datasets),
                )

                for dataset_idx, dataset in enumerate(self.datasets):
                    _log.info(f"Processing dataset: {self.dataset_names[dataset_idx]}")

                    if self.cfg.max_samples_per_dataset is not None:
                        dataset = dataset.select(
                            range(self.cfg.max_samples_per_dataset)
                        )
                        _log.info(
                            f"Limited to {self.cfg.max_samples_per_dataset} samples"
                        )

                    # Progress bar for samples within each dataset
                    sample_task = progress.add_task(
                        f"[yellow]Processing samples for {self.dataset_names[dataset_idx]}...",
                        total=len(dataset),
                    )

                    for item_idx, item in enumerate(dataset):
                        audio_path = item["audio"]["path"]
                        sample_rate = item["audio"]["sampling_rate"]
                        audio_array = np.array(item["audio"]["array"])
                        _log.debug(f"Processing audio: {audio_path}")

                        try:
                            prediction = model.transcribe(
                                audio_array=audio_array, sample_rate=sample_rate
                            )
                            transcription = item["unannotated_text"]

                            # Calculate metrics
                            wer = jiwer.wer(transcription, prediction)
                            mer = jiwer.mer(transcription, prediction)
                            wil = jiwer.wil(transcription, prediction)
                            wip = jiwer.wip(transcription, prediction)
                            cer = jiwer.cer(transcription, prediction)

                            # Fix: Use jiwer.visualize_alignment correctly - it takes reference and hypothesis directly
                            try:
                                visualize_alignment = jiwer.visualize_alignment(
                                    transcription, prediction
                                )
                            except Exception as alignment_error:
                                _log.warning(
                                    f"Failed to create alignment visualization: {alignment_error}"
                                )
                                visualize_alignment = (
                                    f"Alignment failed: {alignment_error}"
                                )

                            metrics = EvaluationMetrics(
                                wer=wer,
                                mer=mer,
                                wil=wil,
                                wip=wip,
                                cer=cer,
                                visualize_alignment=visualize_alignment,
                            )
                            evaluation_result_row = EvaluationResultRow(
                                model_name=model.model_name,
                                dataset_name=self.dataset_names[dataset_idx],
                                metrics=metrics,
                            )
                            print(evaluation_result_row)
                            results.append(evaluation_result_row)

                            _log.debug(
                                f"Sample {item_idx + 1}: WER={wer:.4f}, CER={cer:.4f}"
                            )

                        except Exception as e:
                            _log.error(f"Error processing audio {audio_path}: {str(e)}")

                        progress.advance(sample_task)

                    progress.remove_task(sample_task)
                    progress.advance(dataset_task)

                progress.remove_task(dataset_task)
                progress.advance(model_task)
                _log.info(f"Completed processing for model: {model.model_name}")

        _log.info(f"Completed inference and recording. Total results: {len(results)}")
        return results

    def evaluate(self):
        # model = self.models[0]
        # dataset = self.datasets[0]
        # dataset = dataset.select(range(10))
        # model.load_model()
        # for item in dataset:
        #     sample_rate = item["audio"]["sampling_rate"]
        #     audio_array = np.array(item["audio"]["array"])
        #     prediction = model.transcribe(
        #         audio_array=audio_array, sample_rate=sample_rate
        #     )
        #     print(prediction)
        #     break
        _log.info("Starting ASR evaluation...")
        results = self.inference_and_record()
        _log.info("Evaluation completed successfully")
        return results


@hydra.main(config_path="config", config_name="base.yaml", version_base=None)
def main(cfg: DictConfig):
    # Ensure logging is set up before Hydra processing
    _log = setup_logging()
    _log.info("Starting ASR evaluation script...")

    cfg = ASREvaluatorConfig(**cfg)
    evaluator = ASREvaluator(cfg)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
