"""
Core ASR evaluation engine.

This module contains the main evaluation logic, separated from UI concerns
and configuration management.
"""

import logging
from typing import List

import numpy as np

from experiments.config.asr_evaluator_config import ASREvaluatorConfig
from experiments.utils.datasets.dataset_registry import dataset_registry
from experiments.utils.inference_models.base_multimodal_asr_model import (
    BaseMultimodalASRModel,
)
from experiments.utils.inference_models.inference_model_registry import (
    inference_model_registry,
)
from experiments.utils.metrics import EvaluationResultRow, calculate_asr_metrics
from experiments.utils.progress_manager import ProgressManager

logger = logging.getLogger(__name__)


class ASREvaluator:
    """Main ASR evaluation orchestrator."""

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
        logger.info(
            f"Initialized ASR Evaluator with {len(self.models)} models and {len(self.datasets)} datasets"
        )

    def _validate_setup(self) -> None:
        """Validate configuration and setup before starting evaluation."""
        logger.info("Validating evaluation setup...")

        # Validate models
        for model_name in self.cfg.models_to_evaluate:
            if model_name not in inference_model_registry:
                raise ValueError(f"Model '{model_name}' not found in registry")
            logger.info(f"✓ Model '{model_name}' found in registry")

        # Validate datasets
        for dataset_name in self.cfg.datasets_to_evaluate:
            if dataset_name not in dataset_registry:
                raise ValueError(f"Dataset '{dataset_name}' not found in registry")
            logger.info(f"✓ Dataset '{dataset_name}' found in registry")

        # Log dataset sizes (this will trigger lazy loading of metadata only)
        for dataset_name, dataset in zip(self.dataset_names, self.datasets):
            total_samples = len(dataset)  # This only loads metadata, not audio files
            if self.cfg.max_samples_per_dataset:
                actual_samples = min(total_samples, self.cfg.max_samples_per_dataset)
                logger.info(
                    f"✓ Dataset '{dataset_name}': {actual_samples}/{total_samples} samples"
                )
            else:
                logger.info(f"✓ Dataset '{dataset_name}': {total_samples} samples")

        # Log batch processing configuration
        if self.cfg.batch_size is not None and self.cfg.batch_size > 1:
            logger.info(
                f"✓ Batch processing enabled with batch size: {self.cfg.batch_size}"
            )
        else:
            logger.info("✓ Processing samples one by one")

        logger.info("✓ Setup validation completed")

    def _process_single_sample(
        self, model: BaseMultimodalASRModel, item: dict, dataset_name: str
    ) -> EvaluationResultRow | None:
        """
        Process a single audio sample and return evaluation results.

        Args:
            model: The ASR model to use for transcription
            item: Dataset item containing audio and reference text
            dataset_name: Name of the dataset being processed

        Returns:
            EvaluationResultRow if successful, None if failed
        """
        audio_path = item["audio"]["path"]
        sample_rate = item["audio"]["sampling_rate"]
        audio_array = np.array(item["audio"]["array"])

        try:
            # Get model prediction
            prediction = model.transcribe(
                audio_array=audio_array, sample_rate=sample_rate
            )
            reference = item["unannotated_text"]

            # Calculate metrics
            metrics = calculate_asr_metrics(reference, prediction)

            # Create result row
            result = EvaluationResultRow(
                model_name=model.model_name,
                dataset_name=dataset_name,
                metrics=metrics,
            )

            logger.debug(
                f"Sample processed: WER={metrics.wer:.4f}, CER={metrics.cer:.4f}"
            )

            return result

        except Exception as e:
            logger.error(f"Error processing audio {audio_path}: {str(e)}")
            return None

    def _process_batch(
        self,
        model: BaseMultimodalASRModel,
        items: List[dict],
        dataset_name: str,
    ) -> List[EvaluationResultRow]:
        """
        Process a batch of audio samples and return evaluation results.

        Args:
            model: The ASR model to use for transcription
            items: List of dataset items containing audio and reference text
            dataset_name: Name of the dataset being processed

        Returns:
            List of EvaluationResultRow objects
        """
        results = []

        # Prepare batch inputs
        audio_arrays = []
        sample_rates = []
        references = []
        audio_paths = []

        for item in items:
            audio_path = item["audio"]["path"]
            sample_rate = item["audio"]["sampling_rate"]
            audio_array = np.array(item["audio"]["array"])
            reference = item["unannotated_text"]

            audio_arrays.append(audio_array)
            sample_rates.append(sample_rate)
            references.append(reference)
            audio_paths.append(audio_path)

        try:
            # Get model predictions for the batch
            predictions = model.transcribe_batch(
                audio_arrays=audio_arrays,
                sample_rates=sample_rates,
            )

            # Process each prediction
            for i, (prediction, reference, audio_path) in enumerate(
                zip(predictions, references, audio_paths)
            ):
                # Calculate metrics
                metrics = calculate_asr_metrics(reference, prediction)

                # Create result row
                result = EvaluationResultRow(
                    model_name=model.model_name,
                    dataset_name=dataset_name,
                    metrics=metrics,
                )

                logger.debug(
                    f"Batch sample {i + 1} processed: WER={metrics.wer:.4f}, CER={metrics.cer:.4f}"
                )

                results.append(result)

        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            # Fall back to individual processing
            for item in items:
                result = self._process_single_sample(model, item, dataset_name)
                if result is not None:
                    results.append(result)

        return results

    def _process_dataset(
        self,
        model: BaseMultimodalASRModel,
        dataset,
        dataset_name: str,
        progress_manager: ProgressManager,
    ) -> List[EvaluationResultRow]:
        """
        Process all samples in a dataset for a given model.

        Args:
            model: The ASR model to use
            dataset: The dataset to process
            dataset_name: Name of the dataset
            progress_manager: Progress tracking manager

        Returns:
            List of evaluation results
        """
        results = []

        # Get dataset size for progress tracking
        dataset_size = len(dataset)

        # Limit dataset size if specified
        if self.cfg.max_samples_per_dataset is not None:
            actual_size = min(dataset_size, self.cfg.max_samples_per_dataset)
            dataset = dataset.select(range(self.cfg.max_samples_per_dataset))
            logger.info(f"Limited to {self.cfg.max_samples_per_dataset} samples")
        else:
            actual_size = dataset_size

        # Start sample progress tracking
        progress_manager.start_sample_processing(dataset_name, actual_size)

        # Convert dataset to list for batch processing
        dataset_items = list(dataset)

        if self.cfg.batch_size is not None and self.cfg.batch_size > 1:
            # Process in batches
            logger.info(
                f"Processing {len(dataset_items)} samples in batches of {self.cfg.batch_size}"
            )

            for i in range(0, len(dataset_items), self.cfg.batch_size):
                batch_items = dataset_items[i : i + self.cfg.batch_size]
                batch_results = self._process_batch(model, batch_items, dataset_name)

                for result in batch_results:
                    results.append(result)
                    print(result)  # Print results as they're generated

                # Advance progress by batch size
                progress_manager.advance_sample_by(len(batch_items))

        else:
            # Process one by one
            for item in dataset_items:
                result = self._process_single_sample(model, item, dataset_name)
                if result is not None:
                    results.append(result)
                    print(result)  # Print results as they're generated

                progress_manager.advance_sample()

        progress_manager.finish_sample_processing()
        return results

    def _process_model(
        self, model: BaseMultimodalASRModel, progress_manager: ProgressManager
    ) -> List[EvaluationResultRow]:
        """
        Process all datasets for a given model.

        Args:
            model: The ASR model to process
            progress_manager: Progress tracking manager

        Returns:
            List of evaluation results for this model
        """
        logger.info(f"Loading model: {model.model_name}")
        model.load_model()

        all_results = []

        # Start dataset progress tracking
        progress_manager.start_dataset_processing(model.model_name, len(self.datasets))

        for dataset_idx, dataset in enumerate(self.datasets):
            dataset_name = self.dataset_names[dataset_idx]
            logger.info(f"Processing dataset: {dataset_name}")

            results = self._process_dataset(
                model, dataset, dataset_name, progress_manager
            )
            all_results.extend(results)

            progress_manager.advance_dataset()

        progress_manager.finish_dataset_processing()
        logger.info(f"Completed processing for model: {model.model_name}")

        return all_results

    def evaluate(self) -> List[EvaluationResultRow]:
        """
        Run the complete ASR evaluation pipeline.

        Returns:
            List of all evaluation results
        """
        logger.info("Starting ASR evaluation...")

        # Immediate validation and setup feedback
        self._validate_setup()

        logger.info("Starting model processing...")
        all_results = []

        with ProgressManager() as progress_manager:
            progress_manager.start_model_processing(len(self.models))

            for model in self.models:
                results = self._process_model(model, progress_manager)
                all_results.extend(results)
                progress_manager.advance_model()

        logger.info(f"Evaluation completed. Total results: {len(all_results)}")
        return all_results
