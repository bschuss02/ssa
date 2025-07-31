#!/usr/bin/env python3
"""
Comprehensive ASR Evaluation Script using Hydra

This script evaluates the Phi-4 multimodal ASR model on the FluencyBank dataset
and provides detailed analysis including WER, CER, and other ASR quality metrics.
"""

import hydra
from omegaconf import DictConfig

from experiments.config.models.asr_evaluator_config import ASREvaluatorConfig
from experiments.utils.evaluation_engine import ASREvaluator
from experiments.utils.logging_config import setup_logging


@hydra.main(config_path="config", config_name="base.yaml", version_base=None)
def main(cfg: DictConfig):
    """Main entry point for ASR evaluation."""
    # Setup logging immediately
    logger = setup_logging()
    logger.info("Starting ASR evaluation script...")

    # Parse configuration with immediate feedback
    logger.info("Parsing configuration...")
    cfg = ASREvaluatorConfig(**cfg)
    logger.info(
        f"✓ Configuration parsed: {len(cfg.models_to_evaluate)} models, {len(cfg.datasets_to_evaluate)} datasets"
    )

    # Run evaluation
    logger.info("Initializing evaluator...")
    evaluator = ASREvaluator(cfg)

    logger.info("Starting evaluation pipeline...")
    results = evaluator.evaluate()

    logger.info("ASR evaluation script completed successfully")


if __name__ == "__main__":
    main()
