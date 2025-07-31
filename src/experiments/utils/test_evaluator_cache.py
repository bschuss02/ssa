#!/usr/bin/env python3
"""Test script for evaluator cache integration."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from experiments.config.evaluation_config import EvaluationConfig
from experiments.evaluation.evaluator import Evaluator


def test_evaluator_cache_integration():
    """Test that the evaluator properly integrates with ASR cache."""

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test configuration
        config = EvaluationConfig(
            models={"test_model": temp_path / "model"},
            datasets={"test_dataset": temp_path / "dataset"},
            max_samples_per_dataset=5,
            batch_size=2,
            output_dir=temp_path / "output",
            results_dir=temp_path / "results",
            asr_cache_dir=temp_path / "asr_cache",
            use_asr_cache=True,  # Enable cache
            dataset_cache_dir=temp_path / "dataset_cache",
            load_datasets_from_cache=False,
        )

        # Test with cache enabled
        evaluator = Evaluator(config)
        assert evaluator.asr_cache is not None
        print("✓ Cache initialized when use_asr_cache=True")

        # Test cache statistics
        stats = evaluator.get_cache_stats()
        assert "size" in stats
        print("✓ Cache statistics accessible")

        # Test cache clearing
        evaluator.clear_cache()
        print("✓ Cache clearing works")

        # Test with cache disabled
        config.use_asr_cache = False
        evaluator_no_cache = Evaluator(config)
        assert evaluator_no_cache.asr_cache is None
        print("✓ Cache disabled when use_asr_cache=False")

        # Test cache methods when disabled
        stats_disabled = evaluator_no_cache.get_cache_stats()
        assert "error" in stats_disabled
        print("✓ Cache methods handle disabled state gracefully")

        evaluator_no_cache.clear_cache()  # Should log warning but not crash
        print("✓ Cache clearing handles disabled state gracefully")

        print("All evaluator cache integration tests passed!")


if __name__ == "__main__":
    test_evaluator_cache_integration()
