#!/usr/bin/env python3
"""
Quick test script for ASR metrics calculation
"""

import re
import string
import jiwer
from typing import Dict


class ASRMetricsCalculator:
    """Calculate comprehensive ASR evaluation metrics."""

    def __init__(self):
        pass

    def normalize_text(self, text: str) -> str:
        """
        Normalize text for ASR evaluation by:
        - Converting to lowercase
        - Removing punctuation
        - Normalizing whitespace
        - Handling common ASR variations
        """
        # Convert to lowercase
        text = text.lower()

        # Handle common ASR number/word variations
        text = re.sub(r"\bfour\b", "4", text)
        text = re.sub(r"\btwo\b", "2", text)
        text = re.sub(r"\bone\b", "1", text)
        text = re.sub(r"\bthree\b", "3", text)

        # Handle common contractions and variations
        text = re.sub(r"\bokay\b", "ok", text)
        text = re.sub(r"\byeah\b", "yes", text)
        text = re.sub(r"\byep\b", "yes", text)
        text = re.sub(r"\buh\b", "", text)  # Remove filler words
        text = re.sub(r"\bum\b", "", text)  # Remove filler words

        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        return text

    def calculate_metrics(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """
        Calculate comprehensive ASR metrics for a single sample.
        """
        try:
            # Normalize both texts for fair comparison
            norm_reference = self.normalize_text(reference)
            norm_hypothesis = self.normalize_text(hypothesis)

            print(f"Original ref: '{reference}'")
            print(f"Normalized ref: '{norm_reference}'")
            print(f"Original hyp: '{hypothesis}'")
            print(f"Normalized hyp: '{norm_hypothesis}'")
            print()

            # Calculate word-level metrics on normalized text using process_words
            word_output = jiwer.process_words(norm_reference, norm_hypothesis)

            # Calculate character-level metrics on normalized text
            cer = jiwer.cer(norm_reference, norm_hypothesis)

            return {
                "wer": word_output.wer,
                "mer": word_output.mer,
                "wil": word_output.wil,
                "wip": word_output.wip,
                "cer": cer,
                "substitutions": word_output.substitutions,
                "deletions": word_output.deletions,
                "insertions": word_output.insertions,
                "hits": word_output.hits,
            }

        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return {
                "wer": 1.0,
                "mer": 1.0,
                "wil": 1.0,
                "wip": 0.0,
                "cer": 1.0,
                "substitutions": 0,
                "deletions": 0,
                "insertions": 0,
                "hits": 0,
            }


if __name__ == "__main__":
    # Test the fixed normalization with the example from the user
    calc = ASRMetricsCalculator()

    reference = "okay so can you please talk about the impact of stuttering on your daily life and those people around you ?"
    hypothesis = "OK, so can you please talk about the impact of stuttering on your daily life? And those are the people around you."

    print("Testing fixed metrics calculation:")
    metrics = calc.calculate_metrics(reference, hypothesis)
    print("Final Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
