# Evaluation Script Usage Examples

This document provides practical examples for running the ASR evaluation
pipeline.

## Quick Test Run

For a quick test with minimal samples:

```bash
# Navigate to the experiments directory
cd src/experiments

# Run with test configuration (5 samples, CPU-only)
uv run python run_evaluation.py --config-name=test_eval

# Or override settings on command line
uv run python run_evaluation.py evaluation.max_samples=5 model.force_cpu=true output.dir=./quick_test
```

## Standard Evaluation

For a standard evaluation run:

```bash
# Run with default settings (100 samples)
uv run python run_evaluation.py

# Run with more samples
uv run python run_evaluation.py evaluation.max_samples=500

# Run on full dataset
uv run python run_evaluation.py evaluation.max_samples=0
```

## Configuration Examples

### Use Annotated Text as Ground Truth

```bash
uv run python run_evaluation.py dataset.text_column=annotated_text
```

### Force CPU Usage (for systems without GPU)

```bash
uv run python run_evaluation.py model.force_cpu=true
```

### Custom Model Parameters

```bash
uv run python run_evaluation.py \
  model.max_new_tokens=1000 \
  model.temperature=0.5 \
  model.do_sample=false
```

### Custom Dataset Path

```bash
uv run python run_evaluation.py \
  dataset.parquet_path=/path/to/your/fluencybank_segments.parquet
```

### Custom Output Directory

```bash
uv run python run_evaluation.py output.dir=./my_evaluation_results
```

## Advanced Examples

### Medical Transcription Evaluation

```bash
uv run python run_evaluation.py \
  model.system_message="You are a medical transcriptionist specializing in clinical notes." \
  model.user_prompt="Transcribe this medical dictation with attention to medical terminology." \
  output.dir=./medical_evaluation
```

### Verbose Debugging

```bash
uv run python run_evaluation.py \
  logging.level=DEBUG \
  evaluation.max_samples=10 \
  evaluation.log_interval=1
```

### High-Quality Transcription Settings

```bash
uv run python run_evaluation.py \
  model.temperature=0.3 \
  model.do_sample=true \
  model.max_new_tokens=1000 \
  dataset.max_audio_length=60.0
```

## Batch Processing

For large datasets, you can process in batches:

### Batch 1 (first 1000 samples)

```bash
uv run python run_evaluation.py \
  evaluation.max_samples=1000 \
  output.dir=./batch_1_results
```

### Batch 2 (samples 1001-2000)

```bash
# Note: This would require adding skip_samples parameter to the config
uv run python run_evaluation.py \
  evaluation.max_samples=1000 \
  dataset.skip_samples=1000 \
  output.dir=./batch_2_results
```

## Output Examples

### Expected Console Output

```
2024-01-15 10:30:45,123 - __main__ - INFO - Starting Phi-4 Multimodal ASR Evaluation
2024-01-15 10:30:45,124 - __main__ - INFO - Configuration:
model:
  name: microsoft/Phi-4-multimodal-instruct
  ...
2024-01-15 10:30:50,456 - __main__ - INFO - Initializing Phi-4 multimodal ASR model...
2024-01-15 10:31:15,789 - __main__ - INFO - Model loaded successfully
2024-01-15 10:31:16,012 - __main__ - INFO - Loading FluencyBank dataset from: /Users/Benjamin/dev/ssa/data/fluencybank/processed/fluencybank_segments.parquet
2024-01-15 10:31:18,345 - __main__ - INFO - Limited dataset to 100 samples
2024-01-15 10:31:18,346 - __main__ - INFO - Starting evaluation pipeline...
2024-01-15 10:31:18,347 - __main__ - INFO - Evaluating 100 samples...
Evaluating samples: 100%|████████████| 100/100 [15:32<00:00,  9.32s/it]
2024-01-15 10:46:50,678 - __main__ - INFO - ============================================================
2024-01-15 10:46:50,678 - __main__ - INFO - EVALUATION COMPLETED SUCCESSFULLY
2024-01-15 10:46:50,678 - __main__ - INFO - ============================================================
2024-01-15 10:46:50,678 - __main__ - INFO - Overall WER: 0.234 (23.4%)
2024-01-15 10:46:50,678 - __main__ - INFO - Overall CER: 0.156 (15.6%)
2024-01-15 10:46:50,678 - __main__ - INFO - Samples evaluated: 100
2024-01-15 10:46:50,678 - __main__ - INFO - Real-time factor: 0.87x
2024-01-15 10:46:50,678 - __main__ - INFO - Results saved to: ./evaluation_results
```

### Output Files Generated

After running the evaluation, you'll find these files in the output directory:

1. **evaluation_results.json** - Complete results in JSON format
2. **evaluation_samples.csv** - Per-sample results
3. **evaluation_summary.txt** - Human-readable summary
4. **transcription_examples.txt** - Best/worst examples

### Sample Summary Report

```
================================================================================
ASR EVALUATION SUMMARY REPORT
================================================================================

CONFIGURATION:
Model: microsoft/Phi-4-multimodal-instruct
Dataset: /Users/Benjamin/dev/ssa/data/fluencybank/processed/fluencybank_segments.parquet
Text Column: unannotated_text
Max Samples: 100

OVERALL METRICS:
Word Error Rate (WER):     0.234 (23.4%)
Character Error Rate (CER): 0.156 (15.6%)
Match Error Rate (MER):    0.201 (20.1%)
Word Info Lost (WIL):      0.267 (26.7%)
Word Info Preserved (WIP): 0.733 (73.3%)

DATASET STATISTICS:
Total Samples:     100
Total Duration:    1,250.45 seconds (20.8 minutes)
Avg Sample Length: 12.50 seconds

PERFORMANCE STATISTICS:
Total Inference Time: 1,087.32 seconds (18.1 minutes)
Samples per Second:   0.09
Real-time Factor:     0.87x

ERROR ANALYSIS:
Total Words:    2,456
Correct (Hits): 1,881 (76.6%)
Substitutions:  342 (13.9%)
Deletions:      124 (5.0%)
Insertions:     109 (4.4%)
```

## Troubleshooting

### Memory Issues

If you encounter out-of-memory errors:

```bash
# Force CPU usage
uv run python run_evaluation.py model.force_cpu=true

# Reduce batch size (process fewer samples)
uv run python run_evaluation.py evaluation.max_samples=50

# Limit audio length
uv run python run_evaluation.py dataset.max_audio_length=20.0
```

### Dataset Path Issues

If the dataset is not found:

```bash
# Check if file exists
ls -la /Users/Benjamin/dev/ssa/data/fluencybank/processed/fluencybank_segments.parquet

# Use absolute path
uv run python run_evaluation.py dataset.parquet_path=/full/path/to/your/dataset.parquet
```

### Model Loading Issues

If model loading fails:

```bash
# Try CPU mode
uv run python run_evaluation.py model.force_cpu=true

# Use a different cache directory
uv run python run_evaluation.py model.cache_dir=./my_models
```
