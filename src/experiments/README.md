# ASR Evaluation Pipeline

This directory contains a comprehensive evaluation pipeline for Automatic Speech
Recognition (ASR) models using the Phi-4 multimodal model on the FluencyBank
dataset.

## Overview

The evaluation script (`run_evaluation.py`) provides:

- **Comprehensive ASR Metrics**: Word Error Rate (WER), Character Error Rate
  (CER), Match Error Rate (MER), Word Information Lost (WIL), and Word
  Information Preserved (WIP)
- **Detailed Analysis**: Per-sample results, error analysis, performance
  statistics
- **Configurable Pipeline**: Hydra-based configuration for easy parameter tuning
- **Rich Output**: JSON results, CSV data, summary reports, and transcription
  examples

## Quick Start

### Prerequisites

Make sure you have the required dependencies installed:

```bash
uv add jiwer
```

Note: `polars` is already included in the project dependencies.

### Basic Usage

Run the evaluation with default settings:

```bash
cd src/experiments
python run_evaluation.py
```

### Custom Configuration

Override configuration parameters from the command line:

```bash
# Evaluate only 50 samples
python run_evaluation.py evaluation.max_samples=50

# Use a different text column
python run_evaluation.py dataset.text_column=annotated_text

# Force CPU usage
python run_evaluation.py model.force_cpu=true

# Change output directory
python run_evaluation.py output.dir=./my_results
```

### Configuration File

The main configuration is in `config/phi_4_multimodal_eval.yaml`. Key sections:

- **model**: Model settings, inference parameters, and prompts
- **dataset**: Dataset path and preprocessing options
- **evaluation**: Evaluation settings and metrics to compute
- **output**: Output directory and file format options

## Output Files

The evaluation generates several output files:

1. **`evaluation_results.json`**: Complete results in JSON format
2. **`evaluation_samples.csv`**: Per-sample results in CSV format
3. **`evaluation_summary.txt`**: Human-readable summary report
4. **`transcription_examples.txt`**: Best and worst transcription examples

## Key Features

### ASR Metrics

- **WER (Word Error Rate)**: Primary metric for ASR evaluation
- **CER (Character Error Rate)**: Character-level accuracy
- **MER (Match Error Rate)**: Alternative to WER, more robust to alignment
- **WIL/WIP**: Word Information Lost/Preserved metrics

### Performance Analysis

- Real-time factor (inference time vs. audio duration)
- Samples per second processing rate
- Error breakdown (substitutions, deletions, insertions)
- WER distribution statistics

### Quality Analysis

- Best and worst transcription examples
- Per-speaker analysis (if speaker info available)
- Duration-based analysis

## Configuration Options

### Model Settings

```yaml
model:
  name: "microsoft/Phi-4-multimodal-instruct"
  cache_dir: "./models"
  force_cpu: false
  max_new_tokens: 500
  temperature: 0.7
  system_message: "You are an expert audio transcriptionist..."
  user_prompt:
    "Transcribe the speech that is contained in this audio recording..."
```

### Dataset Settings

```yaml
dataset:
  parquet_path: "/path/to/fluencybank_segments.parquet"
  audio_sample_rate: 16000
  max_audio_length: 30.0
  text_column: "unannotated_text"
  include_timing: true
  include_speaker_info: true
```

### Evaluation Settings

```yaml
evaluation:
  max_samples: 100 # 0 for all samples
  log_interval: 10
```

## Example Output

```
============================================================
EVALUATION COMPLETED SUCCESSFULLY
============================================================
Overall WER: 0.234 (23.4%)
Overall CER: 0.156 (15.6%)
Samples evaluated: 100
Real-time factor: 0.87x
Results saved to: ./evaluation_results
```

## Troubleshooting

### Common Issues

1. **Model loading fails**: Check GPU memory, try `model.force_cpu=true`
2. **Dataset not found**: Verify the `dataset.parquet_path` is correct
3. **Out of memory**: Reduce `evaluation.max_samples` or use CPU mode

### Dependencies

The script requires:

- `jiwer`: ASR evaluation metrics
- `polars`: Data handling and CSV export (already in project dependencies)
- `tqdm`: Progress bars
- `soundfile`: Audio file handling
- `transformers`: Phi-4 model
- `torch`: Neural network framework

### Performance Tips

- Use GPU for faster inference (`model.force_cpu=false`)
- Limit audio length (`dataset.max_audio_length=30.0`)
- Start with a small sample size for testing (`evaluation.max_samples=10`)

## Advanced Usage

### Custom Prompts

Modify the prompts for different transcription styles:

```yaml
model:
  system_message: "You are a medical transcriptionist..."
  user_prompt: "Transcribe this medical dictation accurately..."
```

### Batch Processing

For large datasets, process in batches:

```bash
# Process first 1000 samples
python run_evaluation.py evaluation.max_samples=1000 output.dir=./batch_1

# Process next 1000 samples
python run_evaluation.py evaluation.max_samples=1000 dataset.skip_samples=1000 output.dir=./batch_2
```

### Integration with Experiments

The results can be integrated into larger experimental workflows:

```python
from run_evaluation import Phi4ASREvaluator
from omegaconf import OmegaConf

config = OmegaConf.load("config/phi_4_multimodal_eval.yaml")
evaluator = Phi4ASREvaluator(config)
results = evaluator.run_evaluation()
print(f"WER: {results.mean_wer:.3f}")
```
