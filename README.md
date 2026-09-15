# M-SARCUSE and MUStARD Response Reproduction Package

This package contains two code components:

1. Reproduction and evaluation of five external multimodal baselines for M-SARCUSE under a fixed four-fold, episode-disjoint protocol.
2. MUStARD–Friends transcript alignment, observed next-reply extraction, official media verification, and 16 kHz mono PCM audio extraction.

## Environment

The experiments used an NVIDIA RTX PRO 6000 with 96 GB of memory. The fixed configurations record the Python, PyTorch, Transformers, Accelerate, bfloat16, and SDPA settings. The neural baselines require a compatible CUDA environment. The MUStARD data pipeline mainly uses the Python standard library and FFmpeg for media inspection and audio conversion.

## Directory Structure

- `code/msarcuse/scripts`: data preparation, feature extraction, execution, configuration freezing, validation, and analysis for the five baselines.
- `code/msarcuse/config`: sanitized experiment configurations with relative paths.
- `code/mustard_response/scripts`: MUStARD–Friends alignment, observed reply extraction, and official media processing.
- `code/requirements.txt`: Python dependencies for both components.
- `MANIFEST.json`: source and sanitized code hashes and the file inventory.

## Running the Five Baselines

Run all commands from the package root. Place the MaSaC development-only data, fixed encoder weights, and output directories at the relative paths specified in the configurations. If your directory structure differs, change only the machine-specific path fields and freeze the configuration again. Keep the data folds, random seeds, thresholds, target counts, candidate construction, and held-out data boundaries unchanged.

Execution order:

```text
python code/msarcuse/scripts/materialize_external_baseline_training.py
python code/msarcuse/scripts/materialize_external_baseline_tasks.py
python code/msarcuse/scripts/manifest_external_baseline_models.py
python code/msarcuse/scripts/preflight_external_baselines.py
python code/msarcuse/scripts/run_msh_comics_external.py --help
python code/msarcuse/scripts/run_external_baseline.py --help
python code/msarcuse/scripts/run_external_neural_baseline.py --help
python code/msarcuse/scripts/analyze_external_baselines.py
python code/msarcuse/scripts/validate_external_baseline_completion.py
```

## Running the MUStARD–Friends Pipeline and Audio Audit

Run the scripts in numerical order. The media audit uses media from the official MUStARD release; it does not fill gaps with YouTube or other third-party sources. Target audio is converted to 16 kHz, mono, 16-bit PCM WAV. The independent validation script checks media coverage, file hashes, duration, channels, original sample rates, and the final audio format.
