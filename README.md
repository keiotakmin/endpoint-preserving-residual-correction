# EPOC: Online Residual Correction

This repository contains an implementation of endpoint-preserving online correction for fixed multi-horizon forecasters. It compresses a completed residual block with DCT coefficients, retains its final value, and updates component-wise ridge regressions after the target block becomes available.

## Install and test

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s src -p 'test_*.py'
.venv/bin/python -m unittest test_pipeline
.venv/bin/python reproduce.py
```

## Correct supplied forecasts

Create an NPZ file containing chronological `base` and `target` arrays with shape `[blocks, horizon, channels]`, then run:

```bash
.venv/bin/python replay.py forecasts.npz --output correction.npz
```

The default configuration uses horizon 24, four retained DCT components, ridge strength 1, a 128-block forgetting half-life, and a 32-block blending window. The program issues each forecast before updating from that block's target.

## Train and evaluate

Install the additional training dependencies, then run a small ETTh1 check or the full data and training pipeline:

```bash
.venv/bin/python -m pip install -r requirements-training.txt
.venv/bin/python run_pipeline.py --smoke --fourier
.venv/bin/python run_pipeline.py --fourier
```

The full pipeline downloads or prepares eight numerical series and trains fixed-base and learned-adapter models. Generated datasets, checkpoints, and evaluations are written to ignored `data/` and `runs/` directories. See [reproduction details](docs/REPRODUCIBILITY.md) for data sources, preprocessing, command options, and verification scope.

## Repository map

| Path | Purpose |
|---|---|
| `src/boundary_context.py`, `src/endpoint_review_validation.py` | Correction implementation and input/setting controls |
| `src/elf_real.py`, `src/elf_capacity.py`, `src/elf_transfer.py` | Fourier comparator and storage accounting |
| `src/backbones.py`, `prepare_data.py`, `train.py`, `evaluate.py` | Forecast models, data preparation, training, and chronological evaluation |
| `config/` | Dataset definitions, input hashes, and archived training selections |
| `results/validation/`, `results/rat_exclusion/` | Saved numerical audit records |
| `SOURCE_MANIFEST.json`, `RELEASE_MANIFEST.json` | Source provenance and file hashes |

Code is provided under [LICENSE](LICENSE). Original dataset licenses remain with their providers.
