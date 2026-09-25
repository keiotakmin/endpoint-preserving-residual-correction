# EPOC: Endpoint-Preserving Online Correction

Reference code, numerical records, and manuscript for **EPOC: Endpoint-Preserving Online Correction With Compressed Residual State for Multi-Horizon Time Series Forecasting**, by Takumi Fujimoto and Hiroaki Nishi.

EPOC corrects a fixed multi-horizon forecaster after each completed target block. It retains low-order DCT coefficients and the final residual of the previous block, shares that endpoint across component-wise online ridge regressions, and blends the reconstructed correction with the base forecast.

The current paper evaluates eight series, two backbones, three seeds, and two base-training phases: **96 fixed-base conditions**. EPOC reduces mean condition-wise MSE by 15.40% and MAE by 9.35% from the uncorrected base with 6,352 B median retained auxiliary state. Full ELF reaches 19.29% mean MSE reduction with 474,048 B. The archived CSV identifiers `Proposed` and `proposed` denote EPOC.

## Verify the published records

Python 3.11 was used. From a fresh clone:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python reproduce.py
.venv/bin/python -m unittest discover -s src -p 'test_*.py'
```

`reproduce.py` checks the release hashes and the current 96-condition summaries, then audits the previously released block-level validation records. This numerical audit needs neither a GPU nor the original datasets. The historical validation cohort contains 72 main conditions, four initial Fox/Panther conditions, and 72 width-64 joint-base streams; it is separate from the current 96-condition paper comparison.

The [current PDF](paper/main.pdf), [buildable LaTeX source](paper/main.tex), [outline](paper/OUTLINE_JA.md), and Japanese section translations are in `paper/`. The associated CSVs are in `results/paper/`: `tableS01_all_conditions.csv` has the 96 EPOC/base/ELF condition records, `table03_all_conditions.csv` has all 672 seven-method records, and `table05_endpoint_selection.csv` holds the nine equal-size endpoint comparisons. `SOURCE_MANIFEST.json` links copied files to the research workspace; `RELEASE_MANIFEST.json` hashes the published checkout.

## Recreate data, train, and evaluate

```bash
.venv/bin/python -m pip install -r requirements-training.txt
.venv/bin/python run_pipeline.py --smoke --fourier
```

The smoke route downloads ETTh1 and trains seed 0 with both backbones, both base-training phases, both learned-adapter warmups, and widths 2/4/64. It evaluates the correction candidates and Fourier comparator against the archived records. CUDA is used when available; `--device cpu` is supported. The full route is substantially more expensive:

```bash
.venv/bin/python run_pipeline.py --fourier
```

The full route prepares all eight series, trains the 96 fixed bases and 288 learned-group runs, and writes outputs under ignored `data/` and `runs/`. Completed training runs are reused. The published seven-method comparison also includes transferred comparator records in `results/paper/`; this command does not retrain every external comparator. See [reproduction scope and data preparation](docs/REPRODUCIBILITY.md). Raw datasets and checkpoints are downloaded or generated locally and are not bundled in Git.

## Apply EPOC to supplied forecasts

Create an NPZ with `base` and `target` arrays of shape `[blocks, horizon, channels]` in chronological order:

```bash
.venv/bin/python replay.py forecasts.npz --output correction.npz
```

The replay uses horizon 24, `K=4`, ridge strength 1, a 128-block forgetting half-life, and a 32-block blending window. It issues each corrected forecast before updating from that block's target. The output includes corrected predictions, block losses, blending coefficients, and retained-array counts.

## Repository map

| Path | Contents |
|---|---|
| `paper/` | Current 13-page manuscript PDF, compilable source, figures, tables, outline, and Japanese translations |
| `results/paper/` | Condition-level and aggregate numerical records for the manuscript |
| `src/boundary_context.py`, `src/endpoint_review_validation.py` | EPOC implementation and input/setting controls |
| `src/elf_real.py`, `src/elf_capacity.py`, `src/elf_transfer.py` | Fourier comparator and storage accounting |
| `src/backbones.py`, `prepare_data.py`, `train.py`, `evaluate.py` | Base models, pinned preprocessing, training, and chronological evaluation |
| `config/archived_bases.json`, `config/site_main96_bases_manifest.json` | Audited training selections for the 96 fixed-base conditions |
| `results/validation/`, `results/rat_exclusion/` | Retained retrospective audits from the earlier release |
| `docs/REPRODUCIBILITY.md` | Dataset sources, verified scope, and detailed pipeline commands |

Dataset sources and license conditions are listed in [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). The compared learned adapter follows [TEFL v1](https://arxiv.org/abs/2602.22520v1); the Fourier comparator follows [ELF v3](https://arxiv.org/abs/2502.12920v3). The code is provided under [LICENSE](LICENSE). Cite this repository with the commit used.
