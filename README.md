# Endpoint-Preserving Compressed Residual Correction

Reference code and numerical records for **Endpoint-Preserving Compressed Residual Correction for Multi-Horizon Time Series Forecasting**, by Takumi Fujimoto and Hiroaki Nishi.

The method augments compressed DCT residual coefficients with the final observed residual and current base-forecast coefficients. A separate online ridge regression predicts each retained component and channel; a causal rolling controller blends the reconstructed correction with a frozen base forecast.

## Reproduce the reported numerical summaries

Python 3.11 was used. From a fresh clone:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python reproduce.py
.venv/bin/python -m unittest discover -s src -p test_endpoint_review_validation.py
.venv/bin/python -m unittest discover -s src -p test_elf_real.py
```

`reproduce.py` checks the published source/result hashes, recomputes the principal paper summaries, and audits all 31,080 additional-validation result rows and 1,184 temporal-selection records against 148 block files. No GPU or original dataset is needed for these checks. The output distinguishes main legacy/refit conditions from separately trained MAE/SF joint bases.

This command audits numerical records. For fresh training and evaluation, use the separate pipeline below. Raw datasets and trained checkpoints are downloaded or generated locally, rather than bundled in Git.

## Recreate data, train, and evaluate

```bash
.venv/bin/python -m pip install -r requirements-training.txt
.venv/bin/python run_pipeline.py --smoke --fourier
```

The smoke run downloads ETTh1 and trains seed 0 with both backbones, legacy/refit variants, both learned-adapter warmups, and widths 2/4/64 (20 checkpoints). It evaluates all 21 correction candidates, temporal selection, learned adapters, and seven Fourier configurations, then compares matching losses with the paper tables. CUDA is used when available; `--device cpu` is supported. The full study is substantially more expensive:

```bash
.venv/bin/python run_pipeline.py --fourier
```

This prepares all eight series and trains the 72 main base conditions, four additional-site conditions, and 288 learned-group runs (including matched base-only runs). Outputs go to ignored `data/` and `runs/` directories. Completed training runs are reused; incomplete run directories raise an error. Fresh results never replace the archived paper records. See [commands, preprocessing qualifications, and tested scope](docs/REPRODUCIBILITY.md).

Base training defaults to archived **validation-selected** update counts. To repeat the original selection grid, use `train.py --selection validation-grid --output runs/grid-training`; this trains through 20,000 updates and selects only on the first-half validation interval. Supply that output path to `evaluate.py --input runs/grid-training`. The 78 explicitly recorded Rat snapshot adjustments preserve the evaluated input; their historical cleaning rationale is not recoverable.

## Apply the correction to supplied forecasts

Create an NPZ with `base` and `target` arrays of shape `[blocks, horizon, channels]`. Targets must be ordered chronologically, and each completed target block becomes available only after its forecast was issued.

```bash
.venv/bin/python replay.py forecasts.npz --output correction.npz
```

This uses the paper configuration: horizon 24, K=4, ridge=1, half-life=128 blocks, and blending window=32. The output includes issued predictions, raw/global block MSE and MAE, issued blending coefficients, and retained-array counts. The evaluator scores a block before updating from its target. The NPZ interface is an offline replay of that order, not permission to access future observations during live use.

## Contents

| Path | Contents |
|---|---|
| `src/boundary_context.py` | Original packed online regression and endpoint correction |
| `src/endpoint_review_validation.py` | Parameterized implementation, input ablations, and 21 fixed sensitivity candidates |
| `src/elf_real.py`, `src/elf_capacity.py`, `src/elf_transfer.py` | Transferred Fourier comparator and equivalent storage representations |
| `src/backbones.py` | Evaluated compact DLinear/PatchTST definitions and preprocessing |
| `prepare_data.py`, `config/` | Pinned downloads, ordered meter IDs, numerical hashes, Rat snapshot adjustments, selected base-training steps |
| `train.py`, `src/base_training.py` | Base selection/refit and paired learned-adapter training |
| `evaluate.py`, `verify_training.py`, `run_pipeline.py` | Chronological evaluation, comparison with archived losses, and complete pipeline |
| `src/learned_residual.py`, `src/joint_residual.py`, `src/spectral_residual.py`, `src/spectral_seed.py`, `src/small_joint.py` | Learned adapter and paired training components |
| `results/paper/` | Main and supplementary numerical tables for the manuscript |
| `results/endpoint_positions.json` | Condition/seed-level first, middle, mean, and endpoint comparisons |
| `results/validation/` | Additional input ablations and retrospective selection sensitivity |
| `docs/manuscript_*.tex` | Method, experimental setup, and supplementary-record definitions from the associated manuscript snapshot |
| `SOURCE_MANIFEST.json` | Source-relative provenance and SHA-256 for copied artifacts |
| `RELEASE_MANIFEST.json` | SHA-256 of the complete published snapshot, excluding Git internals and this manifest itself |

For the PyTorch components and spectral-flatness tests:

```bash
.venv/bin/python -m pip install -r requirements-training.txt
.venv/bin/python -m unittest discover -s src -p test_spectral_residual.py
```

## Interpretation

Release `v0.2.1` clarifies that the block-mean control duplicates the DC residual input only in the DC regression; it supplies the DC coefficient to each non-DC regression. It also adds Table S7 and an audited Rat-exclusion analysis. Without Rat, mean MSE reductions are 17.9887% from Static and 4.6646% from endpoint-only regression, with wins in all 60 remaining main conditions. The width-64 MAE/SF joint-base endpoint-only comparisons give 6.6544%/6.8083% reductions over ten seed-averaged pairs each. These are retrospective subset results, not validation of Rat's unrecovered cleaning decisions.

Run `python report_rat_exclusion.py` to regenerate all 108 full-period summaries from the archived CSV and block losses under `runs/rat_exclusion/`. `reproduce.py` also checks these summaries. `results/rat_exclusion/` contains the published summary and source hashes; original forecasts and loss records are unchanged.

The main 72-condition mean MSE reduction is 16.9917% relative to the static base. The full Fourier comparator is more accurate on average (19.6846%). Retained numerical arrays are not peak process memory, latency, or energy measurements.

The additional validation is incorporated into manuscript Tables VII/VIII and S5/S6. It supports a benefit over endpoint-only regression while retaining exceptions to the benefit of individual inputs. The main 72-condition reductions are 16.9917% for the complete method and 12.8513% for endpoint-only regression; their median retained sizes are 4,528 and 2,288 bytes. The direct paired MSE reduction from endpoint-only regression is 4.7066%, with wins in 72/72 conditions; removing residual DCT inputs gives five exceptions on ETTh2/DLinear.

Evaluation series were used during method development. Splitting those series retrospectively does not create an untouched test set. Prefix selection uses only preceding block losses, but its comparison with a suffix-selected oracle is a finite-candidate diagnostic, not a measurement of the entire research process's selection bias. The tested K=4 is an accuracy/storage choice, not the accuracy-optimal configuration.

## Sources and license

Dataset sources and preprocessing scope are listed in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). The compared learned adapter is a transfer of [TEFL v1](https://arxiv.org/abs/2602.22520v1), and the Fourier comparator is a transfer of [ELF v3](https://arxiv.org/abs/2502.12920v3). These are not reproductions of every experiment or configuration in those papers.

See [LICENSE](LICENSE). Cite this repository with the release tag or commit used. Release `v0.2.0` added the raw-data-to-evaluation pipeline; `v0.2.1` adds the Rat-exclusion analysis and corrects the block-mean interpretation in the manuscript snapshot.
