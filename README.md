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

This is numerical-record reproduction and reference implementation testing. It does **not** claim to retrain all archived base checkpoints from raw data with this one command. Raw datasets, trained checkpoints, and per-horizon base/target caches are not included in this release. Backbone and learned-adapter training components are included for inspection and reuse; a fully packaged raw-data-to-checkpoint runner remains outside this release. See [reproduction scope](docs/REPRODUCIBILITY.md).

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

The main 72-condition mean MSE reduction is 16.9917% relative to the static base. The full Fourier comparator is more accurate on average (19.6846%). Retained numerical arrays are not peak process memory, latency, or energy measurements.

The additional 2026-09-20 validation is kept separately from the manuscript's original tables. It supports a benefit over endpoint-only regression while retaining exceptions to the benefit of individual inputs. The main 72-condition reductions are 16.9917% for the complete method and 12.8513% for endpoint-only regression; their median retained sizes are 4,528 and 2,288 bytes.

Evaluation series were used during method development. Splitting those series retrospectively does not create an untouched test set. Prefix selection uses only preceding block losses, but its comparison with a suffix-selected oracle is a finite-candidate diagnostic, not a measurement of the entire research process's selection bias. The tested K=4 is an accuracy/storage choice, not the accuracy-optimal configuration.

## Sources and license

Dataset sources and preprocessing scope are listed in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). The compared learned adapter is a transfer of [TEFL v1](https://arxiv.org/abs/2602.22520v1), and the Fourier comparator is a transfer of [ELF v3](https://arxiv.org/abs/2502.12920v3). These are not reproductions of every experiment or configuration in those papers.

See [LICENSE](LICENSE). Cite this repository with the release tag or commit used. The initial release is `v0.1.0`; subsequent manuscript updates may add further material.
