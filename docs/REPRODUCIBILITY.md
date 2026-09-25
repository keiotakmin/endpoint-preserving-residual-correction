# Reproduction scope and record definitions

## What this snapshot permits

1. Recompute the stored 96-condition numerical summaries from unrounded condition-level tables.
2. Recompute endpoint-position ablations from all 576 condition/policy records.
3. Audit all additional-validation period means and candidate choices from saved block losses.
4. Run and test the online correction and Fourier comparator implementations.
5. Download pinned data and verify the eight evaluated numerical series.
6. Train both frozen-base variants and the paired learned-adapter groups, then replay corrections and Fourier comparators.

Raw datasets, checkpoints, and per-horizon forecasts are generated locally by `run_pipeline.py`; they are not included in Git. `reproduce.py` remains a lightweight audit of archived numerical records. The pipeline is a separate training/evaluation path, not an automatic regeneration of every historical exploratory experiment or publication layout. `replay.py` also accepts user-supplied base/target caches.

## Input datasets and preparation

- ETT: https://github.com/zhouhaoyi/ETDataset (ETTh1, ETTh2, ETTm1, ETTm2).
- Appliances: https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction. Remove timestamp and `rv1`, `rv2`; retain 26 numerical channels.
- Building Data Genome 2: https://doi.org/10.1038/s41597-020-00712-x. Rat is the main site; Fox and Panther are additional sites. The study selected 15 electricity meters per site by lowest whole-series missingness. `config/datasets.json` fixes all 45 IDs and their order, including historical tie choices. Downloads use a pinned Git LFS media URL for the cleaned electricity table. Fox and Panther reproduce by forward filling and then backward filling. Rat additionally requires the 78 explicit cell replacements in `config/bdg2_rat_snapshot_adjustments.csv` to match the archived input. The source table's SHA-256 is `b6ffc9b4dfcefe5c753594730a08ae822b0d50fec6815abb8f185591e6c630a3`.

The Rat replacements were obtained by comparing the archived prepared CSV with the pinned source after filling. Each row identifies the zero-based row index, meter, filled source value, and archived value. The script checks the source value before replacing it. The historical cleaning decisions that produced this discrepancy are not recovered; these are compatibility records for the evaluated snapshot, not a newly inferred general cleaning rule. No experimental results were changed. All eight prepared arrays must match their recorded SHA-256 after conversion to contiguous little-endian float32, both before and after CSV serialization. Hashes identify numerical inputs, not timestamp formatting.

Respect the terms attached to the original dataset releases. Apart from the small Rat compatibility record, observations are obtained from the original providers. Dataset licenses remain those of their providers, not this repository's code license.

Lookback/horizon/stride are 96/24/24 samples. The base-training/evaluation boundary is `floor(T/2)`. The final 20% of the first half is validation; normalization statistics use the whole first half. These are custom chronological partitions, not the standard ETT partitions. Training selections are in `config/archived_bases.json`; implementation settings are in `train.py` and `evaluate.py`.

## Individual pipeline commands

After installing `requirements-training.txt`, run from the repository root:

```bash
python prepare_data.py
python train.py --group base
python train.py --group learned
python evaluate.py --sensitivity --fourier
python verify_training.py
```

Use `--datasets ETTh1 --seeds 0` on each training command for the tested small subset. Default base training repeats the archived validation-selected update count, with original initialization, optimizer, and sampling. `--selection validation-grid --output runs/grid-training` reruns all seven validation milestones; evaluate it with `--input runs/grid-training --output runs/grid-evaluation`. A validation-grid run may choose different steps if numerical differences change a close validation decision. No evaluation target is used in step/epoch selection.

Each training directory contains `checkpoint.pt`, `forecasts.npz`, and `training.json` (metadata, selected step/epoch, training or validation logs, elapsed training time, versions, and input hash). Forecast arrays have shape `[blocks,24,channels]`; `adapter` is the raw learned-adapter forecast when present. `evaluate.py` writes per-stream block arrays, per-condition metrics, and temporal selections under `runs/evaluation/`. It evaluates the shared global blend for Fourier, not the auxiliary ELF-specific fast/slow weighting-rule analysis. Storage accounting for Fourier remains available in the reference modules and archived records.

The full pipeline creates 384 checkpoints: 96 base runs (eight series, two backbones, three seeds, two training phases) and 288 learned runs (six series, two backbones, three seeds, two warmups, and base-only plus three joint widths). The 24 Fox/Panther choices added to the fixed-base group are recorded in `config/site_main96_bases_manifest.json` and merged into `config/archived_bases.json`. The older retrospective validation cohort uses 76 base runs and 72 width-64 joint runs; its 148-stream audit is retained separately from the 96-condition comparison. Main results average condition-wise ratios; learned results average seed losses before forming ratios. `verify_training.py` compares individual matching MSE/MAE records rather than pooling cohorts.

## Verification scope for v0.2.0

All eight downloads/prepared inputs were checked against their archived numerical hashes. On an NVIDIA A100 with Python 3.11, NumPy 1.26.4, pandas 2.1.4, and PyTorch 2.7.0, fresh ETTh1/seed-0 training covered both backbones, legacy/refit, both warmups, all three widths, and matched base-only runs (20 checkpoints). The 16 runs with archived per-run base MSE references matched to a maximum absolute difference of `1.12e-16`; see `training_validation.json`. The remaining four are the separately trained base-only controls. Evaluation checks are recorded in `evaluation_validation.json`. This validates the packaged route on that subset; it is not a claim that all 384 checkpoints were retrained for this release. Different hardware/software may produce larger floating-point differences. `verify_training.py` defaults to an absolute tolerance of `1e-6` and reports deviations for inspection.

The 44 matching corrected MSE/MAE comparisons (including full Fourier and learned-adapter raw/global forecasts) agree to a maximum absolute difference of `6.11e-16`. All 21 candidates' block losses agree exactly on the eight smoke streams represented in the archived retrospective cohort (`sensitivity_validation.json`). Reloading all 20 saved checkpoints reproduces their base and adapter predictions exactly (`checkpoint_validation.json`). The full seven-milestone validation grid was additionally rerun for ETTh1/DLinear/seed 0, selected 2,000 updates as archived, and reproduced both legacy/refit forecasts (`grid_validation.json`). Environment details are in `environment.json`.

Run `python -m unittest test_pipeline` for portable input/order safeguards, and the `src/test_*` suites described in the README for regression, causality, Fourier equivalence, and SF behavior. Nine tests passed for this release.

## Additional-validation arrays

`results/validation/blocks_*.npz` contains:

- `names`: candidate order (the first 13 are one-factor setting sensitivities).
- `losses`: `[candidate, block, policy, metric]`; policy is `[raw, global]`, metric is `[MSE, MAE]`, averaged over horizon/channel within each block.
- `alpha`: issued global blending coefficients.
- `static`: `[block, metric]` reference losses.
- `sizes`: `[candidate, policy]` retained numerical-array bytes for a single fixed candidate.
- `meta`: dataset, backbone, seed, phase, cohort, and stream ID.

`all_results.csv` contains full-period, half (suffix), quarter2, quarter3, and quarter4 results. Indices use `[start, stop)`. Selection records use `[0,start)` to choose the global-MSE-minimizing candidate, then score `[start,stop)`. Regression/controller state continues across the boundary; targets remain available only after issuance. The candidate itself is fixed within each scored interval. Oracle selection reads the scored interval and is explicitly retrospective.

The candidate sets are 13 setting variants and all 21 variants including summaries/input controls. They are not memory-matched. Selection costs are not represented by any one candidate's retained-byte count.

The main summaries average relative reductions over 96 conditions. The retained retrospective validation summaries average over the earlier 72-condition main cohort; its MAE/SF groups first average seed losses within each of 12 dataset/backbone pairs. Temporal selection occurs per stream; selected losses are subsequently seed-averaged. Seeds and architectures sharing a dataset are not independent datasets. No significance claim follows from win counts.

## Versioned provenance

### Rat-exclusion sensitivity (v0.2.1)

`report_rat_exclusion.py` reads only the previously archived full-period results and block losses. It audits 6,048 candidate/policy rows from 144 main and width-64 joint streams, then generates 108 comparisons: three cohorts, with/without Rat, raw/global policies, and nine reference constructions. Main cohorts average condition-wise relative reductions; joint cohorts first average the three seed losses per dataset/backbone. The Rat-excluded sizes are 60 main conditions and ten pairs for each warmup. Additional Fox/Panther sites are not part of these cohorts.

Run `python report_rat_exclusion.py`; the regenerated `runs/rat_exclusion/summary.csv` must match `results/rat_exclusion/summary.csv`. `audit.json` identifies the source CSV, all 144 block files, aggregation script, and output hashes. No models were retrained or input observations modified for this analysis. It establishes persistence of the reported input-construction gains without Rat, not robustness to alternative Rat preprocessing. The archived Rat evidence remains conditional on its documented snapshot adjustments.

The mean-summary interpretation is also clarified: after scaling, the mean equals the retained DC coefficient. Since a component-wise regressor otherwise sees only its own residual coefficient, this is a duplicate residual feature only for the DC output component. For non-DC components it provides access to the DC coefficient; recoverability from the full retained state is different from redundancy within an individual regression. The original mean-control predictions and results are unchanged.

`SOURCE_MANIFEST.json` preserves original repository-relative paths as provenance identifiers, not runnable paths. The snapshot's executable code and data use only paths within this repository. `RELEASE_MANIFEST.json` hashes every distributed file except itself. Use the annotated Git tag/commit to identify the version and run `reproduce.py` to check integrity.
