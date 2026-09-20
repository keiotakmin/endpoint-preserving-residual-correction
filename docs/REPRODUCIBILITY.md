# Reproduction scope and record definitions

## What this snapshot permits

1. Recompute the main numerical claims from unrounded condition-level tables.
2. Recompute endpoint-position ablations from all 576 condition/policy records.
3. Audit all additional-validation period means and candidate choices from saved block losses.
4. Run and test the online correction and Fourier comparator implementations.
5. Inspect the exact backbone definitions and learned-adapter training components used in the study.

It does not include raw datasets, all base checkpoints, or original per-horizon prediction caches. Consequently a new clone cannot rerun every base-training experiment or regenerate every paper plot directly from raw data. This limitation is explicit so that numerical-record reproduction is not mistaken for complete end-to-end replication. `replay.py` accepts user-supplied base/target caches.

## Input datasets and preparation

- ETT: https://github.com/zhouhaoyi/ETDataset (ETTh1, ETTh2, ETTm1, ETTm2).
- Appliances: https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction. Remove timestamp and `rv1`, `rv2`; retain 26 numerical channels.
- Building Data Genome 2: https://doi.org/10.1038/s41597-020-00712-x. Rat is the main site; Fox and Panther are additional sites. The study selected 15 electricity meters per site by lowest whole-series missingness and forward-filled then backward-filled the prepared series. Specific meter IDs/raw-data acquisition files are not bundled in this snapshot.

Respect the terms attached to the original dataset releases. The available files here contain derived error records, not raw observations.

Lookback/horizon/stride are 96/24/24 samples. The base-training/evaluation boundary is `floor(T/2)`. The final 20% of the first half is validation; normalization statistics use the whole first half. These are custom chronological partitions, not the standard ETT partitions. Detailed training settings and departures from source methods are in `manuscript_04_experimental_setup.tex`.

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

Main summaries average relative reductions over 72 conditions. MAE/SF summaries first average seed losses within each of 12 dataset/backbone pairs. Temporal selection occurs per stream; selected losses are subsequently seed-averaged. Seeds and architectures sharing a dataset are not independent datasets. No significance claim follows from win counts.

## Versioned provenance

`SOURCE_MANIFEST.json` preserves original repository-relative paths as provenance identifiers, not runnable paths. The snapshot's executable code and data use only paths within this repository. `RELEASE_MANIFEST.json` hashes every distributed file except itself. Use the annotated Git tag/commit to identify the version and run `reproduce.py` to check integrity.
