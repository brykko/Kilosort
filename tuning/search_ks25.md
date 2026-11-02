# KS2.5 comparison: the winner against the lab's Kilosort 2.5 pipeline

Plan, frozen before the first KS2.5 sort. Rules: [evaluation_spec.md](evaluation_spec.md) v0.17, §14
(how KS2.5 is run, adapted and scored); everything else in the spec applies unchanged. Reported, not
a gate.

## 1. Aim

The search compared KS4 settings with KS4 defaults. The sorts the lab actually uses come from the
neuropixels pipeline's Kilosort 2.5 (Gardner et al. 2022), preset "standard". This comparison asks
whether H8 ([round4_results.md](round4_results.md)) beats that pipeline, and where default KS4 stands
against it, on both excerpts:

- **Kobe (search excerpt, hybrid):** S, S_track, every guard, hybrid recovery, units reproduced across
  replicates and the stricter-cleaning check.
- **MEC (transfer excerpt):** tracked-position S and its two guards.

## 2. Sorter

- **Code:** `C:\code\github\neuropixels` at `6626f68` (clean), KS2.5 under
  `ExternalToolboxes/KilosortToolbox/Versions/kilosort_v2.5`, mex binaries built for CUDA 12.9;
  MATLAB R2025a with CUDA forward compatibility (the RTX 5070 Ti, compute 12.0, is newer than
  R2025a's CUDA libraries; MATLAB recompiles its GPU libraries for it).
- **Preset "standard":** `Th` [10 6], `nfilt_factor` 8, `nblocks` 5, `sig` 20, `fshigh` 300,
  `whiteningRange` 64, `momentum` [5 400], `initNBatch` 16000 with auto-stopping (500-batch lookback),
  `nInnerIter` 500, `triageMergeThresh` 0.975, `curationSplitThresh` 1.0, `minFR` 0.05,
  `templateMinSpikes` 50, bad-batch exclusion at 5 MAD; `autocurate` runs merges, splits and cut-off
  twice, then a final merge.
- **Run:** `ks25_sort.py` calls `ks25_sort.m` (spec §14 lists its differences from the pipeline:
  only the seed reaches the sort) and adapts the output (cluster labels, templates, spike positions).
- **Replicates:** seeds 1, 2, 3 (seed 1 is the pipeline's sort).
- **Smoke test** (first 60 s of the kobe hybrid excerpt, 300 learning batches): every step runs on
  the GPU; 676 clusters, of which 597 pass the waveform screen; spike times within a sample of the
  hybrid truth; cleaning and the waveform screen read the adapted output without change.

## 3. Runs

Six sorts, one at a time, each into its own folder (the sort folder itself):

| Excerpt | Folder | Seeds |
|---|---|---|
| Kobe (hybrid) | `C:\data\ks4_tuning\26648_kobe_2020-08-21\runs\ks25_standard_r<k>` | 1, 2, 3 |
| MEC | `C:\data\ks4_tuning\25843_roger_2019-09-11\runs\ks25_standard_r<k>` | 1, 2, 3 |

    python ks25_sort.py --data <excerpt .bin> --meta <data>/search/source.ap.meta --probe <data>/search/probe_3a_opt3.json --out <folder> --seed <k>

Kobe seed 1 runs first and is timed; the smoke test learned at ~0.6 s per batch, so up to ~3 h per
sort if auto-stopping never triggers. `fproc.dat` (~27 GB) is written into the sort folder and deleted
when the sort ends; `pc_features.npy` and `template_features.npy` are deleted after adaptation.

## 4. Scoring

Kobe (`<K>` = `C:\data\ks4_tuning\26648_kobe_2020-08-21\runs`):

1. Each replicate against the five default replicates, logged:
   `python run_candidate.py run --existing-sort <K>/ks25_standard_r<k> --name ks25_standard --seed <k> --phase comparison --settings '{"sorter": "ks2.5 standard"}' --rationale "..."`
2. H8 against KS2.5: `compare.py --candidate <H8 seed k sort> --default <K>/ks25_standard_r1 r2 r3`,
   for H8 seeds 1–3 (guards with τ from KS2.5's spread; matched units against KS2.5 replicate 1).
3. Units reproduced across replicates (`reproducibility.py`, all three pairs) for KS2.5 and for
   default (seeds 1–3); H8's are in round 4.
4. Stricter duplicate cleaning (`strict_cleaning.py`) for KS2.5's replicates; default's and H8's are
   in `strict_cleaning_round3.json` and `strict_cleaning_round4.json`.

MEC (`<M>` = `C:\data\ks4_tuning\25843_roger_2019-09-11\runs`):

5. KS2.5 against default: `transfer.py --default <M>/default_s1 s2 s3 --winner <M>/ks25_standard_r1 r2 r3 <M>/H8_s1 s2 s3`
   (α scanned on default as in the transfer test, so H8 is scored as in `transfer.json`).
6. H8 against KS2.5: `transfer.py --default <M>/ks25_standard_r1 r2 r3 --winner <M>/H8_s1 s2 s3 --alpha <α from 5>`.
7. Units reproduced across KS2.5's replicates (`reproducibility.py`, no hybrid truth).

## 5. Report

`ks25_comparison.md`: per replicate and mean, S, ΔS and ΔS_track against default and H8's against
KS2.5, every guard, hybrid recovery, eligible and reproduced units, stricter-cleaning S, KS2.5's
run-to-run SD, sort time; the same on MEC for position S. Then the README status. Nothing here feeds
back into the KS4 settings.

## 6. Addendum: Bayesian decoding (spec v0.18)

Added after the comparison above, frozen before it is run on the comparison's sorts (a functional
test on kobe default seed 1 only checked that the script runs). For every sort of the comparison
(kobe: default seeds 1–5, KS2.5 r1–3, H8 seeds 1–3; MEC: default, KS2.5 and H8, seeds 1–3),
`bayes_decode.py` decodes tracked head direction and 2D position from the eligible units, cross-validated
over the two time halves (spec §14). Reported: the median error at 20, 100 and 500 ms windows (100 ms
in the results table), and at 100 ms the error with 25–1600 randomly drawn units, to show where
direction saturates and how the sorters compare at equal unit counts.

    python bayes_decode.py --group default <sorts> --group ks25 <sorts> --group H8 <sorts> --tracking <npz> --raw-artefacts <npz> --t0 <s> --duration 1200 [--hybrid-truth <npz>] --out <data>/runs/bayes_decode.json

## 7. Addendum: correlation-based decoding (spec v0.19)

Added after the Bayesian decoding of §6, which ranked the sorters differently from S on MEC position;
an independent-unit Poisson decoder may lose accuracy when units repeat one another's information.
Frozen before it is run on the comparison's sorts (a functional test on kobe default seed 1 only
checked that the script runs). `corr_decode.py` decodes the same sorts, on the same windows, by
population-vector correlation, in two variants fixed now: 'raw' and 'z' (spec §14). Reported as in §6:
median error at 20, 100 and 500 ms, and at 100 ms the error with 25–1600 randomly drawn units.

    python corr_decode.py --group default <sorts> --group ks25 <sorts> --group H8 <sorts> --tracking <npz> --raw-artefacts <npz> --t0 <s> --duration 1200 [--hybrid-truth <npz>] --out <data>/runs/corr_decode.json
