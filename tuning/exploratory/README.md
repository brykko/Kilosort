# Exploratory analyses

Analyses outside the frozen harness. `run_candidate.py` hashes only `tuning\*.py`, so the files here can
change without a refreeze, and nothing here feeds the search or the spec's results. All experimental code
lives here, never only in a temporary folder; outputs go to `C:\data\ks4_tuning\exploratory\`, in the
same subfolders. Data paths are written into the scripts; run them with the `kilosort` environment. None
of these results is in a results document yet.

## `isolation\`: spikes in well-isolated units

Contamination is the refractory-violation estimate (Hill et al. 2011, τ_c 0.25 ms, τ_r 1.5 ms); 'plain'
assumes independent spikes, 'local' takes the pair density from the unit's own autocorrelogram at
10–30 ms (allowing for theta co-modulation; conservative).

| File | Question |
|---|---|
| `isolation.py` | Correctly assigned spikes, Σ N(1 − ĉ), per sort; hybrid check and perturbation audit (`--out`) |
| `rate_mix.py` | How much of the difference in % misassigned between sorters is their firing-rate mix (Kitagawa split by rate band) |
| `comod.py` | The same with co-modulation allowed for (plain, binned and local estimates) |
| `rate_regression.py` | Share of the difference predicted by unit firing rates alone (pooled regression on log rate) |
| `matched_contam.py` | Contamination of the same cells in two sorts (one-to-one matched units) |
| `low_vs_standard.py` | KS2.5 standard against standard_low (seed 1): which units are found again, how their spikes and contamination change, where the extra spikes are; standard seeds 1 and 2 as a control |
| `where_misassigned.py` | Misassigned spikes in cells shared with KS2.5 against units only one sorter has |
| `rate_decomp.py` | Rate and per-spike terms of score differences (S, isolation); reads `isolation.py`'s outputs |
| `tradeoff.py`, `plot_tradeoff.py` | Correctly assigned against misassigned spikes as units are admitted cleanest first, cross-validated over the halves; KS2.5 also before `set_cutoff` (the traced rerun of `ks25_setup\`) and with preset "standard_low" |
| `trace_units.py` | Where KS4's contamination comes from relative to KS2.5 (spike-level labels), and where KS2.5's own units went |
| `amp_check.py` | Amplitudes of the KS4 spikes that KS2.5 doesn't output |
| `cutoff_trace.py` | Whether the KS4 spikes that KS2.5 doesn't output were never extracted by KS2.5 or deleted by its `set_cutoff` (uses the traced reruns of `ks25_setup\`) |
| `amplitude.py`, `plot_amplitude.py` | Contamination against unit amplitude (KS4 default, KS2.5 standard and standard_low, H8), measured from the raw data the same way for every sorter (one GPU pass per excerpt) |

## `uniqueness\`: locally unique, strongly tuned units

| File | Question |
|---|---|
| `unique_tuning.py` | Per-unit tuning tables and pairs within 500 µm, for scores counting strongly tuned units without a nearby tuning twin (`--out-dir`) |
| `analyse.py` | Calibration, scores across ρ*, audit, resolution sweeps |
| `twin_profile.py` | Twin rate against distance, by pair type and sorter |

## `decoding\`: decoding checks in the KS2.5 comparison

| File | Question |
|---|---|
| `sparse_units.py` | Whether KS2.5's sparsely firing units are what helps its MEC position decoding |
| `matched_decode.py` | MEC position decoded from the cells KS2.5 and H8 share (each sorter's version) and from their unshared units |

## `autocurate\`: KS2.5's autocurate, and a `set_cutoff`-like step for KS4

| File | Question |
|---|---|
| `stage_breakdown.py` | What each step of KS2.5's autocurate does (traced reruns of `ks25_setup\`): units, spikes and contamination after every merge, split and cut |
| `set_cutoff.py` | Python port of the lab pipeline's `set_cutoff` (with `calcCcg`, `ccg_rg` and `aCorrPointProcess`) |
| `validate_port.py` | The port against MATLAB on the traced reruns: same spikes kept and same per-template cut-offs in both cycles, both excerpts |
| `ks4_sort_scored.py`, `queue_scored.sh` | KS4 sorts as `run_sort.py` runs them, copying an existing sort's settings, plus each spike's template-matching score (`score.npy`, from `save_extra_vars`); default and H8, seed 1, both excerpts, outputs in `autocurate\sorts\` (each reproduces its original sort exactly, `same_sort.py`) |
| `apply_cut.py`, `plot_cut.py` | The ported `set_cutoff` (strict, one pass) on those sorts, cut-off lowered from `Th_learned` + 1-4 to `Th_learned`, on `tradeoff.py`'s curves next to KS2.5 standard and standard_low |

## `ks25_setup\`: setting up the KS2.5 sorts

| File | Role |
|---|---|
| `gpu_smoke.m` | MATLAB sees the GPU (CUDA forward compatibility) and KS2.5's mex files are on the path |
| `chanmap_check.m` | The neuropixels pipeline's channel maps for both excerpts (`chanmap_*.mat`, written to the current folder), checked against the probe json |
| `ks25_sort_traced.m`, `ks25_sort_traced.py` | `ks25_sort.m` with `autocurate`'s steps run one by one and every stage's spikes saved (`stages.npz`), then adapted and cleaned; outputs in `ks25_traced\` |
| `queue.sh` | The KS2.5 sorts after the first, one at a time (the commands of [ks25_comparison.md](../ks25_comparison.md)) |
| `queue_low.sh` | The six KS2.5 "standard_low" sorts, one at a time ([search_ks25.md](../search_ks25.md) §8) |
| `score_low.sh`, `summarise_low.py` | Their scoring (§8's commands, one job at a time) and its tables, with "standard" alongside |
