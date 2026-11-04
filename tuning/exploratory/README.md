# Exploratory analyses

Analyses outside the frozen harness. `run_candidate.py` hashes only `tuning\*.py`, so the files here can
change without a refreeze, and nothing here feeds the search or the spec's results. All experimental code
lives here, never only in a temporary folder; outputs go to `C:\data\ks4_tuning\exploratory\`, in the
same subfolders. Data paths are written into the scripts; run them with the `kilosort` environment. None
of these results is in a results document yet.

**No hybrid data** (from 2026-11-03): new sorts are of the recordings as downloaded, and kobe sorts are
scored without hybrid truth (`evaluate.py` without `--hybrid-truth`). Earlier analyses here used the hybrid
kobe excerpt.

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

## `iteration\`: a third round of template matching

| File | Question |
|---|---|
| `borrowed_templates.py`, `borrowed_templates.md` | How many final units have no extraction template of their own (their spikes caught by templates other units own), how much worse those spikes fit, and roughly how many spikes a third round would add; reads the scored sorts of `autocurate\` |

## `autocurate\`: KS2.5's autocurate, and a `set_cutoff`-like step for KS4

| File | Question |
|---|---|
| `stage_breakdown.py` | What each step of KS2.5's autocurate does (traced reruns of `ks25_setup\`): units, spikes and contamination after every merge, split and cut |
| `set_cutoff.py` | Python port of the lab pipeline's `set_cutoff` (with `calcCcg`, `ccg_rg` and `aCorrPointProcess`) |
| `validate_port.py` | The port against MATLAB on the traced reruns: same spikes kept and same per-template cut-offs in both cycles, both excerpts |
| `ks4_sort_scored.py`, `queue_scored.sh` | KS4 sorts as `run_sort.py` runs them, copying an existing sort's settings, plus each spike's template-matching score (`score.npy`, from `save_extra_vars`); default and H8, seed 1, both excerpts, outputs in `autocurate\sorts\` (each reproduces its original sort exactly, `same_sort.py`) |
| `apply_cut.py`, `plot_cut.py` | The ported `set_cutoff` (strict, one pass) on those sorts, cut-off lowered from `Th_learned` + 1-4 to `Th_learned`, on `tradeoff.py`'s curves next to KS2.5 standard and standard_low |
| `make_variants.py`, `score_variants.sh`, `summarise_variants.py` | S of the cut sorts (H8 at widths 2 and 4, default at 4; seed 1) with the harness's `evaluate.py`/`compare.py` (kobe) and `transfer.py` (MEC), outside `run_candidate.py`, against a matched random control (in each unit, as many spikes as the cut removed there, drawn at random; three draws) |
| `placement_audit.md` | Where in KS4 a `set_cutoff`-like step could go and what it would do at each point: a five-agent brainstorming audit with read-only checks on the scored sorts (the check scripts were not kept), and a staged test plan |

## `duplicates\`: duplicate spikes in KS4 and KS2.5

`duplicates_audit.md` is the five-agent audit (whether KS4 duplicates more than KS2.5; how each handles
duplicates; how to choose the lag threshold). The scripts were written by the agents, one subfolder per lens.
Unlike the rest of this folder, they write their outputs next to themselves; their outputs as run are in
`C:\data\ks4_tuning\exploratory\duplicates\<lens>\` (the `empirics` histogram cache, ~400 MB, was not kept).

| Folder | Question |
|---|---|
| `empirics\` | Within- and cross-unit short-lag profiles for every sort (KS4 default, H8; KS2.5 standard, standard_low; both excerpts), before/after each removal step and the harness's cleaning; matched cells; Hill contamination under dedup windows of 0.25-1 ms; hybrid pairs |
| `mechanisms\` | How each sorter makes duplicates; echo pairs in large units; within- vs cross-template pairs; cross-unit profiles by distance; hybrid ground truth |
| `methods\` | KS4's `remove_duplicates` reproduced and varied (which member kept, chaining, wider windows); raw joint refit of pair members; batch-edge enrichment; template-level lags in KS2.5's traced stages |
| `threshold\` | Mixture model and per-unit empirical-Bayes windows; score-order asymmetry estimate of true duplicates; raw one-vs-two-spike classification; the score-order rule; Hill under different censored periods |
| `skeptic\` | Competing explanations: removal-step simulation, raw one- vs two-spike fits with controls, KS2.5's view of KS4's pairs, local co-activity and artefacts, cross-unit raw test, consequences for the comparisons |

## `long_data\`: the 2-hour segments

| File | Role |
|---|---|
| `download.sh` | Kobe 354-7554 s and MEC 7575-14775 s (each containing its 20-min excerpt) with `extract_segment.py`, into each excerpt's `search\long\` |
| `queue_long.sh` | Kobe's segment, one sort at a time, seed 1: KS4 default and H8 with per-spike scores (`autocurate\ks4_sort_scored.py --data`), then KS2.5 standard and standard_low (stopped during KS2.5; those sorts were redone by `queue_ks25.sh`) |
| `queue_ks4.sh` | KS4 default and H8 with per-spike scores, seed 1: kobe's raw 20-min excerpt (the length comparison's baseline), then MEC's segment |
| `queue_ks25.sh` | KS2.5 standard and standard_low, seed 1, after `queue_ks4.sh`: kobe's raw 20-min excerpt, then kobe's and MEC's segments |
| `memlog.py` | System RAM and the largest python/MATLAB process every 30 s while a queue runs |
| `window.py` | A long sort restricted to its 20-min excerpt's window (times shifted), scored by the harness like a 20-min sort; `--cut W` applies the set_cutoff-like cut to the whole long sort first |
| `score_long.py` | Every sort of the round (20-min and 2-hour; KS4 also cut) as a window folder, then kobe S (`evaluate.py`, no hybrid truth), MEC position S (`transfer.py`) and both decoders; skips finished steps |
| `summarise_long.py` | `summary_long.txt` and the figures `long_S.png`, `long_tradeoff.png` (uncut) and `long_tradeoff_cut.png` (KS4 cut at w=4), in `long_data\windows\` |
| `score_after_sorts.sh` | `score_long.py` then `summarise_long.py`, once `queue_ks25.sh` has finished |

## `ks25_setup\`: setting up the KS2.5 sorts

| File | Role |
|---|---|
| `gpu_smoke.m` | MATLAB sees the GPU (CUDA forward compatibility) and KS2.5's mex files are on the path |
| `chanmap_check.m` | The neuropixels pipeline's channel maps for both excerpts (`chanmap_*.mat`, written to the current folder), checked against the probe json |
| `ks25_sort_traced.m`, `ks25_sort_traced.py` | `ks25_sort.m` with `autocurate`'s steps run one by one and every stage's spikes saved (`stages.npz`), then adapted and cleaned; outputs in `ks25_traced\` |
| `queue.sh` | The KS2.5 sorts after the first, one at a time (the commands of [ks25_comparison.md](../ks25_comparison.md)) |
| `queue_low.sh` | The six KS2.5 "standard_low" sorts, one at a time ([search_ks25.md](../search_ks25.md) §8) |
| `score_low.sh`, `summarise_low.py` | Their scoring (§8's commands, one job at a time) and its tables, with "standard" alongside |
