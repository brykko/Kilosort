# KS4 behaviour vs recording length: audit (2026-11-02)

Code audited: branch `ks4-tuning` at `3b4683f` (upstream base `0e07514`). Question: which KS4 steps are invariant to
recording length / spike count, given that benchmarks (ours: 20-min excerpts; the KS4 paper: 45-min simulations) are
much shorter than the 1–5 h recordings sorted in practice.

Method: five parallel static audits (preprocessing/drift, graph clustering, template matching, spike-train
statistics, compute/memory + the authors' claims), then CPU experiments on cached stage outputs, existing sorts and
synthetic data, then an adversarial re-check of the main claims against the code (corrections folded in below),
then end-to-end GPU tests: sorts of 5/10/20-min windows scored against hybrid ground truth, a pass-1 duration sweep,
and a template-age test (section 6). Nothing here was run on data longer than 20 min; hour-scale numbers are
extrapolations and are labelled as such.

Data: kobe excerpt `26648_kobe_2020-08-21` (20 min, hybrid version with 40 inserted units) and MEC excerpt
`25843_roger_2019-09-11` (20 min); 225 existing KS4 logs, all 600 batches (20 min). Scripts: `scaling_audit/scripts/`
(copied from the session scratchpad; some hardcode scratchpad paths for their inputs/outputs). Sort outputs:
`C:\data\ks4_tuning\26648_kobe_2020-08-21\runs\scaling_audit\`.

## Summary

**KS4 is not length-invariant, but most of the dependence is benign or self-correcting; three things are not.**

| Stage | Verdict | What changes on a 1–5 h recording | Severity |
|---|---|---|---|
| Filtering, CAR, per-batch processing | invariant | — | — |
| Whitening (one matrix per session) | length-dependent (data) | thresholds are in session-average noise units; non-stationary noise shifts effective thresholds over time | low–moderate, data-dependent |
| Universal templates / wPCA | **length-dependent** | learned only from the first ~100 min (500k-clip buffer, `break` when full) | low (generic shapes) |
| Drift registration (rigid) | **length-dependent** | batches > ~125 µm from the majority frame can lock to a wrong offset, silently | **high when it happens** |
| Drift: spike binning | runtime O(T²) | ~190 s at 3 h vs 1.5 s at 20 min | runtime only |
| Graph clustering: landmarks | **length-dependent** | landmarks per group grow ∝ T until the 25k cap (pass 2 median group: ~2 h; pass 1: ~4 h) → more leaves; pass-1 clusters 843 / 1008 / 1137 at 5 / 10 / 20 min (measured) | medium |
| Final unit count | **length-dependent (measured)** | 716 / 784 / 849 units at 5 / 10 / 20 min from the same persistent neurons; ~100 more separations at 20 than at 5 min; ground-truth units unaffected | see section 6 |
| Splitter bimodality test | length-dependent | inflated scores at small n → spurious splits of nodes < ~500–700 spikes at 20 min, fewer at long T | low–medium (default); none in the winner (threshold 0) |
| Tree refractory veto / final CCG merge | **length-dependent (power)** | sparse-shortcut and dead-zone regimes shrink ~T-fold; low-rate fragments get merged far more often | medium; high for the winner |
| Time-split units | **length-dependent (data)** | time-disjoint fragments of one neuron are never repaired by the final merge; the tree collapses time-disjoint sub-clusters even if they are two neurons | medium, frequency unknown |
| Templates (whole-recording mean, frozen) | **length-dependent (data)** | mismatch grows with template age: own-half advantage 0.2% (kobe) vs 0.7% (MEC) at ~10 min age (measured); ~3–7% spikes lost at 5 h in MEC (extrapolated) | data-dependent, low–medium |
| Detection thresholds, peel limit, features | invariant per batch | — | — |
| 'good' label | weakly length-dependent | ~4% (default) / ~8% (winner) of labels differ from a 20-min sort; band sharpens around 10.6% contamination | low–medium |
| Host RAM | **length-dependent, fails** | doubling buffers: ~75 GB transient at ~2.4 h (default) / ~1.8 h (winner); 5 h ≈ 150 GB | **high (OOM on 63 GB)** |
| Time | ~linear (+30% at 5 h) | ~24 / 82 / 140 min at 1 / 3 / 5 h on this machine (projection) | — |
| VRAM | fine to 5 h, one exception | `align_U` OOMs at ~13k pass-1 templates | low unless pass-1 count grows |
| int32 overflow | none to 5 h | — | — |

**For the tuned settings ("winner", H8: `cluster_downsampling` 10, bimodality 0, Th 7/7):** `cluster_downsampling=10`
stops mattering by ~3 h (every pass-2 group hits the 25k cap at either 10 or 20); the winner's tree decisions are
made through the refractory veto, whose regime changes most with length (30% of same-group pairs undecidable at 20 min,
~6% at 3 h); and its higher spike counts bring the RAM failure point forward to ~1.8 h. The winner was selected on
20-min behaviour that a long recording won't reproduce exactly — direction of the effects is mostly benign
(more powered refractory decisions), but the selection margin doesn't transfer quantitatively.

**What KS4's authors claim** (compute auditor, via docs/paper/issues): all published quality benchmarks are 45-min
simulations; multi-hour statements are anecdotes about speed/memory (e.g. ~300 GB RAM for ~13 h). The 25k landmark
cap is justified as keeping neighbourhoods "comparable for short and long recordings" — argued, not measured.

## 1. Preprocessing and drift correction

- **Per-batch filtering and CAR** (`io.py:1114-1126`, `preprocessing.py:121-154`): invariant.
- **Whitening** (`preprocessing.py:96-119`): averages every 25th batch over the whole session (precision saturated at
  20 min). One `Wrot` for the session sets the noise unit for `Th_single_ch`, `Th_universal`, `Th_learned`; if noise
  level or correlation changes over hours (movement, state, reference), effective thresholds drift. A 20-min excerpt
  gets a matrix matched to itself and can't show this.
- **Universal templates / wPCA** (`spikedetect.py:52-72`, called with a literal `nskip=25` at `:225`): buffer of
  500,000 clips; sampling stops at the first batch that would overflow. Measured 4110 clips per sampled batch on kobe
  (3194–4685) → buffer full after ~122 sampled batches ≈ **100 min**. Longer recordings take wPCA/wTEMP (the feature
  basis for both clustering passes and matching) from the first ~100 min only. Low severity — shapes are generic and
  round 4 found template settings irrelevant — but real.
- **Rigid registration** (`datashift.align_block2`): initial target is batch `min(300, Nbatches//2)` (minute 10 of any
  recording ≥ 20 min); each of 9 shifting iterations searches ±15 bins × 5 µm = ±75 µm via circular `torch.roll`,
  shifts accumulate, and the target is re-set to the mean of all aligned batches each iteration (so it follows the
  majority frame). No warning when a batch's best shift is at the window edge.
  Synthetic checks (two independent ones, CPU, real `bin_spikes`/`align_block2`):

  | injected drift | batches with error > 10 µm |
  |---|---|
  | step 100 µm | 0 (recovered by later iterations) |
  | step 125 / 150 / 200 µm | 24% / 11–16% / 33% |
  | linear 0–100 µm | 0 |
  | linear 0–200 µm | 16% |
  | linear 0–300 µm | 43% |

  Failing batches lock to wrong offsets (errors up to ~190 µm), so a unit sits at two depths. Real data is harder
  than these synthetics (fingerprints also change with unit turnover and state). **This is the one place where a long
  recording can fail badly and silently.** The kobe excerpt's own drift spans −6.5 to +4.5 µm, so round 3's finding
  that drift settings don't matter says nothing about multi-hour recordings.
- **Non-rigid residual**: ±25 µm per block, smoothed (`drift_smoothing` 0.5 on shift/batch/block); irrelevant at the
  default `nblocks=1`, a cap on differential drift if `nblocks>1`.
- **Reference frame**: `dshift` is never re-centred; with one-way drift D the largest |shift| ≈ D rather than D/2, and
  channels shifted past the probe ends are attenuated (2/4/8/14 of 374 channels below half gain at 25/50/100/150 µm).
- **`bin_spikes`** (`datashift.py:33-35`): one full scan of all spikes per batch → O(Nbatches × N): 1.5 s at 20 min,
  ~190 s at 3 h. Runtime only.

## 2. Graph clustering

- **Landmarks** (`clustering_qr.neigh_mat:28-35`): every `cluster_downsampling`-th spike (20), capped at
  `max_cluster_subset` (25,000) per group; time-uniform. At 20 min the cap never binds (runs 019 and 020 — cds 10 with
  and without `max_cluster_subset=50000` — are identical). Group sizes (96 groups, 40 µm slabs): pass 1 median 38k,
  pass 2 median 89k (default). Cap binds (default) in the median pass-2 group at ~112 min and the median pass-1 group
  at ~4.4 h; winner (cds 10): ~56 min and ~133 min. At 5 h, 94% (default) / 100% (winner) of pass-2 groups are capped.
- **Doubling duration ≈ halving `cluster_downsampling`** at the graph stage: same landmark density and kNN radius.
  Not identical: spikes per landmark (votes) are 2× in the duration case, and everything downstream (splitter,
  correlograms) sees 2× the spikes only in the duration case. The stock `assign_iclust` penalty depends on
  n_nodes/n_spikes but only breaks ties (< 1 vote); the branch's `weighted`/`cluster_resolution` path is scale-free.
- **Measured** (CPU, winner settings, pass 2, 6 groups × 5 seeds, cached features; `scratchpad\clu_scale\`):

  | | 5 min | 10 min | 20 min | 10 min, cds 10 | 20 min, landmarks capped at the 10-min count |
  |---|---|---|---|---|---|
  | leaves (sum of 6 groups) | 263 | 305 | 331 | 340 | 294 |
  | final clusters | 115 | 123 | 123 | 124 | 119 |

  Leaves rise with duration (10→5 min ×1.15 [1.03, 1.31]; 20→10 ×1.09 [0.94, 1.26]), match the lower-cds condition,
  and the rise disappears when landmarks are capped (×0.96) — landmark-driven, as predicted. Last 10 min ≈ first 10
  min, so it is duration not content. **But the final pass-2 count is flat** (20→10: ×0.99 [0.93, 1.04]; same at
  stock settings): the splitter's merges absorb the extra leaves. Pass 1 has no refractory veto and, with bimodality
  0, ends with ≈ its leaf count, so the effect should reach pass 1 (more templates on long recordings), consistent with
  cds 20→10 raising pass-1 clusters 1137→1381 at 20 min. No run hit the 200-iteration cap (max 55).
- **kmeans++ seeds, tree `crat`, connectivity 0.2, spatial groups**: invariant (fractions / geometry). 200 seeds per
  group ≫ units per group. `cluster_min_group_spikes` never binds on this probe.
- **Bimodality** (`swarmsplitter.check_split`/`bimod_score`, threshold 0.6): in-sample ridge regression (fixed
  `0.01·I`) on the node's spikes, D = 60–90. One Gaussian unit (D=90): median score 0.79 / 0.58 / 0.40 / 0.21 / 0.01 at
  n = 300 / 500 / 1000 / 3000 / 30000; P(score ≥ 0.6) 0.97 / 0.43 / 0 / 0 / 0. So spurious splits need n ≲ 500–700;
  scores are inflated (but sub-threshold) up to ~3000. Long recordings push nodes out of that range. Inactive in the
  winner (threshold 0).

## 3. Templates and matching

- Per-batch machinery is invariant: `Th_universal`/`Th_learned` are absolute in whitened units (no per-batch
  normalisation), the peel limit is per batch, features are the spike's own waveform in a fixed PC basis. Template
  choice maximises the projection onto unit-norm templates, so **pure amplitude change never reassigns a spike** — it
  only loses spikes below threshold.
- **Templates are the mean over all of a cluster's spikes across the whole recording** (`clustering_qr.py:566-569`)
  and are frozen for matching — no per-window update anywhere (KS2.5's final pass also freezes templates, so the
  KS2.5 comparison is like-for-like here). Mismatch with the current waveform grows with template age. Threshold
  losses are steep: a uniform 10% projection loss drops ~12–14% of a template's spikes.
- **Measured waveform stability over 20 min** (`scratchpad\t126\`; noise floor = same-minute split halves; growth
  term extrapolated only):

  | | shape 1−cos growth, lag 1→15 min | fitted exponent | amplitude RW SD per √h (p50/p90) | predicted spike loss, 5 h recording (random walk) |
  |---|---|---|---|---|
  | kobe (6 sorts) | 0.0004–0.0019 (≈ jitter) | not identified | 0.010–0.016 / 0.07 | < 0.5% (shape); ~1–3% (amplitude) |
  | MEC | 0.0076–0.0098 | p 1.05–1.20 (random walk; trend rejected) | 0.042 / 0.164 | ~3–7% overall, unit p90 ~7–15% |

  Early-vs-late separation (cross-validated d′) in MEC reaches 2 for the mean unit at ~150 min lag; ~40% of MEC units
  would be d′ > 2 apart across 5 h (random walk). No sign of time-epoch splits within 20 min in either dataset (low
  power). Assumptions: growth continues at the fitted exponent (upper bound if drift saturates); a whole-recording
  mean template has an average age of ~D/6 (random walk). **Data-dependent: negligible for kobe-like recordings, a
  few percent for MEC-like ones.** 20 min cannot tell whether the random walk continues or levels off.
- Pass-2 consequence: a slowly changing neuron is a trajectory in feature space whose length scales with duration;
  spikes reassigned to a template in another 40 µm slab are clustered in another group.

## 4. Spike-train statistics (CCG decisions)

- **All correlogram decisions use raw counts in fixed bins** (1 ms, ±500 ms). R12 is a ratio to the correlogram's
  own baseline (T and slow common rate changes cancel); Q12 tests whether a refractory dip is present. Ratio thresholds
  are asymptotically invariant; significance gates and the `Q00 < .25` sparse shortcut move with T.
- **'good' label** (R12 < 0.2 and Q12 < 0.2): a soft threshold at 20 min that sharpens at 3 h to ~10.6% contamination
  (2c − c² = 0.2). The width depends on rate (≈5–15% at 5 Hz; ~2–20% at 2 Hz; ~9–12% at 20 Hz). Clean units can't lose
  'good' with more data; units at ~11–15% contamination can. Rate floor for clean units: ~0.3 Hz at 20 min →
  ~0.1 Hz at 3 h. Real data (first 10 min vs 20 min): 3.6–4.4% of labels flip in default sorts, 7.8–8.0% in winner
  sorts, balanced in direction — the expected disagreement between a 20-min and a multi-hour sort of the same units.
- **Tree veto** (`swarmsplitter.check_CCG`, pass 2 only) and **final merge** (`CCG.check_CCG`): three regimes by
  Q00 = r1·r2·T·1 ms:

  | regime | 20 min | 3 h |
  |---|---|---|
  | sparse shortcut (collapse regardless of identity) | r1·r2 < 0.21 Hz² | < 0.023 Hz² |
  | dead zone (same-neuron fragments neither collapsed nor merged) | between | between |
  | powered (≥ 89% correct) | ≳ 2.5 Hz² | ≳ 0.28 Hz² |

  Halves of a 1 Hz neuron: collapsed 7% at 20 min, 91% at 3 h. Winner sort, same-group pairs: sparse/dead/powered =
  5/27/67% at 20 min → 2/13/85% at 1 h → 1/6/94% at 3 h. The winner's tree merges come from this veto, so its 20-min
  behaviour on low-rate units does not transfer. `n = K/2` in `CCG_metrics` is right for ACGs and makes the CCG test
  conservative by √2 (upstream).
- **Time-split units** (real spike trains, seed 1, split at 600 s): final merge rejoins a same-neuron split with 60 s
  of overlap (0.95) but not with none (0.00 below 20 Hz). The stock tree veto collapses time-disjoint sub-clusters
  with certainty — one neuron or two (different-neuron control: 1.00) — unless the connectivity test splits first or
  `tree_refractory_significance`/`correlogram_shape_criterion` is on. Hours with units appearing/disappearing produce
  more such pairs. Also found: an abrupt handover between two trains makes a one-sided CCG that looks refractory, so
  above ~20–40 Hz even different neurons can pass the merge test (length-independent artefact).
- **Non-stationarity**: a common slow gain cancels; a contaminant modulated differently from the unit (e.g. across
  sleep/running) hides contamination (R12 0.148 → 0.044 in simulation), so multi-state hours make 'good' more lenient.
- `remove_duplicates` and the `len > 10` gate: invariant. Merge seeding uses `is_ref` computed once (stale after
  merges; the saved label is recomputed). The `np.unique`-count indexing in `merging_function` is safe at both call
  sites (ids contiguous and non-empty) — a latent fragility, not a bug.

## 5. Compute and memory

- **Host RAM is the binding constraint.** `st`/`tF` buffers grow by doubling from 1e6 rows with
  `concatenate(old, zeros_like(old))` → transient 4× the old capacity (verified on CPU against a model to 0.1%).
  tF = 10 ch × 6 PCs × float32 = 240 B/spike. The universal-pass `tF[:k]` view keeps its whole capacity buffer alive
  through learned extraction; the drift-pass `st0` and `clu0` are kept for plots even with `make_plots=False`; the
  end-of-extract `tF[isort]` copies again.
  Peak live arrays (default settings, kobe spike rates 4.02M universal / 8.74M learned per 20 min; process baseline
  ~4 GB extra): ~11 GB at 20 min, ~22 GB at 1 h, ~43 GB at 2 h; the learned-buffer doubling to 128M rows happens past
  64M learned spikes (**~2.4 h default, ~1.8 h winner**) with a transient of **~75 GB**; ~150 GB at 5 h. On 63 GB RAM +
  48 GB pagefile: heavy paging or `DefaultCPUAllocator: not enough memory` from ~2.4 h, certain failure by 5 h. Even
  without the doubling, the final merge / positions / pc_features steps need 67–96 GB at 5 h. Consistent with the
  docs' "~300 GB for ~13 h". Cheap fixes (not applied): grow by 1.5× or preallocate from the batch count; free the
  universal `tF` before `template_matching.extract`; don't keep `st0`/`clu0` without plots; gather pc_features per
  cluster.
- **KS4's logged "sys" memory** (14–17 GB → 28–30 GB over a 20-min sort) is system-wide used RAM, a snapshot not a
  peak; ~8–10 GB of the rise is not live arrays (file cache / allocator / other processes).
- **Time**: ~linear; superlinear terms are faiss kNN (∝ n·landmarks per group until the cap) and `bin_spikes` (T²).
  Projection with fused kernels: ~24 / 82 / 140 min at 1 / 3 / 5 h (pure linear: 21 / 64 / 107). From 1 h the file
  no longer fits in the page cache; NVMe bandwidth is sufficient.
- **VRAM**: fine through 5 h (pass-2 clustering ~9 GB at 5 h in the largest group), except `align_U` at ~1.1 MB per
  pass-1 template → OOM at ~13k templates (winner: ~3.9k at 20 min). Relevant if pass-1 counts grow with duration.
- **int32**: no overflow risk found to 5 h (int64 sample indices, faiss/torch int64, CSR nnz far below 2³¹).

## What a 20-min benchmark can and cannot tell us

- **Transfers**: anything per batch (filtering, detection, matching thresholds, features), ratio-based decisions in
  expectation, recovery of well-isolated units (hybrid units: identical at 5, 10 and 20 min).
- **Does not transfer quantitatively**: the number of units (rises with length from the same neurons: 716 → 849 from
  5 to 20 min), low-rate unit handling (veto/merge regimes, 'good' rate floor, bimodality at small n), pass-1 template
  counts, `cluster_downsampling`'s effect (moot once capped), drift-setting conclusions (the excerpts have ~no
  drift), template-age losses (depend on the recording's waveform stability).
- **Implication for benchmarking on excerpts**: a setting tuned on 20 min is tuned at a point on a moving curve —
  unit counts and low-rate decisions continue to change with length. Comparisons between settings at a fixed
  excerpt length are fair to each other, but their absolute behaviour on 1–5 h recordings will differ.
- **Cannot be seen at all**: registration failure beyond ~125 µm, time-split units, non-stationary noise or
  contamination across behavioural states, RAM failure.

## 6. End-to-end duration tests (GPU)

Default settings, seed 1. Scripts in `scaling_audit/scripts/`: `e1_windows.py`,
`e1_score.py`, `e1_match.py`, `e1_seedfloor.py`, `e1_splitpairs.py`, `gpu_tests.py`, `gpu_queue.sh`,
`tage_analyze.py`; outputs in `runs\scaling_audit\`. The 20-min sort at HEAD is bit-identical in counts and per-unit
agreement to `hybrid_default_seed1` (849 units, 572 good).

**Window sorts of the hybrid excerpt** (`tmin`/`tmax`; 4 × 5 min, 2 × 10 min, 20 min):

| length | pass-1 clusters | pass-2 clusters | units | good | hybrid recovered / 40 | median agreement |
|---|---|---|---|---|---|---|
| 5 min (4 windows) | 821–842 | 734–772 | 695–730 | 459–503 | 29–30 | 0.929–0.956 |
| 10 min (2) | 1005–1006 | 836–859 | 768–800 | 519–542 | 31–32 | 0.948–0.949 |
| 20 min | 1137 | 932 | 849 | 572 | 29–31 in each window (5 seeds: 29–32) | 0.949–0.958 |

- **Ground-truth units are length-invariant.** Per inserted unit and window, window-sort agreement minus 20-min-sort
  agreement (same window): mean −0.004, median 0.000; 10 units better and 25 worse by > 0.05 out of 240 unit-windows;
  oversplit inserted units (≥ 2 sorted units holding ≥ 10%) 0–5 per window in both.
- **Unit counts rise with length, from the same neurons.** By rate band (5 → 10 → 20 min): < 0.3 Hz 3 → 7 → 24;
  0.3–1 Hz 57 → 70 → 86; 1–3 Hz 213 → 239 → 254; 3–10 Hz 264 → 292 → 308; > 10 Hz 178 → 177 → 177. Every 20-min unit
  fires throughout (none has > 70% of its spikes in one 5-min quarter; 842–844 of 849 have ≥ 30 spikes in each 5-min
  window), so the extra units are not neurons that are only present part of the time.
- **Unit-by-unit matching** (shared spikes within ±0.4 ms, inside each window), against a seed-to-seed floor
  (20-min seed 1 vs seeds 2/3):

  | | 5-min windows vs 20-min | 20-min seed vs seed (floor) |
  |---|---|---|
  | short units split by the 20-min sort | 111–141 | 128–150 |
  | 20-min units split by the short sort | 161–189 | 115–141 |
  | 20-min units with no counterpart (< 50% in one unit) | 91–115 | 31–36 |

  So a 20-min sort separates ~100 more units out of the same persistent activity than a 5-min sort (unmatched units:
  median ~700 spikes per 5 min, i.e. ~2.3 Hz — not the low-rate tail). Where one 5-min unit cleanly divides into two
  20-min units (24–32 per window vs 16–20 seed floor), none of the pairs is cross-refractory (median R12 0.70–0.85):
  they look like distinct neurons that the short sort merged. This check is partly circular — KS4's final merge
  already joins template-similar cross-refractory pairs — so it supports, but does not prove, that longer recordings
  resolve real units rather than oversplitting.
- **Pass-1 sweep on the same detected spikes** (`gpu_tests.py sweep`): 843 / 1008 (first 10) / 1020 (last 10) / 1137
  clusters at 5 / 10 / 20 min; first 10 min at `cluster_downsampling` 10 gives 1240. The landmark effect reaches pass 1
  (+13% from 10 to 20 min). Halving cds adds more than doubling duration at equal landmark count, consistent with the
  bimodality test: a longer recording also gives each tree node more spikes, so fewer small-sample spurious splits
  partly offset the extra landmarks. Peak clustering VRAM 0.10 / 0.17 / 0.33 GB (∝ N).
- **Template age** (`gpu_tests.py early|late`): pass-1 templates learned from the first or the last 10 min only,
  matched over all 20 min:

  | | spikes early/late arm, first half | spikes early/late arm, second half | own-half advantage (interaction) | hybrid detection recall |
  |---|---|---|---|---|
  | kobe | 1.0107 | 1.0067 | +0.4% | 0.996 in both arms and halves; no unit differs by > 2% |
  | MEC | 1.0195 | 1.0047 | **+1.5%** | — |

  Each arm catches ~0.7% more spikes in its own half in MEC (~10-min template-age difference) vs ~0.2% in kobe —
  matching the CPU prediction from waveform stability (MEC ~0.6% at this age; kobe ≈ 0). This supports the
  random-walk model only over this short range; the hour-scale figures (MEC ~3–7% of spikes at 5 h) remain an
  extrapolation.

### Not tested

Registration failure on real long drift, whitening non-stationarity, time-split units and RAM failure need longer
data (the 3.19 h source of the kobe excerpt is on `moser:ro`). The CCG/tree regime shifts were simulated and tested
on truncated real spike trains, not on longer sorts.
