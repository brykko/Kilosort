# Kilosort 4 settings search: evaluation spec

**Status:** draft v0.6, not frozen. Items marked **[DECIDE]** need your call; items marked **[PHASE 0]** are numbers set by the validation runs and then frozen.

**Freezing:** the spec is frozen when (1) every [DECIDE] item is settled, (2) Phase 0 has passed and its [PHASE 0] numbers are filled in, and (3) the spec and the evaluation harness are committed together, with the commit hash and the SHA-256 of every harness file recorded at the top of the search log. Any later change gets a new version number and a log entry stating what changed and why. A change to the score, guards or thresholds after the search has started invalidates comparisons made under the old version, and those candidates must be re-scored.

### Changes in v0.6

- **KS4 base is `perf-optimizations` (`e28143c`)** (§2): same output as `0e07514` plus the seed patch except where KS4 raced, bit-reproducible for a given seed, about 2.5× faster, no plots.

- **Raw-data pass runs after KS4's referencing** and is a diagnostic plus a safety net only (§6.1). Before referencing it is dominated by common-mode ringing that KS4's referencing removes; after it, nothing reaches 128 channels, the largest events are real population bursts, and the muscle index carries no information (median subtraction induces negative correlations). The muscle-noise mask is dropped; transients are masked only at ≥ 128 channels.
- **Duplicate test uses a conditional binomial test** (§6.3) instead of a Poisson test against the flank mean, which produced false flags for small units with empty flanks.
- **Flagged pairs more than 150 µm apart are artefact at any lag**; coincident spikes are removed from both units (§6.3). In the prototype every distant flagged pair joined MUA units at opposite ends of the probe at fixed lags of 0 or ±20 samples, not axonal propagation.
- **G2 (tracked-heading anchor) is demoted to a loose backstop** (fails only below default − 5τ; §7.2, §7.5): it measures a signal deliberately not optimised, so a tight gate could block genuine improvements.
- **Phase 0 decisions:** full-population θ_ref kept despite a narrow §8.4 failure; §8.10 waveform adjudication dropped as uninformative for KS4.
- **S is null-corrected per unit and α = 0.25 s** (§5.2, §8.5): untuned units score ≈ 0 instead of paying a cross-validation penalty that would reward dropping them.
- **Waveform screen is Bombcell's noise class only** (§6.4), plus the footprint limit; no somatic-shape requirement. **G5 uses a stricter screen**, not a relaxed one: a relaxed screen could never flag an eligible unit (§7.1).
- **`do_CAR = True` is fixed** (§4): without it the common-mode ringing enters the sort.

### Changes in v0.5

- **θ_ref is quantised to theta cycles** detected from the reference sort's population rate (elv's method), restricted to moving cycles, and estimated by per-cycle Poisson decoding (posterior mean) with tuning refitted for 3 iterations and no temporal prior (§5.1). This replaces the ring HMM on fixed 125 ms bins.
- **The variant was chosen by held-out-unit prediction, not by agreement with tracked heading** (§5.1). Agreement with tracking is a sanity floor only (§8.2).
- **S, S_track and C use moving theta cycles** as their time base (§5.3, §7.2, §7.5).
- **Reference sort uses a reserved seed (1000)** that no compared run uses, avoiding a ~2 bits/s self-leak advantage; §8.3 is now a reference-invariance test of comparisons rather than a correlation threshold.

### Changes in v0.4

- **One 20-minute excerpt only** (1668–2868 s). The sealed confirmation excerpt and its rules are removed. All work, including the final comparison, uses this excerpt; the claim is limited to it (§1).
- **Hybrid units** use time-shifted spike trains of reference-sort units from the same excerpt, so no separate source windows are needed (§7.4).
- **Tracking-quality mask** uses the Motive rigid-body flags (§6.5).
- Target recording filled in (§2).
- **No spike-to-heading time offset** in S_track (§7.2), so the brain region is no longer needed and no decisions remain open (§13).

### Changes in v0.3

- **No KS2.5 dependency.** The reference sort is the default KS4 sort of the excerpt without hybrid insertion (§5.1). §8.3 checks that θ_ref doesn't depend on that sort's seed.
- **Suggested defaults adopted** for all open decisions except the target recording's region (§13).

### Changes from v0.1

- **Primary score is now head direction, not position.** S is scored against a fixed reference heading θ_ref inferred once from a reference sort's population activity (§5). Position decoding moves to a transfer test on the MEC recording (§10, Appendix A).
- **Tracked head direction becomes a guard** (G2), anchoring θ_ref to the world.
- **Calibration guard demoted** to a loose backstop (G9, 5τ instead of 2τ). Correlated internal dynamics (theta sweeps, internal-direction alternation) add noise but don't bias paired comparisons, and duplicates and artefacts are now removed directly (§6).
- **No depth split of the reference.** The self-leak from a unit's own spikes into θ_ref is ~1/N of its information; a one-off Phase 0 check (§8.4) confirms it's negligible.
- Hybrid merge test uses HD pairs with different preferred directions (§7.4).

---

## 1. Goal and claim

Find KS4 settings (and, if needed, code changes) that sort the target recording better than KS4 defaults, for both single-cell and population analyses.

The claim the procedure supports is narrow: *on this 20-minute excerpt, the winning settings extract more heading-informative, non-duplicated single-unit spiking than defaults, without detectable increases in splits, merges, duplicates or artefacts, or loss of untuned cells, consistently across seed replicates.* It does not claim the winner is the best possible sort, that it is better for every analysis, or that it transfers to other recordings (§10 tests one other recording separately).

## 2. Data

| Item | Value |
|---|---|
| Target recording | Rat 26648 ("kobe"), 2020-08-21, open-field session 1 (354–4285 s of a 3.19 h recording) |
| Source | `Z:\ro\moser-intdirection\data\neuropixels\26648_kobe\recordings\2020-08-21_20-05-19\` (rclone remote `moser:`), read only through `rclone cat`/`rclone copy`, never through the mount |
| Probe | Neuropixels 3A (option 3); 384 channels plus sync saved (int16, interleaved); 374 recording sites after excluding the 10 reference sites; 30 kHz. `probe_1` is the sync master: synced time of sample i is (i + 1) / 30000 s. |
| Behaviour | Motive rigid-body tracking, 120 Hz (used only by the harness, for G2 and Phase 0) |
| Reference sort | Default KS4 sort of the excerpt without hybrid insertion, reserved seed 1000 (§4, §5.1, §8.11) |
| KS4 base | Branch `perf-optimizations` at `e28143c`: this fork at `0e07514` (includes the `swarmsplitter.check_CCG` refractoriness fix, 549ec32, which no release tag contains yet), plus the seed patch (§4, `5c84988`) and performance changes (`e28143c`; `perf_audit/README.md`). The performance changes are bit-identical to `0e07514` for default settings except where KS4 itself raced (≈ 0.01% of cluster labels on the excerpt), and make a sort bit-reproducible for a given seed; a default sort takes ≈ 6.5 min instead of ≈ 17. The fused CUDA kernels reproduce cuBLAS's summation order only for KS4's default `nearest_chans`, `template_sizes` and `n_templates`; other values use a generic path that can differ from stock KS4 by ~1 ulp. Plots are skipped (`make_plots=False`). Sorts import KS4 from this checkout only (editable install). Each run folder records its commit, code diff and kernel-library hash in `provenance.json`. The reference sorts (§5.1) were made at `0e07514` plus the seed patch; θ_ref is frozen, so this does not affect comparisons. |
| Transfer recording | MEC open-field recording: Neuropixels 3A, 374 channels, ~2 h, 1.5 × 1.5 m arena (§10) |

### 2.1 Excerpt

| Item | Value |
|---|---|
| Synced time | 1668.0–2868.0 s (open-field minutes 22–42) |
| Samples | [50,039,999, 86,039,999), 385 channels, 27.72 GB |
| Segment A / B | 1668–2268 s / 2268–2868 s |
| Local copy | `C:\data\ks4_tuning\26648_kobe_2020-08-21\search\ap_1668-2868s.bin`, with a manifest (source byte range, SHA-256, spot checks) written by `tuning/extract_segment.py` |
| Tracking | `tracking_1668-2868s.npz` in the same folder, written by `tuning/extract_tracking.py` |

The excerpt is sorted as one run. The hybrid version (§7.4) is written once from the local copy. Nothing outside the excerpt is used.

## 3. Information barrier

- The sorter and the optimising agent never see tracking data, θ_ref, tuning curves or per-unit scores.
- The optimising agent sees, per evaluation: the aggregate score S (§5), each guard's value and pass/fail (§7), and behaviour-blind diagnostics (unit counts, waveforms, auto- and cross-correlograms, amplitudes, drift traces, runtime).
- Only the harness reads tracking data and θ_ref. It runs as a separate process from a frozen, hash-checked copy; each evaluation re-verifies the hashes and refuses to score if they differ.
- The harness, not KS4, applies all post-sort processing (§6). KS4's own `good`/`mua` labels are not used anywhere, because settings under search (`acg_threshold`, `ccg_threshold`) change them.

## 4. What may vary, what is fixed

**May vary:** any KS4 setting, and KS4 code under `kilosort/`, except as below. Every code change is stored as a diff with its hash in the log.

**Fixed:**
- `duplicate_spike_ms = 0.25` (default). Otherwise a sort could pass the refractory criterion (§6.4) by deleting violating spikes.
- `do_CAR = True` (default). The excerpt carries large common-mode ringing (about 76 events per minute reaching ≥ 256 channels before referencing) that referencing removes; with it off, the ringing would enter the sort as artefact units.
- Input files (the excerpt binaries of §2.1), probe file, sampling rate, data type, `tmin`/`tmax`.
- Everything under `tuning/` (harness, reference heading, masks, hybrid generator, this spec).
- Seed handling: a new setting `random_seed` (default 1, as before) replaces the hard-coded numpy/torch seeds in `run_kilosort`. Replicate k uses `random_seed = k`, `cluster_init_seed = k + 4` (k = 1 gives KS4 defaults). This patch is part of the baseline, applied identically to default and candidates.
- Reserved seeds: 1000 for the reference sort, 1001 for the second reference of §8.3. No default or candidate run uses a seed ≥ 1000, so no compared run is scored against a reference built from itself (self-leak ≈ 2 bits/s, §8.3–8.4).

**Bombcell:** not used in the loop; the harness's own screens (§6.4, §7) stand in for it.

## 5. Primary score: S against a fixed reference heading

### 5.1 Reference heading θ_ref (built once, frozen)

θ_ref is an internal-direction signal: one value per moving theta cycle, read out of the reference sort's population. It is expected to depart from tracked heading where the network's own dynamics do; that is the reason for extracting it.

- **Source:** the reference sort (§2): a default KS4 sort of the excerpt *without* hybrid insertion, reserved seed 1000, after the same cleaning as candidates (§6.1–6.3) and the eligibility criteria (§6.4).
- **Speed and moving periods** (elv `RecordingData`, `speed005`): position smoothed with a 0.5 s Gaussian, speed on a 10 ms grid; moving = speed ≥ 0.05 m/s.
- **Theta cycles** (elv `ThetaPhase`, `peak_interp`):
  - theta-source units: reference-sort units labelled sharp or medium by elv's frozen raw-ACG burst-shape classifier (no anatomical ROI filter; this recording has no anatomy);
  - population rate: each source unit's spikes smoothed with a 20 ms Gaussian on the 10 ms grid, summed, spline-upsampled ×10;
  - every local peak is one cycle; boundaries lie midway between successive peaks;
  - a cycle is *moving* if every 10 ms sample inside it is moving with valid tracking.
- **Units:** all eligible units. No selection by HD tuning.
- **Estimation** (after elv `IntDirEstimate`, with a Poisson decoder):
  1. Tuning curves (60 bins of 6°) fitted against tracked heading on moving cycles, shrunk towards each unit's mean rate by 1 s of pseudo-occupancy.
  2. Each cycle decoded independently from its spike counts (Poisson likelihood; no temporal prior, so cycle-to-cycle dynamics are kept). The estimate is the circular mean of the posterior.
  3. Tuning curves refitted against the estimate, and the cycles decoded again; 3 decoding passes in total.
- **Output:** θ_ref per moving cycle, frozen with its hash together with the cycle boundaries.
- The decoder's own tuning curves are used only to infer θ_ref, never to score candidates.

**Why this variant** (reference sort seed 1000 after §6 cleaning, Phase 0 prototype): variants were compared by how well each, built from one half of the units (alternating 100 µm depth blocks), predicts the *other* half's spiking (cross-validated over time halves). Tracked heading scored 452.3 bits/s. Per-cycle Poisson references beat it: +8% with no refit, +17% (530.5 bits/s) for the posterior mean after 3 passes, 62% of held-out units improved. The ranking was the same before cleaning. Gains stop after pass 3 (flat to pass 10) while agreement with tracking keeps falling, so further passes add drift but no information. Population-vector correlation on single-cycle counts did not beat tracked heading. Agreement with tracked heading was *not* a selection criterion: a signal chosen to match tracking would just be tracking.

### 5.2 Per-unit model

For each eligible candidate unit i: λ_i(θ) over 60 bins (6°), circularly smoothed (σ = 12°), fitted on the training segment's moving cycles against θ_ref, and shrunk towards the unit's mean rate by a pseudo-occupancy weight α = 0.25 s (set in Phase 0, §8.5).

**Null correction.** A tuning curve fitted to noise predicts held-out data slightly worse than the mean rate, so an untuned unit scores below zero (about −0.09 bits/s each at α = 0.25 s) and shrinkage only shrinks, never removes, that penalty. Uncorrected, a sort would gain S by dropping untuned units. So each unit's S_i has its own null subtracted: its mean score against θ_ref rotated among the scored cycles by k/7 of their number (k = 1…6; each rotation ≥ ~100 s of moving time). Rotating within the scored cycles keeps the null's cycles and time identical to the real score's, so the penalty matches. (Rolling θ_ref in time and intersecting the moving masks does not: it leaves fewer cycles, a larger penalty, and a +20 bits/s over-correction.) Untuned units then score ≈ 0 for any α, and α is free to be chosen for sensitivity. There is no hard floor at zero on S_i, because a floor would give pure-noise units a small positive expected score.

### 5.3 Score

For each eligible unit i, fit on segment A and evaluate on segment B, then the reverse:

S_i = (1 / T_test) Σ_c [ n_ic · log2(λ_i(θ_ref,c) / λ̄_i) − (λ_i(θ_ref,c) − λ̄_i) d_c / ln 2 ]

- c runs over the test segment's moving theta cycles (§5.1); d_c = cycle duration; n_ic = spike count in cycle c; λ̄_i = training-segment mean rate over moving cycles; T_test = summed duration of the test cycles.
- A cycle more than 50% masked (§6.1–6.2) is dropped; otherwise d_c is scaled by its unmasked fraction.
- **S = Σ_i S_i**, in bits/s, averaged over the two directions weighted by test time.

Units matched to inserted hybrid units (§7.4) are excluded from S.

### 5.4 Comparison

All comparisons are paired on the same excerpt: ΔS = S_candidate − S_default.

- σ_S: SD of S across ≥ 5 default replicates with different seeds; 8.0 bits/s (§8.1), so δ = 16.0 bits/s.
- δ = max(δ_practical, 2σ_S), with δ_practical = 1% of default S.
- A candidate *improves* if mean ΔS ≥ δ over its replicates and every guard passes.

Median decoding error (degrees) is reported for every evaluation but never used for decisions.

## 6. Harness processing (applied identically to every sort, in this order)

### 6.1 Raw-data transient pass (once, independent of the sort): diagnostic and safety net

Computed from the raw excerpt after KS4's referencing (subtract each channel's mean, then the median across channels) and a 300 Hz high-pass (`tuning/raw_artefacts.py`):

- **Transient events:** samples where ≥ 8 channels spanning ≥ 1 mm exceed 6 × their MAD within the same 0.2 ms, merged into events. All are kept for the §8.9 description.
- **Safety-net mask:** events reaching ≥ 128 channels, ±1 ms. Neural activity does not reach that extent after referencing; on this excerpt there are none (largest 83 channels; 475 events at 64–127 channels, which are dense population bursts of real spikes). The mask exists for gross artefacts, not to clean this excerpt.
- **No muscle-noise mask.** Before referencing, 300–600 Hz correlation across ≥ 1 mm is dominated by common-mode ringing; after it, median subtraction makes the index negative everywhere, and no sustained high-frequency noise is visible in either.
- Masked time is **missing data**: removed from time and spike counts alike, never treated as silence. Masks can be behaviour-locked; deleting spikes but keeping the time would bias real cells' rates down in those states.

Why the raw pass masks so little: the artefact that does reach the sort is residual ringing at the probe ends, which becomes MUA units. §6.2–6.3 (multi-unit events, distant duplicate pairs), the waveform screen (§6.4) and G5 catch these. A lower raw threshold would start deleting real population bursts. Heading-locked artefacts (for example, electrical pickup that changes with head orientation) would correlate with θ_ref exactly as wall artefacts correlate with position; §6.2–6.3, §6.4 and G5/G9 are the defence.

### 6.2 Spike-level multi-unit artefact events

- Any 0.2 ms window in which ≥ k = 17 distinct units whose median spike positions span ≥ 1 mm all have spikes.
- k was set on the default replicates as the smallest value at which an interval-jittered copy (5 ms windows) yields < 1 such event per 10 min (17 in 4 of 5 replicates, 18 in one), and is fixed for every sort, the reference sorts included. The jittered rate is still reported per sort.
- All spikes in the window, from every unit, are removed, and the window is masked as missing time.
- Reported per evaluation: event rate. Prototype (default, seed 1): k = 17, 23 events, 401 spikes removed.

### 6.3 Pairwise duplicate detection and removal

Run on **all** unit pairs, via one pass over a time-sorted list of all spikes (not a loop over pairs).

- **Statistic:** lag histogram at sample resolution over ±1.5 ms. For each 5-sample window centred at lag L ∈ [−1, +1] ms: observed count vs expected count from that window's own flanks (10–30 samples either side of L).
- **Flag** a pair if, for some L, both:
  - the conditional binomial p-value is below 0.01 / (number of pairs × number of lag windows): the window's count given the window-plus-flanks total, against a null share of 5/47 samples. This accounts for noise in the flank estimate; a plain Poisson test against the flank mean flags small units (≈ 30–150 spikes) whose flanks happen to be empty (prototype: 5–11 false flags per jittered copy), and
  - the excess is ≥ 3% of the smaller unit's spikes.
- **Null check:** run the identical procedure on an interval-jittered copy of the sort (5 ms windows). Phase 0 requires flagged pairs in the jittered copy ≤ 1% of those in the real sort (or ≤ 5 in absolute terms).
- **Footprint classes** (distance between the units' median spike positions):

| Class | Lag of excess | Interpretation | Spikes removed |
|---|---|---|---|
| ≤ 150 µm | any | Same neuron detected twice | Coincident spikes from the unit with fewer spikes |
| > 150 µm | any | Artefact split across units | Coincident spikes from both units |

Distant pairs are not treated as axonal or dendritic propagation: in the prototype every flagged pair more than 150 µm apart joined KS4 "mua" units near the two ends of the probe (≈ 600–1040 µm and ≈ 4080–4360 µm) at fixed lags of 0 or ±20 samples, the signature of residual common-mode ringing.

Only spikes inside the flagged lag window are removed, so a partial duplicate keeps its genuine spikes and a pure duplicate ends up empty.

### 6.4 Eligibility (single-unit criteria)

After §6.1–6.3, a unit is eligible for S if all hold:

- ≥ 100 spikes in each 10-min segment.
- Refractory-violation contamination estimate (1.5 ms refractory period, 0.25 ms censored period) ≤ 10%.
- Passes the waveform screen (`tuning/waveform.py`, on KS4's un-whitened templates): not Bombcell "noise" under py_bombcell's default thresholds (≤ 2 peaks, ≤ 1 trough, peak–trough duration 100–1150 µs, baseline ≤ 0.3, second peak ≤ 0.8 × trough, exponential spatial decay 0.01–0.1 per µm), and channels with ≥ 20% of the peak channel's peak-to-peak amplitude span ≤ 300 µm. Bombcell's other classes (non-somatic, MUA) are not used: they are of limited reliability, and positive-going (non-somatic by Bombcell's rule) units here are clearly neural and often strongly heading-tuned. Prototype: 4–6 of ~360–376 spike-eligible units fail.
- Not matched to an inserted hybrid unit.

### 6.5 Tracking-quality mask (for G2 and Phase 0 only)

Tracked head direction is marked invalid where the Motive rigid body is flagged invalid or not tracked. The primary score does not use tracking, so this mask affects only G2 and the θ_ref validation.

## 7. Guards (pass/fail, each relative to default on the same excerpt)

Tolerances marked τ are 2 × the SD of that quantity across the five default replicates (§8.1, `runs/phase0_replicates.json`) unless stated.

### 7.1 Main guards

| # | Guard | Pass condition | What it catches |
|---|---|---|---|
| G1 | Unit count | Eligible units within 0.8–1.25 × default | Wholesale fragmentation or loss |
| G3 | Suspected splits | Count ≤ default + τ_split (§7.3) | Oversplitting, which S under-penalises |
| G4 | Hybrid recovery | §7.4 | Splits, merges, misses on known units; merges of differently tuned HD pairs |
| G5 | Artefact share | Share of S from eligible units that fail a *stricter* version of the waveform screen (a margin inside each noise threshold: duration ≤ 1100 µs, baseline ≤ 0.25, second peak ≤ 0.7 × trough, decay 0.0125–0.09 per µm, footprint ≤ 250 µm) ≤ default + τ | Near-noise units that only just passed eligibility |
| G6 | Cleaning burden | Fraction of spikes removed by §6.2–6.3, and flagged-pair count, ≤ default + τ | Settings that generate duplicates or artefacts, even when the harness neutralises them |
| G7 | Untuned units | Number of eligible units with S_i below the 25th percentile of default's S_i ≥ 0.9 × default | Selection pressure against non-HD cells (circularity) |

### 7.2 Tracked-heading score S_track

Identical to §5.2–5.3, with tracked head direction (circular mean within each moving theta cycle) in place of θ_ref, the §6.5 mask applied, no time offset between spikes and heading. S_track is expected to be lower than S, since θ_ref carries internal dynamics that tracking lacks. S_track is reported every run; G2 (a backstop, §7.5) fails only if it drops by more than 5τ. Across the prototype sorts S_track / S was 0.83 for all three, so the two normally move together.

If S rises while S_track falls, that is a warning sign even when G2 passes, and the log entry should say so.

### 7.3 Suspected split pairs

A pair counts if all hold:
- template peak channels ≤ 50 µm apart;
- template similarity ≥ 0.8 (KS4's `CCG.similarity`, maximum over lags);
- cross-correlogram rate within ±1.5 ms < 0.25 × rate at 2–10 ms.

The sharp ±1.5 ms window is deliberate. Cells with different tuning have broad cross-correlogram dips (tens to hundreds of ms) that depress both windows equally, so they are not counted. Nearby genuine pairs whose overlapping spikes the sorter cannot resolve also show a narrow dip; they appear in default and candidates alike, so the comparative guard tolerates them.

### 7.4 Hybrid ground truth

**Construction (once):**
- 40 units, sources drawn from well-isolated reference-sort units (eligible, refractory contamination ≤ 5%, in no flagged duplicate or suspected split pair, ≥ 600 spikes): 10 as HD pairs (below) and 30 singles, one from each amplitude quantile of the rest, so small units are included, not just the easiest ones.
- Waveforms: mean of 300 spikes after KS4-style referencing and a 300 Hz high-pass, −1 to +2 ms, on channels within 120 µm of the peak channel, tapered at both ends. Inserted ≥ 200 µm from the source location, moved by a multiple of 40 µm (the probe's column period) so the spatial pattern is unchanged, with the footprint inside the probe (reference sites get no signal), and ≥ 60 µm from other inserted units (40 units at ≥ 100 µm would not fit on the 3.8 mm probe).
- Spike trains: each source unit's spike train circularly shifted within the excerpt by a fixed offset of at least 60 s, with per-spike amplitude scaling drawn from the source unit's amplitude distribution. The shift removes any relation to heading while keeping firing statistics.
- Includes 5 pairs of HD cells (mean vector length of the tracked-heading tuning curve ≥ 0.25) with preferred directions ≥ 90° apart, inserted ≤ 60 µm apart. Both members of a pair share one shift, so their real cross-correlogram structure (including broad dips) is preserved.
- Drift: inserted units follow the reference sort's drift estimate for the excerpt (partly circular; noted as a limitation): each spike is placed at its target depth minus that batch's `dshift`, interpolated linearly along the probe column.
- Fixed insertion seed (0).
- Built by `tuning/make_hybrid.py`. `kilosort/simulation.py:hybrid_simulation` was not adapted: it hard-codes another probe, its file paths and its own spike-train generator.

**Scoring** (`tuning/hybrid.py`, on the sorter's output before cleaning, so it measures the sort): agreement = matched / (n_true + n_sorted − matched), ±0.4 ms tolerance, best-matching sorted unit per inserted unit. Sorted units whose spikes are ≥ 50% inserted spikes are hybrid copies, excluded from eligibility and S.

**G4 passes if all hold:**
- median agreement ≥ default − τ_hyb;
- units recovered (agreement ≥ 0.8) ≥ default − τ (originally default − 1, but the default replicates themselves range 30–32 around a mean of 31.4, so that rule failed a default replicate);
- no inserted HD pair merged (a merge = one sorted unit holding ≥ 30% of each member's spikes).

### 7.5 Backstops (loose; fail only on large changes)

| # | Backstop | Fail condition | Why it's loose |
|---|---|---|---|
| G2 | Tracked-heading anchor | S_track < S_track,default − 5τ (§7.2) | A good sort is expected to fit θ_ref better than tracked heading, and gains concentrated where θ_ref departs from tracking may lower S_track slightly. Only a large drop, while S rises, signals gains from non-heading structure in θ_ref (§12). |
| G8 | Mask integrity | Raw mask hash differs from the frozen one | Should never trip; catches harness tampering or drift |
| G9 | Calibration | C < C_default − 5τ_C | Duplicates and artefacts are now removed directly by §6. Correlated internal dynamics make the decoder overconfident by the same amount for every sort, so this is a coarse tripwire, not a tight guard. |

**C:** Poisson naive-Bayes decoding of θ_ref from the candidate's eligible units in each moving theta cycle, flat prior, cross-validated as in §5.3. C = mean over cycles of log2(posterior at the θ_ref bin × 60), in bits relative to uniform. Never maximised: dropping units can raise it.

### 7.6 Matched-unit check (reported, not a guard during search)

Units in the candidate are matched to default by spike-train agreement (±0.4 ms tolerance, agreement ≥ 0.5). Report S restricted to matched units, and the S_i distribution of unmatched units on each side. This shows whether a gain comes from better versions of the same cells or from swapping in different (more HD-tuned) ones.

## 8. Phase 0: validation before any search

Not counted in the search budget. The search may not start until every item passes or is explicitly waived in the log.

| # | Check | Pass condition |
|---|---|---|
| 8.1 | Seed replicates: default × 5 on the excerpt | σ_S and all τ values computed. **Done** (hybrid excerpt, seeds 1–5, `e28143c`; `runs/phase0_replicates.json`): with the final score (null-corrected, α = 0.25 s) and k = 17: S = 576.2 ± 8.0 bits/s (σ_S 1.4% of S), so δ = 2σ_S = 16.0 bits/s (2.8%). τ: S_track 15.3 bits/s, C 0.16 bits, eligible units 4.6, suspected splits 2.2, near-noise share 0.002, spikes removed 0.16 percentage points, flagged pairs 39, hybrid median agreement 0.0055. Hybrid: median agreement 0.95, 30–32 of 40 recovered (unrecovered units are mostly the smallest), no HD pair merged; 35–37 hybrid copies excluded per sort. |
| 8.2 | θ_ref sanity floor | On moving cycles, report circular correlation (Fisher–Lee, which needs no mean direction; open-field heading is near-uniform) and angular offset between θ_ref and tracked heading. Pass if circular correlation ≥ 0.8, i.e. θ_ref is still clearly a heading signal. This is a floor, never an objective: departures from tracking are expected. Prototype: 0.81–0.83. |
| 8.3 | Reference invariance | Build θ_ref from two default sorts without hybrid insertion (reserved seeds 1000 and 1001). Score at least three other sorts against both references. Pass if, for every pair of sorts, ΔS differs between the two references by < 0.25 δ. (What matters is that the choice of reference sort doesn't change comparisons; agreement between the two θ_ref series themselves is reported only.) Prototype (references from seeds 1000/1001; sorts: default seeds 1 and 2, and an `extra-settings`-branch sort; all cleaned per §6): spreads 0.2–0.7 bits/s, against 2.0–4.3 bits/s when the reference sorts were themselves among the compared sorts (self-leak, §8.4). θ_ref series: Fisher–Lee 0.944; median difference < 0.001°, 2.3% of cycles differ by > 45° (low-spike cycles). |
| 8.4 | Self-leak check | On the default sort, compute S against the full-population θ_ref and against a depth-split version (θ_ref from the top half scores bottom-half units and vice versa, with a 150 µm buffer). Pass if they differ by < 0.25 δ. If not, adopt the split for all runs. **Result** (5 hybrid replicates, `runs/phase0_self_leak.json`): levels drop 21% under the split (each half-reference has half the units), so pairwise ΔS between replicates was compared instead, as in 8.3: the largest change was 5.3 bits/s against 0.25 δ = 4.0. After allowing for the proportional shrinkage (ΔS_split ≈ 0.87 ΔS_full, r = 0.97) residuals are ≤ 4.9 bits/s and concentrated on one replicate, consistent with the split's own noise (units near the median depth fall on different sides in different sorts) rather than leak. Adopting the split would cost ~13% of sensitivity (mean/SD 62 vs 72). **Decision:** keep the full-population θ_ref; reference invariance (8.3) covers the comparisons. |
| 8.5 | Time-shift control: θ_ref circularly shifted ≥ 60 s | \|S\| within 2σ_S of 0 after α is set. Confirms no leakage. **Pass** (`runs/phase0_alpha_scan.json`): sensitivity across replicates (mean/SD) barely depends on α (raw 70.8 at 0.1 s to 67.5 at 4 s; null-corrected 1–2% higher at every α), so α = 0.25 s. Control rotations half-way between the null rotations: S = +0.8 ± 3.2 bits/s (range −4.9 to +5.8) against 2σ_S = 16.0. (8.4 and 8.6 were run against the reference built with k = 18 before k was fixed; the change removes ~0.003% of spikes and moves S by ~1 bit/s.) |
| 8.6 | Perturbation audit on the default sort (no re-sorting) | At 5% rates: merges between differently tuned units, dropped units, contamination and thinning each lower S by ≥ 3σ_S. Duplicates: \|ΔS\| < σ_S after §6.3 and G6 trips. Heading-locked artefact units: masked, or trip G5/G9. Splits: G3 trips. Merges between similarly tuned cells are expected to cost little (§12); report their effect rather than requiring it. **Pass** (default seed 1, 18 of 361 units, 5 draws; `runs/phase0_perturbation.json`): merges of differently tuned units −44.6 bits/s (−5.6σ_S), drops −27.8 (−3.5σ_S), contamination −285 (most units then fail the refractory criterion), thinning −36.0 (−4.5σ_S). Splits: ΔS −2.1 (S barely penalises them, as expected), and 16 of 18 meet the G3 rule (τ = 2.2). Duplicates (80% copies, ±2 samples): +27.7 bits/s before cleaning, −1.0 after; 2.8% of spikes removed, so G6 trips. Heading-locked artefact units were not tested: they need synthetic waveforms in the raw data. |
| 8.7 | Saturation | S still rising with unit count at the full count (unit-subsampling curve). **Pass by construction:** S is a sum of per-unit scores, so it rises with every informative unit; it cannot saturate the way a population decoder can. |
| 8.8 | Duplicate-test null | Jittered-copy flags within the §6.3 limit. **Pass:** 0 flags in every jittered copy of 5 default sorts (285–332 real flags). |
| 8.9 | Raw transients and masks | Raw events at ≥ 128 channels (§6.1), if any, inspected by you; random 64–127-channel events plotted to confirm they remain neural. Masked time fraction (§6.1–6.2) reported; > 5% is flagged for discussion. Describe (not tune) how multi-unit event rate relates to heading, speed and immobility. Prototype: no ≥ 128-channel events; 64–127-channel events are population bursts. |
| 8.10 | Waveform adjudication of flagged ≤ 150 µm pairs | At coincident events, fit both templates jointly to the filtered data. Duplicate if the smaller unit's conditional amplitude has median < 0.3 × its usual amplitude. Report agreement with §6.3. **Not informative for KS4** (`runs/phase0_adjudication.json`, `tuning/adjudicate.py`): only 2 of 211 flagged pairs fall below 0.3 (median ratio 1.03). KS4 makes cross-unit duplicates by peeling: when one template leaves a residual (timing jitter, amplitude variation), a second template fits it, so a joint fit reproduces that decomposition. Example: pair 702/707, where 98% of the smaller unit's 661 spikes coincide with the larger unit's and its template is a residual-like shape, scores 0.36. **Decision:** dropped; false flags are covered by the jittered-copy null (8.8). |
| 8.11 | Reference sort and hybrid effect | One default run on the excerpt without hybrid insertion, reserved seed 1000. It serves as the reference sort (§5.1); comparing a no-hybrid default run with the 8.1 runs (same seed, with and without insertion) reports the insertion's effect on S and unit counts. **Done:** seed 1 at `e28143c` without vs with insertion: S 581.9 vs 577.5 bits/s (−0.55σ_S), eligible units 372 vs 361, 35 hybrid copies excluded. |

## 9. Search, selection, reporting

**Budget:** 100 KS4 runs on the excerpt, including replicates and ablations; 15 of them are reserved for §9.2–9.3.

### 9.1 Search
- Each run is logged before its result is read (§11).
- A run that improves on default (§5.4, single replicate) and passes all guards gets 2 further seed replicates before it can be the leader.

### 9.2 Selection
- Winner: highest mean ΔS over ≥ 3 replicates among candidates passing all guards on every replicate.
- The best of ~100 runs beats default by ~2–2.5σ by chance alone, which is why single runs never decide.

### 9.3 Ablation
- Revert each of the winner's changes individually (1 replicate each). Drop changes whose reversion costs < δ/2 and passes all guards; re-replicate the simplified winner (3 seeds).

### 9.4 Reporting
- Report the simplified winner's mean ΔS and ΔS_track over its replicates, all guards, and §7.6 matched-unit results against default.
- Run the key downstream single-cell and population analyses on default and winner sorts of the excerpt, and report where conclusions differ.
- Run the §8.9 inspection on the winner's sort.

## 10. Transfer test: MEC open-field recording (reported, not a gate)

- Sort the MEC search excerpt (minutes 20–40) with default and winner, 3 seeds each.
- Score with tracked-position S (Appendix A) plus G1, G3, G5, G6 and G7.
- If the winner does not beat default there, treat the settings as specific to the target recording: do not apply them to the MEC data without a search of its own under this spec.

## 11. Log (append-only)

One record per run, written before results are read where possible:

- run id, timestamp, phase, seeds;
- settings dict, KS4 code diff hash, harness hashes, θ_ref hash, spec version;
- rationale for the run (one line, from the optimising agent);
- S, ΔS, S_track, C, all guards with values and pass/fail, unit counts, cleaning statistics, hybrid recovery, runtime;
- any deviation from this spec.

## 12. Known limitations

**Material:**
- S rewards heading information. G7 and the matched-unit check limit, but do not remove, pressure against untuned cells.
- Merging two similarly tuned cells loses little information, so such merges cost little in S. This is a general blind spot of tuning-based scores (it applied equally to the v0.1 position score), not specific to head direction. How often it arises depends on how similar the tuning of neighbouring cells on the probe is in this recording. G3 and G4 cover it, and the §8.6 perturbation audit measures its size.
- Hybrid units are cleaner than real ones (stereotyped waveforms, drift taken from the sorter's own estimate). Used only as a guard for this reason.
- Sorts are bit-reproducible for a given seed (§2), so seed replicates measure only the variability that comes from the seeds; nothing is left from GPU nondeterminism.
- θ_ref may carry shared signal other than heading. Held-out prediction rewards any shared structure a single angle can express (for example a position-related component in a mixed population). The §8.2 floor limits this; whether θ_ref's departures from tracking can be predicted from position has not been checked.

**Considered and judged minor (monitored, not designed around):**
- *Correlated internal dynamics* (theta sweeps, internal-direction alternation): add noise to decoding but affect every candidate identically on the same data, so they don't bias paired comparisons. θ_ref follows them, so S treats them as signal.
- *Self-leak into θ_ref:* a unit's own reference-sort spikes shift θ_ref by ~1/N. Estimated at ~0.1% of S beforehand; the prototype invariance test measured about 2 bits/s (~0.4% of S) for a sort scored against a reference built from itself. Checked once (§8.4).
- *Genuine sub-millisecond synchrony* (e.g. gap-junction-coupled interneurons) can be flagged as small duplicates in §6.3. Its effect on S is small, and §8.10 measures how often it happens.
- *Overconfidence of the naive-Bayes decoder:* only affects the backstop C (G9), not S.

## 13. Open decisions

None. Settled in Phase 0: the full-population θ_ref is kept (§8.4), and waveform adjudication is dropped (§8.10).

No spike-to-heading time offset is used anywhere (§7.2).

Adopted defaults (change any by editing the relevant section): θ_ref per moving theta cycle (§5.1); no Bombcell in the loop; δ_practical = 1% of default S; contamination ≤ 10%; Bombcell-style waveform rules in the harness; 40 hybrid units including ≥ 5 differently tuned HD pairs; θ_ref accepted at circular correlation ≥ 0.8.

---

## Appendix A. Tracked-position score (transfer test, §10)

Unchanged from v0.1 §5. Per unit, λ_i(z) = r_i · P_i(x) · H_i(θ) · V_i(v), with position in 5 cm bins (Gaussian smoothing σ = 5 cm), head direction in 10° bins (σ = 15°) and speed in 5 cm/s bins over 5–50 cm/s (σ = 5 cm/s). Factors are fitted on the training segment by iterative proportional fitting and shrunk towards 1 by α, set by time-shift control on the MEC default sort. S_i uses Δ = 1/120 s, samples with valid tracking and speed ≥ 5 cm/s, and the masks of §6.1–6.2 computed on the MEC excerpt. S = Σ_i S_i in bits/s, averaged over both directions.
