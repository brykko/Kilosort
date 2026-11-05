# Round-based against interleaved template learning: controlled simulation

Scripts: `tuning/exploratory/interleaving/simulation/`. Outputs: `C:\data\ks4_tuning\exploratory\interleaving\simulation\`.
No real or hybrid data. Nothing here touches `kilosort/`. Part 1 was written before any code was run.

## Part 1. Preregistration (2026-11-04, before any run)

**Purpose.** Find when each schedule works, including when the answer reverses. The toy is a test of
*scheduling*; it says nothing about KS4's graph clustering or its real detection yield.

### Data model

Whitened-scale data (noise sd about 1 per channel), so KS4's thresholds keep their meaning.

- Probe: 32 channels in 2 columns, 20 µm vertical pitch, 32 µm lateral. fs 15 kHz (half of KS4's, to save
  time), window nt = 31 samples (KS4: 61 at 30 kHz), trough at sample 10.
- Base recording: 48 units, 2-s batches, 120 batches (240 s).
- Rates: log-normal, median 5 Hz, sigma 1.2, clipped to [0.1, 40]; then 10% of units forced to 0.1-0.5 Hz and
  6% to 30 Hz. Refractory dead time 2 ms.
- Waveforms: a 'regular' family (biphasic, varied widths) and a 'poor' family, about 25% of units (triphasic,
  axonal/positive-led, wide). Poor-match status is *measured*, not assumed: best space-time correlation with
  the universal dictionary below 0.75 (threshold fixed now; the distribution will be reported).
- Space: soma at a random (x, y); exponential footprint, length 20-45 µm, small conduction delay. Template
  L2 norm log-normal, median 16, sigma 0.5, clipped to [8, 60]. Spike amplitude scale log-normal, sigma 0.15.
- Near-identical pairs: 6 pairs (25% of units); the partner has the soma moved 6-14 µm, a slightly perturbed
  waveform, and a norm within 15%. Template correlations reported.
- Noise: Gaussian with temporal colouring (band-limited) and residual spatial correlation (length 25 µm), sd 1,
  plus a 'hash' of 150 weak unsorted units (norm 2.5-5, 5 Hz, not in the truth).
- Temporal overlaps arise naturally. Subsample jitter: 4 fractional shifts.
- Variants: **stationary**; **drift** (rigid, slow, peak-to-peak 30 µm = 1.5 pitches, standing for what KS4's
  registration leaves behind; no sorter here corrects it); **nonstat** (35% of units silent for one epoch of
  30-50% of the recording; 20% bursting, 2-5 spikes at about 5 ms with 0.85 amplitude decrement per spike;
  some slow amplitude change).

### Shared primitives (identical code for every schedule)

- Universal detection: KS4's `template_match` logic. Temporal dictionary = `kilosort/wTEMP.npz`'s wTEMP resampled
  to 15 kHz (6 shapes); spatial = Gaussian weights at half-pitch centres, 2 widths; local maxima of the
  absolute score above Th_universal.
- Matching pursuit: KS4's `run_matching` peel (relu(B)²/nm, peaks in ±nt, simultaneous subtraction, dense ctc),
  Th_learned, features = residual + amp × template on the template's 10 nearest channels. Representation: 6 temporal
  PCs (resampled wPCA) × channels.
- Thresholds: Th_universal 7, Th_learned 7 (H8's), fixed for all arms, not tuned.
- Clustering (local in space, global in time): spikes grouped by depth row of their template or detection centre;
  PCA to 8 dimensions; k-means to K_init; Ward tree on centroids; top-down splits accepted when a bimodality score
  on the LDA axis exceeds `bim` and both sides have ≥ n_min spikes (n_min = 12). This stands in for KS4's graph +
  tree + swarmsplitter.
- Templates: per-channel cluster means, aligned against wTEMP as `align_U` does.
- Merge by shape: `merging_function` mode 'mu' (max-lag correlation > rho1, norms within 20%).
- Final stage for **all** arms: CCG merge (correlation > 0.9 and a refractory dip) on cluster labels.

### Arms

- **R(K)**: detect, cluster, merge → round 2: extract, cluster, merge → ... K = 2, 3, 4 (R2 is stock KS4; R3, R4 are
  the `ks4-round3` branch's repeat rounds). Full-pass equivalents (FPE) = K (detection counts as one).
- **Rd(K)**: as R but each extraction also runs universal detection on the residual and adds those spikes to the
  clustering. **Control added by this audit**: I contains residual discovery and R does not, so any win for I on missed
  units confounds scheduling with discovery. Rd separates them.
- **S(f, K)**: R(K) restricted to fraction f of batches (evenly spread), then a final extraction + clustering on all
  data. f = 0.25, K = 3. FPE = 3f + 1.
- **I(p)**: per-batch matching with the current dictionary; template update T ← φT + (1−φ) mean(features), φ =
  exp(−n_b/m); universal detection on the residual into a rolling buffer (window 20 batches); every 5 batches cluster
  the buffer and birth clusters of ≥ n_birth spikes not already explained (correlation < 0.95 with a template);
  every 5 batches merge (rho_merge, norms within 20%) and drop; p passes, then one final extraction. p = 0.25, 0.5, 1
  with shuffled batches; p = 1 time-ordered. FPE = p + 1. The first batch's dictionary is empty, as in KS2.5.
- Drop rule: KS2.5's rate test (EMA of spikes per batch below minFR 0.02 Hz) but only for templates with fewer than
  n_keep = 50 lifetime spikes. This protects I against silent epochs and is more generous than KS2.5; a sensitivity arm
  runs n_keep = 0.

Every arm gives two outputs on its last extraction: **T labels** (the spike's template) and **C labels** (a global
clustering of that extraction, then the CCG merge). R and S are C labels, as KS4 is; I's natural output is T labels, and
I+FC is its C labels. I (T) against R (C) and I+FC (C) against R (C) are both reported.

### Tuning (equal budget)

Grids fixed now, 12 configurations each, on tuning seeds never used for reporting (7001 stationary, 7002 drift, 7003 nonstat;
3 runs per configuration, 36 per family).

- R: rho1 ∈ {0.90, 0.95, 0.98} × bim ∈ {0.4, 0.6} × K_init ∈ {8, 16}. Rd and S inherit R's choice.
- I (p = 1 shuffled): schedule ∈ {batches 20→400 as KS2.5, spikes m = clip(n_tot, 20, 200), spikes m = clip(n_tot, 20,
  2000)} × n_birth ∈ {15, 30} × rho_merge ∈ {0.93, 0.97}. I+FC's final clustering uses R's tuned bim and K_init (a small
  favour to I). All I arms inherit I's choice.
- Objective for both: mean over true units of best accuracy (0 if none) minus 0.5 × false units / true units, C labels
  for R, T labels for I (each one's natural output). Ties go to the lower index.

### Metrics

- Template error: for each true unit, min over dictionary and lags (±6 samples) of ‖T − truth‖/‖truth‖ in PC space, capped
  at 1, truth at the unit's mean amplitude and, under drift, time-averaged. Median and mean over units; also on units found
  by all arms. Checkpointed against FPE, matched-filter ops (templates × samples) and clock.
- Spike match window ±4 samples (0.27 ms). accuracy = TP/(TP+FN+FP) per (true, output) pair; each true unit's best output.
  Found: accuracy ≥ 0.5. Recall and precision of that pair. By rate stratum (< 0.5, 0.5-3, 3-15, > 15 Hz) and poor-match status.
- Split: ≥ 2 outputs each ≥ 50% pure for the unit holding ≥ 10% of its spikes. Merged: an output holding ≥ 20% of the spikes of
  ≥ 2 true units. False: output with ≥ n_min spikes and < 50% purity for any true unit. Pair resolved: both members
  found, by different outputs.
- Order dependence: I shuffled against time-ordered; and 3 algorithm seeds on 3 data seeds for run-to-run variability.

### Reporting

Held-out data seeds 1-10 in each of 3 variants (seed paired across arms). 95% CIs: bootstrap over seeds of paired differences.
Sensitivity of I (on 4 held-out seeds, stationary and drift): schedule, batch length, n_birth, birth window, n_keep.
Scale check: units per channel, median rate, batch length, recording length, arms R3, I(1), S, with fewer seeds.
Runtime budget 40 min on 4 CPU workers; if a pilot (seed 9001, not reported) shows that is unreachable, seeds and sizes will be
cut and the cut stated here, keeping at least 10 data seeds in the main comparison. The pilot is for debugging and runtime
only; any model change it forces is recorded under Deviations.

### What would count as evidence

Verdicts are read from paired CIs across seeds; 'unclear' if the CI spans zero or the sign depends on the variant.

- **H1** supported if, in the stationary and nonstat variants, I reaches R2's template error at FPE < 2 and R3's at FPE < 3
  (paired, CI of the difference in error at those costs on I's side). Refuted if its best error stays above R3's.
- **H2** supported if I's found rate on poor-match units exceeds best-R's with a CI above 0 **and** Rd does not explain it (I
  also above Rd). If Rd ≈ I, the verdict is 'discovery, not interleaving'. Extra rounds recover them if R4 − R2 has a CI above 0.
- **H3** supported if I has more merged or false units, or a lower found rate for < 0.5 Hz units, than R, with a CI excluding 0.
- **H4** supported if S is within 0.02 of I's mean accuracy (non-inferiority: lower CI bound ≥ −0.02) at no more FPE.
- **H5** supported for a side if its mean-accuracy CI under drift is above 0 and the gap is larger than in stationary.
- **H6** supported if R's pair-resolved rate exceeds I's (T labels) with a CI above 0; whether I+FC closes the gap is reported.

Any result that depends on a single choice of grid value, or that flips within a sensitivity sweep, is reported as conditional.

### Deviations made after the pilot (seed 9001, before any tuning or held-out run)

The pilot was used to debug the code, calibrate the data model against an oracle (true templates through the same matching)
and size the runtime. No arm comparison was used to set these. Each changes both sides alike.

1. Footprint length 20-45 µm → 10-26 µm. With 20-45 the 10-channel templates left residuals above threshold around large units, so
   the *oracle* double-counted them (echoes on neighbouring templates). This is the scale of whitened footprints.
2. Noise: spatial length 25 → 12 µm and Gaussian sd 1 → 0.8; total data sd is then about 1, as in whitened data, and a unit-norm
   matched filter on noise has sd about 1.3. Hash 150 units, 5 Hz, norm 2.5-5 → 120, 3 Hz, norm 2-4; norm median 16 → 18.
3. Subsample jitter: ±0.25 sample at 15 kHz, the time jitter of ±0.5 sample at KS4's 30 kHz.
4. Pair partner moved 8-22 µm instead of 6-14 µm: at 6-14 µm with the narrower footprints even the oracle could not separate pairs
   (pair-resolved 0.17).
5. Spatial filters for universal detection sit on the channel grid (16 rows × 2 columns), not at half pitch, 2 widths; runtime.
6. Spike-time scoring removes each (true, output) pair's median lag first (search ±9 samples, then ±4), so a template aligned a
   few samples off isn't punished for that.
7. Base recording 240 s → 180 s (90 batches): a 0.1 Hz unit has 18 spikes, near n_min. Sparse results are pessimistic for both sides;
   the T480 scale condition checks this.
8. 6 workers rather than 4, to fit the runtime (16 physical cores).
9. **New arms, added because of a design gap, not a result:** KS2.5's final pass keeps tracking templates, which matters under drift.
   `I1_*_track`: I with a time-ordered final pass that keeps updating (m = 200 spikes). `Tr3`: R3 then the same tracked pass, the
   control that separates tracking from interleaving, as Rd separates discovery. Drift variant only.
10. An **oracle arm O** (true templates, same matching, same final clustering) gives the ceiling on seeds 1-5.
11. I(0.5) dropped; I(0.25) and I(1) kept.
12. Scale check at 120 s except T480, 3 seeds, arms R3, I(1), I(0.25), S. Sensitivity on 3 seeds. Algorithm-seed variability: seeds 1-3,
    2 extra algorithm seeds, R3 and I(1), stationary.
13. Tuning of R uses R3, as the comparison point.

### Deviations made after tuning, before any held-out run

Tuning ran as registered (72 runs on seeds 7001-7003 at 180 s, 12.5 min on 6 workers; machine at 100% load from other jobs, so runs took
about 1.5× the pilot's time). Chosen: R rho1 0.98, bim 0.4, K_init 8; I spike-count schedule capped at 2000, n_birth 30, rho_merge 0.93. Full
table in `tuning_table.txt`. Selection margins are thin: I's top four configurations are within 0.01 in objective.

14. **Caveat on tuning:** both winners sit on a grid edge (R: lowest bim and K_init, highest rho1; I: highest n_birth). The optimum for each may
    lie outside its grid. The sensitivity stage extends n_birth to 60; R gets no such extension, a small asymmetry in I's favour.
15. Held-out recordings 120 s (60 batches), not 180 s, to stay near the time budget. A 0.1 Hz unit has 12 spikes, at n_min, so **sparse
    results here are floors for both sides**; the T360 condition checks them. Settings were tuned at 180 s.
16. Oracle on seeds 1-3 only. Only one tracked I arm (`I1_shuffled_track`). Sensitivity on seeds 1-2 at 120 s (stationary; schedule also on
    drift; n_keep on nonstat). Scale check: seeds 1-3, arms R3 and I(1) only; recording length 360 s in place of 480 s. The base row
    of the scale table is the main stationary run.
17. **Added condition drift10** (10 µm peak to peak; seeds 1-3; R3, I(1), I(1) time, I(1)+track, R3+track). The tuning objective under 30 µm drift
    was about 0.2 for R and 0.05 for I, because deviation 1 narrowed footprints to 10-26 µm after the 30 µm figure was fixed. 30 µm now
    moves a unit by more than its footprint, which is much harsher than what registration leaves behind. H5 is read at both sizes.

Oracle calibration on the pilot seed: T labels mean accuracy about 0.7, found 0.8 (weak, wide and near-identical units stay hard even
with true templates), so results should be read relative to the ceiling.

### What the toy leaves out (known before running)

KS4's graph clustering and its neighbour structure, real probe geometry beyond 32 channels, drift registration, real waveform
heterogeneity and non-Gaussian amplitude structure, 2000-template crowding (only approached by the scale check), and
GPU-scale cost models. Clock numbers are CPU numpy/torch and only compare arms within this code.

## Part 2. Results

Compute: 54 min at 6 workers against a target of about 40 with "a few" cores (tune 12.5, main 23, variability 1.3, sensitivity 3.4, scale 7.2, post hoc 6.4;
the machine was at 100% load from other jobs). The registered stages alone took 47 min. Nothing in `kilosort/` was touched; nothing committed.
Full tables, including every paired comparison and per-unit percentiles: `C:\data\ks4_tuning\exploratory\interleaving\simulation\results.md`. `analyse.py` rebuilds them;
`decompose.py` and `extra_cis.py` give the extra intervals quoted below.

Notation: T labels = the spike's template; C labels = global clustering of the last extraction + CCG merge. O = oracle, true templates (seeds 1-3 only, so
compare it loosely). All intervals are 95% bootstrap over 10 data seeds unless stated. FPE = full-pass equivalents.

### Verdicts

| | Hypothesis | Verdict | Key numbers |
|---|---|---|---|
| H1 | I reaches R's template error in fewer passes | **Refuted** | After 1 pass, I − R1 median error +0.057 [+0.032, +0.086] stationary, +0.078 [+0.064, +0.093] nonstat. On median error 0/10 seeds reach R1 within the pass |
| H2 | I recovers poor-match units R misses | **Refuted** | I(1) C − R2 C −0.025 [−0.100, +0.049]; I(1) T − R2 C −0.104 [−0.324, +0.047]. Extra rounds do not recover them either |
| H3 | I has more merges, false units or sparse loss | **Supported** | Merged +0.135 [+0.075, +0.198] (T), +0.081 [+0.031, +0.137] (C); false units +3.8 [+1.3, +6.5] (T only); sparse found −0.11 [−0.19, −0.03] (C) |
| H4 | S matches I at less complexity | **Supported** | S C − I(1) C +0.027 [+0.008, +0.049], +0.004 [−0.006, +0.018], +0.003 [−0.019, +0.024]; 1.75 FPE against 2, a third of the matched-filter ops |
| H5 | With drift one side wins | **Refuted at 30 µm, unclear at 10 µm** | I(1) C − R3 C −0.006 [−0.019, +0.008]; tracking changes nothing on C labels. At 10 µm, 3 seeds: 0.519 against 0.511 |
| H6 | Global clustering gets structure I can't | **Supported, partly closed by a final clustering** | R3 C − I(1) T +0.10 [+0.05, +0.15], +0.13 [+0.07, +0.22]; against I(1) C +0.07 [+0.02, +0.12], +0.03 [−0.05, +0.10] |

### Main comparison (means over seeds 1-10)

| | | acc | found | false | merged | pairs | poor found | < 0.5 Hz found | te med |
|---|---|---|---|---|---|---|---|---|---|
| stationary | O T (seeds 1-3) | 0.694 | 0.78 | 5.0 | 0.04 | 0.56 | 0.58 | 0.13 | 0.058 |
| | R2 C | 0.617 | 0.64 | 10.3 | 0.34 | 0.27 | 0.65 | 0.25 | 0.185 |
| | R3 C | 0.633 | 0.67 | 9.2 | 0.33 | 0.30 | 0.60 | 0.24 | 0.182 |
| | R4 C | 0.634 | 0.67 | 8.5 | 0.33 | 0.28 | 0.57 | 0.19 | 0.181 |
| | Rd3 C | 0.622 | 0.65 | 17.0 | 0.35 | 0.25 | 0.60 | 0.17 | 0.187 |
| | S C | 0.640 | 0.68 | 7.4 | 0.34 | 0.28 | 0.64 | 0.23 | 0.186 |
| | I(0.25) C | 0.603 | 0.62 | 7.9 | 0.38 | 0.18 | 0.49 | 0.22 | 0.307 |
| | I(1) T | 0.570 | 0.59 | 13.0 | 0.47 | 0.20 | 0.55 | 0.16 | 0.249 |
| | I(1) C | 0.613 | 0.63 | 8.7 | 0.41 | 0.23 | 0.63 | 0.14 | 0.249 |
| nonstat | R3 C | 0.573 | 0.57 | 8.1 | 0.35 | 0.25 | 0.48 | 0.28 | 0.229 |
| | S C | 0.563 | 0.59 | 6.8 | 0.34 | 0.27 | 0.52 | 0.19 | 0.254 |
| | I(1) T | 0.527 | 0.54 | 11.0 | 0.47 | 0.12 | 0.59 | 0.12 | 0.320 |
| | I(1) C | 0.560 | 0.58 | 8.0 | 0.38 | 0.22 | 0.52 | 0.23 | 0.320 |
| drift 30 µm | O T (seeds 1-3) | 0.386 | 0.35 | 19.0 | 0.34 | 0.06 | 0.54 | 0.05 | 0.093 |
| | R3 C | 0.325 | 0.29 | 17.2 | 0.52 | 0.08 | 0.27 | 0.06 | 0.363 |
| | I(1) T | 0.321 | 0.25 | 28.5 | 0.61 | 0.07 | 0.25 | 0.05 | 0.394 |
| | I(1) C | 0.319 | 0.29 | 16.9 | 0.57 | 0.12 | 0.24 | 0.03 | 0.394 |

Compute, stationary (matched-filter ops in 10⁹ template-samples, clock on one loaded core): R2 0.8, 23 s; R3 0.9, 34 s; R4 1.0, 44 s; S 0.3, 20 s; I(0.25) 0.3, 14 s;
I(1) 0.9, 31 s. I(1) is not cheaper than R2: it runs a larger dictionary (66 templates against 54) plus a detection on every residual.

### H1, template error

Rounds plateau at once (R1 0.192, R2 0.185, R3 0.182, R4 0.181 stationary), far above the oracle's 0.058; most of that gap is crowded and merged units, which more rounds
don't fix. I is still improving at the end of its pass (0.306 at 0.25, 0.276 at 0.5, 0.249 at 1.0).

A fair check on my own comparison: R's dictionaries are rebuilt from clusters, I's end-of-pass dictionary isn't. So I reran I (seeds 1-6, post hoc):

| median error, stationary / nonstat | R2 (2 FPE) | I(1) rebuilt from final clusters (2 FPE) | I(2) end (2 FPE) | R3 (3 FPE) | I(2) rebuilt (3 FPE) |
|---|---|---|---|---|---|
| | 0.183 / 0.233 | 0.210 / 0.287 | 0.228 / 0.297 | 0.185 / 0.236 | 0.199 / 0.272 |

I(1) rebuilt − R2 is +0.027 [+0.010, +0.047] and +0.054 [+0.035, +0.083]; I(2) rebuilt − R3 is +0.013 [+0.001, +0.025] and +0.035 [−0.028, +0.085]. This narrows the gap a lot and doesn't close it.
**At the same cost, one global clustering of the final extraction lowers error more than a second interleaved pass does** (0.210 against 0.228). That is the clearest single result here.

Where the deficit sits (I(1) − R3, stationary): units > 3 Hz outside pairs +0.029 [+0.012, +0.045]; units < 3 Hz outside pairs +0.153 [+0.093, +0.210]; units both arms find +0.058 [+0.038, +0.076].

I first read this as starvation by total spike count, and that was **wrong**. Tripling the recording barely helps (T360: I(1) 0.241, R3 0.185, and I at a third of that pass already 0.252). Raising the
rate does: at 15 Hz median the gap is 0.219 against 0.204, at 1 Hz it is 0.433 against 0.192. The governing quantity looks like **spikes per unit within the birth and update window**, not spikes per
unit per recording. Three seeds per condition, so a pattern and not a measurement.

### H2, poor-match units

- 9.8 of 48 units are poor-match; 3.1 of those have a universal score below 8.
- R2 is the best R arm on them, so against R3 alone I would look better than it is. I(1) T − R3 C is +0.10 [+0.02, +0.19] in nonstat, but I(1) T − R2 C is +0.039 [−0.057, +0.121] there, −0.104 [−0.324, +0.047] stationary, −0.045 [−0.126, +0.031] drift. C labels: −0.025, −0.028, −0.050, all CIs spanning 0. This comparison picks R2 after seeing it is best, which favours R; the R3 comparison is reported for that reason.
- **Extra rounds don't recover them and may lose them.** R4 C − R2 C on all poor units −0.08 [−0.20, +0.04]; R3 − R2 on those with score ≥ 8 −0.08 [−0.16, −0.01]. Wide units fall from 0.50 at R2 to 0.22 at R4, −0.28 [−0.57, −0.005], few units, wide interval. They are the units the temporal PCs represent worst (as little as 28% of energy).
- **Residual discovery isn't the lever.** Rd3 − R3 +0.00 [−0.04, +0.06] on poor units, with +7.8 [+6.5, +9.1] false units. Since Rd doesn't help and I doesn't beat R2, there is no interleaving gain to separate from discovery.
- Units below the universal threshold stay mostly lost in every arm (≤ 0.25; oracle 0.33 stationary). The limit is detection, not schedule.

### H3, confirmation bias, false units, sparse loss

- Merges are the robust cost: +0.135 [+0.075, +0.198] T, +0.081 [+0.031, +0.137] C (stationary); +0.121 [+0.060, +0.175] and +0.029 [−0.004, +0.065] (nonstat). The likely mechanism is a template that grabs two units and keeps averaging both, since nothing in the loop splits it. I did not test that directly.
- False units are a T-label problem: +3.8 [+1.3, +6.5] stationary, +11.3 [+8.5, +15.0] drift. On C labels −0.5 [−2.1, +1.1].
- Sparse (< 0.5 Hz): −0.11 [−0.19, −0.03] C stationary; −0.16 [−0.28, −0.04] T nonstat. Results for these units are floors (deviation 15).
- **The feared silent-epoch loss did not appear:** silent units −0.04 [−0.09, +0.00] C, and removing the n_keep protection changes accuracy by at most 0.011. The recording is short; silence of 40-60 s may matter more over 20 min.

### H4, subsampled rounds

S beats or ties I on all three variants at 1.75 FPE and a third of the ops, and needs no new machinery. It even edges R3 in stationary (+0.007 [+0.001, +0.014]) at 1.75 FPE against 3. Costs: mean template error +0.013 [+0.003, +0.022] and +0.029 [+0.017, +0.042]
against R3, and sparse found −0.088 [−0.193, +0.005] in nonstat. It learned from 30 s of data here; a real run at f = 0.25 would see 5 minutes.

### H5, drift

At 30 µm every arm sits at 0.32-0.33 and even the oracle reaches only 0.386, so this condition mostly shows that static dictionaries fail. I's T-label deficit disappears (−0.063 stationary to −0.005 [−0.022, +0.013]), but that is R falling to meet it, not I winning. Time order: +0.009 [−0.002, +0.021]. Tracking: I+track C − R3+track C −0.002 [−0.009, +0.005], and
R3+track C − R3 C −0.005 [−0.018, +0.008], so the tracked final pass gives neither side anything on C labels. On T labels I+track beats R3+track by +0.026 [+0.012, +0.041], with about 11 more false units. At 10 µm (3 seeds; C acc): R3 0.519, I 0.511, I time 0.519, I+track 0.500, R3+track 0.510. No side wins.

### H6, near-identical pairs

Pair-resolved rates are low for everyone (O 0.56, R3 0.30, I(1) T 0.20, I(1) C 0.23; stationary). Clustering after the last extraction gives back part of the gap. The same effect recurs across arms: C labels beat T labels by +0.043 [+0.021, +0.066] for I and +0.035 [+0.018, +0.054] for R4. Any interleaved design that drops the final clustering gives that up.

### Sensitivity of I (seeds 1-2 unless stated; Δ from tuned)

| change | Δ acc T | Δ acc C | te med | note |
|---|---|---|---|---|
| n_birth 8 | +0.046 | +0.010 | 0.218 | **tuning's choice of 30 was not I's best on held-out seeds** |
| n_birth 60 | −0.025 | −0.047 | 0.334 | |
| 0.5-s batches | +0.021 | +0.011 | 0.233 | scale stage, per seed against R3: +0.063, −0.026, −0.006, so unresolved |
| 8-s batches | +0.003 | −0.025 | 0.259 | scale stage, per seed against R3: −0.039, −0.013, −0.054 |
| buffer 5 batches | −0.082 | −0.030 | 0.363 | short buffers starve births |
| buffer 60 batches | 0.000 | 0.000 | 0.261 | |
| schedule batches 20→400 | +0.011 | +0.004 | 0.295 | drift −0.015 / −0.007 |
| schedule spikes, cap 200 | 0.000 | +0.024 | 0.266 | drift +0.005 / +0.005 |
| n_keep 0 (nonstat) | +0.007 | −0.011 | 0.293 | time-ordered; shuffled ≤ 0.004 |

The birth threshold matters most, then buffer length; the step-size schedule hardly matters.

**Best-case I, post hoc, favours I** (n_birth 8, cap 200, 0.5-s batches, picked from the table above; seeds 3-8), against R extended past its own grid edge (bim 0.3, chosen on seeds 1-2):
Ibest C − R3 C −0.021 [−0.050, +0.013] stationary, +0.000 [−0.022, +0.020] nonstat; against extended R −0.023 [−0.045, −0.004] and −0.012 [−0.036, +0.008]. Template error stays above R3's, +0.017 [−0.007, +0.033] and +0.053 [+0.021, +0.082].
Sparse found stays lower (0.12 against 0.22, 0.18 against 0.27). So with favourable settings **I can about tie R on accuracy when it also gets a final clustering**, but doesn't beat it, and H1 and H3 don't reverse.

### Order dependence and run-to-run variability

Shuffled against time order: C −0.010 [−0.027, +0.006], drift +0.009 [−0.002, +0.021], nonstat +0.009 [−0.019, +0.033]. No measurable dependence, including under silent epochs.
Across algorithm seeds (3 data seeds, 3 algorithm seeds): accuracy sd R3 0.0095, I 0.014, both far under the 0.057 between data seeds; but the *set* of found units is less stable for I, Jaccard 0.82-0.84 against 0.95.

### Scale check: which side of each boundary are the real data on?

| condition | units/chan | spikes/unit/batch | R3 C | I(1) T | I(1) C | I(1) C − R3 C, by seed | te med R3 / I(1) |
|---|---|---|---|---|---|---|---|
| base | 1.5 | 12.4 | 0.634 | 0.549 | 0.623 | +0.021 −0.036 −0.018 | 0.181 / 0.256 |
| u24 | 0.75 | 10.5 | 0.790 | 0.692 | 0.789 | −0.036 +0.029 +0.003 | 0.138 / 0.225 |
| u96 | 3 | 8.6 | 0.449 | 0.399 | 0.425 | −0.015 −0.038 −0.018 | 0.278 / 0.336 |
| u192 | 6 | 9.8 | 0.159 | 0.240 | 0.179 | +0.023 +0.016 +0.020 | 0.522 / 0.408 |
| rate 1 Hz | 1.5 | 2.5 | 0.653 | 0.465 | 0.576 | −0.038 −0.098 −0.093 | 0.192 / 0.433 |
| rate 15 Hz | 1.5 | 36.9 | 0.584 | 0.546 | 0.545 | −0.022 −0.081 −0.013 | 0.204 / 0.219 |
| 0.5-s batch | 1.5 | 3.1 | 0.621 | 0.571 | 0.631 | +0.063 −0.026 −0.006 | 0.186 / 0.233 |
| 8-s batch | 1.5 | 49.5 | 0.631 | 0.556 | 0.596 | −0.039 −0.013 −0.054 | 0.174 / 0.261 |
| 360 s | 1.5 | 12.4 | 0.628 | 0.577 | 0.603 | −0.033 −0.054 +0.012 | 0.185 / 0.241 |

- **Spikes per unit per batch:** real about 10 (5 Hz, 2 s), toy base 12.4, so same side, where I lags.
- **Rate:** I's deficit is largest at low rates. Real recordings have many low-rate units; this is the boundary that matters most, and real data sit on the unfavourable side for those units.
- **Spikes per pass:** real 5-10× base. T360 suggests this won't rescue I, on 3 seeds and only 3×.
- **Density: the toy cannot answer.** Real is about 5 templates per channel, i.e. u192, the only condition where I wins (I T − R3 C +0.067 to +0.088 at all 3 seeds, but 129 false units against 27). I suspected R's cap of 8 clusters per row (128 clusters for 192 units). I was **wrong again**: with K_init 32 R still builds only 70 templates and R and I tie on C labels (0.177, 0.176). The oracle shows the real reason: true templates give T 0.369 but C **0.197**, so the toy's clustering primitive itself fails at this density. KS4's graph clustering doesn't collapse at that density on real data, so this regime says nothing about real data in either direction. I T still beats R C there (0.235 against 0.177 at all 3 seeds), for 137 false units against 28: I keeps growing its dictionary where clustering breaks down, and I can't tell whether that would be useful or noise.

### What this cannot say about real KS4

1. **The toy doesn't reproduce why rounds help on real data.** Real rounds cut H8's dictionary from 2450 to 1760 and clean most units; here K goes 54 → 52 and R3 − R2 is only +0.016 [+0.002, +0.029] (nonstat +0.002). The clearest toy result about R is a plateau, which real data contradict. R's real advantage may be larger than shown here, or I may have a redundancy-removal role the toy can't show.
2. Toy clustering is a tree over k-means, not KS4's graph clustering, and fails at real density. Every result about clustering (H3 merges, H6, C labels) inherits this.
3. 32 channels, 48 units, 120 s. Sparse units have 12-60 spikes, near n_min. Real ones have 120-600.
4. fs 15 kHz, no real waveform heterogeneity or noise; poor-match units are built to be poor, and their rate relative to real data is unknown. wPCA is fixed, whereas KS4 can derive it from data.
5. Drift here is uncorrected, and the 30 µm condition exceeds footprint width after deviation 1; it is an extreme.
6. Clock numbers are CPU under load and only compare arms within this code. FPE ignores that real extraction and clustering have different costs (about 100 s and 115 s per 20 min).
7. Both tuned settings hit grid edges and the held-out recording length differs from tuning's. Best-case I and extended R address this after the fact, on few seeds. All scale and sensitivity results are 2-3 seeds and show direction only.
8. Everything labelled post hoc was prompted by seeing the results. Treat those as leads, not tests.

### Bottom line

For the proposal as specified, the toy gives no scheduling win: I costs about what R2 does, learns worse templates than one round of global clustering, merges more and loses sparse units, with no gain on poor-match units or drift. Its best case is a tie, and that depends on keeping a final global clustering. S gets at least as much without the machinery. The result most worth carrying to real data: **test subsampled rounds before interleaving, and don't drop the final clustering.** Conclusions about density need KS4 itself, because the toy fails there for reasons unrelated to scheduling.
