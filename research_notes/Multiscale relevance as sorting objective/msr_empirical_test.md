# Multiscale relevance (MSR) as a sorting objective: empirical test on the kobe sorts

All measurements below are new results from this test: CPU only, on existing sorts of the 20-minute kobe hybrid excerpt (1668–2868 s). No new sorts were run and the GPU was not used. Before starting I checked running processes: two CPU analyses from other sessions were running (`matched_contamination.py`, `bayes_decode.py`), and the GPU was at 1%. This work only reads sort folders and writes to its own folders, so it cannot affect a queue. Compute time was about 9 minutes in all: synthetic tests 7 s, per-unit correlations 60 s, perturbations 132 s plus 209 s, sort-level summaries 118 s.

**Code** (`C:/code/github/Kilosort/tuning/exploratory/msr/`):
- `msr.py`: the MSR itself, plus a Poisson baseline.
- `common.py`: loaders and statistics. Cleaned trains are checked to equal `cleaning.load_clean_units`.
- `synthetic.py`, `real_units.py`, `perturb_msr.py`, `sort_level.py`: the four analyses.

**Outputs** are in `C:/data/ks4_tuning/exploratory/msr/`, with copies in `tuning/exploratory/msr/results/`: `synthetic.json`, `real_units.json`, `real_units_<sort>.csv`, `hybrid_<sort>.csv`, `perturb_default_s1.json`, `perturb_H8_s1.json`, `sort_level.json`, `sort_level.csv`, `sort_level.log`.

## How MSR is defined and implemented, and how it behaves on synthetic trains

### Takeaway
MSR as published is easy to reproduce. On synthetic trains it rises when the firing rate is modulated on slow, behavioural timescales: heading tuning, or a 60 s on/off rate. It falls steadily with spike count M, and that drop across 0.2–50 Hz (0.028) is about as large as the gain from strong tuning (0.02–0.04). MSR must therefore be corrected for M before use. Above a modest tuning strength it saturates, so it does not track how much heading information a unit carries.

### Cited Findings
- **Definitions (paper):** time is split into bins of duration Δt, and k_s is the spike count in bin s.
  - Resolution: H[s] = −Σ_s (k_s/M) log_M(k_s/M).
  - Relevance: H[K] = −Σ_k (k·m_k/M) log_M(k·m_k/M), where m_k is the number of bins holding k spikes.
  - MSR (R_t) is the area under the H[K]–H[s] curve as Δt varies.
  - The mEC spikes were binned at Δt = 1 ms (binarised in the Boolean-function analysis); 0.5 ms gave qualitatively similar results.
  - [Cubero, Marsili & Roudi 2020, arXiv 1802.10354](https://www.arxiv.org/abs/1802.10354)
- **Reference implementation** (`relevance.py`):
  - Partitions: N = unique(int(logspace(0.4, round(log10(0.99·T), 2), 100))), plus the finest partition, with contiguous bins whose lengths differ by at most one.
  - Both entropies are divided by log2 M.
  - The anchors (H[K], H[s]) = (0, 0) and (0, 1) are added, points are sorted by H[s], and MSR is the trapezoid integral of H[K] over H[s].
  - [github.com/rcubero/MSR](https://github.com/rcubero/MSR)
- **The paper's own caveats:** MSR has a negative relation with spike count M, which is partly associated with burstiness. In the mEC data, residual MSR (after a linear regression on log M) was still correlated with spatial and HD information. [Cubero et al. 2020](https://www.arxiv.org/abs/1802.10354)
- **Reported correlations** in the mEC data (65 neurons): Spearman of MSR with spatial information 0.825 and with HD information 0.755 (Pearson 0.738 and 0.460; Fig. 2). [Cubero et al. 2020](https://www.arxiv.org/abs/1802.10354)
- **Choices made here** ([msr.py](C:/code/github/Kilosort/tuning/exploratory/msr/msr.py)):
  - 1 ms time points, T = 1,200,000. Partitions as in `relevance.py`, 98 in all, the finest being 1 ms. Same anchors and integration.
  - Bins hold spike counts, not binarised. The binarised version gives nearly the same values: per-unit Spearman 0.998–0.999 between the two versions, largest absolute difference 0.006–0.010 ([real_units_*.csv](C:/data/ks4_tuning/exploratory/msr/)).
  - Masked time (under 0.02 s on H8) is not removed.
  - The rate-corrected MSR is **dMSR = MSR − mean MSR of homogeneous Poisson trains with the same M and duration**, interpolated in log M over 40 values of M from 50 to 400,000, 8 replicates each.
- **Homogeneous Poisson, 1200 s** ([synthetic.json](C:/data/ks4_tuning/exploratory/msr/synthetic.json), 10 replicates):

  | Rate (Hz) | 0.2 | 1 | 5 | 20 | 50 |
  |---|---|---|---|---|---|
  | MSR | 0.262 | 0.254 | 0.244 | 0.238 | 0.234 |

  The SD across replicates is 0.001–0.005. At a fixed 5 Hz, a longer recording lowers MSR in the same way (150 s: 0.258; 2400 s: 0.241), so the dependence is on M, not on the rate as such.
- **Matched mean rate, 1200 s:** synthetic HD cells driven by the real kobe tracked heading, compared with Poisson at the same rate (MSR 0.2454 at 5 Hz):

  | Tuning κ | 0.5 | 1 | 2 | 4 | 8 |
  |---|---|---|---|---|---|
  | MSR at 5 Hz | 0.2649 | 0.2787 | 0.2828 | 0.2812 | 0.2818 |
  | True HD information (bits/spike) | 0.09 | 0.32 | 0.88 | 1.53 | 2.03 |

  The pattern is the same at 1 and 20 Hz (20 Hz: Poisson 0.2372, κ 2 0.2787). MSR saturates from κ ≈ 2 while the true information keeps rising more than twofold. ([synthetic.json](C:/data/ks4_tuning/exploratory/msr/synthetic.json))
- **Other modulations** at 5 Hz:
  - No change from Poisson: 8 Hz theta modulation at depth 1.0 (0.2450), a 2 ms dead time (0.2444), and a 30% contiguous dropout (0.2456).
  - A 60 s on/off square wave at depth 0.5 raises MSR to 0.2710.
- **Errors on a synthetic HD cell** (5 Hz, κ 2):
  - Contamination with Poisson spikes changes MSR non-monotonically: 0.2825 at 0%, 0.2834 at 10%, 0.2830 at 30%, 0.2812 at 50%.
  - Thinning changes it by less than 0.001 up to 50% (0.2837).

### Inferences
- MSR responds to rate variance on timescales from seconds to minutes. Heading tuning produces that variance because heading changes slowly, but so would any slow non-behavioural fluctuation (the square wave). Theta-scale modulation and refractoriness are invisible to it.
- MSR is a saturating detector of whether a unit is modulated, not a graded measure of information. Its dependence on M is as large as its tuning signal, so raw MSR cannot be compared across units or sorts with different spike counts without a baseline like dMSR.

### Gaps
- I did not extract the paper's supplementary synthetic tests (Fig. S1) quantitatively. The direction of the M dependence agrees with the paper; magnitudes were not compared.
- The paper's exact anchor and partition conventions were taken from the code, not the text. The text defines the curve but not the bin grid.

## (a) Does MSR track the harness's per-unit heading information?

### Takeaway
Yes, moderately to well, once spike count is accounted for:
- Raw MSR against per-unit S_i: Spearman 0.37–0.59.
- Rate-corrected dMSR, or MSR with log rate partialled out: 0.63–0.80, and 0.67–0.82 against bits per spike.

This holds in all four sorts and within every firing-rate quartile. As a covariate-free screen for "tuned vs untuned" units it is good (AUC 0.85–0.96).

### Cited Findings
- **Sorts:** default KS4 seed 1 (`runs/hybrid_default_seed1`), H8 seed 1 (`runs/search/198_H8_cds10_s1/sort`), KS2.5 "standard" r1 and "standard_low" r1. Units are the eligible units of each `evaluation.json` (hybrid copies excluded), with trains cleaned per §6.1–6.3. ([real_units.json](C:/data/ks4_tuning/exploratory/msr/real_units.json), [real_units.py](C:/code/github/Kilosort/tuning/exploratory/msr/real_units.py))
- **Spearman correlations with S_i:**

  | Sort (eligible units) | MSR vs rate | MSR vs S_i | MSR vs S_i, log rate partialled out | dMSR vs S_i | dMSR vs S_i, log rate partialled out |
  |---|---|---|---|---|---|
  | default s1 (777) | −0.655 | +0.369 | +0.643 | +0.625 | +0.681 |
  | H8 s1 (1712) | −0.388 | +0.594 | +0.784 | +0.783 | +0.800 |
  | KS2.5 standard r1 (899) | −0.180 | +0.425 | +0.671 | +0.720 | +0.705 |
  | KS2.5 standard_low r1 (1030) | −0.144 | +0.483 | +0.703 | +0.753 | +0.730 |

  ([real_units.json](C:/data/ks4_tuning/exploratory/msr/real_units.json))
- **Against tracked-heading S_track_i**, dMSR with log rate partialled out: +0.648, +0.783, +0.674, +0.713. **Against S_i per spike** (S_i/rate): raw MSR +0.798, +0.818, +0.727, +0.757. ([real_units.json](C:/data/ks4_tuning/exploratory/msr/real_units.json))
- **Within firing-rate quartiles**, dMSR vs S_i: default 0.69, 0.57, 0.64, 0.56; H8 0.78, 0.68, 0.81, 0.67; KS2.5 standard 0.71, 0.61, 0.42, 0.68. ([real_units.json](C:/data/ks4_tuning/exploratory/msr/real_units.json))
- **Separating tuned from untuned units:** median dMSR by S_i band (bits/s):

  | Sort | S_i < 0.02 | 0.02–0.1 | 0.1–0.5 | 0.5–2 | ≥ 2 |
  |---|---|---|---|---|---|
  | default | 0.019 | 0.016 | 0.029 | 0.038 | 0.042 |
  | H8 | 0.010 | 0.012 | 0.025 | 0.038 | 0.042 |

  AUC of dMSR for S_i > 0.5 against S_i < 0.02: default 0.845 (526 vs 18 units), H8 0.961 (751 vs 134), KS2.5 standard 0.963 (516 vs 39). (computed from [real_units_*.csv](C:/data/ks4_tuning/exploratory/msr/))
- **Top-decile overlap** (top 10% by dMSR against top 10% by S_i): 27%, 36%, 36%, 35%, where chance is 10%. ([real_units.json](C:/data/ks4_tuning/exploratory/msr/real_units.json))
- **Almost every real unit is above the Poisson baseline:** dMSR > 0 for 98.8–99.5% of eligible units. Median dMSR is 0.029–0.036 and the 5th–95th percentile range is about 0.005–0.045, so even untuned units are far from Poisson. ([real_units_*.csv](C:/data/ks4_tuning/exploratory/msr/))

### Inferences
- On this HD-rich recording, dMSR is a reasonable covariate-free proxy for per-unit heading tuning: comparable to, though weaker than, the paper's mEC values of 0.76–0.83. The raw-MSR correlation is held down by the strong negative dependence on rate.
- Rank agreement at the top is weak (about one third overlap), so dMSR would not pick the same best units as S.

### Gaps
- The correlation may partly reflect any slow modulation that co-varies with heading, such as running speed or position. Here MSR was compared only with heading scores.

## (b) Does MSR track unit cleanliness?

### Takeaway
Only indirectly. Across units, MSR is weakly lower in more contaminated KS4 units:
- Refractory "local" contamination estimate: Spearman −0.10 to −0.50.
- "Plain" estimate with log rate partialled out: −0.12 to −0.21.
- KS2.5 units: no association.

That pattern mirrors S_i's own association with contamination, which suggests the link runs through tuning: contaminated units tend to be poorly tuned multi-unit activity. Against hybrid ground truth, MSR barely changes between a true train and its sorted version, and the changes do not go down with false positives.

### Cited Findings
- **Cross-sectional Spearman, MSR vs refractory contamination** (Hill et al. estimates from `isolation/comod.py`):
  - "plain": default −0.224, H8 −0.246, KS2.5 standard +0.011, standard_low +0.069. With log rate partialled out: −0.120, −0.213, +0.054, +0.110.
  - "local": −0.469, −0.499, −0.135, −0.098. With log rate partialled out: −0.335, −0.458, −0.089, −0.054.
  - S_i itself against plain contamination, log rate partialled out: −0.188, −0.275, +0.024, +0.009.
  - dMSR vs S_i with log rate and contamination both partialled out: 0.686, 0.789, 0.705, 0.732, almost unchanged.
  - ([real_units.json](C:/data/ks4_tuning/exploratory/msr/real_units.json))
- **Hybrid ground truth** (40 inserted units; best-matching sorted unit from the raw output, as matched by `hybrid.recovery`):
  - Median agreement: default 0.953, H8 0.976, KS2.5 standard 0.978, standard_low 0.969.
  - MSR of the true train against MSR of its sorted version: Spearman 0.85, 0.96, 0.84, 0.94.
  - Median change in MSR from true to sorted: +0.0004, +0.0001, 0.0000, −0.0001.
  - Change in dMSR against the false-positive fraction: +0.34 (p = 0.03), +0.22 (p = 0.17), +0.01, −0.04. A positive sign means more false positives come with higher MSR.
  - Change in dMSR against the missed fraction: +0.28 (p = 0.08), +0.07, −0.03, −0.30 (p = 0.06).
  - ([real_units.json](C:/data/ks4_tuning/exploratory/msr/real_units.json), [hybrid_*.csv](C:/data/ks4_tuning/exploratory/msr/))

### Inferences
- MSR contains no isolation information beyond what tuning carries. The negative cross-sectional correlation is not causal: adding contamination raises MSR (next section). The hybrid units are mostly well recovered (median false-positive fraction about 1%), so this test has little range. Within that range, the only significant association has the wrong sign.

### Gaps
- No amplitude-based isolation metrics (isolation distance, L-ratio, the amplitude-cut-off results in `isolation/amplitude.py`) were compared.
- The hybrid check has 40 units per sort with few badly recovered ones.

## (c) Does MSR respond in the right direction to sorting errors?

### Takeaway
No. On 40 well-isolated, well-tuned units per sort (default and H8, 3 draws each), every response is either wrong-signed or carried only by spike count or unit count. MSR is not monotonic and correctly signed for any error beyond what counting spikes or units gives:
- **Contamination**, from another unit or from Poisson spikes, raises MSR and dMSR monotonically from 5% to 30%. That is the wrong sign.
- **Thinning** leaves raw MSR flat. dMSR falls only because the Poisson baseline rises as spikes are removed, so the response is a pure spike-count effect.
- **Near-duplicates** inside a unit leave MSR unchanged.
- **Merges** lower the summed MSR only because the sum loses a unit, the same as dropping one. The merged unit's own dMSR falls below the better parent's in only 43–65% of merges.
- **Random splits and duplicated units** raise the summed MSR, where S is flat for splits and is protected from duplicates by cleaning.

### Cited Findings
- **Targets:** eligible units with plain contamination ≤ 5% and S_i in the sort's top quartile. The pool was 97 of 777 default units and 207 of 1712 H8 units; 40 were drawn from each. Default targets have median rate 3.9 Hz, 4643 spikes, dMSR 0.040 and S_i 2.55 bits/s. ΔS_i was rescored with `score.corrected_scores`, which reproduces `evaluation.json` exactly. ([perturb_msr.py](C:/code/github/Kilosort/tuning/exploratory/msr/perturb_msr.py), [perturb_default_s1.json](C:/data/ks4_tuning/exploratory/msr/perturb_default_s1.json))
- **Single-unit errors, default s1** (median over units; "up" is the fraction of units whose dMSR rose, "↓" the fraction whose dMSR or S_i fell):

  | Error | Fraction | ΔdMSR | ΔMSR | ΔS_i (relative) |
  |---|---|---|---|---|
  | Contamination from another unit | 5% | +0.00097 (up 0.90) | +0.00080 | −6% |
  | | 10% | +0.00167 | +0.00117 | −10% |
  | | 20% | +0.00269 | +0.00145 | −17% |
  | | 30% | +0.00341 (up 0.95) | +0.00154 | −25% |
  | Poisson contamination | 5% | +0.00088 | +0.00062 | −6% |
  | | 10% | +0.00192 | +0.00147 | −11% |
  | | 20% | +0.00288 | +0.00178 | −19% |
  | | 30% | +0.00333 (up 0.77) | +0.00180 | −26% |
  | Thinning | 5% | −0.00015 (↓ 0.62) | +0.00002 | −5% |
  | | 10% | −0.00057 | −0.00009 | −10% |
  | | 20% | −0.00109 | 0.00000 | −20% |
  | | 30% | −0.00180 | +0.00008 | −30% |
  | | 50% | −0.00401 (↓ 0.97) | −0.00003 | −50% |
  | Contiguous dropout | 10% | −0.00018 | +0.00048 | −10% |
  | | 30% | −0.00057 | +0.00137 | −30% |
  | In-unit jittered copies (±2 samples) | 5% | +0.00050 | +0.00013 | +5% |
  | | 30% | +0.00023 | −0.00194 | +30% |

  The relative ΔS_i values are medians of ΔS_i/S_i. ([perturb_default_s1.json](C:/data/ks4_tuning/exploratory/msr/perturb_default_s1.json))
- **H8 s1** gives the same picture:
  - 30% contamination from another unit: dMSR +0.00275; Poisson: +0.00287.
  - 50% thinning: dMSR −0.00405, raw MSR −0.00030.
  - 30% in-unit duplicates: dMSR −0.00023.
  - ([perturb_H8_s1.json](C:/data/ks4_tuning/exploratory/msr/perturb_H8_s1.json))
- **Multi-unit errors, default s1** (120 events each; medians):

  | Error | Δ sum MSR | Δ sum dMSR (fraction down) | Merged or new unit's dMSR below the better original | ΔS_sum |
  |---|---|---|---|---|
  | Merge, random partner | −0.288 | −0.035 (0.99) | 61% | −1.70 |
  | Merge, differently tuned (≥ 90°) | −0.290 | −0.037 | 62% | −2.31 |
  | Merge, similarly tuned (≤ 30°) | −0.288 | −0.036 | 43% (median +0.0007) | −0.91 |
  | Merge, nearest in depth | −0.287 | −0.035 | 57% | −1.76 |
  | Random split | +0.289 | +0.033 (never down) | each half below the original in 100% | +0.006 |
  | 80% jittered copy as a new unit | +0.289 | +0.039 (never down) | 83% | +2.02, before cleaning |

  The merge rows for H8 are similar (sum dMSR −0.030 to −0.035; merged dMSR below the better parent in 48–65%). ([perturb_default_s1.json](C:/data/ks4_tuning/exploratory/msr/perturb_default_s1.json), [perturb_H8_s1.json](C:/data/ks4_tuning/exploratory/msr/perturb_H8_s1.json))
- **Sort-level audit at a 5% rate**, as in `perturb.py` (3 draws; eligibility re-applied by spike counts). Noise scales across the five default replicates: σ_S = 14.3 bits/s and σ(sum dMSR) = 0.343. Default s1:

  | Error | ΔS | Δ sum dMSR |
  |---|---|---|
  | Merge of differently tuned units | −52.0 (−3.6σ) | −1.24 (−3.6σ) |
  | Drop | −47.2 (−3.3σ) | −1.30 (−3.8σ) |
  | Contamination | −56.2 (−3.9σ) | **+0.16 (+0.5σ)** |
  | Thinning | −51.6 (−3.6σ) | −0.25 (−0.7σ) |
  | Split | −0.2 | **+0.86 (+2.5σ)** |
  | Duplicate units, before cleaning | +35.1 | +1.20 (+3.5σ) |

  Mean dMSR moves by at most 0.0004 (σ 0.0002) under any of these: contamination +0.0002, thinning −0.0003, split −0.0003. H8 s1 gives the same signs (contamination ΔS −78.9, Δ sum dMSR +0.10; split ΔS −0.5, Δ sum dMSR +1.43). ([perturb_*.json](C:/data/ks4_tuning/exploratory/msr/), sort_level section, and [sort_level.json](C:/data/ks4_tuning/exploratory/msr/sort_level.json) for the default SDs)
- **The paper's pairwise result:** spike trains built by OR or XOR of two mEC neurons almost always had a lower MSR than the more relevant of the two, and never a significantly higher one (Fig. S8). Here the merged unit fell below its better parent in only 43–65% of merges. [Cubero et al. 2020](https://www.arxiv.org/abs/1802.10354)

### Inferences
- MSR measures how much a train's rate varies, not whether its spikes belong together. Spikes from another unit or from noise add variance, so contamination raises MSR. Mixing two modulated units, or splitting one at random, preserves the modulation, so MSR per unit barely moves.
- Summed over units, MSR behaves like a unit counter (about 0.28 per unit raw, about 0.03 per unit for dMSR). It therefore rewards oversplitting and duplicate units, and it detects merges only as lost units. The mean over units is insensitive to every error.
- Effect sizes are small next to the spread across units. 30% contamination changes dMSR by about 0.003, which is roughly 7% of a typical tuned unit's dMSR, against −25% for S_i.

### Gaps
- Heading-locked artefacts and amplitude-dependent (non-random) splits and thinning were not tested.
- The mechanism behind the rise in raw MSR under Poisson contamination of real units (+0.0018 at 30%) is not established. On a synthetic HD cell the change was non-monotonic.

## Sort-level MSR summaries across scored sorts, and use as an objective

### Takeaway
Across 51 scored sorts, summed dMSR ranks sorts much as S does (Spearman 0.88). It does so mainly by counting eligible units: with unit count partialled out the correlation drops to 0.55, and raw summed MSR to 0.06.
- Mean and median MSR rank sorts in the opposite order to S (−0.69 to −0.75).
- Summed dMSR disagrees with S in exactly the cases that matter: it ranks KS2.5 standard well above default (S ranks it below) and the fragmenting γ 3 above H8.

MSR is not suitable as an objective to optimise. At most it could serve as a behaviour-free per-unit tuning screen.

### Cited Findings
- **Sorts:** the 5 default replicates, the 6 KS2.5 sorts (standard and standard_low, r1–3), and the 40 search runs (all seeds) that still hold spike files. Those are rounds 4 and its ablation, runs 191–230; earlier runs' spike files have been pruned. All are scored under the current S, with S_default 1019.66. ([sort_level.py](C:/code/github/Kilosort/tuning/exploratory/msr/sort_level.py), [sort_level.csv](C:/data/ks4_tuning/exploratory/msr/sort_level.csv))
- **Spearman with S across the 51 sorts** (partial Spearman with eligible-unit count partialled out in brackets):
  - eligible units +0.841
  - sum MSR +0.841 (+0.058)
  - sum dMSR +0.884 (+0.550)
  - sum of positive dMSR +0.886 (+0.565)
  - spike-weighted yield Σ N_i·dMSR_i +0.526 (+0.621)
  - mean MSR −0.687; mean dMSR −0.751 (+0.413); median MSR −0.695
  - KS4 sorts only (45): sum dMSR +0.842 (+0.549)
  - The 16 configurations, means over seeds (default, the KS2.5 presets and the round-4 runs): S against sum dMSR +0.785, S against eligible units +0.712.
  - ([sort_level.json](C:/data/ks4_tuning/exploratory/msr/sort_level.json))
- **Noise across the five default replicates:** SD of sum dMSR 0.343 on a mean of 25.74 (1.3%), comparable to S's 1.4%. H8 s1 against the default mean: S +27.3 SD, sum dMSR +59.3 SD, eligible units +94 SD, mean dMSR −35.9 SD. ([sort_level.json](C:/data/ks4_tuning/exploratory/msr/sort_level.json))
- **Per-configuration means:**

  | Configuration | S (bits/s) | Sum dMSR | Eligible units |
  |---|---|---|---|
  | Default | 1019.7 | 25.74 | 772 |
  | KS2.5 standard | 989.0 (−2.1σ_S) | 27.58 (+5.4 SD) | 889 |
  | KS2.5 standard_low | 1073.5 | 30.84 | 1022 |
  | B1, no bimodality | 1087.6 | 27.69 | 875 |
  | B2, no γ | 1382.0 | 40.55 | 1338 |
  | G1 (γ 1.5) | 1378.7 | 42.53 | 1470 |
  | H8 | 1412.6 | 46.01 | 1695 |
  | G2 (γ 3) | 1403.8 | 46.37 | 1830 |
  | X1 (downsampling 5) | 1419.4 | 47.80 | 1831 |

  (computed from [sort_level.csv](C:/data/ks4_tuning/exploratory/msr/sort_level.csv))
- **For comparison,** the harness's resolution rule picked H8 over G2 and X1 because their extra units are seed-dependent fragments (about 1004 reproduced units for H8 against about 957 for G2 and about 998 for X1). [round4_results.md](C:/code/github/Kilosort/tuning/round4_results.md)

### Inferences
- Summed dMSR has an S-like ranking because both are sums of per-unit terms that are positive for nearly every real unit. dMSR is above zero for about 99% of units, so every extra unit, fragment or duplicate adds to the sum.
- S is null-corrected so that untuned units score about zero; dMSR has no analogous correction against modulation that is irrelevant or shared. A sort-level MSR objective would push towards oversplitting, which the perturbation splits and the G2/X1 ranking both show.
- Possible narrow uses:
  - a behaviour-free per-unit screen of tuning or modulation, for example to cross-check the untuned-units guard;
  - a covariate-free quantity to report on recordings without good tracking.
- It should not be used as the optimisation target, or as a cleanliness metric.

### Gaps
- Round 1–3 sorts could not be included because their spike files have been pruned, so the S range below about 1280 bits/s is covered only by the default, KS2.5 and B1 sorts.
- The MEC transfer excerpt was not tested.
- No variant was tried that subtracts a null keeping each unit's own slow modulation (for example a shuffle within time blocks). Such a variant might remove the unit-count effect, but it would still not fix the wrong sign under contamination.
