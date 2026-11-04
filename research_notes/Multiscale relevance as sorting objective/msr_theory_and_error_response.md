# Multiscale relevance (MSR): theory, biases, and how it responds to spike-sorting errors

Scope: what MSR is mathematically, what it measures, its known dependencies, and how it should behave under the specific errors a spike sorter makes. The underlying question is whether MSR could be a covariate-free objective for tuning Kilosort (KS4 vs KS2.5).

Labelling convention. "Cited Findings" contain only published or code-verified claims. My own derivations are marked **[Own derivation]** and my simulations **[Own simulation]**. Every simulation uses a faithful, vectorised re-implementation of the reference code's logic (see Q1). Scripts and raw outputs are in `C:/Users/richa/AppData/Local/Temp/claude/C--code-github-Kilosort/4b7a3881-25bc-4dc5-a0fb-f747ed8b8103/scratchpad/msr_theory/`:
- `msr.py`: the implementation and spike generators
- `e1_poisson.py`: Poisson processes
- `e2_processes.py`, `e2_out.txt`: renewal, bursty and modulated processes
- `e4_errors.py`, `e4_out.txt`: error types
- `e6_paired.py`, `e6_out.txt`: paired deltas
- `e5_info.py`, `e5_out.txt`: informativeness
- `e7_duration.py`: duration and base-grid checks

Unless stated otherwise the simulations use a 20-minute recording, a 1 ms base grid and numpy/scipy only. The reference code itself was not executed; its logic was reproduced.

---

## Q1. Exact definitions: resolution H[s], relevance H[K], MSR; normalisation, bin range, estimator

### Takeaway
MSR is the area under the curve of relevance H[K] against resolution H[s], traced as the bin width varies over the whole range from the base resolution to half the recording. Both entropies are taken in base M (M is the unit's spike count), so both axes lie in [0, 1] and MSR is in "Mats²". MSR is a plug-in statistic of one spike train. It is deliberately not an entropy estimator of any underlying distribution, and the paper applies no bias correction or null subtraction.

### Cited Findings
- **Setup.** A neuron has spike times t1 < … < tM ≤ t_obs. Time is cut into T bins of width Δt, and k_s is the spike count in bin s. The resolution is H[s] = −Σ_s (k_s/M) log_M(k_s/M), where log_M(·) = log(·)/log M is the logarithm in base M ("Mats"). This is the information gained about a random spike's timing from knowing its bin. — [Cubero, Marsili & Roudi 2020, arXiv:1802.10354, Sec. 2, Eq. 1](https://arxiv.org/abs/1802.10354)
- **Relevance.** H[K] = −Σ_k (k m_k/M) log_M(k m_k/M), where m_k is the number of bins holding exactly k spikes, so k·m_k/M is the fraction of spikes in bins with count k. It is read as the information each spike carries about the neuron's "dynamic state" (its count class) at that resolution. — [Cubero et al. 2020, Eq. 2](https://arxiv.org/abs/1802.10354)
- **Endpoints.** For Δt below the value at which every bin holds 0 or 1 spikes, H[s] = 1 and H[K] = 0. For Δt so large that all spikes share one bin, H[s] = 0 and H[K] = 0. H[K] is non-zero only between these extremes. — [Cubero et al. 2020, Sec. 2](https://arxiv.org/abs/1802.10354)
- **Why H[s] is the x-axis.** The authors argue H[s] is an intrinsic resolution scale "fixed by the available data", unlike Δt, which refers to time scales that differ between neurons. MSR (R_t) is defined as the area under the H[K]-vs-H[s] curve. — [Cubero et al. 2020, Sec. 2](https://arxiv.org/abs/1802.10354)
- **Not an entropy estimator.** The authors stress that Eq. 1 is not an estimate of the entropy of an underlying distribution p_s. They note that the naive plug-in entropy is strongly biased (Treves & Panzeri 1995; Strong et al. 1998), which is why they do not interpret it that way. — [Cubero et al. 2020, footnote 1](https://arxiv.org/abs/1802.10354); the same stance in general form: Ĥ[s] is "a quantitative measure of the coding cost of the specific sample", not an estimate of H[s] — [Marsili & Roudi 2022, Physics Reports, arXiv:2202.00339, Sec. 2.3](https://arxiv.org/abs/2202.00339)
- **Reference implementation** (`relevance.py`, function `parallelized_total_relevance`), which settles the conventions the paper leaves implicit — [rcubero/MSR on GitHub](https://github.com/rcubero/MSR):
  - The input is a spike-count vector on a base grid of `total_time` points. The paper's data used 1 ms.
  - The bin counts T are `unique(int(logspace(0.4, round(log10(0.99·total_time), 2), 100)))`, plus T = `total_time` itself. That gives about 100 log-spaced bin counts, from 2 bins (about half the recording each) down to the base grid. The comment says it "has to be in log-space since the resolution increases exponentially as the number of time bins".
  - Bins are contiguous, near-equal "balanced baskets" of base-grid points. Bin width is therefore a fraction of the recording length, not a fixed time.
  - H[K] and H[s] are computed from (k, m_k) with log base M.
  - The points (H[s], H[K]) = (0, 0) and (1, 0) are appended, points are sorted by H[s], and the area is a trapezoid sum.
- **Base grid and binarisation.** Spikes were binned and binarised at 1 ms (also tried 0.5 ms) for the Boolean pair analysis. The main MSR analysis works from spike timestamps. — [Cubero et al. 2020, Sec. 5.7](https://arxiv.org/abs/1802.10354)
- **Theory behind H[K].** Relevance (the entropy of the frequency distribution) "provides an upper bound to the number of informative bits" about the generative process, while the entropy of the data measures resolution. Samples that maximise H[K] at fixed H[s] ("maximally informative samples") have m_k ∝ k^(−μ−1), where −μ is the slope of the H[K]–H[s] curve. Zipf's law (μ = 1) marks the optimal trade-off. — [Cubero, Jo, Marsili, Roudi & Song 2019, arXiv:1808.00249](https://arxiv.org/abs/1808.00249)
- **Regimes along the curve.** The slope marks the regimes: μ > 1 is the noisy regime and μ < 1 is the lossy-compression regime. The optimal bin size can be read off where H[K] + H[s] is maximal, which is where the slope is −1. — [Marsili & Roudi 2022, Secs. 3.5 and 3.7](https://arxiv.org/abs/2202.00339)
- **Sampling effects.** Haimovici & Marsili (2015) treat m_k as Poisson random variables to handle sampling effects and maximise E[Ĥ[k]]. The resulting maximal curve lies above the curve for random (uniform) samples, and the gap grows with N. — [Marsili & Roudi 2022, Sec. 3.5](https://arxiv.org/abs/2202.00339); [Cubero et al. 2019, Fig. 1 text](https://arxiv.org/abs/1808.00249)
- **The theory assumes i.i.d. samples.** The relevance framework is built for samples of N i.i.d. draws, where "the order of the data points is irrelevant". In MSR the labels are bin indices of spikes. — [Cubero et al. 2019, Sec. 1](https://arxiv.org/abs/1808.00249)

### Inferences
- **[Own derivation] Time-rescaling invariance.** The bins are fixed fractions of the recording, and both axes depend only on the multiset of bin counts. MSR is therefore invariant under any rescaling of time t → c·t, except at the finest scale, which is set by the 1 ms base grid. For a homogeneous Poisson process, which conditional on M is M uniform points, MSR is then a function of M alone. Firing rate and recording duration matter only through M. Confirmed: at M = 6000, the MSR is 0.2450 for 300 s at 20 Hz, 0.2441 for 1200 s at 5 Hz and 0.2449 for 7200 s at 0.83 Hz (SD about 0.002). **[Own simulation, e1]**
- **[Own derivation] Identity in the coarse-bin regime.** When every occupied bin has a distinct count (few, large bins), m_k = 1 for each count present, so H[K] = H[s] identically. The left end of every curve is therefore the diagonal H[K] = H[s] and carries no information about the neuron. This holds for Poisson data down to H[s] ≈ 0.26–0.3 at M = 6000. **[Own simulation, e1c]**
- **[Own derivation] Poisson regime at intermediate bins.** When counts are approximately Poisson with mean ν ≫ 1, H[K] ≈ ½·log(2πeν)/log M. This gives a near-linear branch H[K] ≈ (1 − H[s])/2 + c/log M, with c = ½·log(2πe) ≈ 1.42 nats. It agrees with the simulated curve near H[s] ≈ 0.75 (0.288 predicted vs 0.278 simulated). The c/log M term is one source of MSR's dependence on log M. The "linear H[s]–H[K] relation over an extended range" that the paper reports for interneurons (Fig. S9d) is what this Poisson-like branch predicts. — interneuron observation from [Cubero et al. 2020, Sec. 3](https://arxiv.org/abs/1802.10354)
- **What a tuner has to fix.** The conventions above (log base M, bins from 2 up to total_time/1 ms in about 100 log-spaced steps, a 1 ms base grid, trapezoid area, the appended endpoints) all change the number. Any comparison between KS4 and KS2.5 must use identical settings and identical recording durations.

### Gaps
- The published paper does not state the bin grid in the text; it comes only from the code. I did not find a published sensitivity analysis to the number of bins, the base-grid resolution or the bin alignment. My own check showed a negligible effect of a 0.1 ms vs 1 ms base grid (0.2792 vs 0.2797). **[Own simulation, e7]**
- The full text of Haimovici & Marsili 2015 (the Poisson-m_k expected-relevance calculation) was not retrieved. Its formulas for the expected H[K] of random samples are known here only through the review.

---

## Q2. MSR of reference processes; dependence on rate, spike count M and duration; sample-size bias

### Takeaway
MSR is not zero for noise. A homogeneous Poisson train scores about 0.23–0.27 Mats², falling slowly with log M. Published real neurons span only about 0.26–0.30, so the scale is dominated by a large M-dependent floor. Bursty or slowly modulated trains sit about 0.02–0.05 above Poisson, and regular (refractory or gamma-like) trains sit below it. The authors acknowledge the dependence on M and handle it only post hoc, by regressing MSR on log M. They do not subtract a null.

### Cited Findings
- **Negative dependence on M.** MSR falls with M in both synthetic and real data, a relation "partly associated with bursty-ness". For randomly generated events, MSR "can be explained by log M". — [Cubero et al. 2020, Sec. 2 and Text S1/Fig. S1c](https://arxiv.org/abs/1802.10354)
- **Residuals beyond log M.** In the mEC data, regressing MSR on log M leaves residuals that still correlate with spatial and head-direction (HD) information (Figs. S2b–d). Grid cell 41 has a higher firing rate and a significantly negative residual. — [Cubero et al. 2020, Sec. 3.1 and footnote 5](https://arxiv.org/abs/1802.10354)
- **Synthetic renewal tests.** Interspike intervals (ISIs) were drawn from a stretched exponential with shape u (u = 1 is Poisson, u < 1 is broad and bursty, u > 1 is regular), giving series of 100,000 time units. MSR increased with the Goh–Barabási burstiness b and with the memory coefficient m, a short-range correlation between successive ISIs. MSR correlates positively with the local variation coefficient L_V, but L_V does not explain what MSR captures. — [Cubero et al. 2020, Text S1, Fig. S1, Figs. S2e–g](https://arxiv.org/abs/1802.10354)
- **Split-half reliability.** MSR from the first half of the data correlated with MSR from the second half at r = 0.942 (Fig. S2a). — [Cubero et al. 2020, Sec. 3 and Fig. S2a](https://arxiv.org/abs/1802.10354)
- **Published range and inclusion threshold.** Across 65 mEC neurons over about 20 minutes, MSR ranged from about 0.26 to 0.30 Mats², with log M between about 6 and 10. ADn/PoS units were included only with at least 100 spikes. — [Cubero et al. 2020, Figs. 2–3 and Sec. 5.1](https://arxiv.org/abs/1802.10354)
- **Covariate-driven synthetic data.** For idealised HD cells simulated as inhomogeneous Poisson processes on a random-walk trajectory, MSR and HD information were computed as functions of the tuning width (Fig. S1g). This was used to argue that MSR captures structure induced by external covariates. — [Cubero et al. 2020, Fig. S1g](https://arxiv.org/abs/1802.10354)

### Inferences
- **[Own simulation, e1] Homogeneous Poisson MSR against M** (20 reps each):

  | M | 120 | 600 | 1200 | 2400 | 6000 | 12000 | 24000 | 48000 |
  |---|---|---|---|---|---|---|---|---|
  | MSR | 0.267 | 0.258 | 0.254 | 0.250 | 0.245 | 0.241 | 0.238 | 0.234 |
  | SD | 0.008 | 0.005 | 0.003 | 0.003 | 0.002 | 0.001 | 0.002 | 0.001 |

  Against the published 0.26–0.30 range, a structureless unit with log M ≈ 6.4 already scores 0.258. Most of the published spread therefore sits within about 0.05 of the Poisson floor.
- **[Own simulation, e2] Reference processes at 5 Hz** (M ≈ 6000). The Poisson value for the same M is 0.245. "ISI-shuffled" is a surrogate that keeps the ISI distribution but destroys slow modulation and serial correlation.

  | Process | MSR | Excess over Poisson | ISI-shuffled surrogate |
  |---|---|---|---|
  | Gamma ISI, shape 0.3 (bursty) | 0.270 | +0.026 | 0.269 |
  | Gamma ISI, shape 0.5 | 0.260 | +0.015 | 0.260 |
  | Gamma ISI, shape 1 (Poisson) | 0.244 | 0.000 | 0.246 |
  | Gamma ISI, shape 2 | 0.227 | −0.018 | 0.227 |
  | Gamma ISI, shape 4 | 0.207 | −0.039 | 0.208 |
  | Gamma ISI, shape 10 | 0.181 | −0.064 | 0.181 |
  | Poisson + 2 ms dead time | 0.244 | −0.000 | 0.246 |
  | Poisson + 5 ms dead time | 0.243 | −0.002 | 0.243 |
  | Poisson + 20 ms dead time | 0.239 | −0.006 | 0.239 |
  | Poisson + 5 ms dead time, 30 Hz | 0.227 | −0.009 | 0.228 |
  | Bursts (Poisson onsets, 2–10 spikes, 4 ms apart) | 0.263–0.268 | +0.018 to +0.023 | — |
  | Inhomogeneous Poisson, OU log-rate, τ = 0.01 s, σ = 1 | 0.249 | +0.004 | 0.248 |
  | OU log-rate, τ = 0.1 s, σ = 1 | 0.266 | +0.021 | 0.258 |
  | OU log-rate, τ = 1 s, σ = 1 | 0.290 | +0.046 | 0.266 |
  | OU log-rate, τ = 10 s, σ = 1 | 0.294 | +0.050 | 0.266 |
  | OU log-rate, τ = 100 s, σ = 1 | 0.290 | +0.046 | 0.262 |
  | OU log-rate, σ = 0.5 | 0.25–0.28 | +0.001 to +0.034 | — |
  | Theta, 8 Hz, modulation depth 1.0 | 0.2455 | +0.001 | 0.244 |
  | Slow 20-minute sinusoid, depth 0.5 | 0.271 | +0.026 | 0.251 |
  | 10-minute on/off square wave, depth 0.9 | 0.255 | +0.009 | 0.270 |

  Readings from this table:
  - Refractoriness itself barely matters at the usual 1–2 ms. Regularity, meaning ISI shape > 1, strongly lowers MSR.
  - The ISI-shuffled surrogate separates two sources. Bursty and renewal effects survive shuffling; the contribution of slow rate modulation largely disappears.
  - Periodic single-timescale modulation (theta) at 5 Hz earns essentially nothing, because per-cycle counts are below 1 and the modulation averages out at coarser bins.
  - MSR rewards modulation that is broad across timescales and slower than about 0.1 s.
- **[Own simulation, e2d/e7] Rate and duration.** For a multi-timescale modulated unit (OU log-rate components at 0.3, 3 and 30 s, σ = 0.6, 2 ms dead time), MSR is almost independent of rate (0.282 at 1 Hz, 0.278 at 5 Hz, 0.277 at 20 Hz). The Poisson floor falls over the same range, so the excess over Poisson grows with M (+0.027, +0.032, +0.039). Over a fixed process, recording duration changes MSR only slightly (0.280 at 5 min, 0.279 at 20 min, 0.274 at 60 min), while the excess grows from +0.027 to +0.035.
- **[Own derivation] The log M "bias" is structural.** It comes from the base-M normalisation and the c/log M term of the Poisson branch (Q1). The base-M normalisation makes raw MSR smaller for high-M units even with identical structure. A natural correction is excess MSR, ΔMSR = MSR − E[MSR_Poisson | M], with the Poisson floor computed from a lookup table of simulations at the same duration and grid. Shuffle surrogates are a finer alternative: ISI-shuffling isolates slow-modulation structure, and a dead-time-matched gamma surrogate isolates pure renewal effects.
- **[Own simulation] Sampling SD.** The SD of MSR for a single unit is about 0.002–0.003 at M ≈ 6000 and about 0.005–0.008 at M of a few hundred. This is comparable to the error effects estimated in Q3. Paired comparisons (the same unit, two sorter outputs) reduce the noise a lot.

### Gaps
- I found no published closed-form expected MSR for Poisson or renewal processes, and no published bias correction for MSR. The paper uses only post-hoc regression on log M.
- I found no published study of MSR's dependence on recording duration for a fixed neuron, beyond the split-half test.

---

## Q3. How MSR responds to spike-sorting errors: (a) contamination, (b) thinning, (c) merges, (d) splits, (e) duplicates

### Takeaway
MSR is a weak and, in places, wrongly signed sorting objective:
- **Contamination** lowers MSR monotonically, but the effect is small: about −0.0026 for 10% Poisson contamination of a structured unit, about one across-seed SD. Contamination by another structured neuron, by bursts or by artefacts is barely penalised, and artefacts are often rewarded.
- **Random missed spikes** leave MSR almost unchanged, and raise it for weakly structured units.
- **Drift-dependent missed spikes and drift- or amplitude-based splits** increase MSR.
- **Merges** lower MSR only when the partner is less structured.
- **Duplicates** have a negligible or slightly positive effect.

MSR therefore does not reward cleanliness monotonically across error types.

### Cited Findings
- **Published merge-like test.** For spike trains built from the Boolean OR of pairs of neurons (at 1 ms), which is equivalent to merging two units with duplicates collapsed, the MSR of the OR or XOR train "is almost always lower (and never significantly higher)" than the MSR of the more relevant of the two neurons. The authors interpret this as individual neurons carrying non-redundant information. The cases with significant signal mostly involved an interneuron as one partner. — [Cubero et al. 2020, Sec. 5.7 and Fig. S8](https://arxiv.org/abs/1802.10354)
- **Split-half test.** MSR from the two halves of the recording correlated at 0.942. A time split leaves MSR rankings stable on real data. — [Cubero et al. 2020, Fig. S2a](https://arxiv.org/abs/1802.10354)
- **No published analysis of sorting errors.** The paper analyses spike-sorted tetrode data (Stensola et al. 2012; Peyrache et al. 2015) but does not analyse contamination, missed spikes, splits, merges or duplicates. I found no follow-up that does. — [Cubero et al. 2020](https://arxiv.org/abs/1802.10354); searches of Google-indexed literature through 2026 turned up only the original paper, its [bioRxiv preprint "Finding informative neurons in the brain using Multi-Scale Relevance"](https://www.biorxiv.org/content/10.1101/316190) and the [2022 review](https://arxiv.org/abs/2202.00339).

### Inferences
All results below are **[Own simulation]** unless marked otherwise.

**Setup.** The ground truth U has a log-normal multi-timescale rate (OU components at 0.3, 3 and 30 s, σ = 0.6), a mean of 5 Hz and a 2 ms dead time. It is clean MSR 0.279 against a Poisson floor of about 0.245. Errors are applied to the same U realisation and averaged over 8 seeds (e4); the paired deltas use 20 seeds (e6). "Excess" is MSR minus the Poisson MSR at the same M. The across-seed SD of clean MSR is about 0.0023. Reference partners used below:
- V: another modulated neuron, clean MSR 0.283
- I: an interneuron-like unit at 30 Hz with weak modulation, MSR 0.252
- Rg: a regular gamma-4 renewal unit, MSR 0.208
- B: a bursty unit, MSR 0.269

**(a) Contamination.** The table gives MSR for U plus extra spikes equal to a fraction f of its own count. The contaminated fraction is f/(1+f).

| Contaminant | f = 0.05 | f = 0.1 | f = 0.2 | f = 0.5 | f = 1.0 |
|---|---|---|---|---|---|
| Poisson noise | 0.2782 | 0.2770 | 0.2752 | 0.2696 | 0.2628 |
| Neuron V | 0.2783 | 0.2773 | 0.2754 | 0.2733 | 0.2735 |
| Interneuron I | 0.2781 | 0.2770 | 0.2752 | 0.2700 | 0.2642 |
| Regular Rg | 0.2783 | 0.2771 | 0.2750 | 0.2691 | 0.2601 |
| Bursty B | 0.2783 | 0.2772 | 0.2759 | 0.2738 | 0.2738 |

- Paired deltas: 10% Poisson contamination gives −0.0026 (SD 0.0008, t = −14). 33% gives −0.0102 (t = −70). On a near-Poisson ground truth the same contamination gives only −0.0009 and −0.0028.
- The direction is monotonic for every contaminant type, so contamination is always penalised. The size is small and depends on the contaminant: a structured or bursty contaminant at 33–50% is penalised by only about 0.005.
- [Own derivation] In bins short enough that the rate is roughly constant, a doubly stochastic count with excess variance σ²_s mixed with independent contaminant spikes has Fano factor F − 1 = (1 − c)(F_U − 1) + c(F_C − 1), where c is the contaminated fraction. Overdispersion, which is what lifts H[K] above the Poisson branch, is therefore replaced rather than merely diluted. The predicted change in excess is about −c·(excess_U − excess_C), which matches the simulations in order of magnitude (Poisson at c = 0.5: predicted −0.017, observed −0.013).
- Artefact contamination (e5): adding bursty lick artefacts (bouts of 7 Hz licks, 1–3 artefact spikes per lick) to a Poisson 5 Hz unit raised MSR from 0.245 to 0.280 (excess 0.000 to +0.038). Contamination by temporally structured artefacts is rewarded.

**(b) Thinning (missed spikes)**
- Random thinning leaves raw MSR essentially unchanged for a structured unit: 0.2794 at keep fraction p = 0.9, 0.2787 at p = 0.5, 0.2785 at p = 0.1. The paired delta at p = 0.5 is −0.0000 (t = −0.1).
- For a near-Poisson unit, MSR rises with thinning: +0.0045 at p = 0.5 (t = +8), because the Poisson floor rises as M falls.
- The excess over Poisson does fall with thinning (+0.034 at p = 0.9, +0.030 at p = 0.5, +0.022 at p = 0.1).
- [Own derivation] Why raw MSR is invariant. For rate modulation slower than the bin, the doubly stochastic part of the count variance at a matched mean count ν (that is, at matched H[s]) is invariant to independent thinning. Thinning is then equivalent to observing the same process with bins 1/p times longer, and MSR integrates over all bins. Only the finest scales and the 1/log M normalisation change.
- Drift-dependent detection loss rises monotonically with loss depth: keep probability in [0.7, 1] gives 0.2806, [0.4, 1] gives 0.2851 and [0.1, 1] gives 0.2910 (base 0.279). The paired delta for [0.4, 1] is +0.0056 (t = +11) on U and +0.024 (t = +62) on a Poisson unit. Amplitude-dependent loss that follows drift is rewarded.
- Burst-dependent loss (second spikes of bursts kept 40% of the time, others 95%) barely changes MSR (−0.001).

**(c) Merges** (full trains)

| Merge | MSR | Change from U (0.279) |
|---|---|---|
| U + V (both structured) | 0.2735 | −0.006 |
| U + B (bursty) | 0.2742 | −0.005 |
| U + Poisson | 0.2627 | −0.016 |
| U + Rg (regular) | 0.2598 | −0.019 |
| U + I (high-rate interneuron) | 0.2516 | −0.028, mostly the M effect |
| Rg + Rg (two regular units) | 0.2060 | −0.002 from Rg (0.208) |

- A merge is always at or below the better partner, consistent with the published OR result.
- It is barely below the better partner when the partners have similar structure, so merging two similar neurons is almost free under MSR.
- Merging two regular units is invisible, even though the merged train must show refractory-period violations. MSR is blind to the refractory-violation signature that contamination metrics use, because a 1–2 ms dead time changes MSR by ≤0.0005 (Q2).

**(d) Splits of U** (MSR of each fragment; U = 0.279)

| Split | Fragment A | Fragment B | Sum of fragments | Comment |
|---|---|---|---|---|
| Random 50/50 | 0.2804 | 0.2789 | 0.559 | Fragments ≈ parent: thinning invariance |
| Random 90/10 | 0.2792 | 0.2805 | 0.560 | Fragments ≈ parent |
| Drift-boundary, soft | 0.2860 | 0.2854 | 0.571 | Both fragments gain |
| Drift-boundary, hard | 0.2883 | 0.2878 | 0.576 | Paired +0.008 per fragment, t = +17; +0.024 on a Poisson parent |
| Time split at 10 min | 0.2821 | 0.2807 | 0.563 | Both fragments gain slightly |
| Burst second-spike split (ISI < 20 ms → A) | 0.2946 | 0.2739 | 0.568 | Fragment A gains strongly |

  In the drift-boundary splits, the assignment probability is a logistic function of a slow drift coordinate, as when a cluster boundary in feature space sweeps across a drifting waveform. The burst split mimics attenuated-amplitude burst spikes being split off.
- No split lowers fragment MSR. Splits that follow drift, time or burst state raise it.
- Any sum over units roughly doubles with a split. The mean over units is unchanged by random splits and increased by structured splits.

**(e) Duplicates**
- Duplicating a fraction d of spikes at 0.1–0.5 ms lag (1 ms base grid) leaves MSR flat: 0.2792 at d = 0.01, 0.2798 at d = 0.1, 0.2769 at d = 0.5. The paired delta at d = 0.1 is +0.0002 (not significant).
- On a Poisson unit, duplicates give +0.004 (t = +13), because they create k = 2 bins at the finest scales, which look like bursts.
- Near-duplicates at 1–3 ms lag behave the same (+0.0006).
- A 0.1 ms base grid gives the same result (+0.0003).
- MSR neither detects nor penalises duplicates. With a 1 ms base grid, most sub-millisecond duplicates fall into the same base bin, and in the reference code's count vector they add to k rather than being removed.

**Synthesis for the error types**
- MSR is monotonic in error rate for contamination (decreasing), drift-dependent thinning (increasing) and drift-aligned splitting (increasing).
- It is flat in error rate for random thinning and for duplicates.
- Its sign depends on the contaminant's own structure: Poisson or regular contaminants are penalised, structured or artefactual ones are rewarded or neutral.
- The size of the penalty for realistic contamination (5–20%) is about 0.001–0.004 Mats². That is smaller than the difference between units of different firing rate (the Poisson floor moves 0.02 between 1 and 40 Hz) and smaller than the spread between neurons (about 0.04).

### Gaps
- No published experimental or ground-truth (hybrid) evaluation of MSR under sorting errors exists that I could find. Every statement in this section beyond the Boolean OR result is my own simulation.
- My error models are idealised. Real KS4/KS2.5 errors are amplitude-, waveform- and drift-dependent jointly, and contamination from neighbouring units is correlated in time with the true unit (shared drive, near-synchronous firing). A test on hybrid ground truth using the user's own data would be the decisive check.

---

## Q4. What MSR actually captures: covariate information or non-Poisson temporal structure?

### Takeaway
MSR measures broad, multi-timescale variability of spike counts: overdispersion relative to Poisson, from slow rate modulation and bursting, minus regularity. In the published navigation data this correlates with spatial and HD information. The theory, however, defines relevance as information about the generative process, not about any behavioural variable, and the authors' own data show the implication is one-way. Uninformative structured trains (drift, artefacts, state changes) can score high, and informative trains with fast, weak or timing-based codes can score at the Poisson floor.

### Cited Findings
- **Design intent.** MSR "is designed to capture non-trivial structures in the spike train stemming from the variations in spike rates". It is expected to correlate with burstiness, memory and L_V. — [Cubero et al. 2020, Sec. 2](https://arxiv.org/abs/1802.10354)
- **mEC correlations** (65 neurons). MSR against spatial information: Pearson 0.738, Spearman 0.825. Against HD information: Pearson 0.460, Spearman 0.755. Spatial sparsity correlates (Pearson 0.868); grid score does not (Pearson 0.124, P = 0.32); mean vector length correlates weakly (0.298). — [Cubero et al. 2020, Figs. 2–3](https://arxiv.org/abs/1802.10354)
- **The implication is one-way:**
  - Neurons with high spatial or HD information had high MSR, "but the converse was not true".
  - Low-MSR neurons had low information, but not conversely (neurons 4 and 34).
  - Some high-MSR neurons (3 and 6) had low spatial and HD information. The authors suggest they "responded to other correlates".
  - Grid cells 41, 42 and 61, with significant grid scores, had low MSR.
  
  — [Cubero et al. 2020, Sec. 3.1](https://arxiv.org/abs/1802.10354). The review restates this: "High MSR neurons can have high or low mutual information." — [Marsili & Roudi 2022, Sec. 3.7](https://arxiv.org/abs/2202.00339)
- **Decoding.** The 20 highest-MSR neurons decoded position as well as the 20 most spatially informative neurons, and better than the ensemble of grid cells alone. — [Cubero et al. 2020, Discussion](https://arxiv.org/abs/1802.10354); [Marsili & Roudi 2022, Fig. 9](https://arxiv.org/abs/2202.00339)
- **Resampling test.** Resampling spikes as an inhomogeneous Poisson process from the spatial rate map lowered MSR. Adding HD modulation recovered part of it. The differential MSR (original minus resampled) was related to the information content. — [Cubero et al. 2020, Fig. 6 and Sec. 5.5](https://arxiv.org/abs/1802.10354)
- **Relevance is about the generative process, not a covariate.** The relevance-based notion of efficient representation "is independent of what the sample represents", in contrast with information-bottleneck definitions tied to an input–output relation. Statistical criticality "merely serves as a certificate of significance of the data", not an explanation. — [Marsili & Roudi 2022, Sec. 3.5.2](https://arxiv.org/abs/2202.00339)

### Inferences
All results below are **[Own simulation, e5]**, using 1D place cells on a random-walk trajectory over 20 minutes.

**MSR tracks bits per spike, not bits per second, and is nearly rate-independent at fixed tuning shape**

| Place-field width w (track units) | Peak / baseline rate | Information (bits/spike) | MSR |
|---|---|---|---|
| 0.03 | 5 / 0.1 Hz | 1.86 | 0.288 |
| 0.03 | 20 / 0.4 Hz | 1.86 | 0.291 |
| 0.03 | 40 / 2 Hz | 1.12 | 0.290 |
| 0.08 | 5 / 0.1 Hz to 40 / 2 Hz | 1.31 | 0.285–0.286 |
| 0.2 | 5 / 0.1 Hz to 40 / 2 Hz | 0.35 | 0.272 |
| Flat tuning | (any) | 0 | 0.238–0.251, at the Poisson floor |

Within this family, MSR tracks covariate information well.

**High MSR, uninformative**

| Unit | MSR | Excess over Poisson |
|---|---|---|
| Poisson 10 Hz with detection probability following a 15-minute drift cycle | 0.269 | +0.026 |
| Poisson 5 Hz + lick artefacts | 0.280 | +0.038 |
| Lick-artefact cluster alone | 0.256 | +0.010 |

The drift-modulated unit scores in the range of a moderately tuned place cell. The unit with lick artefacts scores in the range of a sharply tuned place cell.

**Informative, low MSR**

| Unit | MSR | Excess over Poisson |
|---|---|---|
| Fast binary covariate switching every 50 ms, rate 3 vs 7 Hz | 0.247 | +0.0025 |
| Same covariate, rate 30 vs 70 Hz (much more total information) | 0.241 | +0.007 |
| Theta phase code, constant rate, phase precessing with a covariate | 0.244 | −0.001 |
| Strict 1-second up/down state alternation (10 vs 0.5 Hz) | 0.247 | +0.001 |

Single-timescale periodic structure, fast weak rate codes and timing codes are essentially invisible to MSR.

**[Own derivation] Why.** MSR is a function only of the bin-count multisets across bin sizes. It is blind to the ordering and identity of bins, to phase, and to anything that does not broaden the count distribution. Within that, it favours count distributions that are broad (heavy-tailed) over a wide range of bin sizes. Slow, scale-free rate fluctuations and bursts do this regardless of cause. For a sorter, the main non-neural causes of slow scale-free fluctuation are:
- probe drift and amplitude changes crossing a detection or assignment threshold
- behavioural-state and movement artefacts (licking, chewing, grooming)
- electrical noise bursts

All of these raise MSR. Theta rhythmicity itself is not rewarded at typical rates; theta-burst firing is rewarded through its burstiness.

**Conclusion for "covariate-free".** MSR is covariate-free in that it uses no covariate. It is not tuning-agnostic evidence of informativeness. It is a measure of non-Poisson, multi-timescale count variability, which in well-sorted navigation data happens to co-vary with tuning.

### Gaps
- No published test of MSR on units known to be artefactual, drift-affected or noise clusters (for example, units labelled as noise in curated Neuropixels datasets).
- No published analysis of MSR in head-fixed task settings or in cortex with many simultaneously recorded units; the published data are tetrode recordings from navigation areas. My searches found no application to Neuropixels or calcium imaging up to November 2026.

---

## Q5. Is MSR comparable or sum-able across units for a population objective, and can an optimizer game it?

### Takeaway
MSR is a per-unit, scale-normalised quantity whose baseline depends on M, so it is not directly comparable across units of different rate without an M correction. It is not additive in any information-theoretic sense. Any population sum grows with the number of units, and structured splits raise even the mean. An optimizer maximising MSR-based totals would be rewarded for over-splitting, especially along drift, and for absorbing structured artefacts. It would be weakly penalised for Poisson-like contamination and indifferent to missed spikes and duplicates.

### Cited Findings
- **Neuron-level ranking only.** MSR was proposed for ranking and selecting single neurons "for their information content". Extension to assemblies is flagged as future work. — [Cubero et al. 2020, abstract and Discussion](https://arxiv.org/abs/1802.10354)
- **Pairwise combinations.** OR and XOR trains of neuron pairs "almost always" have lower MSR than the better neuron. Low-MSR neurons paired together stay low. — [Cubero et al. 2020, Fig. S8](https://arxiv.org/abs/1802.10354)
- **Cross-neuron comparison needs a log M adjustment.** The authors use residual MSR after linear regression on log M to show the information-related part. — [Cubero et al. 2020, Text S1, Fig. S2b–d](https://arxiv.org/abs/1802.10354)
- **Closest population-level construct.** In the relevance framework, comparing clustering algorithms in the under-sampling regime, H[s] + H[k] was the best predictor of the optimal algorithm. This is a different construct (a relevance over cluster labels, not over spike timing), but it is the nearest population-level analogue in the literature. — [Marsili & Roudi 2022, Sec. 3.4](https://arxiv.org/abs/2202.00339)

### Inferences
- **[Own simulation, e4] Sums and means under splitting.** Splitting U into two fragments raises the sum of MSR from 0.279 to 0.559–0.576. A Poisson noise cluster still contributes about 0.24–0.27 (Q2), so a sum rewards every extra cluster, including noise clusters. The mean over units is unchanged by random splits (0.2797) but rises under drift splits (0.288), time splits (0.281) and burst splits (0.284).
- **[Own simulation, e1/e4] Mean and median bias toward low-M units.** Low-M fragments have both a higher Poisson floor (0.267 at M = 120) and a higher variance. A per-unit mean is biased upward by small fragments unless MSR is M-corrected and a minimum M is imposed. The paper used at least 100 spikes.
- **Gaming routes** an optimizer could exploit (**[Own simulation]**, Q3/Q4):
  1. Split along drift or time. A KS4 vs KS2.5 drift-correction difference can produce exactly this: one sorter splits a drifting unit, the other tracks it.
  2. Absorb motion or licking artefacts into units.
  3. Lose spikes when amplitude dips (drift-dependent misses).
  4. Separate burst spikes of attenuated amplitude into their own cluster.

  None of these is penalised by MSR. Each raises fragment MSR by about +0.005 to +0.04, larger than the contamination penalty for realistic error rates.
- **[Own derivation] Usable forms, if MSR is used at all.** None of these protects against the gaming routes; MSR would have to be combined with an explicit contamination term such as refractory-period violations or a template-matching false-positive estimate.
  - Per-unit excess MSR (ΔMSR over the Poisson floor at the same M and duration).
  - An ISI-shuffled surrogate difference, which isolates slow-modulation structure but is then most susceptible to drift.
  - Within-unit paired comparisons of matched units between sorters, matched by spike-time overlap, where the paired SD is about 0.001.
- **[Own inference] Ranking against objective.** MSR is defensible as a covariate-free ranking of matched, well-isolated units by temporal-structure richness. It is not defensible as a sorter objective. It does not reward cleanliness monotonically across error types, it is confounded by spike count, and its largest positive responses come from drift and artefact structure, which are precisely what a sorter must not exploit. Rather than optimising it, a tuner could use MSR as a diagnostic alongside standard quality metrics. For example: does a parameter change raise a unit's ΔMSR only through slow, drift-locked rate changes, as checked against the motion estimate?

### Gaps
- No published population-level MSR objective, and no study of MSR in spike-sorter evaluation, tuning or benchmarking, was found up to November 2026.
- A related but unreviewed line of work defines resolution and relevance through mapping entropy for coarse-graining, which might inform a population construct; I did not read it. — ["Making sense of complex systems through resolution, relevance, and mapping entropy", arXiv:2203.00100; authors not verified here](https://arxiv.org/abs/2203.00100)
- The full texts of Marsili, Mastromatteo & Roudi 2013 (J Stat Mech), Haimovici & Marsili 2015 and Song, Marsili & Jo 2018 were not retrieved. Their content is represented here through the 2022 review and the 2019 paper, which summarise them.
