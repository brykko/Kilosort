# Multiscale relevance detects tuning but rewards contamination

**No: multiscale relevance (MSR) cannot serve as a covariate-free cleanliness metric or as an objective for tuning Kilosort.** On the user's own kobe sorts, the rate-corrected form (dMSR, MSR minus the Poisson value at the same spike count) does track heading tuning without being told about heading: Spearman **0.63–0.80** with the harness's per-unit S_i, and an AUC of **0.85–0.96** for separating tuned from untuned units. That makes it a reasonable behaviour-free *tuning screen*. It fails as a *cleanliness* measure. When spikes from another unit or Poisson spikes are added to 40 well-isolated, well-tuned real units, MSR and dMSR **rise** monotonically, by about +0.003 at 30% contamination, while S_i falls 25%. Thinning moves dMSR only through spike count, merges register only as a lost unit, and random splits and duplicated units *raise* the summed score. Across 51 scored sorts, summed dMSR ranks sorts like S (Spearman 0.88), but mostly because it counts units: the partial correlation drops to 0.55 once eligible-unit count is controlled. In the comparisons that matter it disagrees with the harness. It puts KS2.5 "standard" above KS4 default, and the fragmenting γ 3 configuration above H8. The synthetic simulations in the theory note predicted the opposite contamination sign: a small *fall*. A check run while writing this report shows that both observations hold, and that the sign depends on each unit's fine temporal structure, especially tight bursts. MSR therefore has no fixed sign under the most important sorting error. The better covariate-free path is not a new information score. It is constrained yield: reproducible units or spikes under a contamination budget, plus explicit duplicate and split checks. Any candidate information term should be admitted only after it passes the same perturbation battery that MSR failed.

Throughout, results are labelled by kind: **[kobe]** for measurements on the user's sorts, **[synthetic]** for simulations, **[published]** for the literature, and **[inference]** for the researchers' own reasoning. Where they disagree, [kobe] wins.

## MSR measures slow count variability above a large spike-count floor

MSR is the area under the curve of relevance H[K] against resolution H[s]. The curve is traced as the bin width sweeps from the base grid (1 ms) to half the recording, and both entropies are taken in base M, the unit's spike count ([Cubero, Marsili & Roudi 2020](https://arxiv.org/abs/1802.10354); [reference code](https://github.com/rcubero/MSR)). The authors call it "parameter-free". In practice the value depends on the base grid, the roughly 100 log-spaced partitions, the endpoint anchors and the recording length, so a KS4-versus-KS2.5 comparison has to fix all of them ([repo implementation](file:///C:/code/github/Kilosort/tuning/exploratory/msr/msr.py)).

### The noise floor is most of the published range [synthetic]

Both notes find the same structural problem: **a homogeneous Poisson train already scores 0.23–0.27**, falling with log M. In the published mEC data, real neurons span only about 0.26–0.30 ([Cubero et al. 2020](https://arxiv.org/abs/1802.10354)).

On 1200 s Poisson trains, MSR falls from **0.262 at 0.2 Hz to 0.234 at 50 Hz**. That 0.028 drop is about as large as the gain from strong tuning (0.02–0.04) ([synthetic.json](file:///C:/data/ks4_tuning/exploratory/msr/synthetic.json)). The theory note shows why. Bins are fixed fractions of the recording, so for a Poisson train MSR depends only on M, and the base-M normalisation builds a c/log M term into the curve ([e1_poisson.py](file:///C:/code/github/Kilosort/tuning/exploratory/msr/theory/e1_poisson.py)) [inference, confirmed by simulation].

### What raises MSR and what it misses [synthetic]

Tuning raises MSR, but it saturates early. For synthetic HD cells driven by the real kobe heading trace at 5 Hz, MSR plateaus at about 0.282 from κ ≈ 2. Over the same range the true heading information more than doubles, from 0.88 to 2.03 bits/spike.

MSR does not care where the modulation comes from:

- **Invisible:** 8 Hz theta modulation, a 2 ms dead time, a fast 50 ms binary rate code and a theta phase code all sit at the Poisson floor.
- **Rewarded:** a 60 s on/off square wave raises MSR (0.271 against 0.245). So do a Poisson unit with drift-cycled detection and a unit with lick artefacts. These score like moderately and sharply tuned place cells respectively ([e2_out.txt](file:///C:/data/ks4_tuning/exploratory/msr/theory/e2_out.txt); [e5_out.txt](file:///C:/data/ks4_tuning/exploratory/msr/theory/e5_out.txt)).

The theory note's summary holds: MSR uses no covariate, but it is not tuning-agnostic evidence of information. It measures broad, multi-timescale count overdispersion, which in well-sorted navigation data happens to co-vary with tuning [inference].

### The published evidence is thin [published]

The published support is the originators' own. In one 20-minute session of 65 tetrode units, MSR correlated with spatial information (Spearman 0.825) and with HD information (Spearman 0.755, but Pearson only 0.460). The relation runs one way. High information implied high MSR, but some high-MSR neurons carried little spatial or HD information, and three grid cells with significant grid scores scored low ([Cubero et al. 2020](https://arxiv.org/abs/1802.10354); [review](https://arxiv.org/abs/2202.00339)).

OpenAlex lists 24 citing works. None replicates MSR as a neuron-ranking tool outside the authors' circle ([OpenAlex](https://api.openalex.org/works?filter=cites:W2994877201)). None uses it for spike-sorting QC: the PASER toolbox ships MSR as an encoding analysis, not among its quality criteria ([PASER](https://www.biorxiv.org/content/10.1101/739409v1.full)). Using MSR as a sorter objective would therefore be novel and unvalidated. The perturbation results below show why no one should adopt it as one.

## On the kobe sorts dMSR tracks tuning (ρ ≈ 0.7) but not isolation

### Tuning: a good screen, a weak ranker [kobe]

The per-unit test covers four sorts and every eligible unit, with trains cleaned exactly as the harness cleans them:

| Sort (eligible units) | Raw MSR vs S_i | dMSR vs S_i | dMSR vs S_i, log rate partialled out | AUC, S_i > 0.5 vs < 0.02 |
|---|---|---|---|---|
| KS4 default s1 (777) | +0.37 | +0.63 | +0.68 | 0.85 |
| KS4 H8 s1 (1712) | +0.59 | +0.78 | +0.80 | 0.96 |
| KS2.5 standard r1 (899) | +0.43 | +0.72 | +0.71 | 0.96 |
| KS2.5 standard_low r1 (1030) | +0.48 | +0.75 | +0.73 | — |

Sources: [real_units.json](file:///C:/data/ks4_tuning/exploratory/msr/real_units.json); [real_units_*.csv](file:///C:/data/ks4_tuning/exploratory/msr/real_units_default_s1.csv).

Raw MSR is held down by its strong negative dependence on rate: Spearman with rate is −0.66 in the default sort. Correcting for spike count recovers correlations comparable to the published mEC values, and they hold within every firing-rate quartile.

As a *ranker*, though, dMSR is weak. Only **27–36% of the top decile by dMSR is also in the top decile by S_i**, against 10% by chance. And **dMSR is above zero for 98.8–99.5% of eligible units**, so even untuned units sit well off the Poisson floor. Two consequences follow:

- dMSR can tell "modulated" from "flat" without behaviour.
- It cannot pick the same best units as S. It has no null that brings an untuned unit to zero, which S has by construction ([evaluation_spec.md](file:///C:/code/github/Kilosort/tuning/evaluation_spec.md)).

### Isolation: the link runs through tuning [kobe]

Across units, MSR is weakly *lower* in more contaminated KS4 units. Against the "local" refractory estimate, Spearman is −0.47 to −0.50; against the "plain" estimate with log rate partialled out, −0.12 to −0.21. KS2.5 units show no association.

That pattern mirrors S_i's own association with contamination (−0.19 and −0.28). Partialling contamination out leaves the dMSR–S_i correlations almost unchanged (0.69–0.79). The researchers' reading is that the link runs through tuning, because contaminated units are mostly poorly tuned multi-unit activity [inference]. The perturbations below confirm this: contamination does not lower MSR causally.

### Hybrid ground truth: no signal, and a narrow test [kobe]

Hybrid ground truth adds no support. For the 40 inserted units the median change in MSR from true train to sorted train is **+0.0004 to −0.0001**. The only significant association between the change in dMSR and the false-positive fraction has the wrong sign: Spearman +0.34, p = 0.03, in the default sort ([hybrid_default_s1.csv](file:///C:/data/ks4_tuning/exploratory/msr/hybrid_default_s1.csv)).

This test has little range. The median false-positive fraction is about 1%, there are only 40 units, and no new hybrid data are being produced.

## Every sorting error moves MSR the wrong way or only by counting

### Single-unit errors on real units [kobe]

The decisive test perturbed 40 well-isolated (plain contamination ≤ 5%), top-quartile-S units per sort, with 3 draws, and rescored S_i through the harness's own `score.corrected_scores` ([perturb_msr.py](file:///C:/code/github/Kilosort/tuning/exploratory/msr/perturb_msr.py)). Results for KS4 default s1 (medians; "fraction" is added or removed spikes as a share of the unit's own count):

| Error | Fraction | Change in dMSR | Change in raw MSR | Change in S_i |
|---|---|---|---|---|
| Spikes from another unit | 10% / 30% | +0.0017 / +0.0034 (up in 95% of units) | +0.0012 / +0.0015 | −10% / −25% |
| Poisson spikes | 10% / 30% | +0.0019 / +0.0033 | +0.0015 / +0.0018 | −11% / −26% |
| Random thinning | 10% / 50% | −0.0006 / −0.0040 | −0.0001 / −0.0000 | −10% / −50% |
| Jittered in-unit copies | 30% | +0.0002 | −0.0019 | +30% |

Source: [perturb_default_s1.json](file:///C:/data/ks4_tuning/exploratory/msr/perturb_default_s1.json). H8 gives the same signs ([perturb_H8_s1.json](file:///C:/data/ks4_tuning/exploratory/msr/perturb_H8_s1.json)).

The thinning row is a pure spike-count artefact. Raw MSR does not move. dMSR falls only because the Poisson baseline rises as M falls.

The effects are also small: 30% contamination shifts a typical tuned unit's dMSR by about 7%, against −25% for S_i.

### Multi-unit errors [kobe]

Multi-unit errors behave like unit counting:

- **Merges.** A merge lowers summed dMSR by about 0.035, the same as dropping a unit. The merged unit's own dMSR falls below its better parent in only **43–65%** of merges, and in only 43% when the partners are similarly tuned.
- **Random splits.** These raise summed dMSR by +0.033 and never lower it. S moves by +0.006.

At the sort level (5% error rate, σ taken from five default replicates), the two scores diverge on exactly the errors where S works:

| Error | ΔS | Δ sum dMSR |
|---|---|---|
| Contamination | −3.9σ | **+0.5σ** |
| Split | ~0 | **+2.5σ** |
| Duplicate units, before cleaning | +35.1 bits/s (cleaning removes this) | +3.5σ |

The mean dMSR over units moves by at most 0.0004 under any error. A per-unit average is blind to all of them.

### The synthetic simulations predicted the opposite sign for contamination [synthetic]

In the theory note's simulations, contamination *lowered* MSR monotonically. The ground truth there was a multi-timescale Ornstein–Uhlenbeck (OU) log-rate unit at 5 Hz with a 2 ms dead time. Adding 10% Poisson spikes changed MSR by **−0.0026** (paired t = −14), and 33% by −0.0102. Every contaminant type gave the same direction ([e6_paired.py](file:///C:/code/github/Kilosort/tuning/exploratory/msr/theory/e6_paired.py); [e6_out.txt](file:///C:/data/ks4_tuning/exploratory/msr/theory/e6_out.txt); [e4_errors.py](file:///C:/code/github/Kilosort/tuning/exploratory/msr/theory/e4_errors.py)).

The theory note supported this with a derivation: independent contaminants *replace* rather than merely dilute overdispersion. The alternatives note made a related argument. Predictive information cannot rise under independent Poisson contamination, and it extended the "right sign" to MSR-type relevance by analogy ([covariate_free_alternatives.md](file:///C:/code/github/Kilosort/research_notes/Multiscale%20relevance%20as%20sorting%20objective/covariate_free_alternatives.md)) [inference].

The empirical note's own synthetic HD cell (5 Hz, κ 2) sat in between. Its MSR changed non-monotonically: +0.0009 at 10%, then −0.0013 at 50% ([synthetic.json](file:///C:/data/ks4_tuning/exploratory/msr/synthetic.json)).

### A check run for this report reconciles the two

The disagreement is real, and it is not an implementation difference. Both scripts add f·M uniform spikes. The repo implementation reproduces the theory note's fall on its OU unit: −0.0061 at 30%.

On the same 40 real targets, raw MSR rises (+0.0017 default, +0.0013 H8, up in 68% of units). Two surrogates show where the rise comes from:

- **ISI-shuffled trains** keep the interspike-interval distribution and destroy slow modulation. They rise *most*: +0.0026 and +0.0030.
- **Trains jittered by ±0.5 s** keep slow modulation and destroy sub-second structure. They rise only +0.0004.

Controlled synthetic processes then show that the sign depends on burst structure:

| Synthetic process | Change in raw MSR at +30% Poisson spikes |
|---|---|
| Poisson | −0.0026 |
| Gamma-0.3 renewal | −0.0037 |
| OU unit with partial burst companions | −0.0050 |
| Poisson-onset tight bursts (2–10 spikes, 4 ms apart) | **+0.0071** |
| OU-modulated tight bursts (2–6 spikes, 5 ms apart) | **+0.0035** |

The real targets are strongly bursty: a median **39–48% of ISIs are shorter than 20 ms**, and LV is 1.23–1.29. Within H8, the size of the rise correlates with burst fraction (Spearman 0.35, p = 0.03). Within default it does not (0.02), although it trends with LV (0.29, p = 0.07).

Scripts and output: [contam_mechanism_real.py](file:///C:/code/github/Kilosort/tuning/exploratory/msr/contam_mechanism_real.py), [contam_mechanism_synth.py](file:///C:/code/github/Kilosort/tuning/exploratory/msr/contam_mechanism_synth.py), [contam_mechanism.txt](file:///C:/data/ks4_tuning/exploratory/msr/contam_mechanism.txt).

The most defensible conclusion is that **the sign of MSR's contamination response depends on the unit's own temporal structure**. Trains dominated by tight bursts move up, and smooth rate-modulated trains move down. The real kobe units sit on the "up" side.

The explanation is only partial. The half-bursty synthetic still fell, and the jittered real trains still rose slightly. It is, however, enough to settle the practical question: a metric whose sign under contamination flips with cell type cannot certify cleanliness.

Two smaller points sharpen the comparison:

- **The two notes use "dMSR" for different things.** In the theory note it means a paired change in raw MSR; in the empirical note it means rate-corrected MSR. The rate-corrected version adds a further rise of about +0.0005 per 10% added spikes, because the Poisson baseline falls as M grows.
- **Rate correction alone cannot fix the sign.** The rise in *raw* MSR on real units is the real conflict.

### Other agreements and disagreements across the notes

| Topic | Synthetic, published or reasoned | Kobe measurement | Status |
|---|---|---|---|
| Poisson contamination | Falls: −0.0026 at 10% [synthetic]; "right sign" [inference] | Rises: raw +0.0015, dMSR +0.0019 at 10% | Conflict; sign depends on burst structure (check above) |
| Merges | OR-train almost always below the better neuron [published]; always at or below the better partner [synthetic] | Below the better parent in only 43–65% | Partial conflict; similar partners are near-free in both |
| Random thinning | Raw MSR flat [synthetic] | Raw MSR flat; dMSR falls via the baseline | Agree |
| Duplicates | Flat; +0.004 on a Poisson unit [synthetic] | In-unit copies flat; duplicated *units* raise the sum | Agree |
| Splits | Sum doubles; drift- or burst-aligned splits raise each fragment [synthetic] | Sum +0.033 per split, never down | Agree (drift splits untested on kobe) |
| Saturation with tuning | MSR flattens above about 1.3 bits/spike [synthetic, e5] | Plateau from κ ≈ 2 | Agree |
| "Splitting or dropping spikes raises per-unit MSR" | Inferred in the applications note | Random thinning leaves raw MSR flat | Note's inference not supported |

## Summed dMSR ranks sorts by counting units, and flips KS2.5 against KS4

### Sort-level correlations [kobe]

Across 51 scored sorts (5 default replicates, 6 KS2.5 sorts, 40 round-4 search runs), summed dMSR correlates with S at Spearman **0.884**. Eligible-unit count alone reaches 0.841.

With unit count partialled out, the picture changes:

| Summary | Spearman with S | Partial, unit count controlled |
|---|---|---|
| Summed dMSR | 0.884 | 0.550 |
| Raw summed MSR | — | **0.058** |
| Mean MSR | **−0.69** | — |
| Median MSR | **−0.70** | — |

Source: [sort_level.json](file:///C:/data/ks4_tuning/exploratory/msr/sort_level.json).

Means rank in the opposite order to S because larger sorts add weaker units. Per-unit dMSR falls from **0.033 in KS4 default to 0.027 in H8 and 0.025 in G2** (computed from [sort_level.csv](file:///C:/data/ks4_tuning/exploratory/msr/sort_level.csv)). Replicate noise is not the problem: summed dMSR varies 1.3% across default seeds, comparable to S's 1.4%.

### Where it disagrees with the harness [kobe]

The disagreements fall exactly where the harness has made decisions:

| Comparison | S | Summed dMSR | Eligible units |
|---|---|---|---|
| KS4 default | 1019.7 | 25.74 | 772 |
| KS2.5 standard | 989.0 (−2.1σ_S) | 27.58 (**+5.4 SD**) | 889 |
| KS4 H8 | 1412.6 | 46.01 | 1695 |
| KS4 G2 (γ 3) | 1403.8 | **46.37** | 1830 |
| KS4 X1 (downsampling 5) | 1419.4 | **47.80** | 1831 |

The harness chose H8 over G2 and X1 because their extra units are seed-dependent fragments: about 1004 reproduced units for H8, against about 957 and 998 ([round4_results.md](file:///C:/code/github/Kilosort/tuning/round4_results.md)). Summed dMSR prefers both fragmenting configurations. It also reverses the KS4-default-versus-KS2.5-standard comparison.

### Why: no null correction, no split neutrality [inference]

The researchers' explanation fits all of this. S is null-corrected, so an untuned unit contributes about zero, and it is split-neutral, because heading information is linear under random thinning. Summed dMSR has neither property. With about 99% of units positive, **every extra cluster adds about 0.03**, whether it is a neuron, a fragment or a duplicate.

The empirical note suggests a block-shuffle null that keeps each unit's own slow modulation. That might remove the unit-count effect, but it cannot repair the contamination sign.

This matters most for the KS4-versus-KS2.5 question. An MSR objective would favour whichever sorter emits more clusters, and KS4 at sensitive settings emits more clusters for reasons the reproducibility rule exists to discount.

## Covariate-free alternatives fail in predictable directions

None of the alternatives below has been measured on the kobe sorts. Their response signs come from derivation and reasoning in the alternatives note, and MSR's reversal on real bursty units is a warning that derived signs can fail on real data.

### Spike-train measures [inference]

The alternatives note sorts single-train measures into three families:

- **Randomness measures** (entropy rate, Lempel–Ziv complexity) *rise* with added Poisson spikes, so they point the wrong way.
- **Dispersion measures** (Fano and Allan factors, CV, LV) drift toward their Poisson value under contamination, thinning, merging and splitting alike. A Poisson-like clean neuron then looks the same as a contaminated one.
- **Structure measures** (predictive information, excess entropy, autocorrelation amplitude) have the right sign in theory for independent contaminants. They are data-hungry, though, and the kobe MSR result shows the theory can fail in practice.

Lempel–Ziv as an entropy-rate estimator comes from [Amigó et al. 2004](https://mlanthology.org/neco/2004/amigo2004neco-estimating).

### Population prediction [published for the metric; inference for its sorting behaviour]

The ready-made population-level metric is **co-smoothing bits/spike**: the cross-validated Poisson likelihood gain for held-out neurons predicted from held-in neurons ([Neural Latents Benchmark](https://arxiv.org/abs/2109.04463)). It is universal and needs no covariate.

As a sorting objective it has the most dangerous sign errors. Duplicates, same-neuron splits, cross-contamination from a unit that is itself in the population, shared artefacts and drift co-modulation all make units *more* predictable from their peers, so the score rises. Those are precisely the template-sharing and echo errors this harness has already audited.

SVCA-style spatial separation of the two halves supplies a published template for the exclusions this metric would need ([Stringer et al. 2019](https://mouseland.github.io/research/posts/spontaneous.html)): peers more than about 150 µm away, lags of at least 2 ms, a slow per-unit baseline and artefact masking. Even then it remains unvalidated [inference].

### Waveform and refractory purity measures [published]

Purity measures are covariate-free by construction and need no information term. Refractory-violation contamination estimators (Llobet, Hill) and amplitude- and feature-space isolation are the standard ones ([SpikeInterface ISI-violation docs](https://spikeinterface.readthedocs.io/en/0.101.0/modules/qualitymetrics/isi_violations.html)). Resampling stability offers a further truth-free check of each unit ([Barnett, Magland & Greengard 2016](https://arxiv.org/abs/1508.06936)).

Their failure mode is the mirror image of information metrics: they reward discarding ambiguous spikes. That is why the earlier report framed the objective as **information yield under a stated contamination budget**, compared as swept frontiers ([earlier report](file:///C:/code/github/Kilosort/reports/Spike%20sorting%20sensitivity%20versus%20quality.md)). Nothing in the MSR work changes that framing. MSR does not fill S's known blind spot (false spikes that leave tuning undistorted); on real units it inverts it.

### Ventura's caution [published]

Ventura's work adds a caution that applies to every such score. Sorting that ignores covariates yields biased tuning estimates, and correlation estimates are biased even more ([Ventura 2009](https://www.stat.cmu.edu/~vventura/6921.full.pdf); [Ventura & Gerkin 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3358902)). Any spike-train score computed after sorting inherits that bias, so these scores are better at ranking configurations than at measuring absolute quality.

## Could this be an objective? No, and here is what to do instead

**Direct answer: no.** MSR, in raw, rate-corrected, summed, mean or median form, should not be an objective for tuning KS4 or KS2.5, and it should not be used to estimate cleanliness:

- On the user's units it rises with contamination.
- It ignores thinning except through spike count.
- It detects merges only as lost units.
- It rewards splits and duplicate units.
- At the sort level it reproduces S mainly by counting clusters, and it reverses the harness's two most consequential judgements.

Its legitimate use is narrow: a behaviour-free, per-unit indicator of whether a unit's rate is modulated on slow timescales.

| Priority | Recommendation | Evidence | How in this harness |
|---|---|---|---|
| 1 | **Keep MSR out of the objective and out of quality gates.** | Contamination raises dMSR (+0.5σ at the sort level, while S falls −3.9σ). Splits +2.5σ. KS2.5 standard ranked +5.4 SD over default. [kobe] | No change to S or the resolution rule. Record the decision in `evaluation_spec.md` beside the existing blind-spot notes. |
| 2 | **Make the perturbation battery the admission test for any covariate-free term.** | MSR's sign reversed between synthetic and real units, so derived signs cannot be trusted. [kobe + this report's check] | Generalise `perturb_msr.py` and `perturb.py` into a reusable test. A candidate must fall with 5–30% contamination from another unit or from Poisson spikes in at least 90% of target units. It must not rise in sum under random splits or duplicated units before cleaning. It must fall under a merge by more than under a drop. Run it on default and H8 targets, and on bursty and non-bursty subsets separately. |
| 3 | **For behaviour-free comparisons, use constrained yield, not an information score.** | Purity terms are covariate-free and correctly signed. The seed-reproducibility rule already caught what MSR missed. [kobe; published; earlier report] | Score seed-reproduced units or spikes subject to a held-out, population-level refractory contamination budget plus CCG and duplicate checks. Compare KS4 and KS2.5 as swept frontiers. Use this on the MEC transfer excerpt and on any recording without good tracking. |
| 4 | **Use per-unit dMSR as a diagnostic only.** | AUC 0.85–0.96 for tuned against untuned units. About one third top-decile overlap with S. Drift and artefacts raise MSR in simulation. [kobe; synthetic] | Cross-check the untuned-units guard: list units with S_i < 0.02 but high dMSR, and inspect them for drift, artefacts or other covariates such as speed or position. Compare dMSR of units a configuration *gains* against shared units, and against the motion estimate. Never sum or average it across sorts with different unit counts. |
| 5 | **If a covariate-free information term is still wanted, prototype exclusion-controlled population gain, then put it through priority 2.** | Co-smoothing is the only universal published candidate. It is unrestrictedly gameable by duplicates, splits and cross-contamination. [published; inference] | Peers more than 150 µm away, lags of at least 2 ms, a 30–120 s per-unit rate baseline, masking of synchronous events, reported in bits/s. Expect it to fail on cross-contamination unless spatial exclusion is strict. |
| 6 | **If MSR is reported at all, fix its conventions.** | The value depends on grid, partitions, duration and M. The two notes use "dMSR" inconsistently. [synthetic; kobe] | Use a 1 ms grid, identical durations and the `msr.py` partitions. Require at least several hundred spikes per unit (the paper's floor was 100). Always use the Poisson-baseline-corrected form, and call paired changes "ΔMSR". Optionally test a block-shuffle null to remove the unit-count effect. |

### Open gaps

Several gaps remain:

- **Untested on kobe:** heading-locked artefacts, amplitude- or drift-dependent splits and thinning (the synthetic cases where MSR rose most), the MEC transfer excerpt, and amplitude-based isolation metrics.
- **Sort coverage:** round 1–3 sorts could not be included because their spike files were pruned.
- **Mechanism:** the burst explanation of the contamination sign is partial. A cleaner test would contaminate real units after removing burst companions only.

## Conclusion

The MSR episode is a case study in why a covariate-free information score is hard to make into a sorting objective. Any quantity that rewards "structure" in a spike train will also reward structure that the sorter creates: spikes borrowed from bursty neighbours, fragments, drift-locked dropout. And the sign of its response to contamination can depend on cell type rather than on cleanliness. Theory and simulation predicted the right sign for MSR. The user's real, bursty units gave the wrong one, and only the perturbation battery on real units revealed it. That battery is the most reusable product of this work. It turns "is this metric a valid objective?" into a cheap empirical test that any future candidate must pass.

The practical consequence is a division of labour rather than a single score. Heading information S carries the information term where tracking exists. Contamination budgets, duplicate checks and seed reproducibility carry cleanliness everywhere. Covariate-free spike-train statistics such as dMSR flag units for inspection but never vote on which sorter wins. For the KS4-versus-KS2.5 question this means that, without behaviour, the defensible comparison is a frontier of reproducible yield against estimated contamination. A higher summed relevance mostly means a sorter emitted more clusters.
