# Principled sensitivity-vs-purity thresholds in other detection-and-classification fields, as analogues for spike sorting (Kilosort KS4 vs KS2.5)

Scope note: 15 search/fetch calls. Items marked "(not fetched)" are well-known references cited from prior knowledge with stable DOI/arXiv links; their content was not re-checked in this session. Everything in "Inferences" is my reasoning, not a sourced claim.

## Q1. Calcium imaging cell detection and spike inference (CaImAn, Suite2p, spikefinder, CASCADE): how are thresholds chosen and justified?

### Takeaway
Calcium imaging chose thresholds by a mix of (a) fixed, hand-set defaults used across all datasets, (b) reporting F1 against a consensus of several human annotators, with human-vs-consensus F1 as the ceiling, and (c) an explicit, stated preference for precision over recall because false-positive cells do more harm downstream. Spike inference benchmarks avoided thresholds altogether by scoring continuous rate predictions (correlation at 40 ms bins), and CASCADE handled the operating point by matching the training ground truth to the target data's noise level rather than by tuning a threshold.

### Cited Findings
- CaImAn component evaluation rejects a candidate if the correlation between its spatial footprint and the average frame falls below θsp (example value 0.5) or its trace peak SNR falls below θSNR (example value 2). A CNN classifier (batch version, plus a stricter online one) also scores candidates — [Giovannucci et al. 2019 eLife](https://elifesciences.org/articles/38173)
- The headline CaImAn results use one parameter set for every dataset (no per-dataset tuning), which the authors present as evidence of generality. Per-dataset grid searches against ground truth were reported separately: four evaluation parameters for batch, and three for online (SNR threshold, CNN threshold, candidates per frame) — [Giovannucci et al. 2019](https://elifesciences.org/articles/38173)
- Scores against consensus labels: CaImAn batch with fixed parameters, F1 0.75 ± 0.03 (precision 0.80, recall 0.72). Online, F1 0.76 ± 0.05 (precision 0.82, recall 0.73). Per-dataset optimisation raised batch only to 0.774 and online to 0.78. Suite2p in the same comparison scored F1 0.55 ± 0.12 (range 0.41–0.75) — [Giovannucci et al. 2019](https://elifesciences.org/articles/38173)
- Ceiling from human agreement: 3–4 labellers per dataset. Individual labellers' F1 against the consensus averaged 0.83–0.87 (range 0.69–0.94), and 0.82 ± 0.06 under leave-one-out — [Giovannucci et al. 2019](https://elifesciences.org/articles/38173)
- Justification for the operating point: precision was higher than recall by design. The authors prefer high precision because false-positive traces can do more damage downstream than missing weak true cells, and they note that relaxing the thresholds gives more balanced results — [Giovannucci et al. 2019](https://elifesciences.org/articles/38173)
- Spikefinder used Pearson correlation between the true and inferred spike trains at 25 Hz (40 ms bins) as its main score. It also checked sensitivity to bin size (40, 83, 167 and 333 ms), split train and test by neuron, and warned about overfitting to training-set quirks. The best entry raised mean test correlation only from 0.36 to 0.44 — [Berens et al. 2018 PLoS Comput Biol](https://www.doi.org/10.1371/JOURNAL.PCBI.1006157)
- CASCADE trains on a ground-truth database (more than 35 h, 298 neurons) resampled to match the target recording's noise level ("noise-matched ground truth improves spike inference", Fig. 2), and it outputs calibrated absolute spike rates rather than thresholded events — [Rupprecht et al. 2021 Nat Neurosci](https://www.nature.com/articles/s41593-021-00895-5.pdf); [eScholarship copy](https://escholarship.org/content/qt9sw746d5/qt9sw746d5.pdf?t=rfdwrz)
- Suite2p's ROI curation uses a classifier on ROI statistics (not fetched) — [Pachitariu et al. 2017 bioRxiv](https://doi.org/10.1101/061507)

### Inferences
- The CaImAn recipe transfers directly to spike sorting: (1) score F1, precision and recall per unit against ground truth (hybrid or paired recordings); (2) report the gain from per-dataset tuning against fixed defaults (CaImAn's was only about 0.02 F1, a sign that defaults near the optimum are robust); (3) state the downstream cost asymmetry explicitly. Note that CaImAn's stated asymmetry (false positives are worse) is a judgement call, not something derived.
- The "human ceiling" idea maps onto spike sorting as agreement between independent sorters, or between reruns of one sorter (see Barnett et al. under Q5). If KS4 and KS2.5 disagree with each other about as much as either does with the hybrid ground truth, tuning finer than that is chasing noise.
- CASCADE's noise matching suggests that if the KS thresholds are tuned on hybrid data, the hybrid data's SNR distribution should match the target recordings, or the chosen operating point will not transfer.
- Spikefinder's choice of a threshold-free correlation metric is a reminder that the right figure of merit depends on the downstream use. For rate or tuning analyses, a per-unit firing-rate error may matter more than spike-level F1.

### Gaps
- I did not extract CaImAn's CNN probability cutoffs (in the methods section past what was read), nor Suite2p's default classifier probability cutoff.
- I found no calcium-imaging paper that derives its threshold from an explicit cost model. All the operating points found were set by judgement or by maximising F1.

## Q2. Astronomy, radar, SMLM, genomics/proteomics and particle physics: what principles set the threshold?

### Takeaway
Each field sets the threshold by fixing an error rate it can estimate, then taking whatever sensitivity that rate allows. Astronomy and genomics fix a false discovery rate (Benjamini–Hochberg, or decoy-based), radar fixes the false-alarm probability (Neyman–Pearson/CFAR), and ENCODE fixes a reproducibility rate (IDR). Only particle physics and SMLM collapse the trade-off into a single figure of merit: s/√b, the Asimov significance or Punzi's ε/(a/2+√B) in physics, and an "efficiency" that combines the Jaccard index with localisation RMSE in SMLM.

### Cited Findings
**Astronomy (FDR source detection; reliability from negative detections)**
- Miller et al. 2001 introduced Benjamini–Hochberg FDR control to astrophysical source detection as an alternative to fixed n-sigma cuts (not fetched) — [Miller et al. 2001 AJ 122:3492, arXiv:astro-ph/0107034](https://arxiv.org/abs/astro-ph/0107034)
- Hopkins et al. 2002 built FDR into the Sfind 2.0 radio source finder and compared it with Imsad and SExtractor. The specified bound on the fraction of false detections held even for highly correlated, realistic images. For real sources (groups of pixels) the bound was "somewhat less strict" but still reliable enough for a priori estimates of the false fraction — [Hopkins et al. 2002 AJ 123:1086](https://arxiv.org/abs/astro-ph/0110570)
- Benjamini–Hochberg procedure (not fetched) — [Benjamini & Hochberg 1995 JRSS-B](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x)
- Reliability from negative detections: if the noise is symmetric and real sources have positive flux, running the source finder on the negated cube gives N false detections for every P positive ones, so reliability R = (P − N)/P, equivalent to T/(T+F). It was validated on a test cube with real WSRT interferometer noise (including imaging artefacts) plus real WHISP sources, and R can be estimated as a function of source parameters (total flux, peak flux, voxel count) — [Serra, Jurek & Flöer 2012 PASA](https://arxiv.org/abs/1112.3162); [journal page](https://www.publish.csiro.au/as/AS11065)

**Radar (CFAR)**
- Radar CFAR detectors set the threshold adaptively from local noise or clutter to hold the false-alarm probability Pfa constant (Neyman–Pearson: fix Pfa, maximise detection probability). Ordered-statistic CFAR was introduced for multiple-target and clutter-edge situations, where cell-averaging CFAR's noise estimate is inflated by neighbouring targets (not fetched) — [Rohling 1983 IEEE TAES](https://doi.org/10.1109/TAES.1983.309350)

**Single-molecule localisation microscopy**
- The SMLM challenge ("Super-resolution fight club") scored software on localisation RMSE, on the Jaccard index (fraction of molecules correctly detected) and on a combined efficiency = 100 − √((100 − J)² + α²·RMSE²), designed to rank software by a single number that trades detection against localisation precision — [Sage et al. 2019 Nat Methods (PDF)](https://bigwww.epfl.ch/publications/sage1903.pdf); [Henriques lab copy](https://henriqueslab.org/publications/2019__sage__nature_methods.txt)
- A follow-up preprint describes the efficiency metric as tied to that detection/localisation trade-off and to the matching radius that decides whether a detection counts as correct, with α an empirical choice — [arXiv:2010.13423](https://pdf.arxiv.org/pdf/2010.13423)

**Genomics/proteomics**
- Target–decoy: search spectra against real ("target") plus reversed or shuffled ("decoy") sequence databases. The decoy hit count estimates false identifications above any score threshold, and the threshold is set at a target FDR (typically 1%) (not fetched) — [Elias & Gygi 2007 Nat Methods](https://doi.org/10.1038/nmeth1019)
- Percolator (not fetched) uses decoys as labelled negatives to train a semi-supervised rescoring classifier, which raised identifications at fixed FDR — [Käll et al. 2007 Nat Methods](https://doi.org/10.1038/nmeth1113)
- q-values / positive FDR (not fetched) — [Storey & Tibshirani 2003 PNAS](https://doi.org/10.1073/pnas.1530509100)
- IDR: fits a copula mixture model to the ranks of the same signals in two replicates, gives each signal a probability of being reproducible, and selects on the irreproducible discovery rate. It does not need a prespecified significance cutoff, which makes it a criterion for setting selection thresholds when a scoring system's threshold is hard to determine. It became part of the ENCODE/modENCODE data standards — [Li, Brown, Huang & Bickel 2011 Ann Appl Stat](https://arxiv.org/pdf/1110.4705)
- The IDR software's default soft threshold is 0.05 — [nboley/idr GitHub](https://github.com/nboley/idr)
- ENCODE ChIP-seq guidelines (not fetched) — [Landt et al. 2012 Genome Res](https://doi.org/10.1101/gr.136184.111)

**Particle physics**
- Punzi's figure of merit for optimising a selection when the signal strength is unknown, ε/(a/2 + √B), where ε is signal efficiency, B the expected background, and a the number of sigmas wanted (simplified form for a = b). It does not depend on an assumed signal cross-section, and it stops the optimiser driving the background to zero at the cost of signal yield, unlike s/√b — [Punzi 2003 PHYSTAT, arXiv:physics/0308063](https://www.arxiv.org/pdf/physics/0308063); [Punzi et al. multichannel extension, arXiv:2011.11770](https://ar5iv.labs.arxiv.org/html/2011.11770)
- A worked use with a = b = 4 (CLAS Λ-decay search) — [arXiv:1507.03859](https://arxiv.org/pdf/1507.03859)
- The Punzi FOM has since been used as a training loss for neural-network selections ("Punzi-loss") — [arXiv:2110.00810](https://arxiv.org/pdf/2110.00810)
- Asimov median discovery significance Z_A = √(2[(s+b)ln(1+s/b) − s]), which reduces to s/√b when s ≪ b (not fetched) — [Cowan, Cranmer, Gross & Vitells 2011 EPJC, arXiv:1007.1727](https://arxiv.org/abs/1007.1727)
- Look-elsewhere effect: the global significance after scanning many positions or hypotheses is much lower than the local one (not fetched) — [Gross & Vitells 2010, arXiv:1005.1891](https://arxiv.org/abs/1005.1891)
- Rationale for the 5σ convention (not fetched) — [Lyons 2013, arXiv:1310.1284](https://arxiv.org/abs/1310.1284)

### Inferences
- These fields fall into two families. Error-rate-controlled fields (astronomy FDR, proteomics, radar, ENCODE) pick the threshold that holds an estimable error rate at a target value and accept whatever sensitivity results. Figure-of-merit fields (physics, SMLM) collapse the trade-off into one scalar. The second family needs the downstream goal to be stated: physics wants to discover or exclude a signal, SMLM wants image resolution.
- Spike sorting most resembles proteomics: a detection stage (KS4 universal-template matching at Th_universal, KS2.5 Th[0]) is followed by a learned classification stage (KS4 Th_learned, KS2.5 Th[1], lam, AUCsplit). Proteomics controls FDR at the final classified-identification level, not at the raw detection level. By analogy, the KS detection threshold could be set loose, with error control applied per unit after clustering.
- Ordered-statistic CFAR's motivation (neighbouring targets inflate the noise estimate) parallels spike sorting's noise estimate being inflated by multi-unit "hash". Median/MAD estimators of the noise are the spike-sorting counterpart (see Q4).
- The look-elsewhere effect applies to template matching over about 10^7 time points × hundreds of templates: a per-sample threshold of 8–10 (in KS's projection units) is not a per-recording false-alarm rate. A decoy run (Q4) estimates the global rate empirically rather than analytically.
- The SMLM efficiency score is the closest analogue to a combined spike-sorting score: a detection term (Jaccard, i.e. SpikeForest-style accuracy) combined with a waveform/assignment-quality term, with a weight α that has to be chosen. It does not remove the arbitrariness; it makes the weight explicit.

### Gaps
- I could not fetch the full text of Hopkins et al. 2002 for quantitative FDR-vs-nσ comparisons, or Miller et al. 2001 for its worked examples.
- Astronomy's "completeness limit" conventions (for example, quoting the flux at which a survey is 90% or 50% complete, estimated by injecting artificial sources) are widely used, but I did not retrieve a citable primary source this session.
- SMLM detection-threshold conventions (for example, thresholds as multiples of background noise in ThunderSTORM/SMAP) were not retrieved, nor the challenge's matching radius and α value.

## Q3. Which figures of merit resolve the trade-off, and under what assumptions is each optimal?

### Takeaway
No figure of merit is optimal in general. Each is optimal for a stated downstream goal: s/√b and Asimov Z for discovering a known-shape signal on Poisson background, Punzi for an unknown signal strength, F1 for symmetric weighting of precision and recall (and it has odd properties), FDR at level q when the cost of a false positive is proportional to the size of the list reported, and expected utility whenever the costs can be stated. With calibrated scores, the F1-optimal threshold is half the maximum F1.

### Cited Findings
- With calibrated probabilities, the threshold that maximises F1 equals half the optimal F1. An uninformative classifier maximises F1 by predicting everything positive, which is undesirable when positives are rare — [Lipton, Elkan & Narayanaswamy 2014](https://arxiv.org/pdf/1402.1892)
- Punzi's ε/(a/2+√B) is optimal for "sensitivity": the smallest signal that would be discovered at a-sigma with power set by b, independent of the unknown signal normalisation. s/√b requires an assumed s and degenerates when B → 0 — [Punzi 2003](https://www.arxiv.org/pdf/physics/0308063); [B Factories review discussion via search](https://arxiv.org/pdf/1406.6311)
- Asimov Z_A is the median significance under the Poisson likelihood-ratio test. s/√b is its small-s/b limit and overstates significance when s is comparable to b (not fetched) — [Cowan et al. 2011](https://arxiv.org/abs/1007.1727)
- Benjamini–Hochberg FDR controls the expected false fraction among discoveries under independence or positive dependence (not fetched) — [Benjamini & Hochberg 1995](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x). It held empirically on correlated astronomical images — [Hopkins et al. 2002](https://arxiv.org/abs/astro-ph/0110570)
- PR and ROC curves: a curve dominates in ROC space if and only if it dominates in PR space, and interpolating PR curves linearly is invalid (not fetched) — [Davis & Goadrich 2006 ICML](https://doi.org/10.1145/1143844.1143874)
- AUC is incoherent as a single-number summary: it implicitly averages over misclassification-cost distributions that depend on the classifier itself. The H-measure was proposed instead (not fetched) — [Hand 2009 Mach Learn](https://doi.org/10.1007/s10994-009-5119-5)
- SpikeForest's spike-sorting accuracy is n_match / (n_match + n_miss + n_FP), a Jaccard-type score equivalent to F1/(2 − F1) (not fetched) — [Magland et al. 2020 eLife](https://elifesciences.org/articles/55167)

### Inferences
- Under Bayes decision theory, with costs c_FP and c_FN per spike and calibrated posteriors, the threshold is p* = c_FP/(c_FP + c_FN). F1 and the Jaccard/SpikeForest accuracy correspond to an implicit, data-dependent cost ratio (by Lipton et al., p* = F1*/2, so the implied c_FP/c_FN = F1*/(2 − F1*)). Maximising accuracy therefore weights false positives less as performance worsens. That is a hidden assumption worth stating when comparing KS4 and KS2.5 at their accuracy-optimal thresholds.
- Map to spike sorting at the unit level: s = a unit's true spikes recovered, b = contaminating spikes (from other neurons or noise). s/√(s+b) is the SNR of a spike count or rate estimate when contamination behaves like independent Poisson noise. But contamination from neighbouring neurons is correlated with stimulus and behaviour, so it biases rates rather than adding variance, which makes FDR-style control (contamination fraction below q) more defensible than s/√(s+b) for single-unit claims.
- For population decoding, where many slightly impure units help, an efficiency-weighted measure (recall-heavy, F-beta with beta > 1) is defensible. For single-neuron or connectivity (cross-correlogram) claims, a precision-controlled rule (FDR or contamination at a fixed q, such as a 10% refractory-violation contamination, then maximise yield) is defensible. This is the same split as physics' discovery vs measurement.
- Recommended framing for KS4 vs KS2.5: compare the curves (yield of units vs contamination, or recall vs precision, across the threshold sweep), not single points. If one sorter's curve dominates the other's, the choice of figure of merit does not matter (Davis & Goadrich dominance). Only when the curves cross does the downstream goal have to choose.

### Gaps
- I found no paper that formally derives a spike-sorting figure of merit from downstream inference goals. This looks like an open niche.

## Q4. Decoy and negative-control approaches: estimating false positives by running the detector on null data. Used in spike sorting? How could it apply to Kilosort?

### Takeaway
Decoys are the most transferable idea. Astronomy's negative-detection reliability R = (P − N)/P and proteomics' target–decoy FDR estimate the false-positive rate empirically without ground truth. Spike sorting has partial versions: MountainSort's "noise overlap" (random noise clips as a decoy cluster), stability under perturbation (Barnett et al.), and threshold setting from robust noise statistics (Quiroga 2004). I found no published use of whole-pipeline sign-flipped or time-reversed decoy sorting to estimate the false-positive rate. The key design constraint is that the decoy must be scored by a fixed scoring function; Kilosort learns its templates from the data, so a naive decoy run would learn decoy templates.

### Cited Findings
- Negative-detection reliability (astronomy): assumes noise is symmetric and sources are positive; R = (P − N)/P, estimable as a function of source properties — [Serra et al. 2012](https://arxiv.org/abs/1112.3162)
- Target–decoy (proteomics): reversed or shuffled databases, scored by the same search engine, estimate false identifications at each threshold (not fetched) — [Elias & Gygi 2007](https://doi.org/10.1038/nmeth1019); decoys used as training negatives — [Käll et al. 2007](https://doi.org/10.1038/nmeth1113)
- MountainSort noise overlap: builds a synthetic noise cluster from random voltage snippets of the recording, removes the projection onto the noise-snippet mean to correct sampling bias, and computes a nearest-neighbour isolation score between the noise cluster and the unit. Noise overlap = 1 − isolation. It was introduced by Chung et al. 2017 and adapted in SpikeInterface by Siegle and Kyu Hyun Lee — [SpikeInterface docs](https://spikeinterface.readthedocs.io/en/latest/modules/qualitymetrics/nearest_neighbor.html); [Chung et al. 2017 Neuron, eScholarship](https://escholarship.org/content/qt0cj4d194/qt0cj4d194.pdf?t=rulvz9)
- Barnett, Magland & Greengard: rerun the sorter two or more times under perturbations consistent with the data and measure stability, with no ground truth and no noise-model assumptions — [arXiv:1508.06936](https://arxiv.org/abs/1508.06936). A search-engine summary said this work adds surrogate spikes and re-sorts with the noise sign flipped, but the abstract I fetched does not mention sign flipping, so treat that as unverified.
- Hill, Mehta & Kleinfeld 2011 estimate false positives and false negatives from cluster overlap using Gaussian fits and refractory-period violations — [PMC3123734](https://pmc.ncbi.nlm.nih.gov/articles/PMC3123734/)
- SpyKING CIRCUS defines the false-positive rate on ground-truth recordings as FP / spikes extracted — [Yger et al. 2018 eLife](https://elifesciences.org/articles/34518)
- A review cautions that results on simulated noise may not generalise, because noise differs between recording systems — [arXiv:1809.01051](https://arxiv.org/pdf/1809.01051)
- Robust noise-based detection threshold: Thr = 4σ_n with σ_n = median(|x|)/0.6745, chosen so that spikes do not inflate the noise estimate (not fetched) — [Quiroga, Nadasdy & Ben-Shaul 2004 Neural Comput](https://doi.org/10.1162/089976604774201631)
- Kilosort4 method paper (not fetched for this question) — [Pachitariu et al. 2024 Nat Methods](https://doi.org/10.1038/s41592-024-02232-7)

### Inferences
- Candidate decoys for Kilosort, with their failure modes:
  1. **Sign-flipped data (−x)**, the Serra analogue. Extracellular somatic spikes are dominated by negative troughs, so detections on −x mostly come from noise, plus genuinely positive-going spikes (axonal, or near dendrites or return currents). Like Serra's symmetric-noise assumption, this gives an upper bound on the false-positive rate. Run it with templates frozen from the real sort (template matching / Th_learned stage only). Re-running the full pipeline on −x would learn positive-peaked templates from real positive spikes and is not a null.
  2. **Time-reversed data.** This preserves the noise power spectrum and spatial covariance but reverses the trough–repolarisation asymmetry, so frozen templates should match poorly. It is a cleaner null for the classification stage, but only if the templates are frozen.
  3. **Phase-randomised or spatiotemporally whitened Gaussian noise** with the measured channel covariance (Kilosort already whitens). This is a pure noise null, but it misses the non-Gaussian multi-unit "hash" that dominates real false positives.
  4. **Noise-only recordings**: a probe in saline, channels outside the brain, or post-mortem. This gives a true instrument null, but brain background activity is missing, so it underestimates the false-positive rate.
  5. **Random-snippet decoy clusters** (MountainSort noise overlap), already computable in SpikeInterface, as a per-unit decoy score.
- With frozen templates, the per-unit decoy-to-target ratio N_k/P_k, swept over Th_learned (KS4) or Th[1] (KS2.5), gives a Serra-style reliability curve R_k(Th). A threshold could then be chosen as the loosest one with median R ≥ 0.95, or the units ranked by decoy-estimated FDR. This mirrors proteomics' "1% FDR, then maximise identifications".
- A decoy measures noise false positives, not misassignment between neurons (merges or contamination by a neighbouring neuron), which is likely the dominant purity error for dense probes. Refractory-violation contamination estimates, or hybrid ground truth, are still needed for that part. A decoy and a refractory-period estimate are complementary: the decoy handles noise and the refractory period handles other neurons.
- Kilosort's "hybrid" ground truth (injecting known units into real data) is the counterpart of astronomy's completeness simulations. Together, injection (completeness/recall) and decoys (reliability/precision) give both axes of the curve without paired recordings.

### Gaps
- I found no published spike-sorting study that runs a full sign-flipped or time-reversed decoy to estimate false-positive rates. This is either unpublished practice or a genuine gap. The sign-flipping claim attributed to Barnett et al. could not be confirmed from the abstract.
- I could not access Chung et al. 2017's own definitions or acceptance thresholds (the PMC fetch was blocked by a CAPTCHA page). The commonly quoted curation thresholds (isolation > 0.95, noise overlap < 0.03, peak SNR > 1.5) are unverified here.

## Q5. Generic ML guidance on choosing an operating point when costs are unknown, and on multi-objective (Pareto) tuning

### Takeaway
When costs are unknown, the defensible practice is to report the whole PR or Pareto front and choose by an explicitly stated constraint or utility, not by AUC. Multi-objective HPO (Karl et al. 2023) formalises this: find the Pareto set, then let the domain expert choose. For KS tuning, this means sweeping (Th_universal, Th_learned) or (Th, lam, AUCsplit), plotting yield against contamination (or recall against precision), and choosing by a downstream-motivated rule.

### Cited Findings
- Multi-objective HPO overview: the goal is the Pareto-optimal set, so experts can choose a trade-off afterwards. It surveys evolutionary and Bayesian-optimisation strategies (for example, hypervolume-based ones) and gives applied examples with objectives such as prediction time, fairness and robustness — [Karl et al. 2023 ACM TELO, arXiv:2206.07438](https://arxiv.org/abs/2206.07438)
- A systematic survey of multi-objective HPO algorithms (2014–2020 literature) — [Morales-Hernández et al. 2023 Artif Intell Rev, arXiv:2111.13755](https://arxiv.deeppaper.ai/papers/2111.13755v3)
- F1-optimal threshold = F1*/2 for calibrated scores, and the degenerate predict-all-positive behaviour at low informativeness — [Lipton et al. 2014](https://arxiv.org/pdf/1402.1892)
- AUC implicitly uses classifier-dependent cost distributions; the H-measure fixes the cost prior (not fetched) — [Hand 2009](https://doi.org/10.1007/s10994-009-5119-5)
- PR/ROC dominance equivalence (not fetched) — [Davis & Goadrich 2006](https://doi.org/10.1145/1143844.1143874)
- IDR as a threshold rule when the scale of the scores cannot be calibrated: use agreement between replicates instead — [Li et al. 2011](https://arxiv.org/pdf/1110.4705)

### Inferences
- **A single quantity worth optimising?** Only once the downstream use is fixed. Two defensible defaults:
  - (a) **Constraint form (recommended).** Maximise the number of units, or total recovered spikes, subject to estimated contamination ≤ q per unit (refractory-based, for example q = 0.1) and decoy-estimated noise FDR ≤ q′ (for example 0.05). This is proteomics' and astronomy's FDR logic, it is robust to unknown costs, and it is what curators already do implicitly.
  - (b) **Scalar form.** The Jaccard/SpikeForest accuracy on hybrid ground truth. It is comparable across sorters but carries the hidden cost ratio described under Q3.
- **Defensible rule for the operating point.** Choose thresholds on the "knee" or plateau of the yield-vs-contamination Pareto front where the marginal decoy FDR of newly admitted spikes reaches q. This is the local-FDR ("lfdr") version of the rule, and it stops at the point where the next increment of sensitivity is mostly false.
- **An IDR analogue for spike sorting.** Split a recording into interleaved halves (or rerun with different seeds or batch orderings), match units across them, and keep units or thresholds at which matched spikes are reproducible. This gives a ground-truth-free threshold criterion that could be applied identically to KS4 and KS2.5.
- **For a KS4-vs-KS2.5 comparison.** Run both on identical hybrid and decoy data, plot both Pareto fronts, test for dominance, and only if the fronts cross compare at matched contamination (FDR-matched yield), not at each sorter's default thresholds.

### Gaps
- I found no published application of multi-objective Bayesian optimisation to spike-sorter hyperparameters, and no IDR-style replicate-reproducibility thresholding in spike sorting. Both appear to be open opportunities, though a more targeted search could find recent preprints (2025–2026) that I did not reach within the tool budget.
