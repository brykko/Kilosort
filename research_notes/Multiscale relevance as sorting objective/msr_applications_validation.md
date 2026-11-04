# Multiscale relevance (MSR): empirical applications, validation, critiques, and use for spike-sorting quality

Scope note: about 17 tool calls were used. The original paper was read in full text (PMC HTML plus arXiv PDF text, which I extracted locally to get the figure-embedded correlation values). The citation list came from OpenAlex, which shows 24 works citing the J Comput Neurosci paper as of 2026-11. The reference code (rcubero/MSR `relevance.py`) was downloaded and read. The Semantic Scholar API returned HTTP 429, so the citation list may be incomplete. Google Scholar was not queried directly.

## Q1. What the original study (Cubero, Marsili & Roudi 2020) showed

### Takeaway
On a single 20-minute mEC session (65 units from one rat), MSR correlated strongly with Skaggs spatial information (Spearman 0.83) and spatial sparsity (0.85). It correlated more weakly with head-direction (HD) information (Spearman 0.76 but Pearson only 0.46), only weakly with mean vector length (Spearman 0.25), and not at all with grid score. The top-20 MSR units decoded position about as well as the top-20 spatial-information units. The relation is one-way: high information implies high MSR, but high MSR does not imply high information. In the thalamic/PoS data (746 units, 31 sessions) the evidence is mostly qualitative, plus decoding on two sessions. This is the originators' own work only.

### Cited Findings
- **Publication.** Cubero RJ, Marsili M, Roudi Y. "Multiscale relevance and informative encoding in neuronal spike trains," J Comput Neurosci 48:85–102 (2020), doi 10.1007/s10827-020-00740-x. The preprint was arXiv 1802.10354 / bioRxiv 316190, titled "Finding informative neurons in the brain using Multi-Scale Relevance." — [Springer](https://link.springer.com/article/10.1007/s10827-020-00740-x); [bioRxiv](https://www.biorxiv.org/content/10.1101/316190v1); [arXiv](https://arxiv.org/pdf/1802.10354)
- **Definition.** Spike times are binned at width Δt, giving spike counts k_s per bin and M spikes in total.
  - Resolution is H[s] = −Σ_s (k_s/M) log_M(k_s/M).
  - Relevance is H[K] = −Σ_k (k·m_k/M) log_M(k·m_k/M), where m_k is the number of bins holding k spikes.
  - Both use base-M logarithms ("Mats"). MSR (R_t) is the area under the H[K]-vs-H[s] curve traced as Δt varies. The authors call it "parameter-free." — [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC7035307/)
- **Datasets.**
  - mEC: 65 units from one male Long-Evans rat (Rat 14147, Stensola et al. 2012) foraging in a 150×150 cm box for about 20 minutes. The units were 23 grid cells, 5 interneurons, 1 putative border cell and 36 unclassified.
  - ADn and PoS: 746 units from 6 mice and 31 sessions (Peyrache & Buzsáki 2015; Peyrache et al. 2015) in a 53×46 cm open field. Only awake epochs and units with at least 100 spikes were used. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7035307/)
- **Correlations in the mEC session (n = 65; Figs 2–3 of the arXiv version).**

  | Measure vs MSR | Pearson ρp | Spearman ρs | p |
  |---|---|---|---|
  | Skaggs spatial information (bits/spike, bias-corrected by subtracting a 1000-shuffle mean) | 0.738 | 0.825 | <0.001 |
  | HD information | 0.460 | 0.755 | <0.001 |
  | Spatial sparsity | 0.868 | 0.850 | <0.001 |
  | HD sparsity | 0.597 | 0.765 | <0.001 |
  | Grid score | 0.124 (p = 0.323) | 0.092 (p = 0.468) | not significant |
  | Mean vector length R | 0.298 (p = 0.016) | 0.247 (p = 0.047) | weak |

  — [arXiv 1802.10354](https://arxiv.org/pdf/1802.10354)
- **Range of values.** MSR spanned only about 0.26–0.30 Mats² across all 65 mEC units. The bins in Fig 3 held n = 6, 9, 11, 17 and 22 units. — [arXiv](https://arxiv.org/pdf/1802.10354)
- **Split-half reliability.** MSR from the first half of the session against MSR from the second half gave Pearson ρp = 0.942, p < 0.001 (Fig S2a). — [arXiv](https://arxiv.org/pdf/1802.10354)
- **Asymmetric relationship.** Neurons with high spatial or HD information had high MSR, "but the converse was not true." Neurons with low MSR had low spatial and HD information, but again not conversely (e.g. neurons 4 and 34).
  - Some high-MSR neurons (neurons 3 and 6) had relatively low spatial and HD information. The authors speculate that they respond to "other correlates." This is the paper's main claim that MSR flags units the tuning measures miss, and it is speculation, not a demonstration. — [arXiv](https://arxiv.org/pdf/1802.10354)
  - Grid cells 41, 42 and 61 had significant grid scores but low MSR and low spatial information. Grid cells 33 and 41 had equal grid scores but very different MSR. — [arXiv](https://arxiv.org/pdf/1802.10354)
- **Decoding in mEC.**
  - The top 20 MSR units ("RNs") and the top 20 spatial-information units ("INs") shared 14 units. The RNs decoded position as well as the INs, and better than a grid-cell-only ensemble.
  - The 6 non-overlapping RNs decoded better within 5 cm than the 6 non-overlapping INs.
  - The top-20 LV (local variation) units decoded much worse.
  - Decoding used a 20×20 grid of 7.5 cm cells and 20 ms bins. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7035307/)
- **HD decoding.**
  - Mouse 12, session 120806: 30 RNs against 30 HD INs, with 22 of the 30 INs also among the RNs. The RNs decoded HD as well as the INs and far better than 1000 random 30-unit subsets.
  - Mouse 28, session 140313: all 30 ADn RNs were HD INs, and 23 of 30 PoS RNs were HD INs.
  - HD decoding used 100 ms bins. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7035307/)
- **ADn/PoS evidence.** For the 746-unit dataset the main text reports no correlation coefficients. It says that "in all of the 6 mice … neurons having HD specific firing, i.e. high HD sparsity and high mean vector lengths, were RNs" (Fig S6). — [arXiv](https://arxiv.org/pdf/1802.10354)
- **Resampling control.** Spikes for neuron 47 were resampled as an inhomogeneous Poisson process using only the rate map. This lowered MSR even though spatial information was kept. Adding a multiplicative HD factor raised MSR almost back to the original value. The authors take this as evidence that MSR reflects joint tuning. — [arXiv](https://arxiv.org/pdf/1802.10354)
- **Intended use.** The authors propose MSR "as a pre-processing tool to impose a less stringent criteria" than mean vector length, sparsity or grid score, for high-throughput recordings where the encoded correlates are unknown. They list calcium imaging and fMRI as future directions. — [arXiv](https://arxiv.org/pdf/1802.10354)

### Inferences
- The whole mEC validation rests on one 20-minute session with 65 tetrode units. The thalamic validation is qualitative apart from decoding on two sessions. The effective sample size for "MSR tracks HD information" is small.
- The Pearson–Spearman gap for HD information (0.46 vs 0.76) points to a monotone but nonlinear and noisy relationship. MSR ranks units roughly by HD informativeness but is a poor linear proxy. For a user scoring by cross-validated HD information, this matters: MSR would agree on the coarse ranking but not on fine differences.
- MSR's dynamic range is very narrow (about 0.04 Mats² across 65 heterogeneous units). Small biases from spike count or contamination could therefore move a unit across much of the range.

### Gaps
- I found no per-unit correlation coefficients for MSR against HD information or mean vector length in the 746-unit ADn/PoS data. They may exist only in supplementary Fig S6 panels, which I could not read as numbers.
- No ROC-style or rank-agreement statistics (e.g. top-k overlap) are reported beyond the single 14/20 and 22/30 overlaps.

## Q2. Follow-up applications and independent replications

### Takeaway
I found no independent replication of MSR as a neuron-ranking tool on any neural dataset (hippocampus, PFC, songbird, retina, calcium imaging) up to November 2026. The one reuse under the name "MSR" is on natural images, co-authored by Marsili. The other citing works are mostly the originators' theory papers or loosely related work. One spike-sorting toolbox (PASER) packages an MSR implementation as an analysis option.

### Cited Findings
- **Citation count.** OpenAlex lists 24 works citing the J Comput Neurosci paper. None of them applies MSR to rank recorded neurons outside the original authors' circle. — [OpenAlex query, cites:W2994877201](https://api.openalex.org/works?filter=cites:W2994877201)
- **Natural images.** Lakhal, Darmon, Mastromatteo, Marsili & Benzaquen (Sci Rep 2023) redefine MSR for images, as robustness to compression across scales.
  - Natural images resemble critical random textures.
  - MSR was "more robust and informative of image content than classical methods such as power spectrum analysis."
  - It was used to calibrate colour mapping and denoising.
  - Marsili is a co-author, so this is not an independent validation. — [Sci Rep 2023](https://www.nature.com/articles/s41598-023-41714-0)
- **Theory papers on the relevance framework.** These are by the originators and contain no new neural validation. Examples:
  - Marsili & Roudi, "Quantifying relevance in learning and inference," Physics Reports 2022 (arXiv 2202.00339). It reviews relevance, maximally informative samples, and the resolution–relevance trade-off with Zipf's law at the optimum. The arXiv abstract does not describe neural-data applications.
  - Cubero et al. 2019, "Statistical criticality arises in most informative representations," J Stat Mech.
  - Song, Marsili & Jo, "Resolution and relevance trade-offs in deep learning," J Stat Mech 2018.
  - Marsili, "Simplicity science," Indian J Phys 2024.

  — [arXiv 2202.00339](https://arxiv.org/abs/2202.00339); [OpenAlex citing list](https://api.openalex.org/works?filter=cites:W2994877201)
- **Potestio group (Trento), independent.** This group builds on resolution and relevance for coarse-graining, not for ranking neurons:
  - Holtzman, Giulini & Potestio, Phys Rev E 106:044101 (2022): applied to toy spins, the stock market and a protein.
  - Aldrigo, Menichetti & Potestio, Phys Rev E 111:044315 (2025): Hopfield-model neurons, using the "mapping entropy optimization workflow" (MEOW) to choose informative subsets of nodes.
  - A 2026 J Chem Theory Comput paper on protein resolution.
  - None uses recorded spike trains. — [OpenAlex citing list](https://api.openalex.org/works?filter=cites:W2994877201)
- **Jangjoo (NTNU Kavli, acknowledging Roudi and Marsili), PLoS ONE 2026.** Uses minimally complex models on Neuropixels grid-cell data (Gardner et al.; 2 rats, 23 sessions) to find an "informative" population bin size, with a median of about 181 ms. This is a population time-resolution method from the same group's orbit, not MSR per unit. — [PLoS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0322347)
- **Other citers.** Other works cite the paper only in passing: Morales, Di Santo & Muñoz, PNAS 2023, on criticality; an Entropy 2020 paper on a time-varying entropy measure; a NeuroImage 2026 paper on EEG criticality. — [OpenAlex citing list](https://api.openalex.org/works?filter=cites:W2994877201)
- **A related covariate-free approach (not MSR).** Tolossa et al., eLife: classifiers on spike-train statistics alone (no behavioural covariates) recover the anatomical location of neurons in Allen Neuropixels data. This shows spike timing carries cell-identity structure without covariates. — [eLife](https://elifesciences.org/articles/101506)

### Inferences
- Six years after publication, MSR has not been taken up as a neuron-selection tool by other experimental groups. Its empirical support for neural data is still essentially the 2020 paper.
- I found no direct comparison of MSR with other covariate-free measures (multiscale Fano factor, entropy rate, Lempel–Ziv, temporal sparseness) in the literature. The only covariate-free comparison is internal to the original paper: MSR beat LV. LV correlated with spatial information at ρs = 0.446 and with HD information at ρs = 0.582, and the top-LV units decoded much worse than the top-MSR units. — [arXiv](https://arxiv.org/pdf/1802.10354)

### Gaps
- Google Scholar may list more citations, including theses, the Cubero SISSA/NTNU thesis, and conference abstracts. Semantic Scholar was rate-limited and Google Scholar was not queried.
- I could not confirm whether any calcium-imaging or songbird study has used MSR. Targeted searches returned none.

## Q3. Reported limitations: firing rate, recording length, theta, state, drift, sorting

### Takeaway
The originators themselves show that MSR decreases with log spike count (log M). On synthetic random data, MSR "can be explained by log M." Their defence is a residual analysis: MSR after regressing out log M still correlates with information. High-rate units (interneurons) systematically score low. Theta modulation, behavioural state, drift and sorting quality are not analysed anywhere I could find.

### Cited Findings
- **Spike-count dependence.**
  - Synthetic data: "We found that MSR decreased with log M … This result is indicative that MSR of randomly generated events can be explained by log M" (Fig S1c).
  - Real mEC data: "MSR decreased with the logarithm of the number of observed spikes" (Fig S2b).
  - The residuals of MSR after regressing on log M "still retained the dependence with spatial and HD information" (Figs S2c–d). The residual correlation values are not given in the text. — [arXiv](https://arxiv.org/pdf/1802.10354)
  - A footnote says grid cell 41 "has a higher firing rate, but also the residual MSR wrt to log M is significantly negative." — [arXiv](https://arxiv.org/pdf/1802.10354)
- **Burstiness and memory.** MSR rises with burstiness and with short-term ISI memory (Goh–Barabási coefficients) in synthetic data. It is positively correlated with LV. — [arXiv](https://arxiv.org/pdf/1802.10354)
- **Interneurons.** The fast-spiking interneurons (e.g. interneuron 8 at about 25–34 Hz) were among the five lowest-MSR units. Most interneurons show a linear H[s]–H[K] curve over an extended range. — [arXiv](https://arxiv.org/pdf/1802.10354)
- **Minimum spike count.** Units with fewer than 100 spikes were excluded from ADn/PoS. Boolean pair analyses needed at least 100 co-firing events. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7035307/)
- **Recording length.** The only check is split-half reliability (ρp = 0.942) on the 20-minute session. There is no analysis of how MSR scales with duration. — [arXiv](https://arxiv.org/pdf/1802.10354)
- **Theta and sorting quality.** I found no discussion of theta modulation or spike-sorting quality in the full text. Theta appears only in reference titles. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7035307/)

### Inferences (my reasoning, not from sources)
- Implications for a KS4 vs KS2.5 objective:
  - MSR is lower for units with more spikes. A sorter that splits units, or drops spikes, would tend to raise per-unit MSR. A sorter that merges units or adds contaminating spikes would lower it.
  - Spike count therefore confounds a comparison across sorters, unless MSR is residualised on log M, or each unit is compared with spike-count-matched surrogates (e.g. a rate-matched inhomogeneous Poisson or ISI-shuffled control, as in the paper's differential MSR).
- Contamination by background spikes that look like a homogeneous Poisson process should flatten the H[s]–H[K] curve toward the interneuron-like linear shape and reduce MSR. In principle that makes MSR sensitive to contamination. This is untested.
- Bursting, theta rhythmicity and state changes (immobility, sleep, sharp-wave ripples) all add multiscale variability that MSR rewards whether or not it relates to heading. Drift-induced amplitude loss, which makes a unit's spikes appear and disappear over time, also creates slow-timescale variance. Drift-related unit dropout could therefore inflate MSR, which would reward a sorting failure.
- The narrow range (0.26–0.30) makes these biases large relative to the signal.

### Gaps
- I found no published study that tests MSR specifically against firing-rate-matched nulls across many units, or against theta modulation, behavioural state, or drift.

## Q4. Use of MSR or multiscale entropy for spike-sorting QC, curation, or as an optimization target

### Takeaway
I found no paper that uses MSR, relevance, or a multiscale-entropy measure as a spike-sorting quality metric or a sorter-tuning objective. The closest case is the PASER toolbox, which ships an MSR implementation, but as an encoding-relevance analysis after quality control, not as a QC criterion.

### Cited Findings
- **PASER.** PASER (Brouns & Celikel, bioRxiv 739409) is a MATLAB extracellular-processing pipeline that uses KiloSort by default after benchmarking 9 sorters.
  - Its "spike analysis" step lists "an implementation of the multi-scale relevance to determine how relevant a neuron is for encoding," citing MSR.
  - Its unit-quality classification uses median amplitude above 4σ, a cap of at most 10% sub-threshold spikes, firing-rate stability over 10 s bins, refractory violations within 1.5 ms, and an F1 isolation score above 0.9 from drifting t-mixtures. MSR is not among these quality metrics. — [bioRxiv PASER](https://www.biorxiv.org/content/10.1101/739409v1.full); [OpenAlex citing list](https://api.openalex.org/works?filter=cites:W2994877201)
- **Covariate-free sorting validation in the literature.** This uses waveform or cluster-based or stability-based measures, not spike-train information measures. Examples:
  - Information-theoretic waveform-isolation measures (PMC3247202).
  - Barnett, Magland & Greengard's stability-under-perturbation validation (arXiv 1508.06936).
  - Isolation distance, L-ratio and silhouette as implemented in spikemetrics.

  — [PMC3247202](https://pmc.ncbi.nlm.nih.gov/articles/PMC3247202); [arXiv 1508.06936](https://ar5iv.arxiv.org/html/1508.06936); [spikemetrics](https://pypi.org/project/spikemetrics)
- **The originators' proposed role.** They frame MSR as a covariate-free pre-selection ("pre-processing") tool for choosing informative neurons, not as a sorting-quality metric. — [arXiv](https://arxiv.org/pdf/1802.10354)

### Inferences
- Using MSR as an objective for tuning Kilosort would be novel and unvalidated.
- Because of the spike-count and burst confounds, MSR could be gamed by over-splitting, or inflated by drift dropout, if it were optimized directly.
- A defensible use would be as a supplementary diagnostic:
  - rank-correlate per-unit MSR (residualised on log M) with cross-validated HD information within each sort;
  - check whether units gained by one sorter have MSR comparable to the shared units;
  - do not maximize MSR as such.

### Gaps
- I found no study that measures MSR against ground-truth sorting accuracy (hybrid or synthetic), whether to test that it falls with contamination or rises with splitting. This would have to be tested by the user.

## Q5. Public implementations and parameter choices

### Takeaway
The only public implementation I found is the authors' Python repository, github.com/rcubero/MSR (MIT licence; about 15 commits, 7 stars). PASER contains a MATLAB implementation. The Python code sweeps about 100 log-spaced numbers of contiguous bins, from about 2 up to 99% of the time-grid length, and integrates H[K] over H[s] with the trapezoid rule.

### Cited Findings
- **Repository.** rcubero/MSR is in Python, MIT-licensed, and contains `relevance.py` and `setup.py`. It also includes test scripts that reproduce the paper's figures and one Stensola et al. session (`Flekken_Data/`). The paper states the code is Python 3 and also compatible with Python 2. — [GitHub](https://github.com/rcubero/MSR); [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7035307/)
- **Code details from `relevance.py`** (downloaded from raw.githubusercontent.com/rcubero/MSR/master/relevance.py):
  - **Input.** `parallelized_total_relevance((total_time, spikes))` takes a binary or count spike vector on a fixed fine time grid of length total_time. Δt is set by how many contiguous "baskets" this vector is split into.
  - **Bin counts.** The number of partitions is `np.unique(np.logspace(0.4, round(log10(0.99*T), 2), 100).astype(int))`, plus `spikes.size` (the finest resolution). That is roughly 100 log-spaced bin counts, from 2 bins up to about 0.99·T bins. A code comment says log spacing is needed "since the resolution increases exponentially."
  - **Bin construction.** Bins are contiguous, balanced baskets; their sizes differ by at most one sample.
  - **Entropies.** H[K] and H[s] use log2 normalised by log2(M), so base M. The points (H[s], H[K]) = (0, 0) and (1, 0) are appended before integration, the curve is sorted by H[s], and the area comes from the trapezoid rule.
  - **No further normalisation.** The code applies no normalisation by firing rate or recording length.
  - **Parallelism.** The computation runs in parallel with `multiprocessing.Pool`. — [GitHub repo](https://github.com/rcubero/MSR)
- **PASER.** PASER (MATLAB) includes "an implementation of the multi-scale relevance." Its parameters were not visible. — [bioRxiv PASER](https://www.biorxiv.org/content/10.1101/739409v1.full)

### Inferences
- The method is "parameter-free" only given the choice of the fine time grid (the sampling resolution of the input spike vector) and the log-spaced sweep. The MSR value depends on T, through the sweep limits, and on M.
- The reference implementation loops over 100 partitions with Python list comprehensions. For long recordings at 1 ms resolution (e.g. 2 h = 7.2 million samples), a vectorised reimplementation (np.add.reduceat on spike times) would be needed.

### Gaps
- I did not confirm which fine time resolution the paper used for the input vector (e.g. 1 ms, or the position sampling rate).
- I could not verify the contents of the `MSR/` subfolder, or whether PASER's MATLAB port matches the Python code.
