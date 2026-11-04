# Covariate-free (tuning-agnostic) measures of spike-train information or cleanliness as spike-sorting objectives: alternatives and complements to multiscale relevance (MSR)

Scope note: MSR itself (Cubero, Marsili & Roudi) is covered by another agent. These notes cover the alternatives. The context is the user's harness, which scores KS4 against KS2.5 with a per-unit, cross-validated heading-information score S (bits/s, summed over units) plus contamination checks.

Provenance labels:
- **[fetched]**: confirmed this session from the abstract or full text, or from a search-result summary of the primary source.
- **[prior knowledge, DOI not fetched]**: a standard reference cited from memory. The bibliographic details are believed correct but were not re-checked this session, so the report writer should treat them as lower confidence.

Everything under "Inferences" is my own reasoning, not a published result.

---

## 1. Self-predictability and temporal-structure measures (entropy rate, predictive information, multiscale entropy, Lempel-Ziv complexity, Fano and Allan factors, intrinsic timescales, CV/CV2/LV): what each captures and how it responds to contamination, thinning, merging and splitting

### Takeaway
No single-train temporal measure tells "clean and structured" apart from "contaminated" in a sign-consistent way.
- **Randomness measures** (entropy rate, Lempel-Ziv complexity) rise when Poisson-like noise is added. They point the wrong way for an objective.
- **Structure measures** (predictive information or excess entropy, which is MSR's conceptual neighbour, and the autocorrelation amplitude) fall under independent contamination and under merging. Of these, only predictive information has a clean theoretical bound.
- **Dispersion measures** (Fano and Allan factors, CV, LV) all drift toward their Poisson values of 1 under thinning, merging and contamination, so they measure "distance from Poisson", not purity. A naturally Poisson-like clean neuron then looks identical to a contaminated one.

### Cited findings
- **Entropy rate (direct method).** Strong et al. estimated spike-train entropy and information by counting binned "words" and extrapolating in word length and data size. The method needs a lot of data because the number of possible words grows exponentially with word length. — [Strong, Koberle, de Ruyter van Steveninck & Bialek 1998, Phys Rev Lett 80:197](https://doi.org/10.1103/PhysRevLett.80.197) [prior knowledge, DOI not fetched]
- **Lempel-Ziv (LZ-76) complexity** estimates, or at least lower-bounds, the entropy rate of binned spike trains without knowing the generating probability law. It converges fast in simulation and is competitive with the word-frequency (direct) method. The word-frequency method does better on short sequences, where undersampling is critical. — [Amigó, Szczepański, Wajnryb & Sanchez-Vives 2004, Neural Computation 16(4):717–736](https://mlanthology.org/neco/2004/amigo2004neco-estimating); [abstract](https://pesquisa.bvsalud.org/ses/resource/pt/mdl-15025827); [summary PDF](https://www.ippt.pan.pl/attachments/ewaluacja/informatyka-techniczna-i-telekomunikacja/Szczepanski_Estim_Entropy_Neural_Com_2004_MIT_111cyt_SCOPUS.pdf) [fetched]
- **Predictive information.** I_pred(T) is the mutual information between past and future windows of length T, equivalently the subextensive part of the entropy. It is finite for processes with finite-range correlations. It grows as (K/2) log T for processes described by K parameters and faster for more complex ones. The authors propose it as the unique measure of the complexity of a time series, as opposed to its randomness. — [Bialek, Nemenman & Tishby 2001, Neural Computation 13:2409–2463](https://doi.org/10.1162/089976601753195969) [prior knowledge, DOI not fetched]
- **Multiscale entropy (MSE)** computes sample entropy on coarse-grained versions of a signal across scales. White noise loses entropy as the scale grows, while 1/f-type (long-range correlated) signals keep it. This separates "random" from "complex" signals. The method was developed for heartbeat series, not spike trains. — [Costa, Goldberger & Peng 2002, Phys Rev Lett 89:068102](https://doi.org/10.1103/PhysRevLett.89.068102) [prior knowledge, DOI not fetched]
- **Fano factor and Allan factor across counting windows** reveal fractal (power-law) rate fluctuations in spike trains. The Allan factor (a normalised variance of successive count differences) is less biased by slow nonstationarity than the Fano factor. — Teich, Heneghan, Lowen, Ozaki & Kaplan 1997, J Opt Soc Am A 14:529 [prior knowledge, DOI not fetched; no URL verified]
- **Intrinsic timescales.** Murray et al. fitted an exponential decay to the spike-count autocorrelation of single units in the baseline (fixation) period. Timescales increased along the cortical hierarchy, from sensory to prefrontal areas. — [Murray et al. 2014, Nature Neuroscience 17:1661](https://doi.org/10.1038/nn.3862) [prior knowledge, DOI not fetched]
- **Timescale estimates are biased in finite data.** The sample autocorrelation of a finite series has a negative bias, so direct exponential fits underestimate timescales. The bias depends mainly on sample length, and also on the true autocorrelation at each lag. Zeraati et al. propose adaptive approximate Bayesian computation (abcTau), which returns posteriors over timescales usable for model selection. Their software includes a `check_expEstimates` function for checking the bias of exponential fits. — [Zeraati, Engel & Levina 2022, Nature Computational Science 2:193–204 (preprint)](https://www.biorxiv.org/content/10.1101/2020.08.11.245944v2.full.pdf); [repository record](https://repository.cshl.edu/id/eprint/40582/); [code](https://doi.org/10.5281/zenodo.5949117) [fetched via search summary]
- **CV2** compares adjacent ISIs, 2|ISI(i+1) − ISI(i)|/(ISI(i+1) + ISI(i)), so that slow rate changes do not inflate irregularity the way they inflate CV. — [Holt, Softky, Koch & Douglas 1996, J Neurophysiol 75:1806](https://doi.org/10.1152/jn.1996.75.5.1806) [prior knowledge, DOI not fetched]
- **LV (local variation)** is 3/(n−1) Σ ((I_i − I_{i+1})/(I_i + I_{i+1}))². It equals 1 for Poisson, is below 1 for regular firing and above 1 for bursty firing, and is robust to rate fluctuations. — [Shinomoto, Shima & Tanji 2003, Neural Computation 15:2823](https://doi.org/10.1162/089976603322518759) [prior knowledge, DOI not fetched]

### Inferences (my reasoning, not published results)

**Independent contamination.** Suppose a neuron X is contaminated by an independent Poisson stream N.
- The observed train is a deterministic function of the pair (X, N), so I_pred(X+N) ≤ I_pred(X) + I_pred(N) = I_pred(X), because a Poisson stream has I_pred = 0. Predictive information therefore cannot rise under independent Poisson contamination. This is the theoretical reason structure measures, and by extension MSR-type relevance, have the right sign.
- The same subadditivity argument gives I_pred(X+Y) ≤ I_pred(X) + I_pred(Y) for a merge of independent neurons. Summed over units, merging can only lose predictive information.
- The argument fails for dependent contaminants, such as a correlated neighbour or a duplicate.

**Superposition of independent streams (counts).** For independent processes, variances and means add, so F_merged = (λ1·F1 + λ2·F2)/(λ1 + λ2). The observed Fano factor is a rate-weighted average and moves toward the contaminant's value (1 for Poisson). The same holds for the Allan factor at each window. A bursty clean cell (F > 1) and a regular clean cell (F < 1) both drift toward 1 when contaminated, so neither "high F" nor "low F" is a purity signal.

**Thinning (missed spikes).** For random thinning with retention probability p, the count variance is p²Var + p(1−p)mean, so F' = pF + (1 − p), which also moves toward 1.
- Thinned renewal ISIs tend toward exponential (Rényi's thinning limit), so CV and LV move toward 1.
- The autocorrelation timescale of rate fluctuations is preserved, but its amplitude, normalised by the Poisson variance, shrinks roughly in proportion to p.
- Amplitude-dependent misses are not random. If late spikes in bursts are attenuated and missed, burst ISIs drop out, so LV and CV fall and the cell looks more regular.

**Entropy rate and Lempel-Ziv complexity** (bits/s) rise with any added spikes and are highest for Poisson noise at a given rate. As objectives they reward contamination and adding artefacts, so they should be used only as a normaliser, for example as the denominator in "structure = entropy of a rate-matched shuffle − entropy rate". That normalised form is essentially excess entropy or predictive information again.

**Refractory structure is the only part of the ISI distribution that is strongly diagnostic of purity**: an empty 0–1.5 ms bin. Every other ISI-shape statistic is confounded with cell type, because bursting cells have LV > 1 and fast-spiking cells are regular.

**Splitting one neuron into two random halves.** Each half is a thinned copy. Fano factor and LV move toward 1 in each half, and the halves' refractory dips survive only in their cross-correlogram, not their autocorrelograms. Summed predictive information under splitting can go either way: it is roughly neutral in the low-count (low-SNR) regime and penalised at high SNR because of concavity. I could not find a published result on this.

**Data hunger, roughly from most to least data-hungry:**
- direct entropy and predictive information, because I_pred grows only as log T and words need many samples;
- MSE at large scales;
- intrinsic-timescale fits, which carry the negative bias above;
- Fano and Allan factors at long windows, because the variance of a variance estimate is large;
- LZ complexity;
- LV, CV and CV2, which are stable with a few hundred ISIs.

### Gaps
- I found no paper that applies entropy rate, predictive information, MSE, LZ complexity or the Allan factor to judging spike-sorting quality, or that publishes their response to controlled contamination or splitting. The responses above are derived, not published.
- I could not verify a URL or DOI for the Teich/Lowen Allan-factor paper this session.

---

## 2. Population-based covariate-free information: peer prediction, population coupling, coupling GLMs, SVCA/cross-validated PCA, GPFA/LFADS and co-smoothing. Can "cross-validated log-likelihood gain from the population" be a universal objective, and what are its pitfalls?

### Takeaway
There is a ready-made, widely used covariate-free metric: **co-smoothing bits/spike** from the Neural Latents Benchmark. It is the cross-validated Poisson log-likelihood gain for held-out neurons predicted from held-in neurons. It does not need to know what any neuron is tuned to.

As a sorting objective, though, it has signs that make it gameable:
- It **rewards** duplicates and same-neuron splits, which predict each other near-perfectly.
- It rewards cross-contamination between units that are both in the population, because the contaminant's spikes become predictable from the contaminant's own unit.
- It rewards shared artefacts (synchronous noise events) and drift-induced co-modulation.

It penalises only *independent* noise and missed spikes. Population-level latent structure is also known to be robust to sorting quality, which limits its power to tell sorters apart. It needs spatial exclusion, lag exclusion and a slow-baseline control (below) before it could serve, and even then only as one term alongside purity constraints.

### Cited findings
- **Peer prediction.** Hippocampal pyramidal-cell spike times were predicted better from the animal's position *plus* simultaneously recorded peers' spikes than from position alone. Peer activity predicted best within a 10–30 ms window, with a median optimal timescale of 23 ms across cells. The baseline in this design includes a covariate (position); the gain from peers is what is covariate-free. — [Harris, Csicsvari, Hirase, Dragoi & Buzsáki 2003, Nature 424:552–556](https://www.nature.com/articles/nature01834); [RePEc record](https://ideas.repec.org/a/nat/nature/v424y2003i6948d10.1038_nature01834.html); [ISTA record](https://research-explorer.ista.ac.at/record/3526) [fetched via search summary; full methods not read]
- **Population coupling.** Coupling to the summed population rate varies continuously across neighbouring neurons, from "choristers" to "soloists". It is a stable cell property, largely independent of sensory tuning, and predicts the response to optogenetic drive of local activity and the in-vitro synaptic input from neighbours. The coupling values of n neurons explain a substantial part of the roughly n² pairwise correlations. The main data were 20–80 neurons in mouse V1, with similar diversity in primate V4 and rat auditory cortex. — [Okun et al. 2015, Nature 521:511–515](https://pmc.ncbi.nlm.nih.gov/articles/PMC4449271); [UCL record](https://discovery.ucl.ac.uk/id/eprint/1466596/) [fetched via search summary]
- **SVCA (shared variance component analysis).** Neurons are split into two *spatially separated* halves. The dimensions that maximally covary between halves are estimated on training timepoints, and their reliable variance is measured on held-out timepoints. The first component was about 97% reliable; the 50th was about 50% and the 512th about 9%. The code `SVCA.m` computes cross-validated reliable variance. — [Stringer, Pachitariu, Steinmetz, Reddy, Carandini & Harris 2019, Science (doi 10.1126/science.aav7893)](https://mouseland.github.io/research/posts/spontaneous.html); [code](https://www.github.com/MouseLand/stringer-pachitariu-et-al-2018a) [fetched via search summary]
- **Co-smoothing bits/spike (co-bps)** scores how well held-out neurons' rates are predicted from held-in neurons' spiking. It is the Poisson log-likelihood gain over a baseline that predicts each neuron's mean rate, in bits per spike. NLB frames "unsupervised evaluation" as a common framework across datasets. — [Pei et al. 2021, Neural Latents Benchmark '21, NeurIPS Datasets & Benchmarks (arXiv 2109.04463)](https://arxiv.org/abs/2109.04463); [proceedings PDF](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/979d472a84804b9f647bc185a877a8b5-Paper-round2.pdf); [definition as used in a later paper](https://arxiv.org/pdf/2309.06402) [fetched; exact formula from secondary papers]
- **Typical co-bps values on MC_Maze (20 ms bins):** about 0.21 for SLDS, 0.25 for GPFA and 0.36 for LFADS, against about 0.21 for simple spike smoothing. — [arXiv 2406.16749](https://arxiv.org/pdf/2406.16749) [fetched via search summary]
- **Models with near-optimal co-smoothing can contain extraneous latent dynamics**, so co-smoothing alone under-constrains the model. A few-shot co-smoothing variant has been proposed as a complement. — [PMC12779162, "When predict can also explain"](https://pmc.ncbi.nlm.nih.gov/articles/PMC12779162/) [fetched via search summary]
- **Coupling GLMs.** Cross-validated generalised linear models with spike-history and coupling filters predicted retinal ganglion cell spike trains better than uncoupled models. This is the standard model class for "predict a unit from its peers". — [Pillow et al. 2008, Nature 454:995](https://doi.org/10.1038/nature07140) [prior knowledge, DOI not fetched]
- **Latent-variable models.** GPFA (Gaussian-process factor analysis) is evaluated by leave-neuron-out prediction error. LFADS (latent factor analysis via dynamical systems) is a sequential autoencoder for single-trial dynamics. Both are covariate-free population models whose held-out likelihood could be the "gain". — [Yu et al. 2009, J Neurophysiol 102:614](https://doi.org/10.1152/jn.90941.2008); [Pandarinath et al. 2018, Nature Methods 15:805](https://doi.org/10.1038/s41592-018-0109-9) [prior knowledge, DOIs not fetched]
- **Population dynamics are robust to skipping spike sorting.** Threshold crossings with no sorting recover low-dimensional population dynamics about as well as sorted units do. — [Trautmann et al. 2019, Neuron 103:292](https://doi.org/10.1016/j.neuron.2019.05.003) [prior knowledge, DOI not fetched]
- **Sorting errors and correlations.** On synthetic ground truth with controlled correlations, current sorters miss or mislabel spikes during synchronous events, so measured synchrony or correlation can be a sorting artefact. — [Garcia, Buccino & Yger 2022, eNeuro 9(5) ENEURO.0105-22.2022](https://www.eneuro.org/content/9/5/ENEURO.0105-22.2022.full) [fetched via search summary]
- **Sorting errors and synchrony significance.** Modelling suggests false negatives reduce the significance of synchrony more than false positives do, so sorting errors mostly lead to *under*estimating synchrony. — [Pazienti & Grün 2006](https://pubmed.ncbi.nlm.nih.gov/16927209) [fetched via search summary]
- **Clustering-based sorting can create artefactual short-lag correlations** (for example by missing overlapping spikes). A model-based sorter that handles overlaps removes them. — [Pillow, Shlens, Chichilnisky & Simoncelli 2013, PLoS ONE 8:e62123](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0062123) [fetched via search listing; details from prior knowledge]

### Inferences (my reasoning, not published results)

**Sign of a population-LL-gain objective (bits/s or bits/spike gain over each unit's own baseline) under each error:**
- **Independent false spikes** (noise, distant multi-unit): the gain falls, because the added spikes are unpredictable. Right direction.
- **Missed spikes:** the bits/s gain falls. Right direction. Bits/spike can *rise* if the missed spikes were the less predictable ones, such as low-amplitude spikes during weakly coupled periods.
- **Cross-contamination from unit B into unit A, with B also in the population:** A's extra spikes are exactly predictable from B at zero lag, so the gain **rises**. Wrong direction. This is the most dangerous failure, because it is precisely the KS template-sharing and borrowed-spike pathology the user has audited (make_pc_features overwrites, same-template echoes).
- **Duplicates or double detections** (the 0.3–0.5 ms pile-up) predict each other near-perfectly at zero lag, so the gain rises strongly. Wrong direction.
- **Splitting one neuron:**
  - A *random* split gives two units with identical rate modulation, each predicting the other at the rate level, so the gain rises. The halves' mutual refractoriness (a cross-correlogram dip) is also predictive.
  - A *drift* split (the same neuron split across time) gives halves active in complementary epochs, which a model with slow latents predicts. The gain rises.
  - Either way, the direction is wrong.
- **Merging two neurons:** the merged unit's predictability is a blend. Merges also remove the very predictors that explained each partner. The net effect is ambiguous, probably a decrease, but it is not a reliable merge detector.
- **Shared artefacts** (licking, chewing, stimulation, reference noise picked up as synchronous spikes across many channels): artefact units are the most "population-coupled" units there are (extreme choristers), so the gain rises strongly. Wrong direction.
- **Drift:** coherent amplitude changes push many units below or above threshold together, creating shared slow rate fluctuations. The gain rises unless the baseline includes each unit's own slow rate.

**Mitigations, each borrowed from a published design:**
- **Spatial exclusion.** As in SVCA's spatially separated halves, predict unit i only from units whose peak channels are more than about 100–200 µm away. This removes duplicates, splits, template-sharing echoes and most cross-contamination. It also removes genuine local coupling, which is the cost.
- **Lag exclusion.** Use coupling filters that start at 1–2 ms or more, or bins of 10–30 ms (the Harris timescale) with a zero-lag guard. Zero-lag duplicate effects then cannot be exploited. Coarse bins alone do not fix duplicates, since a duplicate is still in the same bin.
- **Slow-baseline control.** Measure the gain over a baseline of each unit's own smoothed rate (for example 30–120 s), not its global mean, so drift and session-level co-fading are not counted as information.
- **Artefact control.** Regress out or mask global synchronous events (large fractions of channels crossing threshold at once) before scoring.

**Summed over units, the objective remains gameable by unit proliferation.** More units means more predictors and more targets. Per-unit gains can be double-counted when units are redundant, which mirrors why split-neutrality matters for the user's existing score S.
- The heading-information rate of an inhomogeneous Poisson neuron, ∫λ(x) log2(λ(x)/λ̄) p(x) dx, is linear under thinning. A *random* split therefore leaves ΣS unchanged (split-neutral).
- By convexity, a merge of two differently tuned neurons gives I ≤ I1 + I2, so merges are penalised.
- The population-gain score has no such guarantee.

**Discriminating power is likely weak.** Trautmann et al. imply that latent-level population structure barely changes between threshold crossings and sorted units. A population objective will therefore barely separate KS4 from KS2.5 at the latent level, while its sign errors above concentrate exactly where sorters differ: duplicates, splits and template sharing.

**Data hunger.**
- A coupling GLM over N peers with L lag bases has about N·L parameters per unit. With 100+ units this needs strong regularisation or a low-rank or latent approach (GPFA/LFADS-style, or reduced-rank regression).
- Low-rate units (< 1 Hz) give very noisy held-out log-likelihoods. Over 20-minute excerpts, a 0.5 Hz unit has about 600 spikes, which is marginal for a stable bits/spike estimate with roughly 0.05 bits/spike resolution. That figure is an estimate; I found no published power analysis.
- Reporting in bits/s, which weights units by rate, is more stable than bits/spike for low-rate units.

**A cheaper covariate-free variant** is Okun-style population coupling: the correlation of a unit with the summed rate of the *spatially distant* population, at 10–30 ms bins.
- Its useful sorting signal is mostly diagnostic: units with extreme coupling relative to their neighbours are candidate artefacts or duplicates.
- It is not something to maximise, because coupling is a heterogeneous biological cell property.

### Gaps
- I found **no published work** that uses population held-out likelihood (co-bps or coupling-GLM gain) as an objective for *tuning a spike sorter* or ranking sorters. A targeted search found none either.
- I could not check Okun et al.'s methods on whether same-shank or nearby-channel neurons were excluded from the population rate. Full text not read.
- No published quantification of how much duplicates or splits inflate co-bps.

---

## 3. Waveform and feature-space cleanliness measures (isolation distance, L-ratio, silhouette, nearest-neighbour metrics, refractory contamination estimates), and how an information-type measure would combine with them

### Takeaway
The "purity" side is covariate-free by construction and well established. It is the natural constraint, or penalty, partner for any information objective. The combination the literature and the user's prior notes point to is to **maximise (summed, split-neutral) information subject to a contamination budget per unit**:
- contamination estimated from refractory violations (Hill/Llobet-type estimators);
- misses estimated from amplitude cutoff;
- separation from neighbours assessed by L-ratio or isolation distance and nearest-neighbour hit/miss rates.

The information term cannot replace these. Clusterless work shows that information objectives *prefer* keeping multi-unit spikes.

### Cited findings
- **Refractory-violation contamination estimate (Llobet et al.):** C = n_v·T / (2·N²·t_r), with n_v violating pairs (counted over *all* pairs, not just consecutive spikes), recording duration T, spike count N and refractory period t_r. The estimate is derived under two extreme scenarios, random contaminants versus contaminants correlated with the neuron, and the refractory period is adjusted for the system's minimum. — [SpikeInterface ISI-violation docs, citing Llobet, Wyngaard & Barbour 2022, bioRxiv 2022.02.08.479192](https://spikeinterface.readthedocs.io/en/0.101.0/modules/qualitymetrics/isi_violations.html); [docs source](https://spikeinterface.readthedocs.io/en/0.104.6/_sources/modules/metrics/qualitymetrics/isi_violations.rst.txt) [fetched via search summary; secondary description]
- **Hill, Mehta & Kleinfeld quality metrics.** They derived a false-positive rate from refractory-period violations and proposed waveform-space overlap measures to accompany sorted units. — [Hill, Mehta & Kleinfeld 2011, J Neurosci 31:8699](https://doi.org/10.1523/JNEUROSCI.0971-11.2011) [prior knowledge, DOI not fetched]
- **L-ratio and isolation distance** are Mahalanobis-distance-based measures of how well a cluster is separated from other spikes in feature space. — [Schmitzer-Torbert, Jackson, Henze, Harris & Redish 2005, Neuroscience 131:1](https://doi.org/10.1016/j.neuroscience.2004.09.066) [prior knowledge, DOI not fetched]
- **Resampling stability without ground truth.** Barnett, Magland & Greengard estimate per-unit confidence for a black-box sorter by re-running it on subsamples or perturbed data and measuring stability. This is a covariate-free sorting-quality framework. — [Barnett, Magland & Greengard 2016, arXiv 1508.06936 (J Neurosci Methods)](https://arxiv.org/abs/1508.06936) [fetched via search summary]
- **Consensus across sorters.** Estimates of false-positive, false-negative and misclassification probabilities from consensus across sorters correlated significantly with ground-truth errors. — [PLoS ONE 10.1371/journal.pone.0160494](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0160494) [fetched via search summary]
- **Restricting decoding to "good" units loses information.** Zhang et al. found that decoding only from units passing quality metrics lost information and decoded worse than their density-based clusterless decoder. That decoder models spike localisation and amplitude features with a mixture of Gaussians whose mixing proportions vary with behaviour, and it outperformed both thresholded multi-unit activity and sorted-unit decoding. — [Zhang, He et al. 2023, NeurIPS 36](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f499387f191d6be56e68966181095878-Abstract.html); [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.09.21.558869.full.pdf) [fetched via search summary]

### Inferences (my reasoning, not published results)

**Purity metrics and information metrics fail in complementary ways:**
- Purity metrics reward discarding ambiguous spikes. Aggressive thresholds give high isolation distance and low refractory violations but low yield.
- Information metrics, whether heading S, MSR or population gain, reward keeping spikes that carry *any* signal, including another neuron's.

Hence the constrained form "maximise Σ_units I subject to C_i ≤ c_max (Llobet) and amplitude-cutoff misses ≤ m_max". Alternatively, sweep thresholds and compare sorters as information-against-contamination frontiers. This matches the user's earlier "no cost-free optimum / swept frontiers" notes.

**Coverage of each error type:**

| Error type | Information side | Purity side |
|---|---|---|
| Independent false spikes | Heading S, predictive information and MSR (penalise) | Refractory estimate (detects) |
| Missed spikes | Bits/s measures (penalise) | Amplitude cutoff (detects, partially) |
| Cross-unit contamination and duplicates | Population gain does **not** penalise | Cross-correlogram, zero-lag and template-similarity checks (needed) |
| Splits | Must use a split-neutral information sum | Template similarity plus cross-correlogram refractory dip (merge candidates) |

I found no standard published split metric beyond these curation heuristics.

### Gaps
- I did not fetch the Llobet bioRxiv derivation itself; the formula comes from the SpikeInterface docs.
- Silhouette and nearest-neighbour metrics were not re-verified this session. Standard sources would be Rousseeuw 1987 for silhouette, Chung et al. 2017 Neuron (MountainSort) for nearest-neighbour hit/miss rates, and Siegle et al. 2021 Nature (Allen) for amplitude cutoff and presence ratio, all from prior knowledge.

---

## 4. Prior work on spike-sorting objectives that maximise information or likelihood without ground truth (tuning-informed sorting, ensemble sorting, clusterless likelihood, model-based sorting)

### Takeaway
The closest published line is **Ventura's** work.
- Sorting that ignores covariates gives *biased and inconsistent* tuning estimates.
- Sorting jointly with tuning (2009) or jointly across the ensemble (2012) removes that bias.
- Clusterless decoding (Kloosterman; Zhang et al. 2023) skips sorting and maximises a joint mark-and-covariate likelihood.

All of these use covariates or joint likelihoods. I found **no published "information-maximising" or "predictive-likelihood-maximising" spike sorter that is covariate-free**. Model-based generative likelihoods, such as Pillow et al. 2013 and the template-matching objective KS itself optimises, are covariate-free, but they score the voltage reconstruction, not the information in the spike trains.

### Cited findings
- **Ventura 2009 (PNAS):** the covariates that modulate tuning functions also carry information about spike identities. Ignoring tuning information during sorting makes tuning estimates biased and inconsistent unless classification is perfect; even the PSTH of an imperfectly isolated neuron is biased. The bias persists "even if neurons are not tuned" to the covariate (text of the paper). Sorting and tuning estimation done in parallel, with soft assignments, give unbiased tuning curves even from imperfectly sorted neurons. — [Ventura 2009, PNAS 106:6921, "Traditional waveform based spike sorting yields biased rate code estimates"](https://www.stat.cmu.edu/~vventura/6921.full.pdf) [fetched, full text read]
- **Ventura 2009 (Neural Computation), "Automatic spike sorting using tuning information":** an EM algorithm that combines waveform and tuning information, assuming independent Poisson neurons, tested on simulated data. — [Ventura 2009, Neural Computation (doi 10.1162/neco.2009.12-07-669)](https://www.stat.cmu.edu/~vventura/SStun.pdf); [record](https://portal.mardi4nfdi.de/wiki/Automatic_Spike_Sorting_Using_Tuning_Information) [fetched via search summary]
- **Ventura & Gerkin 2012 (PNAS), ensemble sorting:** sorting spikes jointly rather than one at a time gives unbiased estimates of neuronal correlation, even for poorly isolated cells. — [Ventura & Gerkin 2012, PNAS, "Accurately estimating neuronal correlation requires a new spike-sorting paradigm"](https://pmc.ncbi.nlm.nih.gov/articles/PMC3358902) [fetched via search summary]
- **Todorova et al. 2014 (J Neural Eng):** a full analysis of how sorting schemes affect brain-computer-interface decoding, plus decoding from a joint waveform-and-tuning model that skips explicit sorting. — [Todorova et al. proposal/paper](https://www.stat.cmu.edu/proposals/SoniaTodorova.pdf) [fetched via search summary]
- **Clusterless density decoding (Zhang et al. 2023):** see section 3. It maximises a behaviour-conditioned mixture likelihood over spike features, with no explicit sorting. — [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f499387f191d6be56e68966181095878-Abstract.html) [fetched via search summary]
- **Clusterless marked-point-process decoding of hippocampal position** outperformed decoding from sorted units. — [Kloosterman et al. 2014, J Neurophysiol 111:217](https://doi.org/10.1152/jn.01046.2012) [prior knowledge, DOI not fetched]
- **Model-based sorting (Pillow et al. 2013).** A generative model of the recording, with explicit handling of overlapping spikes, removes the correlation artefacts that clustering introduces. Its statistical formulation lets the authors test robustness to prior parameters, a ground-truth-free quality check. — [Pillow, Shlens, Chichilnisky & Simoncelli 2013, PLoS ONE](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0062123) [fetched via search listing]

### Inferences (my reasoning, not published results)
- Ventura's results argue that *any* objective scoring spike trains through a fixed, already-sorted pipeline inherits the sorting bias. A covariate-free information score will be biased by imperfect isolation in the same way the heading score S is. This favours using such scores to *compare* sorter configurations (relative ranking) over reading their absolute values.
- The Ventura & Gerkin result implies that correlation-based objectives, which include population-prediction gain, are especially sensitive to sorting choices. That cuts both ways: high discriminating power, but large exposure to the duplicate and contamination sign errors in section 2.

### Gaps
- No covariate-free information-maximising sorter was found.
- I did not find an "information-theoretic spike sorting" paper that optimises spike-train information, as opposed to entropy-based clustering criteria in feature space. The search was limited.

---

## 5. For each candidate: does it penalise false spikes, missed spikes, merges and splits in the right direction? Can an optimiser game it? How data-hungry is it per unit?

### Takeaway
Ranked by suitability as a *covariate-free complement* to S:
1. **Purity constraints** (refractory contamination, amplitude cutoff, cross-correlogram and template-similarity duplicate checks). Required in any case.
2. **Structure measures** (predictive information, excess entropy, MSR-type relevance, and the refractory-corrected autocorrelation amplitude). Right sign for independent contamination and merges, but data-hungry and with ambiguous split behaviour.
3. **Population held-out gain with spatial and zero-lag exclusion and a slow baseline.** Powerful and universal, but has the wrong sign for duplicates, splits, cross-contamination and artefacts unless those controls are imposed.

Randomness measures (entropy rate, LZ complexity) and dispersion measures (Fano factor, CV, LV) are unsuitable as objectives. They are useful only as diagnostics or normalisers.

### Cited findings
- Supporting evidence, from the sections above:
  - subadditivity and predictive-information definitions — [Bialek et al. 2001](https://doi.org/10.1162/089976601753195969) [prior knowledge]
  - co-smoothing definition — [Pei et al. 2021](https://arxiv.org/abs/2109.04463) [fetched]
  - robustness of population structure to sorting — [Trautmann et al. 2019](https://doi.org/10.1016/j.neuron.2019.05.003) [prior knowledge]
  - quality-filtered units lose decodable information — [Zhang et al. 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f499387f191d6be56e68966181095878-Abstract.html) [fetched]
  - biased tuning under imperfect sorting — [Ventura 2009](https://www.stat.cmu.edu/~vventura/6921.full.pdf) [fetched]
  - negative finite-sample bias in timescale fits — [Zeraati et al. 2022](https://repository.cshl.edu/id/eprint/40582/) [fetched]
  - contamination estimator — [SpikeInterface/Llobet](https://spikeinterface.readthedocs.io/en/0.101.0/modules/qualitymetrics/isi_violations.html) [fetched]

### Inferences (my reasoning, not published results)

Response table. "Right" means the score falls as that error increases, or penalises it. Data hunger is rough and per unit, over the user's 20-minute excerpts.

| Candidate | Independent false spikes | Missed spikes (random) | Cross-unit contamination / duplicates | Merge | Split | Main gaming route | Data hunger |
|---|---|---|---|---|---|---|---|
| Entropy rate / LZ complexity (bits/s) | **Wrong** (rises) | Right (falls) | Rises (wrong) | Rises (wrong) | Roughly neutral | Add noise spikes; merge everything | Medium (LZ) to high (direct) |
| Predictive information / excess entropy (summed) | Right (provably ≤ for independent Poisson) | Right | Uncertain: a dependent contaminant can add structure | Right (subadditive for independent merges) | Ambiguous; can be inflated | Split neurons; keep bursty artefacts | High (grows only as log T) |
| Fano factor / Allan factor | Toward 1 (ambiguous sign) | Toward 1 | Ambiguous | Toward 1 | Toward 1 per half | Any target value can be met by mixing | Medium to high at long windows |
| CV / CV2 / LV | Toward 1; fills the refractory gap | Toward 1 | Ambiguous | Toward 1 | Toward 1 | Same as above | Low (a few hundred ISIs) |
| Intrinsic-timescale amplitude (autocorrelation fit) | Right (amplitude diluted) | Right (amplitude ∝ p) | Shared timescale can inflate it | Mixed timescales | Each half keeps τ, loses amplitude | Drift-induced slow co-fluctuation looks like a long τ | Medium-high; negatively biased fit |
| Population held-out gain (co-bps, coupling GLM), unrestricted | Right | Right in bits/s; bits/spike can rise | **Wrong** (rises strongly) | Ambiguous | **Wrong** (rises) | Duplicates, splits, artefacts, drift | High (many parameters; noisy for low-rate units) |
| Population gain with exclusions (> ~150 µm peers, ≥ 2 ms lags, slow-rate baseline, artefact masking) | Right | Right | Mostly neutralised | Ambiguous | Mostly neutralised | Remaining: shared artefacts and long-range duplicates | High |
| Population coupling (Okun), as a diagnostic | Lowers coupling | Little effect on correlation | Raises it | Averages the partners | Raises it if the halves are nearby | Not an objective (cell property) | Low-medium |
| Refractory contamination C (Llobet/Hill) | Right (detects) | Insensitive | Detects merges and same-neuron duplicates only above the minimum refractory window | Right | Blind | Drop spikes (lower yield) | Low-medium; unreliable at low rates |
| Isolation distance / L-ratio | Right | Rewards dropping borderline spikes | Partial | Right | Blind (each half can look isolated) | Shrink clusters | Low |
| Heading S (current harness, cross-validated, summed bits/s) | Right | Right | Partially right: borrowed spikes dilute the tuning | Right (convex) | Neutral for random splits (linear under thinning) | Splits along heading-correlated drift | Medium |

Recommended combination for an unknown-tuning objective: keep S where heading tuning exists, and add a covariate-free term T_i for every unit. T_i should be either:
- (a) the cross-validated excess-entropy or relevance-type structure score (MSR or predictive information), referenced to a rate-matched shuffle; or
- (b) the exclusion-controlled population gain, in bits/s.

Then maximise Σ_i T_i subject to the purity constraints, and report sorter comparisons as frontiers over the threshold sweeps.

Before trusting any T, validate its sign empirically on the harness's own data with **synthetic perturbations**:
- inject Poisson spikes;
- randomly thin;
- merge random unit pairs;
- randomly split units;
- duplicate spikes at 0.3 ms lags;
- copy a neighbour's spikes into a unit (mimicking the make_pc_features overwrite).

This is cheap, needs no ground truth, and turns the derived signs in this table into measured ones.

### Gaps
- None of the sign entries in the table come from a published perturbation study. They are derived (subadditivity, superposition and thinning algebra) or reasoned, and should be checked with the perturbation tests above.
- I found no published per-unit sample-size requirements for predictive information, MSR or co-bps on sorted Neuropixels units.
