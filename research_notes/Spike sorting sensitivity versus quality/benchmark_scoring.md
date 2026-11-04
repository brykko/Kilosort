# How spike-sorting benchmarks score sorters, and what objective they implicitly optimize

Scope note: I fetched full text, or long excerpts, of SpikeForest (Magland et al. 2020, eLife), SpikeInterface (Buccino et al. 2020, eLife), the Kilosort4 preprint (Pachitariu et al., bioRxiv 2023, published in Nature Methods 2024; the Nature and PMC pages were blocked by redirects and bot checks), Buccino et al. 2026 (eLife VoR, hybrid KS4 vs KS2.5) and Laquitaine et al. 2024/2025 (bioRxiv, biophysical cortical model). I could not get readable text for the 2016 Kilosort preprint (the PDF would not extract), and I did not fetch Neto et al. 2016, Henze et al. 2000, Yger et al. 2018 or Marques-Smith et al. 2018. Claims about those are in Gaps. Where I derive a cost ratio from a metric's formula, it is labeled as an inference: it is my algebra, not a published claim.

## 1. Which metrics the major benchmarks use, how units are counted, and how results depend on SNR

### Takeaway
Almost every benchmark scores spikes with the same counts (TP, FN, FP) after matching each ground-truth (GT) unit to its best sorted unit. They combine them as accuracy = TP/(TP+FN+FP) (SpikeForest, SpikeInterface, Buccino 2026) or as 1 − FP_rate − FN_rate (Kilosort4). Unit-level summaries count GT units above a 0.8 bar. Only SpikeInterface also counts false-positive, redundant and overmerged units explicitly. Results depend strongly on GT-unit SNR for most sorters, less so for Kilosort 2 and 4, and KS4's advantage over KS2.5 is concentrated below SNR 10.

### Cited Findings
**SpikeForest (Magland et al. 2020, eLife)**
- Matching: a sorted event matches a GT event if the two are within Δ = 1 ms ("the results are rather insensitive to its exact value"). Each GT event matches at most one sorted event. Each GT unit's best match is the sorted unit with the highest accuracy. — [SpikeForest, eLife 2020](https://elifesciences.org/articles/55167)
- Accuracy = n_match / (n_match + n_miss + n_fp), described as "a balance between precision and recall" similar to but not identical to F1. The website also lets users rank by precision or recall. — [SpikeForest](https://elifesciences.org/articles/55167)
- SNR = peak absolute amplitude of the mean waveform ÷ noise, where noise = MAD/0.6745 of the 300–6000 Hz filtered data. — [SpikeForest](https://elifesciences.org/articles/55167)
- Aggregation:
  - The main table averages accuracy over GT units with SNR above a threshold, 8 by default. The authors warn "conclusions can depend strongly on this threshold."
  - A second table counts units with accuracy > 0.8, regardless of SNR.
  - Unit metrics are averaged within a study "without weighting by their numbers of events."
  - Crashed or timed-out runs are imputed by regression, because imputing zero yields "deceptively low values." — [SpikeForest](https://elifesciences.org/articles/55167)
- SNR dependence: KiloSort2 "can retain high accuracy down to lower SNR than other sorters, but not for all such low-SNR units". Tridesclous "appears to struggle with lower-SNR units". For IronClust on SYNTH_JANELIA, SNR and log ISI-violation ratio predict accuracy, while firing rate is "much less predictive". — [SpikeForest](https://elifesciences.org/articles/55167)

**SpikeInterface (Buccino et al. 2020, eLife)**
- Definitions:
  - accuracy = tp/(tp+fn+fp)
  - recall = tp/(tp+fn)
  - precision = tp/(tp+fp)
  - miss rate = fn/(tp+fn)
  - false discovery rate = fp/(tp+fp)

  — [SpikeInterface, eLife 2020](https://elifesciences.org/articles/61834)
- Unit classes, based on the agreement score (equal to accuracy):
  - well-detected: matched, with score > 0.8
  - false positive: unmatched, or score < 0.2
  - redundant: score > 0.2 with one GT unit but not the best match (an oversplit or a duplicate)
  - overmerged: score > 0.2 with two or more GT units

  — [SpikeInterface](https://elifesciences.org/articles/61834)
- Simulated 10-min Neuropixels recording with 250 GT units:
  - Detected units: Tridesclous 189, HerdingSpikes2 233, IronClust 283, SpyKING CIRCUS 343, Kilosort2 415, HDSort 458.
  - Kilosort2, default output: 245 well-detected, 147 false-positive, 21 redundant units.
  - Kilosort2 after its automated curation (removing units with > 20% estimated contamination): 241 well-detected, 93 FP, 18 redundant, 2 overmerged. — [SpikeInterface](https://elifesciences.org/articles/61834)
- "The accuracy is modulated by the SNR of the ground-truth units for all spike sorters except Kilosort2." — [SpikeInterface](https://elifesciences.org/articles/61834)
- Sorters lean one way: "Some sorters seem to favor precision (HerdingSpikes, SpyKING Circus, HDSort), others instead have higher recall (Kilosort2, Ironclust). Tridesclous does not show a bias." — [SpikeInterface PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7704107/)

**Kilosort4 (Pachitariu et al., bioRxiv 2023 / Nature Methods 2024)**
- Score = 1 − FP − FN:
  - FP = estimated spikes without a match ÷ all estimated spikes, i.e. 1 − precision.
  - FN = missed GT spikes ÷ all GT spikes, i.e. 1 − recall.
  - A spike matches if it is ≤ 0.1 ms from a GT spike.
  - Each GT unit is compared with the 20 nearest detected units and paired with the one that maximizes the score.
  - A GT unit counts as "correctly identified" if its score is > 0.8 (Table 2). — [KS4 preprint](https://www.biorxiv.org/content/10.1101/2023.01.07.523036v1.full)
- Ground truth:
  - Hybrid-like simulations built from 11 IBL Neuropixels recordings with ≥ 40 µm drift.
  - Each simulation has 600 GT neurons, with amplitudes drawn from a truncated exponential "which matched the amplitudes in real datasets", plus 600 lower-amplitude "multi-units".
  - Spike trains are shuffled ISIs of real units.
  - Five drift conditions: none, medium, high, fast, step. — [KS4 preprint](https://www.biorxiv.org/content/10.1101/2023.01.07.523036v1.full)
- Results:
  - Kilosort 2, 2.5, 3 and 4 "outperformed all other algorithms in all cases".
  - IronClust, the nearest competitor, "generally found ∼50% of all units" vs "the 80-90% found by Kilosort4".
  - KS4 beat KS2.5 and KS3 "consistently", most of all under step drift.
  - KS4's dependence on firing rate, amplitude and spatial extent was "minimal". Some other sorters "had a strong dependence on amplitude, which could not be improved by lowering spike detection thresholds". — [KS4 preprint](https://www.biorxiv.org/content/10.1101/2023.01.07.523036v1.full)

**Buccino, Sridhar, Feng, Svoboda, Siegle (eLife VoR, 5 Aug 2026; "Efficient and reproducible pipelines…")**
- Hybrid units:
  - IBL NP1 templates and Neuropixels Ultra templates (interpolated to the NP2 geometry) injected into real recordings.
  - 10 units per probe or shank, Poisson firing at 15 Hz, amplitudes rescaled to a set range (e.g. 50–200 µV).
  - Templates follow the measured (DREDge) drift.
  - 5 iterations per recording, giving 1,800 NP1 and 3,000 NP2 GT units. — [Buccino et al. 2026 eLife](https://elifesciences.org/articles/110170)
- Scoring:
  - Matching tolerance 0.2 ms.
  - accuracy, precision and recall as in SpikeInterface.
  - Evaluated on all sorter output, with no curation.
  - "Matched" means accuracy ≥ 0.2. — [Buccino et al. 2026](https://elifesciences.org/articles/110170)
- Matched GT units: KS4 3,652 vs KS2.5 3,046, about 20% more. KS4 has significantly higher accuracy (effect sizes 0.276 NP1, 0.408 NP2), precision (0.272, 0.232) and recall (0.319, 0.415). — [Buccino et al. 2026](https://elifesciences.org/articles/110170)
- SNR dependence: the differences are largest for units with SNR < 10. KS4 "tends to have higher accuracy and recall, at the expense of a slight degradation of precision". This statement sits in some tension with the precision effect sizes above; the tool summary did not resolve it. — [Buccino et al. 2026](https://elifesciences.org/articles/110170)
- Other unit properties: KS4's refractory contamination is slightly higher (effect size 0.013) and its presence ratio higher (0.193). KS4 takes almost twice as long to run as KS2.5. — [Buccino et al. 2026](https://elifesciences.org/articles/110170)

**Laquitaine et al. (bioRxiv 2024, revised 2025; biophysically detailed rat S1 model, 30,190 neurons)**
- Scoring:
  - Matching tolerance 1.3 ms.
  - "Well sorted" means > 80% accuracy.
  - Units are classed as good, merged, partial, or false positive (score at chance). — [Laquitaine et al. bioRxiv](https://www.biorxiv.org/content/10.1101/2024.12.04.626805v2.full)
- Yield:
  - Sorters isolated only about 10–15% of neurons within 50 µm.
  - A denser probe gave "five times fewer neurons sorted with above 80% accuracy" (3% vs 15%).
  - Distance to the electrode influenced accuracy "three times more than firing rate". — [Laquitaine et al.](https://www.biorxiv.org/content/10.1101/2024.12.04.626805v2.full)

### Inferences
- Matching tolerances differ by an order of magnitude: 0.1 ms (KS4), 0.2 ms (Buccino 2026), 1 ms (SpikeForest), 1.3 ms (Laquitaine). Absolute accuracies are therefore not comparable across papers. A 0.1 ms window punishes small timing jitter of template-matched spikes as an FP plus an FN.
- The SNR cut is a hidden weighting choice. SpikeForest's default SNR > 8 average largely removes the low-SNR regime, which is exactly where a detection threshold matters most and where Kilosort's advantage lies (SpikeInterface; Buccino 2026).

### Gaps
- I did not get KS4 Table 2 per-sorter counts, or the exact Nature Methods 2024 thresholds ("[9, 8]" rendered ambiguously in the preprint text).
- Henze et al. 2000 (tetrode + intracellular), Neto et al. 2016 / Marques-Smith et al. 2018 (paired juxtacellular/patch + dense probes), and Yger et al. 2018 (SpyKING CIRCUS, in vitro/in vivo ground truth): not fetched this session. A search snippet noted that Yger et al. report errors rising sharply for events below about 50 µV ([Yger 2018 eLife](https://elifesciences.org/articles/34518)), but I did not verify the figure.
- The 2016 Kilosort preprint's hybrid scoring: from memory, it included a "best match after optimal merges" variant that forgives oversplitting, but I did not verify this.
- Rossant et al. 2016 hybrid methodology: not fetched.
- MEArec (Buccino & Einevoll 2021) is a simulator, not a scoring scheme.

## 2. What cost ratio between misses and false positives do these scores imply? Published precision–recall vs threshold curves

### Takeaway
All the standard scalars count a missed spike and a false or misassigned spike equally (symmetric counts). Because they are ratios, though, the threshold that maximizes them is not "marginal precision 50%". Accuracy and F1 are maximized when the spikes added by a small threshold reduction are correct at rate F1/2 = A/(1+A). That is roughly 45% for a unit at accuracy 0.8–0.9, so these metrics favor fairly sensitive settings. KS4's 1 − FP − FN behaves about the same when precision ≈ recall, but asks more of a conservative sorter. I found no published per-sorter precision–recall curves against detection threshold. Benchmarks either use defaults or sweep the threshold to maximize a unit count.

### Cited Findings
- SpikeForest notes that accuracy is a balance between precision and recall, and that researchers "may want to place a greater importance on maximizing either precision or recall", which is why the site exposes all three. — [SpikeForest](https://elifesciences.org/articles/55167)
- KS4 FP and FN are defined as 1 − precision and 1 − recall, so its score = precision + recall − 1. — [KS4 preprint](https://www.biorxiv.org/content/10.1101/2023.01.07.523036v1.full)
- Precision and recall are reported separately in SpikeInterface and Buccino 2026. Buccino 2026 describes KS4 as trading "a slight degradation of precision" for higher recall. — [SpikeInterface](https://elifesciences.org/articles/61834); [Buccino 2026](https://elifesciences.org/articles/110170)
- The KS4 paper's only threshold sweeps were for competing sorters. The criterion was the number of correctly identified units (score > 0.8) on the medium-drift simulation, not a precision–recall curve. — [KS4 preprint](https://www.biorxiv.org/content/10.1101/2023.01.07.523036v1.full)

### Inferences (my derivations; check before quoting)
- **Accuracy and F1 rank identically.** A = k/(k+FP+FN), F1 = 2k/(2k+FP+FN), and A = F1/(2 − F1), which is strictly monotone. So "accuracy vs F1" is not a substantive choice for a single unit, although averages over units can differ slightly.
- **Marginal rule for accuracy.** Hold the GT count N = k + FN fixed and let a threshold decrease add dk true and dFP false spikes. Accuracy is A = k/(N + FP), and dA ≥ 0 iff dk/dFP ≥ A. So the threshold should keep dropping while the newly admitted spikes are true at a rate above A/(1+A).
  - At A = 0.9 that is about 0.47; at A = 0.8, about 0.44; at A = 0.5, 0.33.
  - In words: one extra false spike is "worth" 1/A extra true spikes.
  - This matches the known F1-optimal-threshold result for calibrated scores, threshold = F1*/2 (Lipton, Elkan & Naryanaswamy 2014, arXiv:1402.1892; I cite it from memory and did not fetch it).
- **Marginal rule for KS4's score.** For S = P + R − 1, dS ≥ 0 iff dk/dFP ≥ P²/(R + P(1 − P)).
  - At P = R this gives the same marginal precision, R/2, as accuracy.
  - When P ≫ R (a conservative sorter, e.g. P = 0.99, R = 0.7), S needs marginal precision of about 0.58, whereas accuracy needs about 0.41.
  - So 1 − FP − FN rewards recall gains slightly less than accuracy does in the low-recall regime.
- **Misassigned spikes count twice.** A spike assigned to the wrong unit is an FN for its true unit and an FP for the receiving unit (if that unit is matched to a GT unit). Detection misses count once. So per-spike symmetric metrics already weight confusion errors about twice as much as pure misses at the population level.
- **The unit-count summaries (score > 0.8) are step functions.** A sensitivity increase is rewarded only if it pushes GT units across 0.8. The summaries are indifferent to quality above the bar, and indifferent to new false-positive units unless those are counted separately.

### Gaps
- I found no paper that publishes precision–recall curves of Kilosort (any version) as a function of Th_universal / Th_learned (KS4) or ops.Th (KS2.5). This looks like a genuine gap in the literature, not just a search failure; I ran 2 targeted searches with no hits.

## 3. Known problems with these metrics

### Takeaway
The documented problems are:
- Per-GT-unit scores ignore spikes in sorted units that match no GT unit, so extra noise or "unknown" units are free unless counted separately.
- Ground truth is biased toward the units that are easy to get GT for: large, near, high-rate, pyramidal.
- Synthetic and hybrid data flatter sorters relative to biophysical or paired data. Precision is systematically lower on real paired data than on simulations.
- Unit-level FP counts can wrongly penalize real, unlabeled neurons. Hybrid benchmarks partly avoid this, because native units in the host recording are not scored.

### Cited Findings
- SpikeForest: "the results shown in the main table do not account for false positive units". KiloSort2 had "the largest number of false positive units" in synthetic studies, but this is not reported in the main table. — [SpikeForest](https://elifesciences.org/articles/55167)
- SpikeForest: for paired and manual studies, "precisions are generally a lot lower than recalls", while for synthetic studies this often reverses. This suggests simulations "may not yet be duplicating the firing and noise statistics" of real recordings. — [SpikeForest](https://elifesciences.org/articles/55167)
- SpikeForest: "SpikeForest results are biased towards the types of neurons and conditions for which ground-truth data exist." — [SpikeForest](https://elifesciences.org/articles/55167)
- SpikeForest, small paired sets: PAIRED_KAMPFF has 15 units, 11 of them sortable. "A single false split or merge by a sorter can lead to variations in reported study-averaged accuracy as large as ±0.05." — [SpikeForest](https://elifesciences.org/articles/55167)
- SpikeInterface: Kilosort2's 147 FP units in simulation fell to 93 after contamination-based curation, while well-detected units fell only from 245 to 241. In that simulation, most FP units are therefore removable at little recall cost. — [SpikeInterface](https://elifesciences.org/articles/61834)
- SpikeInterface: the simulation (independent Poisson firing, additive Gaussian noise of 10 µV SD) may not reflect real recordings, and all sorters were run with defaults. — [SpikeInterface](https://elifesciences.org/articles/61834)
- Laquitaine et al.:
  - Synthetic simulations with independent Poisson firing and no connectivity "may inflate sorting accuracy and yield". Sorting accuracy in their biophysical model was about 10× lower than in the synthetic simulation.
  - Sorters over-represent rare interneuron types by 2.9–8.1× (especially Kilosort 1–2.5).
  - Sorted firing rates are overestimated 3–4× in most layers.
  - KS3 and KS4 produced no false-positive units, while older Kilosort versions did. — [Laquitaine et al.](https://www.biorxiv.org/content/10.1101/2024.12.04.626805v2.full)
- Buccino 2026 lists its own hybrid limitations:
  - Poisson trains.
  - Units placed uniformly rather than by neuronal density.
  - Templates not matched to brain region.
  - It is untested whether more realistic hybrids would change the accuracy ranking. — [Buccino 2026](https://elifesciences.org/articles/110170)
- KS4 paper, step drift: many algorithms split units in two, with many units at "exactly a 0.5 score". Kilosort is biased toward over-splitting but "almost never made incorrect merges". — [KS4 preprint](https://www.biorxiv.org/content/10.1101/2023.01.07.523036v1.full)
- SpikeInterface on consensus: a significant fraction of true units found by only one sorter may be wrongly classified as false positives. — [SpikeInterface](https://elifesciences.org/articles/61834)

### Inferences
- "Accuracy rewards conservative sorting" holds only for the unit-count style of score, and only weakly. From Q2, per-unit accuracy actually favors moderately sensitive thresholds. The real conservative bias comes from:
  - SNR > 8 filters, which hide the low-SNR regime.
  - Ignoring FP units, which is the opposite bias: it makes oversensitive sorting free.
  - Best-match scoring of oversplit units. A split unit scores about 0.5 and loses well-detected status, so over-splitting is punished harder than missing spikes uniformly at random.
- Hybrid benchmarks (KS4 paper; Buccino 2026) inject a few GT units into real data and score only those units. Native neurons the sorter finds are neither rewarded nor penalized, which avoids the "real but unlabeled" FP-unit problem. Fully synthetic benchmarks (SpikeInterface 2020) do count FP units, but there every non-GT unit really is spurious. The two designs therefore measure different things about sensitivity.
- Paired recordings (1–15 GT cells per study) give too few units for unit-count metrics, and they cannot measure FP-unit rates at all.

### Gaps
- I found no study that systematically measures how many "FP units" in real-data comparisons are real but unlabeled neurons. Laquitaine et al.'s biophysical model, where every neuron is known, is the closest substitute.

## 4. Systematic hyperparameter tuning against ground truth, and conclusions about threshold-like parameters

### Takeaway
Tuning against ground truth has been limited:
- SpikeForest let developers tune on its data, but ran no automated optimization because of compute cost.
- The KS4 paper swept detection thresholds only for competitors, choosing the threshold that maximized the count of units with score > 0.8. The defaults were best for IronClust and MountainSort4; SpyKING CIRCUS was best at 4.5. KS4 also reports that some sorters' amplitude dependence "could not be improved by lowering spike detection thresholds".
- I found no Bayesian optimization of Kilosort parameters against ground truth in the literature.

### Cited Findings
- SpikeForest:
  - Algorithms marked with asterisks "were updated and optimized using SpikeForest data"; the rest used defaults. "Optimal choices are not guaranteed".
  - IronClust uses the same parameters across all studies, "limiting the potential for overfitting".
  - Automated parameter optimization (as in ClustEval) was not implemented because it would multiply CPU cost. — [SpikeForest](https://elifesciences.org/articles/55167); [SpikeForest PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7237210/)
- KS4 paper:
  - Kilosort 1–3 detection thresholds were set to fixed values instead of each version's defaults.
  - IronClust, MountainSort4 and SpyKING CIRCUS got a threshold sweep, choosing the value that "maximized the number of correctly identified units on the medium drift simulation". Defaults were best for IronClust and MS4; SpyKING CIRCUS used 4.5. A sweep for SpyKING CIRCUS 2 did not help.
  - All other parameters were left at defaults. — [KS4 preprint](https://www.biorxiv.org/content/10.1101/2023.01.07.523036v1.full)
- KS4 paper: some algorithms "had a strong dependence on amplitude, which could not be improved by lowering spike detection thresholds". — [KS4 preprint](https://www.biorxiv.org/content/10.1101/2023.01.07.523036v1.full)
- SpikeInterface 2020 ran all sorters with default parameters for comparability. — [SpikeInterface](https://elifesciences.org/articles/61834)
- Buccino 2026 compares KS4 and KS2.5 at defaults (no threshold sweep was reported in the fetched portion). KS4 recovers about 20% more GT units, and the largest gains are below SNR 10. — [Buccino 2026](https://elifesciences.org/articles/110170)
- Other drift benchmarks: Garcia, Windolf, Boussard, Dichter, Buccino & Yger (eNeuro 2024) benchmark motion-correction algorithms on simulated drifting recordings, tracing errors through the sorting pipeline. I did not fetch details on threshold parameters. — [Garcia et al. eNeuro 2024](https://www.eneuro.org/content/11/2/ENEURO.0229-23.2023.abstract)
- Outside Kilosort, tuning is typically done once on a small GT set with Optuna TPE or a genetic algorithm and then frozen, to avoid overfitting scarce ground truth. Example: a 2025 sparse-coding SNN sorter. — [arXiv 2506.24041](https://arxiv.org/pdf/2506.24041)

### Inferences
- The only threshold-selection criterion used in a Kilosort paper is "number of GT units with 1 − FP − FN > 0.8". It ignores FP units and ignores quality above 0.8. A KS4 vs KS2.5 threshold comparison that wants to match the literature should report this count, and should also report what it ignores.
- Hybrid-injected units at 50–200 µV (Buccino 2026), or with exponential amplitudes matched to real data (KS4), cover the range where the detection threshold matters. SNR < 10 is where the KS4–KS2.5 gap was largest.

### Gaps
- Steinmetz et al. 2021 (Neuropixels 2.0, Science) used hybrid and drift ground truth with KS2.5. I did not fetch it, so its scoring and threshold choices are unverified here.
- I found no published sweep of KS4's Th_universal/Th_learned, or of KS2.5's ops.Th, against ground truth.
- I also found no "spike sorting hyperparameter optimization" paper that tunes Kilosort; 2 targeted searches returned none.

## 5. Proposals for scoring at the population level (agreement, ensemble or consensus, yield above a quality bar)

### Takeaway
The main population-level alternatives are:
- Unit-class counts: well-detected / FP / redundant / overmerged, from SpikeInterface.
- Consensus or agreement sorting across sorters. This mostly removes FP units, at some risk of discarding true units found by only one sorter.
- Production "yield above a quality bar" from label-free metrics, such as ISI violations, amplitude cutoff and presence ratio, or classifiers like UnitRefine.

None of these has been validated as a single scalar to optimize.

### Cited Findings
- SpikeInterface agreement analysis:
  - Real Neuropixels data: of 2,031 units, all six sorters agree on 33, and two or more agree on 263. For most sorters, more than 50% of their units match no other sorter.
  - Simulation: only 139 of 1,921 units (7.24%) are agreed by all sorters. Almost all FP units are found by a single sorter, and agreement rises with SNR. — [SpikeInterface](https://elifesciences.org/articles/61834)
- Agreement between a Kilosort2 and IronClust pair identifies false positives at 96% and true positives at 95% on ground truth. Kilosort2's consensus units match the two human curators at 84.6% and 89.6%, its non-consensus units at 18.6% and 24.3%. The two experts themselves agreed on only 64.2% of curated units. — [SpikeInterface](https://elifesciences.org/articles/61834)
- SpikeInterface tooling: multi-sorter comparison builds a graph of pairwise matches. "Agreement sorting" keeps units found by at least k sorters. The docs warn that consensus can suffer from the weak performance of single algorithms. — [SpikeInterface docs](https://spikeinterface.readthedocs.io/en/0.96.0/modules/comparison/plot_2_compare_multiple_sorters.html)
- An earlier consensus-clustering method shows that its estimated misclassification proportion correlates with the true proportion measured against ground truth, offering a label-free error estimate. — [Consensus-based sorting, PMC4990262](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4990262/)
- Production yield (Buccino 2026):
  - Default quality thresholds: ISI violation ratio < 0.5, amplitude cutoff < 0.1, presence ratio > 0.8, then UnitRefine classifies units as noise, SUA or MUA.
  - Across 3,126 probes, more than 1 million units were labeled neural and more than 700,000 passed the thresholds.
  - KS4 had higher presence ratio and slightly higher refractory contamination than KS2.5 on hybrid units. — [Buccino 2026](https://elifesciences.org/articles/110170)
- Laquitaine et al. argue that population-level scientific value, not just unit counts, should be the criterion:
  - Selection biases cut stimulus-information capacity by about 21% on average.
  - Overall, sorting cut it by about 50%.
  - Undersampling alone did not distort the manifold. — [Laquitaine et al.](https://www.biorxiv.org/content/10.1101/2024.12.04.626805v2.full)

### Inferences (answering "is there a defensible single quantity?")
- No benchmark offers a principled single scalar that covers the population. The common scalar (mean per-GT-unit accuracy, or the count above 0.8) is symmetric between missed and false spikes, but blind to FP units and to quality above the bar.
- The most defensible single objective for choosing a detection threshold, consistent with the literature, is a **pooled, event-level accuracy over all sorted spikes**:
  - Count every spike in a sorted unit that matches no GT unit as an FP, and every GT spike not in its best match as an FN.
  - Or count well-detected GT units minus λ × FP units, with λ stated explicitly.
- The symmetric choice (λ = 1, per-spike FP cost = FN cost) is the implicit convention of SpikeForest, SpikeInterface and KS4. Any departure from it should be justified by the downstream analysis, as SpikeForest itself says.
- A practical, literature-aligned report for a KS4 vs KS2.5 threshold sweep on hybrid data:
  1. Count of GT units with accuracy ≥ 0.8. This is SpikeInterface's well-detected count, and the KS4 paper's Table 2 criterion with its score substituted.
  2. Median precision and recall separately, stratified by SNR (< 5, 5–10, > 10).
  3. FP / redundant / overmerged unit counts. On hybrid data these can only be estimated for units near injected ones.
  4. Label-free yield: units passing ISI < 0.5, amplitude cutoff < 0.1, presence > 0.8.

  A threshold that raises (1) and (4) without raising (3) is an improvement under every published convention. One that trades (1) against (3) needs an explicit λ, and the literature gives no consensus value.

### Gaps
- I found no paper that proposes or validates an explicit cost ratio between FP units and missed units, or between FP spikes and FN spikes, for neuroscience use cases.
- I found no paper that scores sorters by a downstream population analysis across threshold settings. Laquitaine et al. compare sorters, not thresholds.
