# Name the cost before tuning thresholds

**Theory does not supply a "best" threshold. It supplies a best detector, which is a whitened matched filter compared against a threshold. Where that threshold should sit depends on two things the user has to state: a false-alarm budget, or the relative cost of a false spike against a missed one.** Once one of these is fixed there is a quantity to optimise, and its optimum is well defined. Until then the result is a frontier of trade-offs, and choosing a point on it is a choice about costs, not something to be discovered. That still leaves one question theory can settle without any cost ratio: whether one sorter or configuration *dominates* another. If its precision–recall curve, or its yield-versus-contamination curve, lies outside the other's everywhere, every cost ratio prefers it. Comparing KS4 with KS2.5 should therefore start with a frontier comparison, not a comparison at single points. Three further results limit what tuning can achieve:

- Kilosort's default thresholds (KS4: 9 and 8) sit far above the values a Gaussian-noise derivation gives (roughly 4.5–6). So the practical noise floor is set by background spikes and template mismatch, and it has to be calibrated empirically.
- Detection and assignment have different optima. Lowering a detection threshold does not fix units lost in clustering.
- The "right" operating point depends on the analysis. Decoding favours sensitivity. Correlation and synchrony analyses are hurt most by cross-unit contamination.

For the harness this means three things. Treat the objective as **information yield under a stated contamination budget**. Compare KS4 and KS2.5 on swept frontiers, not tuned-versus-preset points. Recognise that S, as currently defined, leans towards sensitivity: it barely penalises false spikes that leave tuning curves undistorted.

## Detection theory fixes the detector, not the threshold

In classical detection theory the optimal detector for a known waveform in Gaussian noise is a likelihood-ratio test. In practice that means whitening the data, applying a matched filter and comparing the output to a threshold ([ISAC tutorial, arXiv 2609.22314](https://arxiv.org/pdf/2609.22314); [arXiv 1907.01465](https://arxiv.org/pdf/1907.01465)). Theory fixes the *form* of this test, but not the threshold. The threshold comes from one of two external choices:

- **Neyman–Pearson:** a tolerated false-alarm probability.
- **Bayes risk:** priors and per-error costs.

Without either choice, all that is left is an ROC curve.

For spike sorting the clearest derivation is Franke et al.'s Bayes-optimal template matching (BOTM). There the optimal filter is f = C⁻¹ξ, and the noise is treated as one more class with prior 1 − Σpᵢ. The authors state that "the spike detection threshold depends on the templates energy, the noise covariance matrix and the prior probability." They set equal costs, so the objective is FP + FN, and they report that performance was **insensitive to the exact priors** ([Franke et al. 2015, PMC4420847](https://pmc.ncbi.nlm.nih.gov/articles/PMC4420847)).

Even this principled sorter therefore fixes the cost ratio at 1:1 by convention. No spike-sorting paper in these notes states physiologically motivated costs C_FA and C_miss.

### What the theory implies for a single "k-sigma" threshold [inference]

The detection-theory notes work through this case. All of it is the note-taker's algebra, not a published result.

**A universal threshold is a Neyman–Pearson rule.** Under noise, the normalised matched-filter statistic z = wᵀx/‖w‖ is N(0,1). For a spike of whitened norm s it is N(s,1). One threshold therefore serves every unit, and the false-alarm probability alone sets it. This is why a "k-sigma" rule is a Neyman–Pearson (constant false-alarm rate) rule.

**The Bayes threshold depends on the unit.** With equal costs it is z* = s/2 + ln(p₀/p₁)/s, which depends on both the unit's norm and its firing rate. For a 5 Hz unit at 30 kHz:

| Whitened norm s | Bayes threshold z* |
|---|---|
| 6 | about 4.45 |
| 8 | about 5.1 |
| 10 | about 5.9 |

So a single universal z threshold is Bayes-optimal for at most one amplitude. If misses are weighted more heavily, the threshold drops most for *small* units.

The model-selection notes reach a similar figure by a different route. A MAP rule in the style of Pillow and Franke admits a spike when A²/2 exceeds the prior log-odds, which gives A ≈ 4–5 for p ~ 10⁻⁴ [inference].

### Kilosort's statistic and why its defaults sit higher

The local code shows what Kilosort actually thresholds. In this branch's `run_matching`, KS4 computes Cf = relu(B)²/‖U‖² and accepts a peak when Cf > Th_learned². This is a one-sided normalised matched-filter statistic in whitened-SD units ([template_matching.py](file:///C:/code/github/Kilosort/kilosort/template_matching.py)). `Th_universal` is applied the same way to unit-norm universal templates ([spikedetect.py](file:///C:/code/github/Kilosort/kilosort/spikedetect.py)).

The KS4 defaults are therefore **roughly twice the Gaussian Bayes value**. At z = 8, Gaussian noise would predict essentially no false alarms per hour even across about 5×10¹⁰ template-by-lag tests. A threshold near 6 would already give only tens [inference]. The notes conclude that the defaults absorb non-Gaussian structure: background spikes ("hash"), template mismatch, drift, and subtraction residuals. The Kilosort authors do not say this; it is the note-taker's interpretation. It does match the simulations, where background is built from tens of thousands of superimposed spikes per second ([Pedreira et al. 2012, PMC3657693](https://pmc.ncbi.nlm.nih.gov/articles/PMC3657693)). KS4's own simulations add 600 low-norm multi-units for the same reason ([Pachitariu et al. 2024, PMC11093732](https://pmc.ncbi.nlm.nih.gov/articles/PMC11093732)).

The practical point: **a principled threshold for Kilosort must be calibrated against an empirical null.** That null can come from hybrid injection, decoys or refractory evidence. It cannot come from an analytic Gaussian tail.

## Lowering detection thresholds cannot fix assignment errors

Detection asks whether an event is "unit i or noise", which depends on ‖ξᵢ‖. Assignment asks whether it is "unit i or unit j", which depends on ‖ξᵢ − ξⱼ‖. These are different questions with different optima.

**The assignment floor [inference].** Under Gaussian noise the minimum pairwise misassignment rate is Q(‖ξ₁ − ξ₂‖/2). This is the linear-discriminant result BOTM is built on. Two units can each be easy to detect yet hard to separate, as with nearby cells, or one cell before and after drift. A lower detection threshold does nothing to that floor.

**The empirical record agrees.** Pedreira et al. found that correctly identified neurons **plateaued at about 8–10** however many neurons were simulated. Misses came mainly from merged clusters, and sparse units (<0.5 Hz) were missed about 80% of the time ([Pedreira et al. 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3657693)). The KS4 paper reports that for several competing sorters the dependence on amplitude "could not be improved by lowering spike detection thresholds" ([PMC11093732](https://pmc.ncbi.nlm.nih.gov/articles/PMC11093732)). BOTM given the *correct* templates reached 99.7% accuracy under noise where clustering-based sorting fell to 72%. Templates built from fewer than 30 spikes were noisy ([Franke et al. 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4420847)). In practice, then, the floor is set by template completeness and template estimation, not by detection.

### The two KS4 thresholds do different jobs

The two KS4 thresholds act at different stages:

- **`Th_universal`** feeds only template learning. It decides *which units exist*. By the authors' own account the first pass misses about 50% of spikes, which they accept for this purpose ([bioRxiv KS4](https://www.biorxiv.org/content/10.1101/2023.01.07.523036v1.full)).
- **`Th_learned`** decides *how many spikes each existing unit gets*.

The user's own sweep shows the same split ([round3_results.md](file:///C:/code/github/Kilosort/tuning/round3_results.md)):

- Lowering `Th_universal` from 9 to 7 was "the biggest lever", and it saturates (6 ≈ 7).
- `Th_learned` 7 "mostly adds low-amplitude spikes to existing cells" (+25 bits/s). Going down to 6 adds nothing more, and raising it to 9 loses 64 bits/s.

BOTM shows that missing templates **lower specificity but leave sensitivity unchanged** ([Franke et al. 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4420847)). By the notes' inference, a lower `Th_learned` with an incomplete template set therefore draws unmodelled hash into the nearest template, and this hurts most for small, spatially spread units.

### Stock KS4 cannot take those spikes back out

This matters because of a structural difference between the versions, confirmed in local code ([kilosort_parameters notes, Q1](file:///C:/code/github/Kilosort/kilosort/template_matching.py)):

- **KS2.5** limits a low final-pass threshold in two ways. Its matching score includes an amplitude prior weighted by `lam` (`mexMPnu8.cu`). Its `set_cutoff` lowers each unit's cut-off from Th(1) towards Th(2) only while that unit stays refractory ([set_cutoff.m](file:///C:/code/github/neuropixels/ExternalToolboxes/KilosortToolbox/Versions/kilosort_v2.5/postProcess/set_cutoff.m)).
- **KS4** has neither safeguard. `lam` is assigned but never used, and nothing after matching removes the low-score spikes a low `Th_learned` admits ([template_matching.py](file:///C:/code/github/Kilosort/kilosort/template_matching.py)).

So threshold values **do not transfer numerically** between the versions, and in KS4 a sensitivity gain at matching is permanent unless the harness adds a cut.

The user's placement audit adds a further finding. A set_cutoff-like cut on KS4 gained only +6–13% S over a matched random-deletion control, and this was "mostly explained by" low-amplitude spikes carrying less heading information. Most of KS4's contamination is a *grouping* error made in the final reclustering ([placement_audit.md](file:///C:/code/github/Kilosort/tuning/exploratory/autocurate/placement_audit.md)).

### "Detect low, curate later" holds only partly

Does detecting liberally and curating strictly afterwards dominate? The data-processing inequality says the information available can only grow as the threshold falls: I(S; C) ≤ I(S; D) [inference]. Decoding work also confirms that discarded hash carries signal ([Todorova et al. 2014, PMC4454741](https://pmc.ncbi.nlm.nih.gov/articles/PMC4454741)).

But the bound is a ceiling, not an outcome. Real pipelines hard-assign spikes, learn templates from the detected events, and remove contamination imperfectly. The extra false alarms contaminate template estimates and get absorbed into real units. Curation is itself a noisy classifier: in tetrode ground truth, manual operators made 0–30% errors, against about 0–8% near the estimated optimum ([Harris et al. 2000](https://pubmed.ncbi.nlm.nih.gov/10899214/)).

No paper in the notes proves or disproves "low threshold plus curation dominates" for a template-matching sorter. The best-supported version is narrower: **detect liberally, keep the labels graded**, and let each analysis apply its own inclusion rule.

## Benchmark scores already hide a cost ratio near one-to-one

Every major benchmark counts a missed spike and a false spike symmetrically:

| Benchmark | Score |
|---|---|
| SpikeForest | accuracy = TP/(TP+FN+FP) ([Magland et al. 2020](https://elifesciences.org/articles/55167)) |
| SpikeInterface | accuracy, precision, recall ([Buccino et al. 2020](https://elifesciences.org/articles/61834)) |
| Buccino et al. 2026 | accuracy, precision, recall ([eLife 110170](https://elifesciences.org/articles/110170)) |
| Kilosort4 | 1 − FP − FN, i.e. precision + recall − 1 ([KS4 preprint](https://www.biorxiv.org/content/10.1101/2023.01.07.523036v1.full)) |

### What the cost ratio implies [inference]

Symmetric counts do not mean a 50% marginal precision. Because accuracy is a ratio, it keeps improving as the threshold drops until the newly admitted spikes are correct at a rate of only A/(1+A). That is **about 0.44–0.47 for units at accuracy 0.8–0.9**. This matches the F1-optimal threshold of F1*/2 for calibrated scores ([Lipton et al. 2014](https://arxiv.org/pdf/1402.1892)). Accuracy therefore favours fairly sensitive settings.

The implied cost ratio also moves with performance. As accuracy falls, a false positive is weighted less. KS4's P + R − 1 behaves the same as accuracy when precision equals recall. For a conservative sorter it demands more: about 0.58 marginal precision at P = 0.99, R = 0.7. A misassigned spike counts twice, as an FN for its true unit and an FP for the receiving unit, so symmetric metrics already weight confusion errors above pure misses.

### What benchmarks hide

The summaries built on these scores are where the real biases come in.

**Unit counts above 0.8 are step functions.** They reward a sensitivity gain only when it pushes a ground-truth unit over the bar [inference].

**False-positive units are often invisible.** SpikeForest's main table "do[es] not account for false positive units", even though KiloSort2 had the most of them in its synthetic studies ([SpikeForest](https://elifesciences.org/articles/55167)). SpikeInterface's 250-unit simulation shows how many there can be: Kilosort2 produced **245 well-detected and 147 false-positive units**. Contamination-based curation cut the false positives to 93 while losing only 4 well-detected units ([SpikeInterface](https://elifesciences.org/articles/61834)).

**SNR cuts filter out the interesting range.** A cut at SNR > 8 removes the low-SNR regime, which is where thresholds matter. That regime is also where KS4's advantage over KS2.5 is concentrated: below SNR 10 ([Buccino et al. 2026](https://elifesciences.org/articles/110170)).

**Matching windows differ by an order of magnitude** (0.1–1.3 ms across papers), so absolute accuracies cannot be compared between papers.

**Kilosort's own thresholds have never been swept against ground truth.** The KS4 paper swept thresholds only for competing sorters, choosing whichever value maximised the count of units with score > 0.8. It ran Kilosort 1–3 at [9, 8] rather than at their shipped defaults ([KS4 preprint](https://www.biorxiv.org/content/10.1101/2023.01.07.523036v1.full)). The notes found no published sweep of KS4's `Th_universal`/`Th_learned` or KS2.5's `Th`, and no published derivation of any Kilosort default. The user's roughly 124 scored sorts across four rounds are more systematic than anything found in print ([round3_results.md](file:///C:/code/github/Kilosort/tuning/round3_results.md)).

### Real data look worse than simulations

Realism cuts the other way. On paired recordings, SpikeForest found "precisions are generally a lot lower than recalls", whereas synthetic data often show the reverse ([SpikeForest](https://elifesciences.org/articles/55167)). In a biophysically detailed cortical model, sorting accuracy was **about 10× lower** than in simple synthetic data. Sorters isolated only about 10–15% of neurons within 50 µm and over-represented rare interneuron types by 2.9–8.1× ([Laquitaine et al. 2024/25](https://www.biorxiv.org/content/10.1101/2024.12.04.626805v2.full)). Whatever operating point simulations favour is therefore likely to be too optimistic about precision on real data.

## The right operating point moves with the downstream analysis

The literature on downstream effects explains why no universal cost ratio exists: different analyses are hurt by different errors.

### Decoding favours sensitivity

For decoding, spikes that an isolation-maximising pipeline would discard keep carrying signal.

- **Rat CA1, position decoding.** Clusterless decoding was 14% better than decoding sorted clusters plus hash. It was *comparable* (P = 0.90) to decoding sorted clusters with no quality criteria ([Kloosterman et al. 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC3921373)). Most of the loss therefore comes from *excluding* units and spikes, not from misassigning them.
- **Hippocampal decoding, later work.** Clusterless rMSE was 15.93 against 31.97 for sorted. The discarded spikes accounted for **37–48% of the incremental gains**, concentrated where sortable neurons were sparse ([Ghahari & Eden 2025](https://arxiv.org/html/2512.06280v1)).
- **Utah-array BMIs.** Decoding was flat between −3 and −4.5 × RMS. One monkey kept improving down to the lowest threshold tested ([Christie et al. 2015](https://escholarship.org/uc/item/0ww0z6xm)).
- **Population dynamics.** More permissive thresholds preserved a population result that stricter ones lost ([Trautmann et al. 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC7002296)).

There is a floor. A per-waveform likelihood cutoff improved decoding over raw crossings in most sessions, increasingly so in older implants ([Issar et al.](https://www.biorxiv.org/node/1136608)). Keeping everything stops paying once noise events become common.

### Correlation and synchrony analyses are hurt by contamination

Ventura and Gerkin showed that sorting units independently biases correlation estimates, and that "the bias does not disappear in large samples." Misclassification of 5–10% makes pairs with true correlations of −0.05, 0 and 0.1 indistinguishable ([Ventura & Gerkin 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3358902)). Failure to resolve overlapping spikes creates an artificial zero-lag trough ([Bar-Gad et al. 2001](https://cris.huji.ac.il/en/publications/failure-in-identification-of-overlapping-spikes-from-multiple-neu/), abstract-level).

### Which error hurts most depends on the analysis

The two error types trade places depending on the objective:

- **Synchrony significance:** misses reduce significance more than false positives, so conservative sorting tends to lose significant synchrony ([Pazienti & Grün 2006](https://www.bohrium.com/paper-details/robustness-of-the-significance-of-spike-synchrony-with-respect-to-sorting-errors/812062707468468225-11386), abstract-level).
- **Channel capacity:** false positives "severely degrad[e] the capacity, even when the error rate is extremely small", while misses act only linearly ([Goodman & Johnson 2008](https://bmcneurosci.biomedcentral.com/articles/10.1186/1471-2202-9-S1-P122)).

The notes treat these two findings as reflecting different objectives rather than as contradicting each other, and that reading is correct. Neither generalises.

### Which units are found matters more than how many

In the biophysical model, randomly removing neurons had no effect on stimulus information capacity (all p > 0.47). Cell-type-biased selection cut it by about 21% ([Laquitaine et al. 2024](https://www.biorxiv.org/content/10.1101/2024.12.04.626805.full)). A sensitivity gain that recovers under-represented cells can be worth more than its raw unit count suggests.

The practical consequence, stated in several of these papers: produce **one sort and apply different inclusion filters downstream** — all clusters for decoding, strict isolation for place fields and cross-correlograms. A single composite score cannot be right for both.

## Other fields fix an error rate, then maximise yield

Fields that face the same trade-off settle it in one of two ways.

**Error-rate control.** Most fields fix an error rate they can estimate and accept whatever sensitivity follows:

- Astronomy uses Benjamini–Hochberg FDR source detection, which held on correlated images ([Hopkins et al. 2002](https://arxiv.org/abs/astro-ph/0110570)). It also estimates reliability from detections on negated data, R = (P − N)/P ([Serra et al. 2012](https://arxiv.org/abs/1112.3162)).
- Proteomics uses target–decoy FDR at about 1% ([Elias & Gygi 2007](https://doi.org/10.1038/nmeth1019), not fetched).
- Radar fixes the false-alarm probability with CFAR detection.
- ENCODE fixes a reproducibility rate across replicates (IDR) ([Li et al. 2011](https://arxiv.org/pdf/1110.4705)).

**A single figure of merit.** Only particle physics and SMLM collapse the trade-off into one scalar, and both need a stated goal. Punzi's ε/(a/2+√B) is optimal for discovery sensitivity when the signal strength is unknown ([Punzi 2003](https://www.arxiv.org/pdf/physics/0308063)). The SMLM challenge's efficiency score makes its weight α explicit; it does not remove it ([Sage et al. 2019](https://bigwww.epfl.ch/publications/sage1903.pdf)).

**Calcium imaging** sits between the two. CaImAn chose precision over recall by stated judgement, because false-positive cells do more damage downstream. Tuning per dataset gained only about 0.02 F1 over fixed defaults ([Giovannucci et al. 2019](https://elifesciences.org/articles/38173)). That small gain suggests a defensible default is often close to the optimum.

### What carries over to spike sorting [inference]

Two ideas carry over.

**Compare frontiers first.** If two ROC curves dominate one another, so do the corresponding precision–recall curves ([Davis & Goadrich 2006](https://doi.org/10.1145/1143844.1143874), not fetched). So when one configuration's frontier dominates, the choice of figure of merit is irrelevant. The downstream goal has to decide only when the curves cross.

**Use decoys to estimate noise false positives.** MountainSort's noise overlap already uses random noise clips as a decoy cluster ([SpikeInterface docs](https://spikeinterface.readthedocs.io/en/latest/modules/qualitymetrics/nearest_neighbor.html)). A Kilosort decoy would need **frozen templates**: matching sign-flipped or time-reversed data against templates from the real sort. Re-running the full pipeline on −x would learn new templates and would not be a null. A decoy measures noise false positives only. Contamination by neighbouring neurons still needs refractory evidence or hybrid truth. The notes found no published whole-pipeline decoy run in spike sorting.

## The harness's S leans towards sensitivity, by design

### How S responds to errors

The harness's objective, S, is a sum over units of each unit's cross-validated, null-corrected Poisson log-likelihood gain from heading tuning. It is computed after the harness's own duplicate and artefact cleaning ([evaluation_spec.md](file:///C:/code/github/Kilosort/tuning/evaluation_spec.md)). This makes S an information-yield objective.

The harness's perturbation audit (5% error rates) shows which errors S punishes and which it ignores:

| Error | Effect on S (bits/s) |
|---|---|
| Merge of two differently tuned units | −63.6 |
| Contamination | −55.7 |
| Dropped units | −55.1 |
| Thinning | −51.6 |
| Duplicates (after cleaning) | −1.7 |
| Split | −0.0 |

The spec names two blind spots: merges of similarly tuned cells cost little, and false spikes that do not distort a tuning curve cost about nothing ([evaluation_spec.md §8.6, §12](file:///C:/code/github/Kilosort/tuning/evaluation_spec.md)). In the decoding literature's terms, S sits near the sensitivity-favouring end, which suits heading-information questions. It is not a licence for CCG or synchrony analyses, where cross-contamination is the main hazard.

### What the current comparison does and does not show

The current winner, H8, has `Th_universal` 7 and `Th_learned` 7 among its settings. Against the default it gains **ΔS +392.9**, with about 1695 eligible units against 772 and about 1004 reproducible units against 660. Each added unit contributes about 0.21 bits/s ([round4_results.md](file:///C:/code/github/Kilosort/tuning/round4_results.md)).

Against the lab's KS2.5, H8 is +423.6 over "standard" and +339.1 over "standard_low". standard_low ([9 4]) itself gains about 8–9% over standard by emitting about 15% more spikes ([ks25_comparison.md](file:///C:/code/github/Kilosort/tuning/ks25_comparison.md)). That gain shows KS2.5's presets are not at *its* optimum on S either. The current comparison therefore sets a swept, tuned KS4 against KS2.5 presets: points on two frontiers, not the frontiers themselves.

The same document reports that at equal unit counts "H8's units again decode worse". This fits the reading that part of H8's S gain is yield, not per-unit quality.

### Constraints on any new objective

**No new hybrid data.** From 2026-11-03 new sorts are scored without hybrid truth, and the hybrid excerpt remains only in earlier results ([tuning/README.md](file:///C:/code/github/Kilosort/tuning/README.md)). Any ground-truth term therefore has to live on the frozen excerpt.

**The FN side has no good truth-free estimator.** Truth-free FP estimation from refractory violations is the best-developed option:

- Vincent and Economo's estimator is unbiased at the population level: median FDR RMSE about 0.03 over 1,000 clusters ([eNeuro 2024](https://www.eneuro.org/content/11/8/ENEURO.0554-23.2024)).
- Per unit it is very noisy. The same ISI-violation rate means 5% FDR at 20 Hz but 50% at 3 Hz, and the authors warn against per-cluster inclusion rules.

The FN side, which is what sensitivity mostly changes, lacks a validated estimator. `amplitude_cutoff` is tied directly to the threshold being tuned, and drift breaks its symmetry assumption ([AllenSDK notebook](https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_quality_metrics.html)).

**Kilosort's internal labels cannot be the objective.** KS4's refractory metric (R12) drives both the 'good' labels and the split and merge decisions ([CCG.py](file:///C:/code/github/Kilosort/kilosort/CCG.py)), so the harness is right not to use those labels as an objective.

## Prioritised recommendations for the harness

The recommendations below follow from the evidence above. The specific numbers (q, κ, bin choices) are proposals from the notes, labelled as inference. No published study has validated them.

| Priority | Recommendation | Why (evidence) | How in this harness |
|---|---|---|---|
| 1 | **Define the objective as constrained yield**: maximise S, or Σ N(1−ĉ), subject to a stated contamination budget, estimated held-out at the population level. | Fields facing the same trade-off fix an estimable error rate and then maximise yield. Vincent & Economo's estimate is reliable for populations, not for single units. | Report S alongside held-out population FDR (Vincent & Economo, fixed τ, not KS R12). Gate on a budget such as median ĉ ≤ 0.1 (the 0.1 is an example value [inference]). Require ≥ 100 spikes per segment, as now, so low-rate units are not certified by chance. |
| 2 | **Compare KS4 and KS2.5 as frontiers, not points.** Sweep each version's own sensitivity knobs and plot S (or correctly assigned spikes) against estimated misassigned spikes or contamination. | Dominance makes the cost ratio irrelevant. Only crossing curves need a stated goal. KS2.5 presets are not at its S-optimum (standard_low beats standard by about 8–9%). | Sweep KS2.5 Th(1)/Th(2) (and `lam`, with `set_cutoff` on), and KS4 `Th_universal`/`Th_learned`. Use `tradeoff.py` curves on both. If the curves cross, compare at matched contamination. |
| 3 | **Tune the two KS4 thresholds for different jobs.** `Th_universal` controls unit discovery. Choose `Th_learned` by an explicit per-spike FDR. | Detection and assignment have different optima. Stock KS4 cannot remove spikes admitted at matching. The round 3 levers behave this way. | Using the saved per-spike scores (`th_amps`), estimate held-out FDR per score bin from refractory violations. Choose the loosest bin edge whose marginal FDR stays under q (local-FDR rule [inference]). This is the truth-free counterpart of set_cutoff, without the 2× in-sample optimism. |
| 4 | **Produce one sort with graded labels, and choose inclusion per analysis.** | Decoding optima sit above isolation optima. Correlation bias does not shrink with more data. | Keep core-versus-tail (by score) labels. Use all eligible units for heading and decoding analyses, and a strict contamination bar for place fields and cross-correlograms. Report S for both sets. |
| 5 | **Make S's purity price explicit.** | S barely penalises false spikes that leave tuning undistorted. | Report S − κ·(estimated FP spikes), with κ set from the perturbation audit (−55.7 bits/s per 5% contamination) [inference]. Alternatively, require ΔS to survive contamination-corrected re-scoring, as the strict-duplicate check does now. |
| 6 | **Guard against gaming the objective.** | Splits are free in S. Removing violating spikes lowers ISI violations without lowering FDR. Amplitude cutoff is tied to the threshold. | Deduplicate or merge fragments before counting yield (the seed-reproducibility rule already helps). Never use KS-internal acg/ccg labels or amplitude_cutoff as the objective. Flag units under about 2 Hz as uncertifiable. |
| 7 | **Use the frozen hybrid excerpt for calibration only.** | It is the only measure that cannot be gamed, but new hybrid data are off. Hybrid SNR must match the target data for the chosen point to transfer. | On the excerpt, report precision and recall separately by SNR band (<5, 5–10, >10), the count with agreement ≥ 0.8, and a cost-weighted J = Σ(TP − c_FP·FP − c_FN·FN) with c stated explicitly. Use these to check whether the truth-free FDR estimates are calibrated. |
| 8 | **Add a frozen-template decoy.** | Decoys estimate noise false positives without ground truth (astronomy, proteomics). | Match sign-flipped or time-reversed whitened data against frozen templates across `Th_learned`, and plot per-unit reliability R(Th). This covers noise false positives only, not contamination by neighbours. |
| 9 | **Exploratory only: per-unit Bayes thresholds.** | Theory says the threshold should scale with template norm and firing rate. No published Kilosort variant exists. | Prototype it as a post-matching cut using z* = s/2 + ln(p₀/p₁)/s, with empirically recalibrated costs. Judge it against a matched random-deletion control, as the set_cutoff port was. |

## Contradictions and unverified claims flagged across the notes

The notes disagree in several places, and some claims rest on sources that were not fully checked. The table sets these out so they can be resolved before quoting.

| Issue | What the notes say | Status |
|---|---|---|
| **Direction of the KS4 vs KS2.5 precision difference** (Buccino/AIND 2026) | Two notes report a *positive* KS4 precision effect size (0.272 NP1, 0.232 NP2) alongside the paper's sentence that KS4 has higher accuracy and recall "at the expense of a slight degradation of precision" ([eLife 110170](https://elifesciences.org/articles/110170)). | **Unresolved.** The fetched summaries could not reconcile the two. One possibility is that the statement refers only to the SNR-stratified analysis, but that is speculation. Check the figures before quoting either direction. |
| **KS4 paper benchmark type** | One note calls it "simulation-based, not hybrid", citing the paper's criticism of hybrid data. Another groups it with Buccino 2026 as a hybrid benchmark that injects a few units into real data. | The KS4 paper uses 600-unit simulations built from IBL recordings ([bioRxiv](https://www.biorxiv.org/content/10.1101/2023.01.07.523036v1.full)). "Hybrid-like simulation" is the safer description. The claim that native units go unscored applies to Buccino 2026, not clearly to KS4. |
| **KS4 matching window** | One note gives 0.2 ms, another ≤ 0.1 ms ([KS4 preprint](https://www.biorxiv.org/content/10.1101/2023.01.07.523036v1.full)). | Unresolved. 0.2 ms is Buccino 2026's window, so it may have been carried over by mistake. |
| **Todorova et al. 2014** | One note: unsorted crossings beat expert-sorted neurons on 63% of trials. Another: sorted units beat raw crossings. | These are compatible. Expert sorting that discards hash is worst (monkey B RMSE 0.311). Raw crossings are in between (0.255). Sorting that keeps hash is best (0.218) ([PMC4454741](https://pmc.ncbi.nlm.nih.gov/articles/PMC4454741)). |
| **False positives vs misses** | Goodman & Johnson: false positives are most damaging. Pazienti & Grün: misses cost more synchrony significance. | Different objectives, not a contradiction. Pazienti & Grün is abstract-level only. |
| **False-positive units in Kilosort** | SpikeInterface/SpikeForest: KS2 produces many false-positive units. Laquitaine: KS3/KS4 produced none. | These cover different versions and data. Laquitaine is a bioRxiv preprint. |
| **Mapping of `Th_learned` to a z-statistic** | Confirmed from this branch's code. The KS4 paper text says matching uses "the average norm of the template". | Could differ upstream or describe another variant. |
| **Not fetched or unverified** | Lewicki 1994/98, Shoham 2003, Ekanadham 2014, Kay/Van Trees textbooks, Elias & Gygi, Davis & Goadrich, Hand 2009, Lipton 2014 (cited from memory in one note), Wave_clus's 4σ justification (PDF failed), the IBL white paper's [6, 3] rationale, the authorship of the IBL sliding-RP paper, the eLife e61834 claim about Kilosort2's low-SNR robustness (search summary), Yger 2018's "~50 µV" figure, and the Barnett et al. sign-flip claim. | Treat their details as unverified. |
| **Inference-only results** | All Bayes threshold numbers, the Gaussian false-alarm arithmetic, the accuracy and KS4-score marginal rules, the R12-to-contamination mapping (R12 < 0.2 ≈ c ≲ 10–11%), and every proposed objective (constrained yield, κ penalty, decoy FDR, IDR analogue). | Researchers' own derivations. Check them before publishing. |

## Conclusion

The sensitivity question splits into two parts, and the harness can answer both. The first part asks whether KS4 *moves the frontier outward* relative to KS2.5 — more correctly assigned information at every contamination level. That needs no cost ratio and is fully decidable with swept curves. It is a better question than whether H8 beats a KS2.5 preset, which mixes the frontier with the operating point. The second part asks *where on the frontier to sit*, and that is a statement of downstream priorities. Theory then gives a clean rule: lower the threshold until the newly admitted spikes are false at the rate your stated cost ratio tolerates.

The most transferable idea from the literature is the local-FDR form of that rule, applied to the matching scores KS4 already saves: estimate, held out, how contaminated each next score bin is, and stop where it crosses the budget. This turns `Th_learned` from a number tuned by eye into a stated error rate. Two gaps remain open. No validated truth-free estimator of missed spikes exists. And nothing yet establishes what contamination level hippocampal tuning and correlation analyses can tolerate. Until those are settled, report S together with an explicit contamination curve rather than S alone.
