# Interleaving extraction, clustering and template refinement: literature and other sorters (2026-11-04)

Lens: what published work and other sorters' code show about the proposal to interleave extraction, clustering
and template refinement in small batches in KS4. The proposal is treated as a hypothesis.

**Status marks.** **[R]** I read the text or code myself. **[S]** opened, but read through the fetch tool's
summariser, so details could carry its errors; check before quoting. **[A]** abstract or search snippet only.
**[U]** not opened. Raw source files were pulled one at a time with `curl` into the scratchpad to read; no repo
was cloned and nothing was run. The one verbatim quote below is kept short; the rest is paraphrase.

## 1. Bottom line

- **Nobody has shown that fine interleaving beats rounds.** I found no head-to-head on any data, and no study
  of iteration count at all in a benchmark (SpikeForest, Garcia et al. 2026, KS4 paper).
- **Coarse iteration has some support, usually about two rounds.** Marre 2012, YASS, the 2023 DARTsort
  preprint, Combinato, Laboy-Juárez 2019 and KS3/4 itself do this. The evidence is mostly qualitative or
  indirect. The user's table (`round3_results.md`) is the largest effect size I found for a further
  whole-recording round, and it is on real data without ground truth.
- **The trend in the leading sorters runs against the proposal.** KS3/4 moved away from KS2.5's interleaved
  learning. The three SpikeInterface sorters are single-pass. DARTsort's default has been one matching step
  since at least Feb 2024, and it fits its model on a subset, not by streaming.
- **KS2.5 is not a precedent for interleaved clustering.** It interleaves *matching* with a running mean and
  with births from the residual. It has no clustering step in the loop (section 3).
- **The stated motive may not be the main mechanism.** The literature that finds missed waveforms looks in the
  residual. The local orphans are lumped with a neighbour on the same channel, so their spikes are caught and
  subtracted, and the residual holds nothing (`borrowed_templates.md`). Augment (adding the missing templates)
  gave nothing on the curves, while repeat (rebuilding the dictionary) did. Round 3's gain is mostly cleaner
  units. Marre's within-template reclustering matches this mechanism, and KS4's second clustering already does
  something like it.
- **Clustering needs far more spikes than a batch holds.** Published chunks are 10 s to 5 min, or 1,000-20,000
  spikes (section 5). A 2-s batch holds about 11 spikes of a mean-rate unit (section 7).

## 2. Comparison table

"Rounds" means matching → clustering/template update → matching.

| Sorter | Loop between matching and clustering/templates? | Granularity | How missing waveforms are found | Evidence that iteration helps | Mark |
|---|---|---|---|---|---|
| Kilosort 1 (Pachitariu 2016) | Yes: template optimisation alternates with matching, annealed | Mini-batch running average, then a final pass | K-means seeds; none from residual | Online nature credited with speed and with avoiding local minima. No ablation. Full matching during learning was found unnecessary | R |
| KS2, 2.5 | Yes: per-batch matching and running-mean update; **no clustering in the loop** | One pass of ~2-s batches; τ ≈ 400 spikes | **Residual detection every batch** (`mexGetSpikes2` on `drez`) with `spkTh` −6, births capped at `Nfilt`; triage every 5 batches | None for the interleave itself. Pachitariu: number of passes didn't matter (#84) | R (code), S (preprint) |
| KS3 | Learn, match, then recursive pursuit on features | Whole recording, two clusterings | Templates only detect; clustering decides units | Attributed at version level | S |
| KS4 | Serial: universal detect → cluster → match → cluster | Whole recording, 2 clusterings | Assertion only: a neuron is very unlikely to be missed entirely; one template may serve several neurons | Attributes its gain to clustering, not to rounds | S; code R |
| YASS (Lee 2020) | Yes: cluster clean spikes → deconvolve → re-sort cleaned waveforms without triage → final deconvolution | 2 sort rounds; templates **rescaled per 5-min batch**, forward then backward pass | Triage then deconvolution; split step adds intermittently active units; backward pass assigns spikes for units first seen mid-recording | Only the inner coordinate-descent step is quantified (few % more true positives, up to 50% fewer false positives). The re-sort isn't isolated. Detecting on the residual and decoding with those spikes didn't improve decoding | S |
| DARTsort preprint (2023) | Yes: split-merge ↔ drift-aware matching, "until budget" | Whole-recording rounds, count not stated | Greedy NN-denoised subtraction over falling thresholds; early oversplitting so matching sees more cells | Fig. 7C trends only: precision rises each stage, recall rises at merges and matching, falls at splits. 75% vs KS2.5's 57% at highest noise is the whole pipeline | S |
| DARTsort `main` (2026) | Optional: `matching_iterations` default **1**; `recluster_after_matching` default False | Subset (≥ 5,000 spikes/channel, ≥ 2.5% of recording) for detect + model; intermediate rounds on subset; **final round full** | Subtraction; mixture split on spikes, with held-out data | No published result; README says not for production use | R (code) |
| SpyKING CIRCUS 2 / Tridesclous 2 / Lupin | **No loop**: one clustering, one matching | Peaks subset, whole recording | Matched-filter detection from a prototype (SC2, Lupin). TDC peeler re-detects in residual within matching | Lupin wins on simulations with no loop. Authors note loops are possible but untested | R (code), S (paper) |
| SpyKING CIRCUS 1 (Yger 2018) | No matching↔clustering loop; `nb_repeats` passes refine the density estimate | Subset | Optional garbage templates collect unfitted spikes | None on iteration | S (docs), A (paper) |
| MountainSort 5 | No loop; train on subset, classify rest. Scheme 3: blocks linked by classifiers | Blocks | Phase 3 re-detects with a narrower radius to catch overlaps | Claims only | S |
| Tridesclous 1 | No: catalogue from subset, then peeler | Subset then whole | Docs: a good catalogue leaves residual under threshold | None | A |
| Marre 2012 | **Yes, one extra round**: recluster each template's spikes to recover missed cells, refit with expanded set | Whole recording | Within-template reclustering | 228 of 242 labelled cells; ~20% of templates still needed manual splits | S |
| Segev 2004 | No: clean-spike templates, then greedy subtraction | Whole recording | Residual peeling within a matching step | Exhaustive combinations gave the same choices; no learning loop | R |
| Combinato (Niediek 2016) | Yes: cluster, match unassigned, repeat `N_rep` times; large clusters reclustered | Blocks of 20,000 spikes, single wire | Unassigned spikes reclustered | `N_rep` 1 is default, more judged unnecessary, yet 2 → 1 lowered hits in all 24 comparisons | S |
| FAST (Dhawale 2017) | Rounds of clustering (no matching): low-density spikes re-pooled, 4 rounds | Blocks of 1,000 spikes, tetrodes | Sparse units surface once dense ones are removed | Comparable to BEER on paired data; better over days | S |
| SpikeSift (2025) | **Yes, fine**: cluster one template → match → subtract → repeat | Segments ≥ 10 s, linked across | Residual reveals masked small spikes | Speed mainly; no comparison with rounds; preprint | S |
| Laboy-Juárez 2019 | Sort → normalised template matching → recluster | Tetrode groups, whole recording | Unit-specific matching threshold | Simulated detection ~70% → ~90%; real data more spikes per unit, quality unchanged | S |
| Binary pursuit (Pillow 2013) | Designed as coordinate ascent, **stopped after one round** on cost and good performance | 1-s blocks | From clustering | Corrects synchrony artefacts; one round sufficed | R |
| CBP (Ekanadham 2014) | Yes: alternates spikes and waveform refit | Whole | Cell count chosen by hand | Waveform refit gave minor improvement; not significant in simulations | S |
| OSort (Rutishauser 2006) | Online: new cluster if distance large, merge when close | Per spike, single wire | Birth by distance | Authors warn merges are irreversible and can converge wrongly | R |
| Franke 2010 / BOTM 2015 | Templates refit online (last 350 spikes, every 5 s) | Rolling | **Not handled**; proposed running a detector in parallel and clustering unmatched spikes | Good on small data; no ground truth for monkey data | S / A |
| OPASS (Carlson 2013) | Online DP mixture on residual | Per sample | New neuron drawn in the DP | Beat prior art on partial ground truth, small channel counts | R (part) |
| GPUDPM (Gasthaus 2009) | Particle filter, births and deaths | Per spike, tetrode | DP prior | Two 4-min recordings against a pooled DPM: false negatives 1.3% vs 4.2% on one, 18.6% vs 9.4% on the other; no refractory violations | R |
| RT-Sort (van der Molen 2024) | **Frozen** after a pre-recording | 40 s to 5 min, then fixed | None online; updating listed as future work | Precise, but fewest units on Neuropixels (168) | S |
| JRCLUST, IronClust, HerdingSpikes, Klusta | No matching loop | Whole, or ~20-s drift chunks (IronClust) | n/a | None | A |
| MoDT (Shan 2017) | EM, full data; template matching optional after | 1-min frames | Human split/merge | None on iteration | S |

Links are in section 9.

## 3. What KS2.5's interleaving actually is

From `mainLoop/learnTemplates.m` (tag v2.5) and the 2023 preprint:

- Each batch: match with current templates, update a running mean with an effective memory rising to τ ≈ 400
  spikes, detect in the residual (`spkTh` overwritten to −6), add new templates up to `ops.Nfilt`.
- Triage every 5 batches: drop low rate, or near-duplicates (correlation > 0.9 with close means). The merge
  block in `triageTemplates2.m` is commented out.
- It is EM-style inference plus a mean. A template is assumed to be a neuron, since few splits and merges follow.
- The preprint's stated limits: τ means fast drift isn't tracked well, low-rate units are hard below about a
  minute, and step drift fails. My reading: these matter less once drift is corrected beforehand, as in KS4.
- Pachitariu on issue #84 called the first pass an initialisation, said the algorithm forgets quickly so that
  only the latest few hundred spikes per neuron matter, and said of his data that
  "it didn't matter how many back and forth passes we did".

So its fast convergence comes from a simple model (a mean per template), not from interleaving clustering. A
per-batch rebuild of KS4's dictionary needs a clustering, which KS2.5 doesn't have. KS4's templates also have
no one-to-one link to units, so a running mean per template estimates something different.

## 4. What was abandoned, and why

| What | By whom | Stated reason |
|---|---|---|
| Online template learning as the source of units | KS3/4 | Templates relaxed to detection only; clustering refined; the paper attributes the gain to clustering. KS1 bias to over-splitting noted. No ablation of the online step itself |
| Full matching during learning | KS1 | Similar templates without it, so run in full only on the final pass |
| Multiple rounds | Pillow 2013 | Cost, and one round was good enough |
| Waveform refit | CBP | Little gain for well-isolated cells |
| Reclustering after first matching, default | DARTsort | False Jul 2025, True Jan-Apr 2026, False again by Jul 2026. Nothing explains it, so it says little either way. From 12 sampled commits, so dates are approximate |
| Whole-recording alternation, default | DARTsort | `matching_iterations` is 1 at every sampled commit from Feb 2024. It may never have been the default; the preprint's count isn't stated |
| Direct detection of collision templates | YASS | Timing jitter blurs the mean collided waveform |

## 5. Scale at which others cluster

| Work | Unit of clustering | Note |
|---|---|---|
| FAST | 1,000 spikes per block, tetrode | Needs linking across blocks by integer programme |
| Combinato | 20,000 spikes per block | Single wire |
| YASS | 5 min per template batch | Rescaling, not reclustering |
| SpikeSift | ≥ 10 s segments | Small arrays; linking needed; authors suggest sorting a reference segment first |
| MoDT | 1-min frames | Tied by a drift prior; EM on all data |
| DARTsort | ~1.9 M spikes for 384 channels | About 3 min of kobe at 12 M spikes per 20 min |
| KS2.5 | ~400 spikes per template, by mean | No clustering |

General clustering results point the same way. Mini-batch k-means is orders of magnitude faster than batch but
ends at a slightly worse objective (Sculley 2010, [A]). Hughes & Sudderth 2013 [R, part] say stochastic online
fits are sensitive to the learning rate and often reach poor local optima.

**Hughes & Sudderth's remedy is the best fit to the proposal.** They keep per-batch sufficient statistics
(memoised), which guarantees every step improves the full-data objective, unlike stochastic online methods.
Births are learned on a targeted subsample of data poorly explained by one component, adopted over one pass,
then merged away if redundant. This is a method for mixture models on image data, with no spike-sorting
validation that I found.

## 6. Is the first-pass missing-waveform problem real, and how big?

No dedicated study found.

- **KS4 paper:** about half of spikes go undetected at the universal stage, which the authors accept, and they
  argue it is very unlikely that all of a neuron's spikes are missed. Pachitariu made the same point on #809
  for the single-channel waveforms step. Neither gives a measurement.
- **Garcia 2026:** the best static clustering found 420 of 500 units, failures mostly at small SNR. They say it
  is unclear whether missing spikes at detection matters much. Templates estimated from a small subset had no
  clear impact on matching (Fig. 6, static). No incomplete-dictionary experiment.
- **Local:** 12-15% of eligible units are orphans, estimated recall gain 0.1-0.4% of spikes
  (`borrowed_templates.md`). Augment gained about half of repeat in S and nothing on the curves.

GitHub (MouseLand/Kilosort): about 15 search-API queries found no report of anyone running a further round or
re-sorting a residual. Nearest are #860 (reusing templates: Pennington says templates and clusters are separate
and it would need substantial changes), #742 (Pennington: a template learned from another's residual, minor
when rare), #809, #281 (KS2.5 missing spikes against KS2).

## 7. Reading the proposal against this

- **For:** repeat rounds help locally; all the 2-round sorters agree in direction; Combinato shows hits fall
  when a round is removed. Convergence by round looks fast locally (round 4 adds about a third of round 3 on
  the curves), which is what a coarser, cheaper schedule could exploit.
- **Against:** no direct evidence. The H8 gain comes with a dictionary that shrinks by about 25-30% when
  rebuilt, while default KS4 gains about 40% as much on the curves and loses S on MEC, so any scheme needs
  testing on both settings. Online schemes add path dependence: OSort warns of irreversible merges, and the
  first batches' templates steer later ones. Rounds already replace about as many units as a change of seed;
  an online path might not reduce that.
- **Spike budget (my arithmetic, kobe):** at the mean 5.5 Hz a 2-s batch holds ~11 spikes per unit, and at
  the 2 Hz typical of units lost between rounds ~4. Gathering KS2's 400 spikes takes 36-100 batches.
- **Interleaving would need new machinery:** a buffer, a schedule, cross-batch unit identity and a stopping
  rule, each with its own failure modes (Sculley, OSort, the KS2 limits above).

## 8. Ideas most worth taking

1. **Rounds on a subset, then a full final round** (DARTsort: `stop_after_n_spikes`, `ensure_coverage`, subsampling
   off on `is_final`). This gets the speed the proposal wants with no batch-level clustering. On kobe ~1.9 M spikes
   is about a sixth of a 12 M round, if cost scales with spikes. Spread batches over the recording. Risk: low-rate
   units (0.5 Hz gives ~100 spikes), which must be tested. **First thing to try.**
2. **Memoised statistics instead of forgetting** (Hughes & Sudderth). If interleaving is built at all, keep sums per
   template per batch and rebuild from them, with births learned on a buffered subsample of poorly fit spikes and
   adopted at intervals. Don't cluster per batch.
3. **Search within templates rather than the residual** (Marre 2012; also FAST's re-pooling of low-density spikes).
   This matches the local orphans, whose spikes are caught by a shared template. Cheap version: rerun clustering
   only for templates with bad fit or bimodality.
4. **Held-out checks on splits and births** (DARTsort `demolish` destroys units with bad generalisation; validation
   proportion 0.5). Useful with any scheme, to guard against the path dependence above.

Also possible, but a larger change: a mixture classifier for the final assignment, as DARTsort does when subsampling.

## 9. Corrections to the brief, gaps, sources

**Corrections.**
- OPASS is Carlson, Rao, Vogelstein & Carin (NIPS 2013), not Pachitariu.
- Binary pursuit is Pillow 2013; Ekanadham 2014 is continuous basis pursuit.
- DARTsort's current EM is not minibatch in the stochastic sense. `gmm_batch_size` batches the E-step, statistics are
  combined over all spikes, and there is one M-step per pass (`mixture.py`, `em`, `e_step`). It runs on a stored
  subset, not between matching passes.
- No file named for `iterative_peeling` exists in the current SpikeInterface tree. The nearest are
  `IterativePeakDetector` (detect, denoise, subtract; `num_iterations` default 2; not in `detect_peak_methods`,
  and I didn't see it in Lupin's chain) and TDC peeler's `max_peeler_loop`.
- The DARTsort preprint doesn't mention online or minibatch work; the GMM is in code only. No newer paper found.

**Gaps.** Not opened: IronClust paper, HerdingSpikes, Klusta (snippets only), Gasthaus's successors, Carlson's joint
dictionary paper, Steinmetz 2021. Scanned only part of RT-Sort, Marre, Garcia, FAST and the eLife peer-review text.
GitHub issue search is keyword based, so a report phrased differently could exist. Sampling DARTsort commits can miss
brief changes in defaults. Absence of a comparison is not evidence that interleaving fails.

**Sources.**
- Kilosort: [KS1 paper](https://proceedings.neurips.cc/paper_files/paper/2016/file/1145a30ff80745b56fb0cecf65305017-Paper.pdf); [2023 preprint](https://www.biorxiv.org/content/10.1101/2023.01.07.523036v1.full); [KS4 paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11093732); [learnTemplates.m](https://github.com/MouseLand/Kilosort/blob/v2.5/mainLoop/learnTemplates.m); [triageTemplates2.m](https://github.com/MouseLand/Kilosort/blob/v2.5/mainLoop/triageTemplates2.m); issues [#84](https://github.com/MouseLand/Kilosort/issues/84), [#281](https://github.com/MouseLand/Kilosort/issues/281), [#742](https://github.com/MouseLand/Kilosort/issues/742), [#809](https://github.com/MouseLand/Kilosort/pull/809), [#860](https://github.com/MouseLand/Kilosort/issues/860).
- DARTsort: [preprint](https://www.biorxiv.org/content/10.1101/2023.08.11.553023v1.full); [repo](https://github.com/cwindolf/dartsort) (`src/dartsort/main.py`, `config.py`, `util/internal_config.py`, `util/main_util.py`, `clustering/mixture.py`).
- YASS: [Lee et al. 2020](https://www.biorxiv.org/content/10.1101/2020.03.18.997924v1.full).
- SpikeInterface: [Garcia et al. reviewed preprint](https://elifesciences.org/reviewed-preprints/110588v2); [eLife version](https://elifesciences.org/articles/110588); [lupin.py](https://github.com/SpikeInterface/spikeinterface/blob/main/src/spikeinterface/sorters/internal/lupin.py), [tridesclous2.py](https://github.com/SpikeInterface/spikeinterface/blob/main/src/spikeinterface/sorters/internal/tridesclous2.py), [spyking_circus2.py](https://github.com/SpikeInterface/spikeinterface/blob/main/src/spikeinterface/sorters/internal/spyking_circus2.py), [iterative.py](https://github.com/SpikeInterface/spikeinterface/blob/main/src/spikeinterface/sortingcomponents/peak_detection/iterative.py), [tdc_peeler.py](https://github.com/SpikeInterface/spikeinterface/blob/main/src/spikeinterface/sortingcomponents/matching/tdc_peeler.py).
- Others: [MountainSort 5](https://github.com/flatironinstitute/mountainsort5); [SpyKING CIRCUS config](https://spyking-circus.readthedocs.io/en/0.9.4/code/config.html); [Yger 2018](https://elifesciences.org/articles/34518v1/peer-reviews); [Tridesclous docs](https://tridesclous.readthedocs.io/en/latest/overview.html); [Marre 2012](https://www.jneurosci.org/content/32/43/14859); [Segev 2004](https://www.nature.com/articles/nn1323); [Combinato](https://pmc.ncbi.nlm.nih.gov/articles/PMC5145161); [FAST](https://pmc.ncbi.nlm.nih.gov/articles/PMC5619984); [SpikeSift](https://arxiv.org/html/2504.01604v1); [Laboy-Juárez 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6700190/); [Pillow 2013](https://www.cns.nyu.edu/pub/lcv/pillow12.pdf); [Ekanadham 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4075282); [OSort](https://arxiv.org/abs/q-bio/0604033); [Franke 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2950077); [BOTM](https://pmc.ncbi.nlm.nih.gov/articles/PMC4420847); [OPASS](https://proceedings.neurips.cc/paper/2013/hash/944bdd9636749a0801c39b6e449dbedc-Abstract.html); [Gasthaus 2009](https://papers.nips.cc/paper/3606-dependent-dirichlet-process-spike-sorting); [RT-Sort](https://pmc.ncbi.nlm.nih.gov/articles/PMC11620616/); [JRCLUST](https://www.biorxiv.org/content/10.1101/101030v2.full); [IronClust](https://github.com/jamesjun/ironclust); [MoDT](https://www.biorxiv.org/content/10.1101/109850v1.full); [SpikeForest](https://elifesciences.org/articles/55167/figures).
- General: [Hughes & Sudderth 2013](https://proceedings.neurips.cc/paper/2013/hash/d490d7b4576290fa60eb31b5fc917ad1-Abstract.html); [Sculley 2010](https://ccs.neu.edu/home/vip/teach/DMcourse/2_cluster_EM_mixt/notes_slides/sculey_webscale_kmeans_approx.pdf).
- Local: `.claude/worktrees/ks4-round3/tuning/exploratory/iteration/round3_results.md`, `borrowed_templates.md`; `tuning/exploratory/modularity/modular_pipeline_audit.md`; `reports/Kilosort4 quality evidence since 2024.md`.
