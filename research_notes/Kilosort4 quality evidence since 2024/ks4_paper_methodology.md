# Kilosort4 paper (Pachitariu, Sridhar, Pennington & Stringer 2024, Nat Methods 21:914-921): evaluation methodology, confounds, and critiques

Source conventions. "PDF p.N" means the journal page number printed on the local PDF (C:\Users\richa\Downloads\s41592-024-02232-7.pdf; pp. 914-921 are the main text, the Methods and Extended Data follow on unnumbered pages). The paper is open access (CC-BY): [Nature article](https://doi.org/10.1038/s41592-024-02232-7); [PMC11093732](https://pmc.ncbi.nlm.nih.gov/articles/PMC11093732/). Received 16 Jan 2023, accepted 1 Mar 2024, published 8 Apr 2024 (PDF p.914). Statements under "Cited Findings" that start with "Paper states" are taken directly from the PDF. Everything under "Inferences" is my interpretation.

## Q1. How were the benchmarks built (hybrid, full and drift simulations), which sorter's output supplied the injected units, and was any ground truth used that was not derived from Kilosort?

### Takeaway
The paper has three kinds of benchmark, and all of their ground-truth units come from Kilosort sorts of IBL Neuropixels 1.0 data:
- The no-drift hybrid (Fig. 3a-c) uses units found by **Kilosort4 itself** (default parameters, contamination < 0.1). Their average waveforms were re-injected into the same recording, 8 sites above or below where they were found.
- The drift and "full" simulations (Fig. 3d-e, Fig. 4, Fig. 5) use a waveform bank of 597 units built from spikes found by **pyKilosort** (the IBL's Python port of Kilosort2.5-era code) on 11 high-drift IBL recordings. These are placed over synthetic noise plus 600 "multi-unit" waveforms.

No paired (juxtacellular or patch) recording and no simulator independent of Kilosort was used to compare KS4 with KS2.5. The only non-Kilosort ground truth is the SpikeInterface/MEArec biophysical simulation, and it was used only to argue that Kilosort2 false positives there are an artefact (Extended Data Fig. 3). KS2.5 was not compared with KS4 on it.

### Cited Findings
**Hybrid ground truth, no drift (Fig. 3a-c; PDF p.916-917; Methods "Hybrid simulation")**
- Paper states it chose three IBL recordings "which specifically had very low levels of drift, as estimated by Kilosort2.5", from different labs and brain areas (Mainen, Angelaki, Carandini/Harris). Their IDs are 3f6e25ae-..., fe380793-... and 4ddb8a95-... — [Kilosort4 paper, p.916 and Methods](https://doi.org/10.1038/s41592-024-02232-7)
- Paper states (Methods): "for each recording, we ran Kilosort4 with default parameters to get spike times for extracting waveforms, and then re-inserted these waveforms into the same recording at different positions. We only used waveforms from units with a contamination ratio less than 0.1." Average waveforms were computed in the raw recording from the KS4 spike times. — [Kilosort4 paper, Methods](https://doi.org/10.1038/s41592-024-02232-7)
- Paper states that 100 such waveforms per recording were chosen at random and added "at either eight sites above or eight sites below the original position". — [Kilosort4 paper, Methods](https://doi.org/10.1038/s41592-024-02232-7)
- Spike trains in the hybrid:
  - Generated with exponential ISIs, mean ISI about 500 ms (about 2 Hz).
  - Modulated by the recording's local firing rate: 100-ms bins per channel, Gaussian-smoothed across channels with an s.d. of 10 channels, normalised to mean 1, and the ISI divided by it.
  - Minimum ISI 2 ms.
  — [Kilosort4 paper, Methods](https://doi.org/10.1038/s41592-024-02232-7)
- Fig. 3c pools 300 ground-truth units (3 × 100) and plots sorted 1-FP-FN scores for 10 sorters. The text gives no numeric table for this figure. — [Kilosort4 paper, Fig. 3 legend](https://doi.org/10.1038/s41592-024-02232-7)
- Paper states: "All Kilosort versions except Kilosort1 outperformed all other algorithms, with Kilosort4 performing the best (Fig. 3c)." (p.917) — [Kilosort4 paper](https://doi.org/10.1038/s41592-024-02232-7)
- Methods, "Comparison to other benchmarking approaches": the authors themselves note that hybrid datasets "depend on finding the neurons in the first place and they also depend on correcting for the initial drift of the dataset". The same passage describes hybrid insertion "in a different position after being subtracted off". The Methods "Hybrid simulation" paragraph for their own hybrid does not say that the original units were subtracted. — [Kilosort4 paper, Methods](https://doi.org/10.1038/s41592-024-02232-7)

**Drift simulations and the "full simulation" (Fig. 3d-e, Fig. 4; Methods "Drift simulation")**
- Paper states that the drift statistics came from 512 IBL Neuropixels 1.0 recordings "processed by the IBL using pyKilosort". pyKilosort returns depth estimates per batch at 9 positions along the probe, with a batch size of 65,536 samples. The drift range was defined as the 5th-95th percentile of the median drift, and the median across IBL recordings was 9.5 µm (Fig. 4b). — [Kilosort4 paper, Methods and Fig. 4b](https://doi.org/10.1038/s41592-024-02232-7)
- Waveform bank:
  - 11 IBL recordings whose drift spanned at least 40 µm. "We then used the estimated spike times from pyKilosort for each detected unit in these recordings and the estimated depth of the probe to compute the average waveform for the unit at specified depth positions."
  - 20 depth bins of 2 µm; units with fewer than 50 spikes at any depth were excluded.
  - Waveforms were denoised to their top 3 PCs and normalised.
  - Units whose waveform changed too much between -20 µm and +20 µm (shifted by 4 channels; variation > 0.25, about 25% of units) were discarded.
  - The final bank had 597 units.
  — [Kilosort4 paper, Methods "Extraction of waveforms at multiple depths"](https://doi.org/10.1038/s41592-024-02232-7)
- The waveforms were upsampled 100× by kriging (regularisation 0.01, Gaussian s.d. 20 µm). Units with contamination < 0.1 (from their units' estimated spike trains) became "single-units", and those above 0.1 became "multi-units". — [Kilosort4 paper, Methods](https://doi.org/10.1038/s41592-024-02232-7)
- Each simulation has 600 single units and 600 multi-units, placed at random between site 4 and site 380, keeping the best channel modulo 4.
  - Single-unit spike trains are shuffled ISIs from 1,497 detected units with contamination < 0.1 (mean rate 12.6 Hz).
  - Multi-units are Poisson.
  - Single-unit norms are 10 + Exp(mean 7); multi-unit norms are Uniform(4, 10).
  - Flat-spectrum noise up to 300 Hz (s.d. 0.76) was added, then the data were "unwhitened" with the inverse of a whitening matrix from one of the 11 recordings, scaled by 200 and saved as int16.
  — [Kilosort4 paper, Methods "Simulation of spikes" and "Simulation noise"](https://doi.org/10.1038/s41592-024-02232-7)
- Drift profiles (45-min simulations; drift held for 2 s except in the fast-drift case, where it changes every 200 ms):
  - **No drift.**
  - **Medium:** smoothed Gaussian noise, σ = 100 s, per-position factor 0.4, rescaled to ±7 µm, drift range 9.4 µm.
  - **High:** per-position factor 0.26, ±18.5 µm, range 27.9 µm.
  - **Fast:** medium drift plus 300 events of 10 µm with an 80-ms rise and 200-ms decay.
  - **Step:** factor 0.58, ±4 µm, plus a 30 µm jump halfway through the recording.
  - **Step, aligned:** the same drift as step, with waveforms re-interpolated onto aligned sites (20 µm vertical, 32 µm horizontal, NP2-like).
  — [Kilosort4 paper, Methods "Drift simulation"](https://doi.org/10.1038/s41592-024-02232-7)
- Paper states that the "full simulation without drift" (Fig. 3d-e, 600 GT units) uses the same machinery: real waveforms over 1/f-like noise plus MUA. "We found similar performance for all algorithms… with Kilosort4 outperforming all other methods." — [Kilosort4 paper, p.917](https://doi.org/10.1038/s41592-024-02232-7)
- Paper states that the drift-estimation step "from Kilosort2.5 (in the Kilosort4 implementation)" recovered the simulated drift well in all cases except fast drift, which is faster than the 2-s bins (Extended Data Fig. 6). — [Kilosort4 paper, p.918](https://doi.org/10.1038/s41592-024-02232-7)

**Ground truth not derived from Kilosort**
- Paired recordings were not used. Methods argue that paired ground-truth datasets "were acquired in anesthetized animals and are very short", which makes them "much easier to spike sort". They cite SpikeForest's PAIRED results, where IronClust, Kilosort2 and SpyKING CIRCUS "performed similarly", as consistent with the authors' own no-drift results. — [Kilosort4 paper, Methods "Comparison to other benchmarking approaches"](https://doi.org/10.1038/s41592-024-02232-7)
- Biophysical simulation (SpikeInterface paper, DANDI 000028):
  - Used only to argue that its waveforms are unrealistically long and wide (trough-to-peak time and spatial-spread comparisons against KS4 "good" units from a real recording).
  - The authors show that Kilosort2's false positives there drop when the simulation is sped up 2× (Extended Data Fig. 3f-h).
  - KS4 and KS2.5 were not benchmarked on it.
  — [Kilosort4 paper, p.917, Methods "Biophysical simulation", ED Fig. 3](https://doi.org/10.1038/s41592-024-02232-7)
- Non-Neuropixels data (64-channel linear probe, DANDI 000231; tetrodes, DANDI 000410) are shown only qualitatively (Extended Data Fig. 2), with no ground truth. — [Kilosort4 paper, Discussion and ED Fig. 2](https://doi.org/10.1038/s41592-024-02232-7)

**Preprint versus published version**
- The bioRxiv preprint "Solving the spike sorting problem with Kilosort" (10.1101/2023.01.07.523036, v1 posted 7 Jan 2023), as summarised from the full text:
  - Contains the drift simulations (600 GT units plus 600 MUA, five drift conditions plus aligned step, score > 0.8 counts, Kilosort 1-4 versus IronClust, SpyKING CIRCUS, MountainSort4 and others).
  - Contains no hybrid ground-truth benchmark, no systematic ablation, and no separate false-positive-unit analysis.
  - As read, it does not name the sorter whose output supplied the ground-truth waveforms.
  — [bioRxiv preprint v1](https://www.biorxiv.org/content/10.1101/2023.01.07.523036v1.full)

### Inferences
- **Circularity in the hybrid test (Fig. 3a-c) is direct.** The injected units are exactly the units KS4 (default) isolated cleanly, with contamination measured by Kilosort's own ACG metric. Their waveforms are averages over KS4's spike times. So the ground truth is a set of neurons whose waveform, footprint and amplitude KS4's clustering is known to handle. Units that KS2.5 could isolate but KS4 could not are absent by construction. Units KS4 over-merged or split also produce templates that are biased toward KS4's partition. This bias favours KS4 over KS2.5 specifically. The paper does not discuss it, although it does note the general point that hybrids "depend on finding the neurons in the first place".
- **A further consequence of the hybrid design.** If the originals were not subtracted (the Methods paragraph does not say they were), each hybrid unit is a copy of a real neuron shifted by 8 sites (4 rows, about 80 µm on NP1). The copy fires an independent synthetic train and sits over a real background that still contains the original neuron. This tests detection of a known waveform shape at a new location. It does not test the sorter's ability to discover unknown neurons.
- **The drift and full simulations favour template-based Kilosort-family sorters as a group, not KS4 over KS2.5.**
  - The waveform bank comes from pyKilosort (KS2.5-lineage) sorts.
  - Units were filtered by Kilosort contamination, required at least 50 spikes at every 2-µm depth bin, and excluded if their shape changed non-geometrically with depth.
  - Drift was generated as rigid plus smoothly varying nonrigid displacement on the NP1 geometry, which is the model Kilosort2.5/3/4 drift correction assumes. The "Kilosort2.5 drift estimation" step then recovers it well (ED Fig. 6).
  - Within the Kilosort family, KS2.5/3/4 share preprocessing and drift correction (Fig. 4 text), so the KS4-versus-KS2.5 differences in Fig. 4 isolate clustering and postprocessing differences on data generated from a model those front-ends match.
- **The preprint-to-publication change is suggestive.** The hybrid benchmark (Fig. 3a-c), ablations (Fig. 5) and false-positive analysis (Fig. 4k) appear only in the published version, which suggests they were added during review. There is no public peer review file to confirm this (see Q3).

### Gaps
- The Methods do not say whether the original units were subtracted before re-insertion, or whether the shifted copies could overlap other real units.
- The paper gives no per-sorter numbers for Fig. 3c or 3e, only plots. Source data may exist on the Nature site but could not be retrieved, because Nature and PMC pages were blocked by a login redirect or CAPTCHA.
- No exact KS4 version or commit is given for the benchmarks, beyond "latest versions as of December 2022" for the other sorters.

## Q2. Which parameters did each sorter use, which metrics were reported, and what were the headline numbers?

### Takeaway
- **Parameters.** KS4 ran at its own defaults. KS1/2/2.5/3 had their thresholds overridden to [9, 8]; KS2.5's default is [10, 4]. IronClust, MountainSort4 and SpyKING CIRCUS got a detection-threshold sweep on the medium-drift simulation. Everything else ran at defaults through SpikeInterface.
- **Metric.** Per GT unit, the best-matching sorted unit's 1-FP-FN, with "correctly identified" meaning a score above 0.8.
- **KS4 versus KS2.5, units with score > 0.8 out of 600:** 551 vs 488 (no drift), 555 vs 463 (medium), 502 vs 326 (high), 538 vs 465 (fast), 463 vs 280 (step), 531 vs 440 (step, aligned).

### Cited Findings
**Sorter parameters (Methods "Spike-sorting algorithm parameters")**
- Paper states: "All algorithms other than Kilosort4 were run through their respective SpikeInterface wrappers… and parameter adjustments were made in some cases" — [Kilosort4 paper, p.916](https://doi.org/10.1038/s41592-024-02232-7)
- Paper states "The latest algorithm versions as of December 2022 were used in all cases". — [Kilosort4 paper, p.916](https://doi.org/10.1038/s41592-024-02232-7)
- Paper states: "For Kilosort1, 2, 2.5 and 3, we set the detection thresholds to [9, 8] instead of their defaults, which varied across versions." KS1 ran 2 passes instead of 6. — [Kilosort4 paper, Methods](https://doi.org/10.1038/s41592-024-02232-7)
- Defaults for reference:
  - SpikeInterface's KS2.5 wrapper defaults are `projection_threshold: [10, 4]`, `detect_threshold: 6` and `nblocks: 5`. KS3's default is `[9, 9]`. — [SpikeInterface kilosort2_5.py](https://github.com/SpikeInterface/spikeinterface/blob/main/src/spikeinterface/sorters/external/kilosort2_5.py); [kilosort3.py](https://github.com/SpikeInterface/spikeinterface/blob/main/src/spikeinterface/sorters/external/kilosort3.py)
  - KS4's defaults are `Th_universal` = 9 and `Th_learned` = 8 (local repo `kilosort/parameters.py`, lines 61-73). So the [9, 8] override gave the older Kilosorts KS4's threshold values.
- The other sorters:
  - IronClust, MountainSort4 and SpyKING CIRCUS were swept over detection threshold, picking the value "which maximized the number of correctly identified units on the medium drift simulation". The best values were the defaults for MS4 and IronClust and 4.5 for SpyKING CIRCUS.
  - SpyKING CIRCUS 2 was swept without improvement.
  - The MS4 adjacency radius was set to 50 (the IronClust default) for runtime.
  - "All other parameters were set to their default values."
  — [Kilosort4 paper, Methods](https://doi.org/10.1038/s41592-024-02232-7)

**Metrics (Methods "Performance metrics", "Benchmarking")**
- How a sorted unit is scored against a GT unit:
  - Each GT unit is compared with the 40 closest detected units (by best-channel distance).
  - A spike is a match if it is within ≤ 0.2 ms of a GT spike.
  - FP rate = unmatched estimated spikes / total estimated spikes; FN rate = missed GT spikes / total GT spikes.
  - The GT unit is matched to the detected unit that maximises 1-FP-FN. "Correctly identified" means a score > 0.8 (Fig. 4j).
  — [Kilosort4 paper, Methods](https://doi.org/10.1038/s41592-024-02232-7)
- False-positive units (Fig. 4k):
  - A unit was labelled "good" if its ACG-estimated refractory-violation rate was < 0.2: "This is also the default rate in Kilosort4 to call units 'good' and we used the same strategy for labeling units from the other algorithms."
  - Good units matching no GT unit were counted as false-positive units.
  - Paper states that KS4 "had similar numbers of false positive units compared to the other algorithms… generally in the range of 50–100 units", and suggests these "likely correspond to pieces of ground-truth units that were not matched at the 0.8 threshold".
  — [Kilosort4 paper, p.918 and Methods](https://doi.org/10.1038/s41592-024-02232-7)
- Splits and merges:
  - Only the single best match counts. Paper notes that in step drift "many units are identified with exactly a 0.5 score, which corresponds to 50% of the spikes identified", meaning units oversplit into two halves, which then fail the 0.8 threshold.
  - No separate over-merge, over-split or redundant-unit counts are reported.
  — [Kilosort4 paper, Methods "Iterative neighbor assignment"](https://doi.org/10.1038/s41592-024-02232-7)

**Headline numbers (Fig. 4j: units with score > 0.8, out of 600 GT single units per simulation)**

The column order was verified from the PDF text coordinates for KS4, KS3, KS2.5 and KS2. The other columns follow the label order in the figure.

| Drift | KS4 | KS3 | KS2.5 | KS2 | KS1 | IronClust | MS4 | SpyKING CIRCUS | SC2 | HDSort | HerdingSpikes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| No drift | 551 | 529 | 488 | 476 | 182 | 355 | 342 | 372 | 126 | 168 | 25 |
| Medium | 555 | 493 | 463 | 461 | 167 | 323 | 297 | 316 | 137 | 5 | 22 |
| High | 502 | 294 | 326 | 410 | 59 | 256 | 129 | 144 | 63 | 20 | 6 |
| Fast | 538 | 470 | 465 | 449 | 123 | 270 | 281 | 287 | 96 | 12 | 13 |
| Step | 463 | 262 | 280 | 113 | 10 | 1 | 3 | 7 | 7 | 1 | 0 |
| Step, aligned | 531 | 435 | 440 | 74 | 6 | 1 | 1 | 5 | 1 | 1 | 0 |

— [Kilosort4 paper, Fig. 4j](https://doi.org/10.1038/s41592-024-02232-7)

- Paper states: "Kilosort2, 2.5, 3 and 4 again outperformed all other algorithms in all drift conditions". IronClust "generally found ~50% of all units, compared to the 80–90% found by Kilosort4". Among the sorters with drift correction, KS4 "consistently performed better due to its improved clustering algorithm". — [Kilosort4 paper, p.918](https://doi.org/10.1038/s41592-024-02232-7)
- The ablation (Fig. 5b) reports changes in misses and false positives against normal KS4. Removing drift correction cost +435 to +512 misses in the step conditions. Removing deconvolution cost about +273 to +337 misses in every condition. Removing CCG information cost +4 to +220 misses. — [Kilosort4 paper, Fig. 5](https://doi.org/10.1038/s41592-024-02232-7)
- Runtime: all sorters ran "within 2× the duration of the simulations" (Fig. 4l, n = 6 simulations). — [Kilosort4 paper, p.918](https://doi.org/10.1038/s41592-024-02232-7)
- Extended Data Fig. 7: paper states that KS4's dependence on firing rate, norm and spatial extent "was minimal". Some other sorters depended strongly on norm, which "could not be improved by lowering spike detection thresholds". — [Kilosort4 paper, p.918 and ED Fig. 7](https://doi.org/10.1038/s41592-024-02232-7)

### Inferences
- **KS2.5 was not run at its own default.** Its [10, 4] became [9, 8], raising the second-pass (learned-template) threshold from 4 to 8. KS2.5 was therefore run with fewer low-amplitude extractions than its default, which plausibly lowers its recall on low-norm units. Low-norm units make up a large share of the GT, since single-unit norms start at 10 with an exponential tail. The paper gives no reason for [9, 8] and no sweep for any Kilosort version. The non-Kilosort sorters, by contrast, were swept.
- **Under high drift, KS2 (410) beat KS2.5 (326) and KS3 (294).** This is not discussed in the text, which says KS2 and KS2.5 "perform similarly on most simulations except for the step drift conditions". It suggests that on these simulations KS2.5's registration plus its postprocessing loses units relative to KS2's drift tracking. That is relevant to anyone weighing KS2.5's drift handling.
- **The single cut-off at 0.8 is a lever.** The step-drift 0.5 scores show that over-splitting is counted the same as missing the unit. Precision and recall are never reported separately, so a sorter that is high-precision and lower-recall (one later benchmark describes KS2.5 this way) is not distinguished. The "good" label for false-positive units uses KS4's own ACG criterion. Any sorter whose units are refractory but fail KS4's specific estimator is treated differently.

### Gaps
- No per-unit precision or recall, and no merge/split categories, are published in the main paper or Extended Data. Recovering them would need the shared simulations ([Janelia figshare 10.25378/janelia.25298815.v1](https://doi.org/10.25378/janelia.25298815.v1)) and re-running the sorters.
- The exact SpikeInterface version and wrapper parameters for KS2.5 (for example nblocks, minFR, AUCsplit) are not listed beyond "default".

## Q3. Is there a Nature Methods peer review file, or a published critique, commentary or response about confounds in the KS4 evaluation?

### Takeaway
- **No peer review file.** I found no public transparent peer review file. The paper's statement thanks "the anonymous reviewers". The journal's supplementary files are the Supplementary Information (algorithms for KS 2/2.5/3) and the Reporting Summary only.
- **No direct critique.** I found no published commentary, PubPeer thread or social-media thread that explicitly accuses the KS4 benchmark of circularity or unfair tuning.
- **Later independent benchmarks** mostly corroborate KS4 > KS2.5 on recall-weighted accuracy. Several of them contain points that bear on the confounds: KS2.5 has higher precision, results depend on the metric threshold, KS4 has more false positives under motion correction, and KS4's own benchmark is "closer to the assumptions of template-based, geometry-aware sorters".

### Cited Findings
**Peer review**
- The paper's "Peer review information" reads: "Nature Methods thanks the anonymous reviewers for their contribution to the peer review of this work. Primary Handling Editor: Nina Vogt". There is no mention of a Peer Review File. — [Kilosort4 paper, final page](https://doi.org/10.1038/s41592-024-02232-7)
- Springer's static ESM store has MOESM1 (Supplementary Information: "Algorithms for Kilosort 2/2.5/3") and MOESM2 (Reporting Summary). MOESM3 and MOESM4 returned HTTP 403 when probed, which is consistent with no further supplementary file such as a peer review file. — [MOESM1](https://static-content.springer.com/esm/art%3A10.1038%2Fs41592-024-02232-7/MediaObjects/41592_2024_2232_MOESM1_ESM.pdf); [MOESM2](https://static-content.springer.com/esm/art%3A10.1038%2Fs41592-024-02232-7/MediaObjects/41592_2024_2232_MOESM2_ESM.pdf)
- The Supplementary Information contains no benchmark details. It describes KS2 drift tracking, KS2/2.5 global optimisation and KS3 recursive pursuit. — [MOESM1](https://static-content.springer.com/esm/art%3A10.1038%2Fs41592-024-02232-7/MediaObjects/41592_2024_2232_MOESM1_ESM.pdf)

**Later independent benchmarks touching on KS4 versus KS2.5**

*Buccino, Sridhar, Feng, Svoboda & Siegle (Allen Institute for Neural Dynamics), "Efficient and reproducible pipelines for spike sorting large-scale electrophysiology data"*
- Venues and dates: bioRxiv 2025.11.12.687966; eLife Reviewed Preprint 110170, v1 3 Feb 2026, v2 3 Jul 2026.
- Hybrid ground-truth setup:
  - Templates come from the SpikeInterface template library: IBL templates for NP1, and NP-Ultra templates interpolated to NP2.
  - Injected spikes follow drift estimated by DREDge.
  - 10 units per probe or shank, Poisson firing at 15 Hz, 50-200 µV, 5 iterations.
- KS4 beat KS2.5 on:
  - accuracy (effect size NP1 0.276, NP2 0.408)
  - recall (0.319 / 0.415)
  - precision (0.272 / 0.232, with "slight degradation" for KS4 in matched units)
- The text also reports:
  - 3,046 (KS2.5) vs 3,652 (KS4) matched units
  - KS4 slightly higher refractory contamination
  - KS4 about 2× runtime
- The authors say their results "corroborated the results of the original paper".
- Self-stated limitations: Poisson GT spike trains, uniform placement, and templates not matched to brain region.
  — [eLife RP 110170](https://elifesciences.org/reviewed-preprints/110170); [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.11.12.687966v1.full)
- The eLife v1 public reviewers (Reviewers #1-3) accepted the KS4 > KS2.5 finding and raised no concerns about template source, circularity or fairness. Their comments were about curation thresholds, for example a presence ratio > 0.8 and duplicate units. — [eLife 110170 v1 reviews](https://elifesciences.org/reviewed-preprints/110170v1/reviews)

*SpikeInterface documentation, "Benchmark spike sorting with hybrid recordings"*
- Setup: 31 IBL-library templates injected into a Steinmetz NP1 recording with imposed triangular motion, using `InjectDriftingTemplatesRecording`. Compared KS2.5 (default), KS3, KS4 (nblocks = 5) and SpyKING CIRCUS 2.
- Results as summarised from the page:
  - KS2.5, KS4 and SC2 each find about 10-12 hybrid units above 80% accuracy.
  - KS4 has the best overall curve, with almost all units above 50%.
  - KS2.5 has "high precision but lower recall".
  - There is "no clear 'winner'".
- No numeric table is given, and the result rests on one recording with a few dozen units.
  — [SpikeInterface docs](https://spikeinterface.readthedocs.io/en/latest/how_to/benchmark_with_hybrid_recordings.html)

*Garcia, Halcrow, Windolf, McKenzie, Adkisson-Floro, Mayorquin, Dichter, Buccino & Yger, "Opening the black box: a modular approach to spike sorting"*
- Venues and dates: bioRxiv 2026.01.23; eLife RP110588, v1 Apr 2026, VOR 25 Sep 2026.
- Setup: fully synthetic recordings from SpikeInterface's generator (parametric templates, 500 units, NP1-like, static and drifting).
- Findings about KS4: "Kilosort4 finds many false positives, especially for the motion-corrected recording."
- The authors acknowledge their own new sorter was "built with the 'best' algorithms, evaluated on our generated data".
- They note that Bombcell and UnitRefine quality labels used on real data had defaults tuned on Kilosort4 output, which could favour KS4.
- KS2.5 was not benchmarked.
  — [eLife 110588](https://elifesciences.org/articles/110588); [bioRxiv v2](https://www.biorxiv.org/content/10.64898/2026.01.23.701239v2.full)

*Banaie Boroujeni, Womelsdorf & Kastner, "KIASORT", J Neurosci 46(27) e1594252026, 8 Jul 2026*
- Argues that KS4's benchmark datasets rely on smooth waveform libraries and "are closer to the assumptions of template-based, geometry-aware sorters". In the authors' view they do not test neuron-specific drift.
- On KS4's own public benchmark, KS4 kept "a modest lead" under 1-FN-FP > 0.8, but the methods "were more comparable under F1 > 0.75". This illustrates that rankings depend on the metric and threshold.
- The KIASORT authors tuned KS4's detection threshold and parameters for their own comparison, and their custom simulations were designed around the conditions KIASORT targets.
  — [J Neurosci KIASORT](https://www.jneurosci.org/content/46/27/e1594252026)

*Community reports*
- GitHub issue #634 (26 Mar 2024): a user found about equal numbers of Good and MUA clusters between KS2.5 and KS4 on NP1, but "at least 50% less" total spikes in KS4 Good clusters.
- Marius Pachitariu replied that the result "will depend on your particular experimental prep, probe geometry, thresholds etc." and suggested comparing in Phy.
- This is anecdotal and not a controlled benchmark.
  — [MouseLand/Kilosort #634](https://github.com/MouseLand/Kilosort/issues/634)

**Authors' rebuttal of an earlier benchmark (context)**
- Extended Data Fig. 3 argues that the "Kilosort false positives from the Spikeinterface paper are due to unrealistically long spike durations" in the MEArec/biophysical simulation. Kilosort2 false positives fall when that simulation is sped up 2×.
- I found no published SpikeInterface reply to this specific claim.
  — [Kilosort4 paper, ED Fig. 3](https://doi.org/10.1038/s41592-024-02232-7)

### Inferences
- The strongest independent check on the KS4 > KS2.5 claim is Buccino et al. 2025/2026. It avoids KS4-derived templates by using SpikeInterface/IBL library templates and injects with drift.
  - The IBL template library was, as I understand it, built from IBL Brain Wide Map sorts produced with the IBL's KS2.5-lineage pipeline (pyKilosort/iblsorter). If so, any residual template-source bias in that test would favour KS2.5, not KS4. This makes the corroboration more robust on the circularity question. I have not verified the library's provenance in this session.
  - It still evaluates uncurated output, uses Poisson spike trains and one injection model, and reports precision gains for KS4 as well as recall gains.
- The SpikeInterface docs example (KS2.5 high precision and lower recall, KS4 best curve, about equal counts above 0.8) and KIASORT's threshold sensitivity both suggest the KS4-versus-KS2.5 gap depends heavily on whether recall-weighted 1-FP-FN at 0.8 is the yardstick. A precision-first view can narrow the gap substantially.
- Garcia et al. 2026 report many KS4 false positives with motion correction, which bears on the paper's Fig. 4k claim that KS4's false-positive units were similar to other sorters. Those false-positive units were counted only among units passing KS4's own refractory criterion.

### Gaps
- Nature Methods did not publish transparent peer review for this paper, as far as I could determine. Reviewer concerns and author responses are unknown. The Nature article page and PMC were not readable in this session (login redirect or CAPTCHA), so the absence of a peer review file rests on the paper's statement and the ESM probe.
- I found no PubPeer entry, and no X, Bluesky or Mastodon thread or blog post explicitly questioning the KS4-versus-KS2.5 claims or the KS4-derived hybrid templates. Searches returned nothing relevant. PubPeer was not queried directly by URL, so this is "not found", not "confirmed absent".
- I did not read the remaining portions (beyond the first 100k characters) of the long eLife and J Neurosci pages, so some quantitative details may be missing.

## Q4. What does the general literature say about biases in hybrid or synthetic ground-truth benchmarks, especially when the injected units come from one sorter's output?

### Takeaway
The benchmarking literature (SpikeForest, SpikeInterface, and more recent modular and KIASORT work) acknowledges general dangers:
- Benchmark selection can be unintentionally biased toward a new method.
- Ground truth covers only the neurons and conditions it contains.
- Generator assumptions that match a sorter's model favour that sorter.

The specific circularity of building hybrid units from one sorter's own clean units was not found discussed as a named, quantified bias. Even the KS4 paper only notes that hybrids "depend on finding the neurons in the first place".

### Cited Findings
- SpikeForest (Magland et al. 2020, eLife 9:e55167):
  - "the choice of simulations used for evaluation can often be (unintentionally) biased" toward showing that a new method is superior.
  - "SpikeForest results are biased towards the types of neurons and conditions for which ground-truth data exist."
  - Simulations "may not yet be duplicating the firing and noise statistics of real-world electrophysiology recordings".
  - IronClust was substantially developed while its developers had access to the SpikeForest ground truth, a train-on-test risk.
  — [SpikeForest, eLife 2020](https://elifesciences.org/articles/55167)
- The SpikeForest SYNTH_JANELIA set was "Distributed with KiloSort2, with and without simulated drift", using real average waveforms over synthetic noise. This is an earlier example of Kilosort-originated benchmark data entering community benchmarks. — [SpikeForest, eLife 2020](https://elifesciences.org/articles/55167)
- The KS4 paper's own taxonomy (Methods) covers three approaches:
  1. Paired GT, which is short and anesthetised.
  2. Hybrid GT from "ground-truth units… or manually curated units", which "depend on finding the neurons in the first place".
  3. Simulated or biophysical waveforms, which the authors argue are "not… as diverse as real neurons".
  — [Kilosort4 paper, Methods](https://doi.org/10.1038/s41592-024-02232-7)
- SpikeInterface's hybrid how-to says templates "can be from previous spike sorted data". It recommends motion-aware injection ("Such drifts have to be taken into account in order to smoothly inject spikes into the recording") and uses a public template database (601 IBL templates) rather than the sorters under test. — [SpikeInterface docs](https://spikeinterface.readthedocs.io/en/latest/how_to/benchmark_with_hybrid_recordings.html)
- Garcia et al. 2026 make three related points:
  - Pipelines are usually "evaluated as 'black boxes'".
  - Their own sorter's design was selected on their own generator, an explicit admission of the same train-on-test risk.
  - Quality-metric curation tools tuned on KS4 output may favour KS4.
  — [eLife 110588](https://elifesciences.org/articles/110588)
- KIASORT (2026) argues that generator assumptions (smooth waveform libraries, geometry-consistent drift) align with template-based, geometry-aware sorters such as KS4. — [J Neurosci KIASORT](https://www.jneurosci.org/content/46/27/e1594252026)
- Buccino et al. (2025/2026) list their hybrid's own realism limits: Poisson spike trains, uniform placement, and templates not region-matched. They state that whether "more realistic hybrid recordings will have any effect on spike sorting accuracy" remains untested. — [eLife RP 110170](https://elifesciences.org/reviewed-preprints/110170)

### Inferences
- Taken together, the literature supports three distinct bias channels relevant to the KS4 paper:
  1. **Unit-selection bias.** GT units are the ones a particular sorter already isolated. This applies to KS4 in the Fig. 3 hybrid, and to pyKilosort/KS2.5-lineage sorting in the drift simulations.
  2. **Generative-model match.** Fixed or kriged templates, linear superposition, smooth nonrigid drift on NP geometry, and whitened Gaussian-like noise, all of which match Kilosort's model.
  3. **Evaluation-rule choices.** The 1-FP-FN > 0.8 threshold, best-match-only scoring, and false-positive units labelled with KS4's own refractory criterion.
- Channel 1 is the only one that favours KS4 over KS2.5 specifically. Channels 2 and 3 favour the Kilosort family, or recall-heavy sorters, more generally.
- The most persuasive mitigation in the later literature is the use of template libraries not produced by the sorters under test, with motion-aware injection (SpikeInterface hybrid; Buccino et al.). Under that design KS4 still beat KS2.5 on accuracy, though by less clear margins in the small SpikeInterface docs example.

### Gaps
- I found no paper that quantifies the size of "same-sorter template" bias, for example by building hybrids from sorter A's units and from sorter B's units and comparing the ranking flip.
- Rossant et al. 2016 (Nat Neurosci) and the original Kilosort 2016 hybrid methodology were cited but not re-read in this session, so I have no direct quotes on their stated hybrid limitations.
