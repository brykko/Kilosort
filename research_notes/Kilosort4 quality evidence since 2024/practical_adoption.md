# Practical adoption of Kilosort 4 vs Kilosort 2.5 in large pipelines and labs (2024 to Nov 2026)

Scope note: About 20 search and fetch calls. Primary sources were read where they could be fetched: the AIND eLife reviewed preprint, the eLife reviews, the ibl-sorter README and CHANGELOG, the IBL 2025 BWM release notes, the Garcia et al. eLife RP, Power Pixels, and Laquitaine et al. v2. Several sources were blocked: the Nature/PMC full text for Trautmann et al. 2025 (captcha or login), the Laquitaine PDF (too large), and the IBL figshare white paper (not fetched). Some claims below rest only on search-engine snippets, and they are marked "(snippet only)". MouseLand/Kilosort GitHub issues are deliberately excluded except for brief cross-references.

## 1. IBL: iblsorter / pykilosort (KS2.5-derived). Did IBL evaluate KS4?

### Takeaway
As of mid-2026, IBL still runs its own KS2.5-derived sorter (ibl-sorter, formerly pykilosort). The improvements have gone into preprocessing and motion estimation (destriping, channel rejection, DARTsort detection with DREDge registration), not a switch to KS4. I found no published IBL document that evaluates KS4 against iblsorter. IBL data were used heavily in other groups' KS4 benchmarks.

### Cited Findings
- The ibl-sorter README describes itself as "the implementation of the IBL spike sorting pipeline described on this white paper" (figshare doi 10.6084/m9.figshare.19705522.v4). It states that "The clustering part is based on the original MATLAB version of Kilosort 2.5, written by Marius Pachitariu." The README does not mention Kilosort 4. Requirements: NVIDIA GPU with at least 8 GB, more than 200 GB of scratch disk, CUDA 12+, Python 3.10 to 3.12. — [int-brain-lab/ibl-sorter README](https://github.com/int-brain-lab/ibl-sorter)
- The ibl-sorter CHANGELOG shows active development through 2026, all on the KS2.5-derived core. The CHANGELOG never mentions Kilosort 4 or a comparison with it. — [ibl-sorter CHANGELOG](https://raw.githubusercontent.com/int-brain-lab/ibl-sorter/main/CHANGELOG.md)
  - 1.13.0 (2026-06-23) exposes destriping parameters and switches the default spatial filter from k-filter to CAR (`k_filter: false`).
  - 1.12.1 (2025-12-16), 1.12.0 (2025-10-09), 1.11.0 (2025-09-18) and 1.10.0 (2025-06-27) are maintenance and feature releases: EC2/Docker docs, NP2.4 shank fixes, and a parameter-file override.
  - 1.8.x (undated) adds a "passingSpikes" table output, exposes DREDge and channel-detection parameters, and raises an error if more than half the channels are bad.
  - 1.6a.dev5 (undated) "reworked datashift2 to use DARTsort spike detection feeding DREDge motion estimation", meaning IBL replaced the KS2.5 drift-estimation front end with Paninski-lab tools.
  - 1.5.0 set minFR to 0.02 Hz "to reduce near-empty clusters".
- IBL Brain-Wide Map 2025 data release: on 2024-05-15 IBL "decided to re-run the latest spike sorting version on the full dataset" for state-of-the-art results and consistency across years of acquisition. The improvements are described in the "updated spike sorting whitepaper" (figshare 19705522). The release totals are 459 sessions, 699 recordings, 621,733 units, and 75,708 "good quality" units, about 12.2%. The 2024-02-15 entry added 81,229 units, of which 12,319 were good. The page does not name a sorter version and does not mention KS4. — [IBL docs, 2025 Brain Wide Map release](https://docs.internationalbrainlab.org/notebooks_external/2025_data_release_brainwidemap.html)
- The Nature 2025 Brain-Wide Map paper's methods describe "a modified version of the Kilosort 2.5 algorithm, with improvements for scalability, reproducibility and stability" (snippet only). — [IBL, Nature 2025](https://www.nature.com/articles/s41586-025-09235-0)
- The KS4 paper (Pachitariu et al., Nat Methods 2024) built its drift-containing simulations from the properties of 512 IBL Neuropixels 1.0 recordings (snippet only). Its benchmarks therefore indirectly cover IBL-like data. — [Kilosort4, Nature Methods 2024](https://www.nature.com/articles/s41592-024-02232-7)
- A search-engine summary claimed that a preprint says "Kilosort4 outperforms Kilosort2.5 in the IBL framework". I could not verify this. It appears to conflate the AIND pipeline paper, which used IBL NP1 templates for hybrid generation (Section 2), with an IBL statement. Do not cite it as an IBL conclusion.

### Inferences
- IBL's choice reads as a large-consortium pattern: keep the clustering core fixed for reproducibility and cross-year consistency (the re-sort of all data in 2024), and upgrade the parts most tied to data quality (destriping, bad-channel detection, DREDge registration). Switching to KS4 would break comparability with the published BWM dataset. This is inferred from the release notes' stated motivation of "consistency", not from an IBL statement about KS4.
- The low "good" fraction (~12%) reflects IBL's strict single-unit QC (e.g., the passingSpikes / RF-contamination criteria) rather than a sorter-yield comparison.

### Gaps
- I did not fetch the IBL spike sorting white paper (figshare v4 or later), so I cannot say whether it contains a KS4 comparison. This is the most important unverified item. The report writer should flag it as unknown, not as "IBL did not evaluate KS4".
- I found no IBL blog, forum post, or talk explicitly stating why IBL did not adopt KS4.

## 2. Allen Institute (AIND ephys pipeline; Brain Observatory / OpenScope)

### Takeaway
Allen Neural Dynamics switched its default from KS2.5 to KS4 in November 2024, based on its own hybrid ground-truth benchmark. KS4 had higher accuracy and recall, slightly lower precision, slightly higher refractory contamination, more units present throughout the recording, and about 2x runtime. Several projects stayed on KS2.5 "for consistency". This is the best-documented institutional switch.

### Cited Findings
- **Paper and status:** "Efficient and reproducible pipelines for spike sorting large-scale electrophysiology data" (Buccino et al., AIND; bioRxiv 2025.11.12.687966; eLife reviewed preprint 110170). The eLife assessment rates it "valuable" and "compelling". — [eLife RP 110170](https://elifesciences.org/reviewed-preprints/110170); [bioRxiv v2](https://www.biorxiv.org/content/10.1101/2025.11.12.687966v2)
- **Timeline:** The pipeline was deployed on Code Ocean in February 2024 with KS2.5 as default. KS4 became the default in November 2024 "given its better performance", and the benchmark results "motivated our decision". — [eLife RP 110170](https://elifesciences.org/reviewed-preprints/110170); [bioRxiv v1 PDF](https://www.biorxiv.org/content/10.1101/2025.11.12.687966v1.full.pdf)
- **Usage as of August 2025:** 789 sessions sorted with KS2.5 and 468 with KS4. The paper states a total of 1,287, but the two figures sum to 1,257, so the text has an internal inconsistency. "Several projects continued using Kilosort2.5 for consistency." — [eLife RP 110170](https://elifesciences.org/reviewed-preprints/110170)
- **Hybrid benchmark (NP1 and NP2, IBL NP1 templates injected):** — [eLife RP 110170](https://elifesciences.org/reviewed-preprints/110170)
  - Accuracy: NP1 effect size 0.276, NP2 0.408 (both P < 1e-10), favouring KS4.
  - Recall: NP1 0.319, NP2 0.415 (P < 1e-10), favouring KS4.
  - Precision: NP1 0.272 (P < 1e-7), NP2 0.232 (P < 1e-10). The direction is ambiguous: the text separately says KS4 had "a slight loss of precision".
  - Matched ground-truth units at accuracy ≥ 0.2: KS2.5 had 3,046, KS4 had 3,652 (about +20%).
  - Refractory-period contamination was slightly higher for KS4 (P < 1e-10, effect size 0.013).
  - KS4 found significantly more units with high presence ratio, i.e. active throughout the recording (effect size 0.193). The authors attribute this to a more effective merging strategy.
  - The KS4 advantage was largest for low-SNR units (SNR < 10).
- **Runtime:** KS4 takes "almost twice as long to run as Kilosort2.5" (Fig. S2). — [eLife RP 110170](https://elifesciences.org/reviewed-preprints/110170)
- **Lossy compression:** KS4 on the most lossy WavPack setting (BPS = 2.25) "still outperforms" KS2.5 on lossless data. — [eLife RP 110170](https://elifesciences.org/reviewed-preprints/110170)
- **Versions:** the PDF lists KS2.5 v2.5.1 and KS4 v4.0.30 (snippet only). — [eLife PDF](https://elifesciences.org/articles/110170.pdf)
- **Benchmark code** is public and also includes SpykingCircus2. — [aind-ephys-hybrid-benchmark](https://github.com/AllenNeuralDynamics/aind-ephys-hybrid-benchmark)
- **Reach:** the pipeline is reportedly used by about 50 researchers at the Kempner Institute (Harvard). — [eLife RP 110170](https://elifesciences.org/reviewed-preprints/110170)
- **Reviewer comments:** — [eLife reviews](https://elifesciences.org/reviewed-preprints/110170v1/reviews)
  - Reviewer 1: KS4 "significantly outperforms Kilosort 2.5", consistent with the KS4 paper's different ground-truth method.
  - Reviewer 3: "Kilosort4 performs the best out of the tested options".
  - Reviewer 3 also asked why duplicate units appear, suggesting this is a sorter issue rather than a pipeline issue, and cautioned that a presence-ratio ≥ 0.8 curation threshold can wrongly exclude behaviour- or state-specific cells.
  - No reviewer raised a KS4 precision concern.

### Inferences
- The AIND trade-off is the clearest documented profile of KS4 vs KS2.5 on Neuropixels. KS4 gives more and better-recovered units, especially at low SNR, and better whole-session tracking. The cost is a small increase in false positives and contamination and about 2x compute.
- Adoption is incremental even where KS4 is the default: KS2.5 sessions still outnumbered KS4 sessions in August 2025. Ongoing projects value within-project consistency over the accuracy gain.
- Legacy Allen Brain Observatory / Visual Coding / Visual Behavior Neuropixels datasets were sorted with KS2-era tools (pre-2024). I did not re-verify this for this note.

### Gaps
- I found no per-sorter quality-metric pass rates and no curation-burden figures for real (non-hybrid) AIND data.
- The NP1 precision effect direction is ambiguous between the extracted text and the "slight loss of precision" statement. The figure needs checking.
- I did not verify which sorter OpenScope 2024 to 2026 releases use.

## 3. Other large projects (SpikeInterface, NHP, human, non-Neuropixels, chronic)

### Takeaway
Independent 2024 to 2026 benchmarks agree that KS4 beats KS2.5 on recall and yield. Several flag KS4 false positives and redundant units, especially after motion correction and when per-neuron drift varies. Turnkey pipelines (AIND, Power Pixels) ship KS4 alongside KS2.5 options. NHP and human work is mixed: version choice is often driven by drift severity. I found no documented large project reporting KS4 worse than KS2.5 overall.

### Cited Findings
- **SpikeInterface modular sorters** (Garcia, Halcrow, Windolf, …, Buccino, Yger; eLife RP 110588, v1 April 2026, v2 September 2026): — [eLife RP 110588](https://elifesciences.org/reviewed-preprints/110588)
  - On simulated NP1-like data, "Kilosort4 performs well… finds many good units and runs quickly". However, "Kilosort4 finds many false positives, especially for the motion-corrected recording", and motion correction "increases the number of False Positives, especially for Kilosort4."
  - Their new CPU sorter, Lupin, "clearly outperforms all other sorters in all conditions… including Kilosort4". On real data (IBL NP1, Lebedeva NP2 chronic, Cambridge NeuroTech chronic) it is "competitive with Kilosort4, finding on average a similar number of good units and less noisy units."
  - Versions used: KS4 4.1.1 for simulations and 4.1.2 for real data.
  - The paper does not test KS2.5 directly.
- **Power Pixels** (Meijer & Battaglia, Radboud; bioRxiv 2025.06.27.661890 v6, Jan 2026): a turnkey Neuropixels pipeline (SpikeInterface + Kilosort + Bombcell + UnitRefine). It ships recommended parameter files for KS2.5, KS3, KS4 and pykilosort, with KS2.5 and KS3 run in Docker and KS4 installed natively in Python. It gives no rationale for a specific version and reports no comparison. — [Power Pixels bioRxiv](https://www.biorxiv.org/content/10.1101/2025.06.27.661890v6.full)
- **Biophysical ground truth** (Laquitaine, Imbeni, Tharayil, Isbister, Reimann; EPFL/Blue Brain; bioRxiv 2024.12.04.626805, v2 Mar 2025): — [bioRxiv v2](https://www.biorxiv.org/content/10.1101/2024.12.04.626805v2.full)
  - The study evaluated KS 1, 2, 2.5, 3 and 4 on a detailed rat S1 column simulation.
  - KS3 and KS4 "did not produce false positives", while "older kilosort versions did".
  - KS2 had the highest accuracy on the full synthetic model (up to 99%) and an unusually large unit count.
  - KS 1 to 2.5 over-represented rare interneuron types by factors of 2.9 to 8.1.
  - KS4 single-unit yield was about 15% of neurons within 50 µm.
  - The majority of isolated single units were under 80% accurate across sorters, and stimulus information dropped about 50% on average.
  - A search snippet adds that "41% of sorted units were single-units" for KS4 on spontaneous NP-like data (snippet only).
- **KIASORT** (bioRxiv 2025.07.10.664175, later PubMed 42248684): argues that KS4 degrades when neuron-specific drift varies, and reports outperforming KS4 in simulations (snippet only; a method-developer's claim). — [KIASORT bioRxiv](https://www.biorxiv.org/content/biorxiv/early/2025/08/31/2025.07.10.664175.full.pdf); [PubMed](https://pubmed.ncbi.nlm.nih.gov/42248684/)
- **NHP:** — [Trautmann et al., Nat Neurosci 2025](https://www.nature.com/articles/s41593-025-01976-5); [Caltech record](https://authors.library.caltech.edu/120143)
  - Trautmann et al., "Large-scale high-density brain-wide neural recording in nonhuman primates" (Nat Neurosci 28:1562–1575, June 2025; Neuropixels 1.0 NHP, up to 8 probes / 3,072 channels) reportedly used KS2.5 for drift estimation. Drift was often under 10 µm, lower than in human Neuropixels recordings (snippet only; full methods blocked).
  - A search snippet also mentions a separate primate study that used KS2.0 after external drift correction in one monkey and KS4 in a second, low-drift monkey (source not identified; snippet only).
- **NHP and human motion:** the MEDiCINe motion-estimation paper (eNeuro 2025) notes that motion is "typically more extreme in NHP and human recordings". It reports better motion estimation than existing methods and ships SpikeInterface/KS4 examples (snippet only). — [MEDiCINe, PMC11896784](https://pmc.ncbi.nlm.nih.gov/articles/PMC11896784)
- **Bombcell / UnitMatch:**
  - Bombcell release notes add "Handles kilosort4 spike sorting output" (v1.5.0). A Python port was released in 2025 and computes up to 17 metrics. — [Bombcell releases](https://github.com/Julie-Fabre/bombcell/releases); [Fabre on X](https://x.com/basal_gang/status/1933599328909631867)
  - One KS4-based study (Dentate Gyrus place cells, bioRxiv Oct 2025) used Bombcell "noise" labels as proxies for redundant units and "mua" labels as proxies for overmerged units. This is the authors' interpretation (snippet only). — [bioRxiv 2025.10.24.684382](https://www.biorxiv.org/content/10.1101/2025.10.24.684382v1.full.pdf)
  - UnitMatch (van Beest et al., Nat Methods 2024) is reported in a snippet to have used KS4 output (snippet only, unverified). — [UnitMatch GitHub](https://github.com/EnnyvanBeest/UnitMatch); [ResearchGate](https://www.researchgate.net/publication/384407643_Tracking_neurons_across_days_with_high-density_probes)
  - I found no Bombcell or UnitMatch publication that quantifies KS4 vs KS2.5 output differences.
- **Cross-reference only (GitHub, anecdote):**
  - One user found similar Good/MUA counts but at least 50% fewer spikes in Good clusters with KS4. — [Issue #634](https://github.com/MouseLand/Kilosort/issues/634)
  - Another user reports KS4 drift correction making things worse on some NP2 sessions. — [Issue #1006](https://github.com/MouseLand/Kilosort/issues/1006)

### Inferences
- The consistent independent picture is that KS4 gives higher recall and yield, with the main cost in false positives and redundant (oversplit or duplicate) units, sensitive to motion correction. This matches the AIND "slight loss of precision / slightly higher contamination" and the reviewer's duplicate-unit question. It implies more reliance on automated curation (Bombcell, UnitRefine, merge steps) after KS4.
- Groups with bespoke preprocessing (IBL) or with high or nonuniform drift (NHP, human) are the least likely to have simply switched. They tend to swap in external motion estimation (DREDge, MEDiCINe) regardless of the sorter.

### Gaps
- I found no human intraoperative Neuropixels paper (2024 to 2026) explicitly comparing KS4 and KS2.5.
- I found no Neuropixels Ultra, Utah array, tetrode or MEA study comparing the two versions, beyond the Garcia et al. supplementary geometries, which compare KS4 to SpikeInterface sorters only.
- I could not locate the UCL Neuropixels course 2024/2025/2026 slides, Neuropixels Slack summaries, or Reddit/Bluesky threads with substantive KS4-vs-KS2.5 comparisons. These are anecdotal channels, and nothing documented was found.
- Janelia (beyond MouseLand itself), DANDI-wide sorter-version statistics, and BRAIN Initiative project-level choices were not found.
