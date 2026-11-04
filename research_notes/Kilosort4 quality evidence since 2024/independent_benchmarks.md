# Independent benchmarks of Kilosort4 versus Kilosort2.5 and other sorters (2024 to Nov 2026)

Scope note: studies by people other than the Kilosort authors (Pachitariu, Sridhar, Pennington, Stringer), published or posted 2024 to Nov 2026. The KS4 paper itself (Pachitariu et al. 2024, Nat Methods, doi 10.1038/s41592-024-02232-7) is excluded as evidence. Research cut off after about 20 tool calls (searches plus full-text fetches). Several full texts were only partly read because pages were truncated; those points are flagged.

## Which independent benchmarks include KS4, and how did it rank (especially against KS2.5)?

### Takeaway
Only one independent study runs a large, controlled, head-to-head ground-truth comparison of KS4 and KS2.5: Buccino et al. (Allen Institute / SpikeInterface team; eLife 2026, peer reviewed). KS4 won on accuracy, precision and recall on both Neuropixels 1.0 and 2.0 hybrids, with the largest margin for SNR < 10. It had a slight per-unit precision cost, slightly higher refractory contamination, and about twice the run time. The other independent studies that include KS4 (SpikeInterface hybrid tutorial; Laquitaine et al. biophysical model; KIASORT; Lupin/modular sorter; NeuroSort) either do not test KS2.5, test it only lightly, or use proxy metrics. In those that compare KS4 with other modern sorters, KS4 is usually the strongest existing baseline or the runner-up to the authors' new sorter.

### Cited Findings

**Study 1: Buccino, Sridhar, Feng, Svoboda, Siegle. "Efficient and reproducible pipelines for spike sorting large-scale electrophysiology data." eLife 2026;15:RP110170, doi 10.7554/eLife.110170 (PEER REVIEWED, eLife VOR; preprint bioRxiv doi 10.1101/2025.11.12.687966, Nov 2025).**
- Authors are Allen Institute for Neural Dynamics / SpikeInterface developers, not Kilosort authors. Reviewing editor Lisa Giocomo. — [PMC13441294](https://pmc.ncbi.nlm.nih.gov/articles/PMC13441294/)
- Data: mouse in vivo. Neuropixels 1.0: 6 sessions x 6 probes = 36 recordings. Neuropixels 2.0 four-shank: 15 insertions x 4 shanks = 60 recordings. Brain regions not specified. — [PMC13441294](https://pmc.ncbi.nlm.nih.gov/articles/PMC13441294/)
- Ground truth method: hybrid. Templates were injected into real recordings: IBL Brain-Wide Map templates for NP1, and Neuropixels Ultra templates interpolated to NP2 geometry. Each probe or shank got 10 hybrid units with Poisson firing at a 15 Hz mean, amplitudes rescaled to a user range (example 50 to 200 µV), and 5 random iterations. That gives 1800 GT units (NP1) and 3000 GT units (NP2). Injected spikes followed DREDge-estimated motion. Matching tolerance was 0.2 ms. All raw sorter units were used, with no curation. — [PMC13441294](https://pmc.ncbi.nlm.nih.gov/articles/PMC13441294/)
- Sorter versions and parameters: KS2.5 and KS4 from versioned DockerHub containers, with identical preprocessing. Version numbers and parameter values are not given in the text read. — [PMC13441294](https://pmc.ncbi.nlm.nih.gov/articles/PMC13441294/)
- Headline result, quoted: "Across all recordings with both probe types, Kilosort4 is able to find hybrid units with significantly greater accuracy (NP1 ... effect size = 0.276; NP2 ... effect size = 0.408), precision (NP1 ... 0.272; NP2 ... 0.232) and recall (NP1 ... 0.319; NP2 ... 0.415)". "These differences are most pronounced for units with signal-to-noise ratios below 10." (The p-value exponents are lost in the web rendering and appear as "p < 1".) — [eLife article](https://elifesciences.org/articles/110170)
- Mixed-metric caveat, quoted: "When comparing individual ground-truth units, Kilosort4 tends to have higher accuracy and recall at the expense of a slight degradation of precision. This suggests that Kilosort4 recovers more complete spike trains, but also finds more false positive spikes in the process." — [eLife article](https://elifesciences.org/articles/110170)
- Matched GT units (accuracy ≥ 0.2): KS2.5 3046 and KS4 3652, out of 4800 injected units in total (1800 NP1 + 3000 NP2). — [PMC13441294](https://pmc.ncbi.nlm.nih.gov/articles/PMC13441294/); [eLife reviews](https://elifesciences.org/reviewed-preprints/110170/reviews)
- Refractory-period contamination: "Kilosort4 has slightly higher values than Kilosort2.5", effect size 0.013. A search-engine rendering of the same sentence gives P < 1 × 10^-10. Presence ratio: KS4 "finds significantly more units that are active throughout the entire recording", effect size 0.193. The authors suggest this "could be due to" KS4's merging strategy. — [eLife article](https://elifesciences.org/articles/110170); [PMC13441294](https://pmc.ncbi.nlm.nih.gov/articles/PMC13441294/)
- Conclusion: "a clear superiority of Kilosort4 in terms of spike sorting performance, although it does take Kilosort4 almost twice as long to run as Kilosort2.5". Pipeline cost: KS4 sorting took 2.7 ± 0.12 h per hour of recording per 384-channel probe on a g4dn.4xlarge instance. — [eLife article](https://elifesciences.org/articles/110170); [PMC13441294](https://pmc.ncbi.nlm.nih.gov/articles/PMC13441294/)
- Practical adoption: the AIND pipeline switched its default from KS2.5 (from Feb 2024) to KS4 (from Nov 2024). As of Aug 2025, 789 sessions had been sorted with KS2.5 and 468 with KS4. — [PMC13441294](https://pmc.ncbi.nlm.nih.gov/articles/PMC13441294/)
- Author caveats: Poisson hybrid trains may not match real firing statistics; template choice ignores brain region; whether more realistic hybrids would change accuracy "remains to be tested". In response to review, the authors added that idealized template injection "might miss extreme multi-unit overlapping dynamics". Two GT units were missed by both sorters. — [PMC13441294](https://pmc.ncbi.nlm.nih.gov/articles/PMC13441294/); [eLife reviews](https://elifesciences.org/reviewed-preprints/110170/reviews)
- eLife assessment: "valuable" significance, "compelling" evidence. A reviewer wrote that, as in the original KS4 paper, "Kilosort 4 significantly outperforms Kilosort 2.5". No reviewer critique of the sorter parameter choices was found in the review text read. — [eLife article](https://elifesciences.org/articles/110170); [eLife reviews](https://elifesciences.org/reviewed-preprints/110170/reviews)

**Study 2: SpikeInterface documentation, "Benchmark spike sorting with hybrid recordings" (how-to tutorial, NOT peer reviewed, living document).**
- Data: one Neuropixels 1.0 recording from the Steinmetz dataset with imposed triangular probe motion (figshare doi 10.6084/m9.figshare.14024495.v1); 384 ch, about 32.6 min. 31 IBL Brain-Wide Map templates (visual cortex, layers 5 and 6) rescaled to 50 to 150 µV. Hybrids were injected with drift-aware template interpolation. — [SpikeInterface docs](https://spikeinterface.readthedocs.io/en/latest/how_to/benchmark_with_hybrid_recordings.html)
- Sorters: kilosort2_5 (defaults), kilosort4 (nblocks=5), spykingcircus2 (defaults); also Kilosort3 is named. No versions given. — [SpikeInterface docs](https://spikeinterface.readthedocs.io/en/latest/how_to/benchmark_with_hybrid_recordings.html)
- Results, quoted: KS2.5, KS4 and SC2 "all find around 10-12 hybrid units with accuracy greater than 80%". KS4 "has a better overall curve, being able to find almost all units with an accuracy above 50%". KS2.5 "performs well when looking at precision" but "has a lower recall". — [SpikeInterface docs](https://spikeinterface.readthedocs.io/en/latest/how_to/benchmark_with_hybrid_recordings.html)
- Verdict: tied with KS2.5 on the count of well-detected (>0.8) units; KS4 better on recall and on the overall accuracy curve; KS2.5 better on precision. This is one recording and an illustrative tutorial, not a formal study.

**Study 3: Laquitaine, Imbeni, Tharayil, Isbister, Reimann (EPFL Blue Brain). "Spike sorting biases and information loss in a detailed cortical model." bioRxiv doi 10.1101/2024.12.04.626805 (PREPRINT, v2 c. Mar 2025; no journal version found).**
- Data: biophysically detailed simulation of rat non-barrel S1 (400 µm subvolume, 30,190 neurons, 60 m-types), with simulated Neuropixels 1.0 (384 ch) and a dense 128-ch probe (Horvath et al. geometry). Spontaneous and evoked conditions. Ground truth is the simulator's spike times. Results are compared with in vivo data (Marques-Smith; Horvath) and with the Buccino et al. synthetic MEArec-style model. — [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.12.04.626805v2.full)
- Sorters: Kilosort 1, 2, 2.5, 3, 4 and HerdingSpikes via SpikeInterface 0.100.5 (MATLAB KS run in R2022b). KS2.5 was run WITHOUT drift correction (nblocks = 0). minFR and minfr_goodchannels = 0. KS2 was run on only 4 min of evoked data because it failed on longer recordings. MountainSort4 was excluded (>8 h runtime). — [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.12.04.626805v2.full)
- KS4 (the primary sorter) results: about 41% of sorted units were single units. Only about 10 to 15% of nearby neurons were well sorted (>80% accuracy), and the abstract, Sec 2.3 and Sec 2.5 use different metrics for this. Yield was about one tenth of the theoretical maximum. Pyramidal cells were under-represented about 2x in most layers and over-represented about 2x in L5. Sorted firing rates were overestimated about 3 to 4x. Sorting biases cut stimulus-information capacity by about half. — [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.12.04.626805v2.full)
- KS4 vs older Kilosorts: per the text read, single-unit yield was not significantly different between sorters (Supp. Fig. 5a). Supp. Fig. 5 is titled "...more diverse spike assignment biases for older spike sorters". KS1 to KS2.5 over-represented rare interneurons by 2.9 to 8.1x. KS3 and KS4 were described qualitatively as producing no false-positive units, while older Kilosorts did. KS4 and KS3 firing-rate distributions fit lognormals better than older sorters'. For information capacity, 'degraded' units lost about 12% relative to 'good' units for KS4, KS2.5, KS1 and HerdingSpikes alike. — [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.12.04.626805v2.full)
- On the simpler Buccino synthetic dataset, the well-sorted fraction ranged from 50% (HerdingSpikes) to 99% (Kilosort2). The KS4 value on that dataset was not in the text read. — [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.12.04.626805v2.full)
- Verdict: KS4 is better than KS2.5 on bias and false-unit measures, not on yield. Caveats: KS2.5 was handicapped by disabled drift correction (probably irrelevant in a static simulation); simulated firing rates and amplitudes were lower than in vivo; numbers are inconsistent across sections.

**Study 4: Banaie Boroujeni, Womelsdorf, Kastner. "KIASORT: Knowledge-Integrated Automated Spike Sorting for Geometry-Free Neuron Tracking." J Neurosci 2026;46(27):e1594252026, doi 10.1523/JNEUROSCI.1594-25.2026 (PEER REVIEWED; preprint bioRxiv 10.1101/2025.07.10.664175, Jul 2025).**
- Data: simulations with neuron-specific drift and nonlinear waveform change, plus geometry-aware benchmarks; real macaque 128-ch DBC probe recordings (caudate, ACC, lPFC) and Allen mouse Neuropixels data. The only comparator is KS4. KS2.5 is not tested. KS4 parameters were "optimized ... to maximize performance", with values not reported. — [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.07.10.664175v1.full); [J Neurosci](https://www.jneurosci.org/content/46/27/e1594252026)
- Under pronounced neuron-specific drift, KIASORT reports "5-10% higher sorting precision" and "5-15% more high-quality single units (F1 score > 0.75)" than KS4. — [J Neurosci](https://www.jneurosci.org/content/46/27/e1594252026)
- KS4 held its own elsewhere, quoted: "Kilosort4 retained a modest advantage under its native 1-FN-FP > 0.8 quality score, whereas the methods were more comparable under an F1 > 0.75 criterion"; "In geometry-aware simulations, Kilosort4 yields a slightly higher number of units". On geometry-based benchmarks KIASORT "ranked highest after Kilosort4 among existing methods". — [J Neurosci](https://www.jneurosci.org/content/46/27/e1594252026)
- On real data, KIASORT had about 5% more refractory units among high-SNR units, but the authors state "a direct comparison between KIASORT and Kilosort4 on real data without ground truth is not a valid benchmark". KS4 "failed running using only CPU". — [J Neurosci](https://www.jneurosci.org/content/46/27/e1594252026); [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.07.10.664175v1.full)

**Study 5: Garcia, Halcrow, Windolf, McKenzie, Adkisson-Floro, Ramon Mayorquin, Dichter, Buccino, Yger. "Opening the black box: a modular approach to spike sorting." bioRxiv doi 10.64898/2026.01.23.701239; eLife Reviewed Preprint RP110588 (v1 Apr 1 2026, v2 Sep 3 2026). REVIEWED PREPRINT, not a final VOR.**
- Authors are SpikeInterface, SpyKING CIRCUS and DARTsort developers; this is a competing-sorter paper. — [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.01.23.701239v2.full)
- Data: SpikeInterface generative simulations (NP1-like 384 ch, 30 min, 500 units; rigid and non-uniform drift) plus supplementary geometries (CamNeuro 64 ch, NP2-like 128 ch, SiNAPS 128 ch, tetrode). Real data: chronic NP2 (Lebedeva), acute NP1 (IBL), chronic 64-ch CamNeuro (Duszkiewicz), scored with proxy curation tools (Bombcell, UnitRefine, SLAy). — [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.01.23.701239v2.full)
- Sorters: KS4 (4.1.1 in Methods and 4.1.2 in the real-data text, an internal inconsistency), SpyKING CIRCUS 2, TriDesClous 2, and the new Lupin. KS2.5 is NOT benchmarked. — [eLife RP110588](https://elifesciences.org/reviewed-preprints/110588v2)
- Results, quoted: Lupin "clearly outperforms all other sorters in all conditions (static or motion-corrected, including Kilosort4)". "Kilosort4 finds many false positives, especially for the motion-corrected recording". KS4 "finds many good units and runs quickly". On real data Lupin "is competitive with Kilosort4, finding on average a similar number of good units and less noisy units". — [eLife RP110588](https://elifesciences.org/reviewed-preprints/110588v2)
- eLife assessment: "valuable", "solid". The evidence "could be strengthened with extended simulations, broader benchmarking against existing sorters, and validation via expert curation". Caveats noted by the authors and in the text: Lupin was assembled from components selected on the same simulator used for evaluation (overfitting risk), and UnitRefine was trained on KS4 output, which may favour KS4 on the real-data proxies. — [eLife RP110588](https://elifesciences.org/reviewed-preprints/110588v2); [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.01.23.701239v2.full)

**Study 6: Liu, Hu, Zhang, et al. (Duan, Li). "Scalable spike sorting across thousands of neurons by modeling neural dynamics with NeuroSort." bioRxiv doi 10.1101/2025.11.11.687825 (PREPRINT, Nov 2025).**
- Data: rat PFC Neuropixels 1.0 (one 32-min session); macaque 1,024-ch Neuroscroll (one 5-min session); macaque Utah array (manual labels, but not used against KS4). Comparators: KS4 and MountainSort5 at default settings. No KS2.5. — [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.11.11.687825v1.full)
- Proxy results only: about 3.6% of KS4 units had SNR < 4. MountainSort5 found about 20% as many units as KS4. On Neuroscroll, NeuroSort produced 1,964 units with a mean "sorted score" of 0.997, while some KS4 units scored ≤ 0.9 near amplitudes of about 10 µV. KS4 took about 2x NeuroSort's runtime and used 22 vs 12 GB VRAM. Several authors are affiliated with a company (Shenzhen We-Linking Medical Technology). — [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.11.11.687825v1.full)

**Related, no KS4 or KS2.5 (context only): De Preter, Leimer, Sonneborn, Heinricher. "Comparative analysis of spike-sorters in large-scale brainstem recordings." bioRxiv 10.1101/2024.11.11.623089 (PREPRINT, Nov 2024).**
- Rat RVM, Cambridge NeuroTech 64-ch probe. KS3, MS5, IronClust, Tridesclous and SpyKING CIRCUS via SpikeInterface; no ground truth; manual Phy curation plus consensus. Of 671 post-processed units, only 69 (10%) were found by all five sorters and 487 (73%) by one sorter. "Kilosort3 and IronClust required the least curation while maintaining the largest number of units." KS4 and KS2.5 were not tested. — [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.11.11.623089.full)

**SimSort (arXiv 2502.03198, 2025):** its benchmarks compare against Kilosort, Kilosort2 and MountainSort4/5 (MS5 about 0.66 vs MS4 about 0.84 accuracy on simulated BBP L6 data). No KS4 entry was found. — [arXiv](https://arxiv.org/pdf/2502.03198)

### Inferences
- The independent evidence that KS4 beats KS2.5 rests mainly on one peer-reviewed hybrid study (Buccino et al. 2026), which used a large GT set (4800 injected units across 96 NP1 and NP2 recordings). The direction matches the KS4 paper. KS4's advantage is concentrated in low-SNR units (< 10) and in recall and completeness.
- The KS4 vs KS2.5 margin is metric-dependent. Population-level precision favours KS4, but per-unit precision slightly favours KS2.5, and refractory contamination is marginally higher for KS4. The SpikeInterface tutorial shows the same pattern (KS2.5 better precision, KS4 better recall), and the count of >0.8-accuracy units was tied.
- Newer sorter papers (KIASORT, Lupin, NeuroSort) treat KS4 as the reference state of the art and report beating it only in specific regimes: heterogeneous per-neuron drift, fewer false positives under motion correction, and low-amplitude units. All of them have developer bias and none includes KS2.5.
- The Buccino study comes from the AIND/SpikeInterface team, and Buccino is also an author on the Lupin paper. These are independent of the Kilosort authors but are not neutral with respect to SpikeInterface tooling.

### Gaps
- No independent study found that uses paired juxtacellular or patch ground truth to compare KS4 with KS2.5.
- No SpikeForest update including KS4 was found within the search budget.
- No independent KS4 vs KS2.5 comparison was found on Neuropixels Ultra (as a sorted probe), Utah arrays, tetrodes, HD-MEA or retina, primate or human Neuropixels, or in vitro MEA.
- Exact KS4 and KS2.5 version numbers and parameter files for Buccino et al. were not in the text read; they may be in the Methods or the GitHub configs.
- Per-sorter numbers in Laquitaine et al. (Supp. Fig. 5a yield; Fig. 7 information capacity) and the Lupin paper (Figs. 7 and 8) are only in figures and could not be extracted.

## Are there studies where KS2.5 outperforms KS4?

### Takeaway
No independent study found shows KS2.5 beating KS4 overall. KS2.5 wins or ties only on narrow metrics: per-unit precision and refractory contamination (Buccino et al.), precision in the SpikeInterface hybrid tutorial, and number of >0.8-accuracy units (tied in the tutorial). Laquitaine et al. found no significant yield difference between Kilosort versions in a biophysical simulation. Several studies report false positives as KS4's main weakness, especially after motion correction.

### Cited Findings
- KS4 per-unit precision is slightly worse than KS2.5's: "at the expense of a slight degradation of precision ... finds more false positive spikes". Refractory contamination is slightly higher (effect size 0.013). KS4 runs about 2x slower. — [eLife article](https://elifesciences.org/articles/110170)
- SpikeInterface tutorial: KS2.5 "performs well when looking at precision" but has "a lower recall". The number of units at >80% accuracy was similar for KS2.5, KS4 and SC2 (about 10 to 12). — [SpikeInterface docs](https://spikeinterface.readthedocs.io/en/latest/how_to/benchmark_with_hybrid_recordings.html)
- Biophysical simulation: yield was "not significantly different between sorters" (Supp. Fig. 5a). KS4 was better on assignment biases (KS1 to KS2.5 over-represented rare interneurons 2.9 to 8.1x) and on false-positive units. — [bioRxiv Laquitaine et al.](https://www.biorxiv.org/content/10.1101/2024.12.04.626805v2.full)
- On the simpler Buccino synthetic model, the best well-sorted fraction was Kilosort2 (99%). The KS4 value was not extracted, so whether an older Kilosort beat KS4 there is unconfirmed. — [bioRxiv Laquitaine et al.](https://www.biorxiv.org/content/10.1101/2024.12.04.626805v2.full)
- KS4 false positives under motion correction (vs Lupin and SC2, not KS2.5): "Kilosort4 finds many false positives, especially for the motion-corrected recording". — [eLife RP110588](https://elifesciences.org/reviewed-preprints/110588v2)
- When KIASORT authors lowered KS4's threshold, it "tended to introduce false positives". — [J Neurosci](https://www.jneurosci.org/content/46/27/e1594252026)

### Inferences
- The recurring theme across independent work is that KS4 trades a little precision and contamination for substantially better recall and completeness, especially at low SNR. Users who prioritise conservative, low-contamination units may see a smaller KS4 benefit after curation. This is an inference; no study measured post-curation KS4 vs KS2.5 yield directly.

### Gaps
- No independent study found on KS4 over-splitting vs KS2.5, on non-Neuropixels probes (KS2.5 vs KS4), or on drift scenarios where KS2.5 wins.
- No independent report quantifying post-curation (e.g., Bombcell or UnitRefine 'good') unit yield for KS4 vs KS2.5 on the same real recordings.

## What do new sorter papers from 2024 to 2026 (using KS4 as a baseline) report about KS4 versus KS2.5?

### Takeaway
None of the new sorter papers found (KIASORT, Lupin/modular SpikeInterface sorter, NeuroSort, SimSort) benchmark KS2.5 alongside KS4. KS2.5 is dropped as a baseline and described as legacy MATLAB software. KS4 is uniformly treated as the state-of-the-art reference.

### Cited Findings
- Lupin paper: KS2.5 appears only in the introduction as an older MATLAB tool that is hard to run on modern GPUs. KS4 is the main comparator. — [eLife RP110588](https://elifesciences.org/reviewed-preprints/110588v2)
- KIASORT: KS4 is the only comparator, "optimized" by the authors. — [J Neurosci](https://www.jneurosci.org/content/46/27/e1594252026)
- NeuroSort: KS4 and MS5 at default settings; KS2.5 not mentioned. — [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.11.11.687825v1.full)
- SimSort: compares with Kilosort, Kilosort2 and MountainSort4/5, not KS4 or KS2.5. — [arXiv](https://arxiv.org/pdf/2502.03198)
- A live/online sorting system preprint (bioRxiv 10.64898/2025.12.29.696938) used full-recording KS4 output as its reference "ground truth" for real data. This was seen only in a search snippet and not verified in the full text. — [bioRxiv](https://www.biorxiv.org/content/10.64898/2025.12.29.696938v1.full)

### Inferences
- In 2025 to 2026, KS4 is the de facto baseline, so new-sorter papers give little direct evidence on KS4 vs KS2.5. They mainly show where KS4 is beatable: heterogeneous drift, false positives under motion, and low-amplitude units.

### Gaps
- DARTsort (Paninski lab) results vs KS4 were not retrieved within the search budget.
- Other 2024 to 2026 sorters (e.g., Lightning-pose-style or deep-learning sorters, Rastermap-adjacent tools, SpikeInterface SC2 papers) were not checked individually.

## Any meta-analyses, reviews, or perspective pieces summarizing the evidence?

### Takeaway
No dedicated independent meta-analysis or review of KS4 vs KS2.5 performance was found. The closest syntheses are reviewer commentary in eLife (both of the AIND pipeline paper and of the Lupin paper), which accepts that KS4 outperforms KS2.5 while calling for broader benchmarking.

### Cited Findings
- An eLife reviewer of Buccino et al. echoed that KS4 "significantly outperforms Kilosort 2.5", and another said KS4 performs best "out of the tested options". — [eLife reviews](https://elifesciences.org/reviewed-preprints/110170/reviews)
- The eLife assessment of the Lupin paper calls for "broader benchmarking against existing sorters, and validation via expert curation". — [eLife RP110588](https://elifesciences.org/reviewed-preprints/110588v2)
- Laquitaine et al. argue that even the best current sorter (KS4) isolates only about 10 to 15% of nearby neurons well and introduces cell-type and firing-rate biases that roughly halve stimulus information. This is a perspective-level caution that applies to all Kilosort versions. — [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.12.04.626805v2.full)

### Inferences
- The independent literature is thin: one strong peer-reviewed hybrid head-to-head, plus several developer-authored new-sorter papers. Overall it supports "KS4 at least as good as and usually better than KS2.5", with documented precision and contamination trade-offs. It does not establish KS4's superiority on real-data curated yield, non-Neuropixels hardware, or paired-recording ground truth.

### Gaps
- No systematic review or meta-analysis of post-2024 spike-sorter benchmarks was found.
- SpikeForest and SpikeInterface public leaderboards with KS4 entries were not located.
