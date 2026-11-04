# Kilosort4 output quality on MouseLand/Kilosort GitHub (issues, releases, docs), Feb 2024 – Nov 2026

Method note (applies to all sections): read-only, unauthenticated GitHub REST API (search API, 34 queries; comment threads of ~60 issues), the GitHub releases page, PyPI release dates, and the local checkout's `git log`/tags (v4.0.x–v4.1.7, plus `origin/main`). GitHub Discussions are **disabled** on this repo (`has_discussions: false`; /discussions returns 404), so all user evidence is in Issues. Search coverage is keyword-based; ~145 post-Nov-2023 issues matched, of which ~60 quality-relevant threads were read in full. Issue URLs are `https://github.com/MouseLand/Kilosort/issues/N`. Versions listed are those stated by the user or visible in the thread; "?" means not stated in text I could read (often only inside an attached log).

## 1. Issues reporting KS4 finding fewer/worse units than KS2.5 (or KS2/KS3) on the same data, and maintainer explanations

### Takeaway
About a dozen threads directly compare KS4 to KS2.5/pykilosort/KS2/KS3 on the same data; almost all involve non-standard geometry (multi-shank, 2D MEA, tetrodes, sparse/single-column Neuropixels maps, low-channel probes) or carried-over KS2-era parameters. Maintainers almost never concede an algorithmic deficit: the fixes offered were a geometry bug fix (v4.0.4), disabling drift correction (`nblocks=0`), raising detection thresholds, reverting `cluster_downsampling`, or "use defaults"; in the one Neuropixels-standard-map comparison (#634, #667) Marius Pachitariu judged both sorters similar.

### Cited Findings
Catalogue (date, KS4 version, probe → observation → maintainer response → resolution):
- **#964** (2025-07-21; KS4 ?; one 32-ch 4-shank uLED probe + one 16-ch linear silicon probe, treated as 5 shanks): "yield with kilosort4 is roughly 50% of that with kilosort2.5" ("good" units); user has no such problem on 1-shank Neuropixels. KS4 detected ~1.5M total spikes vs 687,732 for KS2.5; KS4 "good" units all had very low spike counts, most spikes clumped into MUA. Removing the 16-ch probe (32 ch, 4 shanks) still gave KS4 more spikes but fewer units and fewer good units. Jacob Pennington: first `nblocks=0` ("unlikely that KS4's drift correction will work well for your probe configuration") — did not help; after receiving data he set `Th_universal=15`, `Th_learned=12.5`, `nblocks=0` → "around 20 good units and a similar spike count to KS2.5". User: "seems to work". Closed, resolution = parameter advice (thresholds far above defaults 9/8). Note: KS2.5 output had all units assigned to shank 0 (channel-map confound noted by the user). — [#964](https://github.com/MouseLand/Kilosort/issues/964)
- **#968** (2025-07-29; v4.0.13, then v4.1.0; Neuropixels with custom sparse map — one of two columns enabled per shank; hour-long freely moving mouse recordings): KS2.5 "consistently yield 30-55 high-quality, well-isolated units per session"; KS4 "results are poor", merges two distinct neurons, "seems to lose units over time". Jacob: v4.0.13 "pretty outdated", a fix for staggered maps exists; on v4.1.0 all clusters looked like noise. Jacob then suggested (separately) `cluster_downsampling = 20` ("the old default (before v4.1.0)") and `Th_universal = 10`, `Th_learned = 9` (log-spike-count vs amplitude plot suggested low-amplitude noise detected as spikes). User: both removed "small alternating noise" but downsampling gave very low waveform amplitudes; thresholds "a bit better". Closed without a stated KS2.5-parity outcome. — [#968](https://github.com/MouseLand/Kilosort/issues/968)
- **#634** (2024-03-26; v4.0.2 era; Neuropixels 1.0): Good and MUA cluster counts "fairly equal" between KS2.5 and KS4, but total spikes in Good clusters "atleast 50% less in KS4". Marius: "This will depend on your particular experimental prep, probe geometry, thresholds etc."; load both in Phy side by side; promised a guide for evaluating/fixing problems. Closed, pointing to #642. — [#634](https://github.com/MouseLand/Kilosort/issues/634)
- **#642** (2024-03-27, still **open** Nov 2026): maintainer feature request "DIY guide to improve the results of Kilosort" — "It's very difficult to anticipate everything that could go wrong with different types of data and experiments." — [#642](https://github.com/MouseLand/Kilosort/issues/642)
- **#607** (2024-03-05; v4.0.x pre-4.0.4; Intan, 64-ch Cambridge NeuroTech, uint16, 30-min file): KS4 with defaults gave no real units (motion artefacts/flat lines) vs good units in pykilosort ("claims to be using Ks2.5 algorithm"); channels with KS2.5 good units had 0 units in KS4. Jacob fixed a uint16 conversion bug (did not fix it); Marius attributed it to multi-shank handling; workaround = stack shanks vertically; closed as the known multi-shank issue (fixed in v4.0.4). A second user (Neuropixels 3B1, int16) said KS4 output "seems completely different" from Matlab KS2.5; Marius asked whether KS2.5 thresholds had been used ("That would result in a lot of noise clusters"). — [#607](https://github.com/MouseLand/Kilosort/issues/607)
- **#619** (2024-03-14; early 4.0.x; 2-shank probe): KS2.0 sorted easily; KS4 found 0 spikes. Fixed by stacking shanks; Jacob: drift correction needs "at minimum, ~32 channels on one shank (preferably more, and with denser spacing)" and "in general doing drift correction with an insufficiently large & dense probe is likely to make things worse ... large drift amounts that don't exist in the data." — [#619](https://github.com/MouseLand/Kilosort/issues/619)
- **#663** (2024-04-11; 4.0.3→4.0.4; Litke retinal MEAs, 60 µm rectangular and 30/120 µm hexagonal): "~1000 good units using kilosort2.5 ... only getting around 130 good units with kilosort4" (135 with `dminx=60`; `dminx=120` → GPU OOM on 48 GB). Jacob: known template placement/grouping issue for multi-shank and 2D arrays; fixed in **v4.0.4** with new `x_centers` (try 10), `dmin=dminx=60`, `max_channel_distance`, `nblocks=0`. User then: "getting a lot of good clusters" (no count given); gaps in spike positions between electrodes deemed physically expected; some runs took >42 h. Closed. — [#663](https://github.com/MouseLand/Kilosort/issues/663)
- **#667** (2024-04-18; v4.0.x; Neuropixels 1): user reports KS4 over-merging (bimodal amplitude, refractory violations) in a low-amplitude region vs pyKilosort. Marius: "Looks similar doesn't it?" — both sorters similarly sparse in that region; "both algorithms perform similarly in terms of how many good units they find". Closed, no change. — [#667](https://github.com/MouseLand/Kilosort/issues/667)
- **#668** (2024-04-18; tetrodes): data sorted normally in KS3, "garbled" waveforms in KS4. Fixed by v4.0.4 + `dminx = 10` (user confirmed well-sorted units). — [#668](https://github.com/MouseLand/Kilosort/issues/668)
- **#815** (2024-11-10 → 2025-03; 4-ch tetrodes, later 4 recordings concatenated to 16 ch): user had used dummy channels; whitening suppressed spikes; KS4 "consistently get less than 20 clusters", while "KS2 previously being able to detect a similar spread of good spikes". Jacob: concatenate tetrode recordings to ≥16 ch; artefact removal; after receiving data, tuned settings (batch_size 1,000,000 samples = 40 s, `whitening_range=4`, `dmin=dminx=20`, `nearest_chans=4`, `nearest_templates=10`) — "improve the results a little". — [#815](https://github.com/MouseLand/Kilosort/issues/815)
- **#995** (2025-10-09; v4.1.1; NeuroNexus 4x16, 14 ch per shank, `nblocks=0`): "ever since upgrading from Kilosort2 to Kilosort4" many clusters with near-identical waveforms (7 in example). Jacob: could be distinct units (Phy shows unwhitened waveforms); may be good units flagged MUA because of very short refractory period; drift correction best with "at least 32 channels on a shank, preferably 64 or more". Unresolved; user kept them separate. — [#995](https://github.com/MouseLand/Kilosort/issues/995)
- **#748** (2024-07-29; NeuroNexus A4x16): strange KS4 waveforms vs normal in KS2; user closed it themselves ("Sorted for now") with no explanation. — [#748](https://github.com/MouseLand/Kilosort/issues/748)
- **#657** (2024-04-05; KS3, NeuroNexus A4x4): KS3 found 8 units vs ~32 by manual sorting; Marius: "Can you please try Kilosort4?" — [#657](https://github.com/MouseLand/Kilosort/issues/657)
- **#1034** (2026-04-20; Plexon 64-ch V-probe, 96-ch Utah, macaque V1): oversplit/undersplit units with KS2-era settings (`n_pcs=3` "a carry over from kilosort2", `Th_learned=6`). Jacob: use defaults; reducing `n_pcs` "will most likely make the results worse"; treat Utah as linear. User after defaults: "the results look fantastic". — [#1034](https://github.com/MouseLand/Kilosort/issues/1034)
- **#594** (comments Oct 2024): a KS2 user: "We haven't found KS4 to work better with our setup, so we're sticking with 2 for now." Marius: KS4 "never had this problem [spike holes] and ... performs better in most cases"; invited a new issue. — [#594](https://github.com/MouseLand/Kilosort/issues/594)

### Inferences
- The largest reported gaps (KS4 ≈10–50% of KS2.5 "good" yield: #663, #964, #968) were all on non-standard geometries; the two resolved ones were fixed by geometry handling (v4.0.4) or by thresholds well above default, not by algorithm changes. No thread demonstrates a standard dense Neuropixels map where KS2.5 was shown to be consistently better.
- "Fewer good units" in KS4 often coexists with *more total spikes* (#964, #634, #966): KS4 at defaults appears to detect more low-amplitude events, which end up in MUA/noise clusters and dilute "good" clusters — i.e. threshold calibration differs from KS2.5 and comparisons of KSLabel counts are confounded.
- Many comparisons are confounded by: KS2-era parameters reused in KS4 (#607 second user, #845, #1034, #675), outdated KS4 versions (#968 v4.0.13, #787 v4.0.4, #897 v4.0.4), pykilosort vs Matlab KS2.5 (#607, #667), channel-map errors (#877, #986, #964), and KS2.5 spike-hole bug in pre-Apr-2024 KS2.5 outputs (#594).

### Gaps
- KS2.5 good-unit counts for #964 were only in images; KS4 version for #964 is only in the attached log.
- #968 ended without a stated final unit count vs KS2.5's 30–55.
- No issue reports a controlled, quantitative KS2.5-vs-KS4 benchmark with ground truth or with matched curation.

## 2. Recurring problem categories (catalogue)

### Takeaway
The recurring categories are: drift correction misbehaving on small/sparse/multi-shank probes (default advice `nblocks=0`), geometry/template-placement problems on non-Neuropixels layouts, threshold sensitivity (both directions), residuals clustered as separate units and similar-waveform "duplicates", occasional over-merging/over-splitting that maintainers treat as acceptable error, and noise/artefacts that maintainers consistently say must be removed before sorting.

### Cited Findings
**Drift correction making things worse / unreliable**
- Docs: for "probes with fewer channels (around 64 or less) or with sparser spacing (around 50um or more between contacts), drift estimates are not likely to be accurate, so drift correction should be skipped by setting `nblocks = 0`." — [parameters.rst](https://github.com/MouseLand/Kilosort/blob/main/docs/parameters.rst)
- #686 (2024-05-07, still open; 2-shank 64-ch): noisy drift estimate; Jacob: "~64 channels (on a single shank) is the minimum we recommend"; led to new `drift_smoothing` parameter (v4.0.8); Jacob later (2024-07-03): "we did identify a bug related to this [time smoothing producing outliers] that we'll have to work on." Marius: for large physical readjustments, split the data and match units with UnitMatch etc. — [#686](https://github.com/MouseLand/Kilosort/issues/686)
- #765 (2024-08-21; v4.0.16; NP1.0 NHP long, two banks as single long column): no spikes detected; drift estimate "~7000 microns"; root cause = template placement with `dminx=103` putting a column of templates where no channels exist; workaround `max_channel_distance = 20` (fixed). — [#765](https://github.com/MouseLand/Kilosort/issues/765)
- #1006 (2025-11-10; NP2.0, 1–2 h): drift estimate moves "4 channels up" while matched clusters are 10–12 channels above; Jacob: recording unstable after ~3000 s; suggested `tmax=3000`, `nblocks=1`, `batch_size=30000`; mismatch persisted at another time point; Jacob: drift traces are a side effect of units dropping in/out. — [#1006](https://github.com/MouseLand/Kilosort/issues/1006)
- #897 (2025-03-24; v4.0.4→4.0.30; Cambridge NeuroTech H9 in spinal cord): same spike detected on multiple channels and not clustered together; Jacob: large noisy drift estimates can interfere with clustering; try `tmax` or `batch_size` ~300000. — [#897](https://github.com/MouseLand/Kilosort/issues/897)
- #934 (2025-05; chronic NP, two probes): ~30%+ of clusters split in time at the same moments; Jacob: no parameter will prevent it; truncate with `tmax`, or `nblocks=5`/per-shank sorting; user found saturation in that timeframe. — [#934](https://github.com/MouseLand/Kilosort/issues/934)
- #1033 (2026-04-15; NP1.0): high-frequency oscillations in drift trace; Jacob: likely recording quality after ~5000 s; `tmax = 5000`. — [#1033](https://github.com/MouseLand/Kilosort/issues/1033)
- #964, #957, #995, #619: `nblocks=0` recommended for multi-probe/uLED, tetrodes, 14-ch shanks, 2-shank probes. — [#957](https://github.com/MouseLand/Kilosort/issues/957), [#619](https://github.com/MouseLand/Kilosort/issues/619)

**Non-Neuropixels / sparse / multi-shank / 2D / tetrode geometry**
- Early 4.0.x mishandled multi-shank probes (workaround: stack shanks vertically, #607, #619) and 2D MEAs (#663); fixed in v4.0.4 (grouping centers per shank, `x_centers`, `max_channel_distance`). — [v4.0.4 release](https://github.com/MouseLand/Kilosort/releases/tag/v4.0.4)
- #891 (2025-03-10; v4.0.24; 1024-ch 4-shank SiNAPS): a few out-of-brain artefact channels caused clean distant regions to be "entirely missed"; Jacob: "expected behavior" because drift correction and universal templates use all channels; remove bad channels first. — [#891](https://github.com/MouseLand/Kilosort/issues/891)
- #1019 (2026-01-09; 8 tetrodes): template-generation bug for some tetrode layouts acknowledged ("a bug in the template generation"), workaround `dmin = 15.99`; fixed in commit d947208 (2026-01-28, shipped in v4.1.6). — [#1019](https://github.com/MouseLand/Kilosort/issues/1019)
- #957 (2025-06-28; 4 probes × 8 tetrodes): units appear to span tetrodes — display artefact of `nearest_chans=10`; `nblocks=0`. #927: paper's tetrode dataset was sorted "with default settings except for `nblocks = 0`". — [#957](https://github.com/MouseLand/Kilosort/issues/957), [#927](https://github.com/MouseLand/Kilosort/issues/927)
- #949 / #986 (2025; Cambridge NeuroTech H-series 32-ch, insects; v4.0.37/4.1.1): 10–30 putative units where 1–2 expected; Jacob: noise interfering with clustering — "Tinkering with KS4 parameters is not going to fix that"; #986 traced partly to a wrong channel map, then to a residual "being clustered as a separate unit" (suggest `Th_learned` 8.5–9) and to unstable recording segments (`tmin=800`, `tmax=2500`). — [#949](https://github.com/MouseLand/Kilosort/issues/949), [#986](https://github.com/MouseLand/Kilosort/issues/986)

**Over-merging**
- #710 (2024-06-06): good unit with clean ACG but two waveform/position clusters (place cells rarely co-active). Marius: probably found separately "and then merged in the pairwise comparison step which is the last thing that happens in the pipeline"; suggested the torus-paper merge modification; "we do not have a lot of bandwidth". Follow-up 2026-10-12 on **v4.1.7**: putative MEC grid cells "classified as MUA before the final merging step", asking for Gardner et al. 2022-style settings (no reply seen). — [#710](https://github.com/MouseLand/Kilosort/issues/710)
- #667 (see §1), #801 (2024-10-19; noisy ISI cluster with core + halo; Jacob: recording artefact, `artifact_threshold`), #990 (2025-09-16; 128-ch multi-shank, "many multi-units, as if Kilosort4 is over-merging"; no data supplied, closed), #997 (2025-10-15; mixed-amplitude clusters; Jacob: "There is nothing to fix here", ambiguous MUA). — [#801](https://github.com/MouseLand/Kilosort/issues/801), [#990](https://github.com/MouseLand/Kilosort/issues/990), [#997](https://github.com/MouseLand/Kilosort/issues/997)
- #685 (2024-05-07): `ccg_threshold` is combined with a hard-coded criterion (`Q12<ccg_threshold and (R12<.05)`), so raising it may do nothing; `cluster_pcs` "currently has no purpose" (later removed). — [#685](https://github.com/MouseLand/Kilosort/issues/685)

**Over-splitting / duplicates / residual units**
- #742 (2024-07-22): bimodal amplitude = residual of one template learned as a separate template, "~2% of the detected spikes ... an acceptable error". — [#742](https://github.com/MouseLand/Kilosort/issues/742)
- #664 (2024-04-12): spikes misaligned within units after merging (~5% of units for one user); fixed in **v4.0.31** ("update merging to include feature shifts"); a 2026 report on v4.1.3 (0.5 ms shifts, ~1 in 40 units) was attributed to residuals clustered separately — "There isn't really a 'solution' or any parameter tweaks". — [#664](https://github.com/MouseLand/Kilosort/issues/664)
- #877 (2025-02-21; v4.0.20; NP2.0 4-shank): ~10% of clusters had a near-identical twin at a distant position with 99% coincident spikes → user found wrong channel mapping. — [#877](https://github.com/MouseLand/Kilosort/issues/877)
- #1030 (2026-03-20; Axion MEA): large waveform split across several clusters; Jacob: likely artefact or ripple-like event. #1034 strangely time-split units with KS2-era parameters. — [#1030](https://github.com/MouseLand/Kilosort/issues/1030)
- v4.1.3 release note: `cluster_downsampling` reverted to 20 partly because of "issues with oversplit units for some datasets". — [v4.1.3](https://github.com/MouseLand/Kilosort/releases/tag/v4.1.3)
- #1042 / PR #1043 (2026-08-21 / 2026-09-01): typo `len(st2 == 0)` in `swarmsplitter.check_CCG` (added in v4.1.5) makes the guard fire for every non-empty train, so the refractory-CCG veto on splits "has been dead code in v4.1.5–v4.1.7. Splits are currently decided by modularity and bimodality alone, which allows oversplitting" (PR author's claim). Merged to main 2026-09-25 (merge commit 0e07514) with regression tests. — [#1042](https://github.com/MouseLand/Kilosort/issues/1042), [PR #1043](https://github.com/MouseLand/Kilosort/issues/1043)

**Missing low-amplitude / specific units; threshold sensitivity**
- #675 (2024-04-23): KS4 missed a fast-spiking unit Plexon Offline Sorter found; Marius: no underlying reason; try `templates_from_data = False`; Jacob: many non-default params (`nt = 801`); user: switching to predefined universal templates "seems to have worked". — [#675](https://github.com/MouseLand/Kilosort/issues/675)
- Docs: `Th_universal`/`Th_learned` "loosely similar to Th(1) and Th(2) in previous versions"; lower `Th_learned` if neurons disappear/reappear; lower `Th_universal` for more units; change by 1–2 at a time. README: "Re-using parameters from previous versions of Kilosort will probably not work well." — [parameters.rst](https://github.com/MouseLand/Kilosort/blob/main/docs/parameters.rst), [README](https://github.com/MouseLand/Kilosort/blob/main/README.md)
- Threshold advice given in threads spans 7.5 to 15: #957 lower to 8.5/7.5; #927 raise `Th_universal` to 9.5–10; #986 `Th_learned` 8.5–9; #968 10/9; #964 15/12.5; #845 user tried 7/7 to 20/20 (defaults 9/8) for synchronous high-rate events — Jacob: matching pursuit fails at very high spike density; try `max_peels` (v4.0.24). — [#957](https://github.com/MouseLand/Kilosort/issues/957), [#927](https://github.com/MouseLand/Kilosort/issues/927), [#845](https://github.com/MouseLand/Kilosort/issues/845)

**Noise / artefact units**
- Consistent maintainer position: KS4's built-in preprocessing (highpass, CAR, whitening) "will not be able to remove all kinds of artifacts"; remove them before sorting or use the blunt `artifact_threshold` (zeros whole batches). — [#684](https://github.com/MouseLand/Kilosort/issues/684), [#801](https://github.com/MouseLand/Kilosort/issues/801), [#815](https://github.com/MouseLand/Kilosort/issues/815), [#1009](https://github.com/MouseLand/Kilosort/issues/1009), [#927](https://github.com/MouseLand/Kilosort/issues/927), [#787](https://github.com/MouseLand/Kilosort/issues/787)

**Short recordings / nondeterminism**
- #900 (2025-03; 8-shank 32 ch): ~2 min of data "likely not enough data to get reliable results for only 32 channels"; concatenated file gave no good units (unresolved). #1019: 5-min files should be combined; unit-count changes between runs "consistent with the amount of noise that can show up in the clustering process". v4.1.1: library version differences "can change sorting output". — [#900](https://github.com/MouseLand/Kilosort/issues/900), [#1019](https://github.com/MouseLand/Kilosort/issues/1019), [v4.1.1](https://github.com/MouseLand/Kilosort/releases/tag/v4.1.1)

**High sampling rate**
- #675: using `nt = 61` at a higher sampling rate shortens waveforms to ~1.5 ms, "required for using the default templates"; #742 (20 kHz): no need to lower `nt` from 61. — [#675](https://github.com/MouseLand/Kilosort/issues/675), [#742](https://github.com/MouseLand/Kilosort/issues/742)

**"good" (KSLabel) and ContamPct**
- #995: units may be flagged MUA "because they have a very short refractory period". #1044 (2026-09-01, open; PR #1045 open): clusters with ≤10 spikes export `ContamPct = 0.0` (best score) because `CCG.refract` only evaluates clusters with >10 spikes. v4.0.15 added a tutorial for relabelling good/mua with custom criteria. — [#995](https://github.com/MouseLand/Kilosort/issues/995), [#1044](https://github.com/MouseLand/Kilosort/issues/1044), [v4.0.15](https://github.com/MouseLand/Kilosort/releases/tag/v4.0.15)
- #1039 (2026-07-26, open): `make_pc_features` may attach PC features to wrong channel IDs when final clusters share a detection template; Jacob: that case "should never happen to begin with". — [#1039](https://github.com/MouseLand/Kilosort/issues/1039)

### Inferences
- The typical support path is: request `kilosort4.log` → spot an outdated version or non-default parameters → advise defaults, `nblocks=0`, or threshold changes → close. Many "quality" threads end without a quantitative before/after.
- Maintainers treat occasional over/under-merges and residual units as an inherent accuracy cost of automation (#742, #997, #1009) rather than bugs, except when systematic (multi-shank, template placement, misalignment, check_CCG).
- Threshold defaults (9/8) seem poorly calibrated for some low-channel/noisy preparations: maintainers themselves recommended values from 7.5 up to 15 depending on data.

### Gaps
- The `drift_smoothing` time-axis bug acknowledged in #686 (July 2024): I found no release note or commit confirming a fix.
- #710's 2026 follow-up (grid cells labelled MUA on v4.1.7) had no maintainer reply visible.
- I could not quantify how much the v4.1.5–4.1.7 check_CCG bug changed unit counts; no issue reports measured impact.

## 3. Quality-affecting bug fixes and default changes by version

### Takeaway
Quality-relevant changes cluster in three periods: April–June 2024 (v4.0.3–4.0.13: geometry/multi-shank/2D handling, template placement, amplitude scaling, drift smoothing, duplicate-spike window), Dec 2024–Apr 2025 (v4.0.22–4.0.31: duplicate templates on multi-shank, `max_peels`, merge misalignment fix), and July 2025–2026 (v4.1.0 clustering-subset default, reverted in v4.1.3; a v4.1.5 split-veto regression fixed on main only in Sept 2026, unreleased as of PyPI 4.1.7).

### Cited Findings
- **v4.0.2** (2024-03-23): "Initial release of Kilosort4, linked with paper." — [releases](https://github.com/MouseLand/Kilosort/releases)
- **v4.0.3** (2024-04-10): `dminx` no longer auto-determined (default 32; fixes checkerboard bug); fixed possible negative spike times in first batch; `pc_features.npy` added. — [v4.0.3](https://github.com/MouseLand/Kilosort/releases/tag/v4.0.3)
- **v4.0.4** (2024-04-13): multi-shank probes work without stacking; distance gating of universal templates (`max_channel_distance`), grouping centers per shank and `x_centers` for 2D arrays (resolved #607, #619, #663, #668). — [v4.0.4](https://github.com/MouseLand/Kilosort/releases/tag/v4.0.4)
- **v4.0.7** (2024-05-09): template amplitude scaling fix (Phy amplitudes); reverted the negative-spike-time "fix" because it caused misalignment. — [v4.0.7](https://github.com/MouseLand/Kilosort/releases/tag/v4.0.7)
- **v4.0.8** (2024-05-25): new `drift_smoothing` parameter (from #686). — [v4.0.8](https://github.com/MouseLand/Kilosort/releases/tag/v4.0.8)
- **v4.0.11** (2024-06-08): template placement handled separately per shank; float32 shift/scale parameters. — [v4.0.11](https://github.com/MouseLand/Kilosort/releases/tag/v4.0.11)
- **v4.0.13** (2024-06-27): `duplicate_spike_bins` → `duplicate_spike_ms` (default 0.25); `highpass_cutoff` added. — [v4.0.13](https://github.com/MouseLand/Kilosort/releases/tag/v4.0.13)
- **v4.0.22** (2024-12-30): "Exclude empty clustering centers"; "Fixed duplicate templates generated for multi-shank probes"; `max_channel_distance` default constant 32; fix for spike positions extending between shanks. — [v4.0.22](https://github.com/MouseLand/Kilosort/releases/tag/v4.0.22)
- **v4.0.24** (2025-01-15, PyPI): `max_peels` parameter (from PR #809) and Cf amplitude exposed in `st` (PR #844) — per local git log commits db1443b, 169d3fb. — [#845](https://github.com/MouseLand/Kilosort/issues/845)
- **v4.0.26–4.0.28** (Feb–Mar 2025): clustering memory fixes (non-consecutive channel maps); v4.0.26 had a bug in `clustering_qr.run` fixed in 4.0.28; a "bug from CCG swap in previous release" fixed 2025-03-18 (commit 47014f3). — [v4.0.26](https://github.com/MouseLand/Kilosort/releases/tag/v4.0.26)
- **v4.0.31** (2025-04-13): "Fixed #664 regarding misalignment of spike waveforms and features after merging clusters." — [v4.0.31](https://github.com/MouseLand/Kilosort/releases/tag/v4.0.31)
- **v4.0.33** (2025-04-22): benchmark code updated + tutorial (#910); note that the `sim_stepNP2_drift` figshare upload had the wrong probe file, yielding "several dozen fewer units than expected". — [#910](https://github.com/MouseLand/Kilosort/issues/910)
- **v4.0.37** (2025-05-13): option to sort shanks separately; `max_cluster_subset` parameter. — [v4.0.37](https://github.com/MouseLand/Kilosort/releases/tag/v4.0.37)
- **v4.1.0** (2025-07-18): clustering now uses all spikes per probe section up to 25,000 (vs every 20th spike, no cap) — "better-aligned with the methods published"; 65% time reduction on two ~13 h NP datasets "with no qualitative change"; "did not impact ... published benchmarks". User #966 saw more spikes/clusters, mostly noise, vs v4.0.2. — [v4.1.0](https://github.com/MouseLand/Kilosort/releases/tag/v4.1.0), [#966](https://github.com/MouseLand/Kilosort/issues/966)
- **v4.1.1** (2025-08-27): NumPy ≥2 / PyTorch ≥2.1 allowed; "Those version differences can change sorting output." — [v4.1.1](https://github.com/MouseLand/Kilosort/releases/tag/v4.1.1)
- **v4.1.2** (2025-11-04, PyPI; no GitHub release note): local git log shows a parameter for randomizing cluster initialization (09311fa) and an option to downsample data batches (42fd52c).
- **v4.1.3** (2025-12-30): `cluster_downsampling` default reverted to 20 because of memory issues and "oversplit units for some datasets"; "values between 5 and 20 also tend to work well". — [v4.1.3](https://github.com/MouseLand/Kilosort/releases/tag/v4.1.3)
- **v4.1.5** (2026-01-15, PyPI): commit 74811f0 "Added empty spike check in swarmsplitter.check_CCG" introduced `if len(st1) == 0 or len(st2 == 0) or T == 0: return False, False` (verified in local git). Fixed on main by PR #1043 merged 2026-09-25; PyPI's latest release is still 4.1.7 (2026-03-05), so released versions 4.1.5–4.1.7 carry the bug. — [#1042](https://github.com/MouseLand/Kilosort/issues/1042), [PR #1043](https://github.com/MouseLand/Kilosort/issues/1043), [PyPI](https://pypi.org/project/kilosort/#history)
- **v4.1.6** (2026-03-03, PyPI): template placement bug for some layouts (tetrodes, #1019) fixed (d947208); check for odd `nt`. — [#1019](https://github.com/MouseLand/Kilosort/issues/1019)
- KS2.5/KS3 "spike holes" (#594): ~7 ms of reduced detection at every batch boundary (every 2.1866 s) in KS2, 2.5, 3 "but not 4"; fixed in v2.5.2/v3.0.2 (2024-04-08) and v2.0.2 (2024-04-16) with new `NT = 64*1024 - ntbuff`. — [#594](https://github.com/MouseLand/Kilosort/issues/594), [README](https://github.com/MouseLand/Kilosort/blob/main/README.md)

### Inferences
- Any KS4-vs-KS2.5 comparison should state the KS4 version: geometry handling changed substantially before v4.0.4/4.0.22, merge alignment before 4.0.31, and clustering subsampling between 4.1.0–4.1.2 vs ≤4.0.39 and ≥4.1.3; v4.1.5–4.1.7 lack the refractory veto in splitting.
- Any KS2.5 baseline made before April 2024 (or with pykilosort) carries the spike-holes bug.

### Gaps
- GitHub release notes do not exist for 4.1.2 and 4.1.4–4.1.7; their contents above come from local git commits and PyPI dates, not maintainer notes.
- I did not verify whether any fixed version later than 4.1.7 has been released between 2026-09-25 and 2026-11-03 beyond PyPI's list (which ended at 4.1.7 when checked).

## 4. Do maintainers acknowledge KS2.5 doing better, and what do they recommend?

### Takeaway
Maintainers acknowledge in general terms that an older version "could work better for your specific probe type or brain area", but in individual threads they attribute KS4 underperformance to geometry, drift settings, thresholds, outdated versions, reused parameters, or data quality; their standing recommendation is KS4 with defaults and no support for KS1–3.

### Cited Findings
- Jacob (2024-06-04): "it's possible an older version could work better for your specific probe type or brain area, but in most cases Kilosort4 should work just as well (or better)." — [#707](https://github.com/MouseLand/Kilosort/issues/707)
- Marius (2024-10-18): KS4 "never had this problem and ... performs better in most cases"; offered to look at setups where KS4 underperforms. — [#594](https://github.com/MouseLand/Kilosort/issues/594)
- Marius (2024-05-21): "it's a good idea to forget everything you know from Kilosort 2.5 and start over with Kilosort4"; Jacob: even parallel parameters "are implemented differently, so using the same values you used before may not work well." — [#689](https://github.com/MouseLand/Kilosort/issues/689)
- Marius (2024-06-11): KS2 does not need probe geometry; from KS2.5 on geometry is used for drift correction and (KS3/KS4) local clustering. — [#707](https://github.com/MouseLand/Kilosort/issues/707)
- Marius (2024-05-10): in the paper's experiments "Kilosort is typically much better than other algorithms especially at low amplitudes." — [#675](https://github.com/MouseLand/Kilosort/issues/675)
- README: "we are no longer providing in-depth support for Kilosort 1-3. If you encounter difficulties using the older versions, we recommend trying Kilosort4 instead." Jacob (#815): "We're not providing support for those anymore." — [README](https://github.com/MouseLand/Kilosort/blob/main/README.md), [#815](https://github.com/MouseLand/Kilosort/issues/815)
- Historical context (v2.0 release note, 2020, pre-KS4): "we expect some users will get better performance from v2.0" than v2.5 for "tetrodes, single-channel recordings such as from a Utah array, probes with >30um vertical spacing." — [v2.0 release](https://github.com/MouseLand/Kilosort/releases/tag/v2.0)
- Specific concessions: Marius in #710 conceded the final pairwise merge step can over-merge distinct cells and pointed to a custom merge modification (Gardner et al. 2022); Jacob in #964 conceded KS4 drift correction is "unlikely" to work for that probe layout. — [#710](https://github.com/MouseLand/Kilosort/issues/710), [#964](https://github.com/MouseLand/Kilosort/issues/964)

### Inferences
- The only open admissions of systematic weakness are geometry-dependent (drift correction on <~64-channel/sparse probes; early multi-shank/2D handling) and the final merge step for rarely co-active cells; there is no thread in which a maintainer says KS2.5 is better on standard Neuropixels.

### Gaps
- No maintainer statement found that quantifies KS4 vs KS2.5 on users' real data (beyond the paper's simulations).

## 5. User-posted quantitative KS2.5 vs KS4 comparisons/benchmarks

### Takeaway
No GitHub Discussions exist and no issue contains a systematic benchmark; quantitative comparisons are anecdotal single-session numbers in support threads (#663, #964, #968, #634, #657).

### Cited Findings
- #663: ~1000 (KS2.5) vs ~130–135 (KS4 4.0.3) good units on Litke retinal MEAs, before the v4.0.4 fix. — [#663](https://github.com/MouseLand/Kilosort/issues/663)
- #964: KS4 ≈50% of KS2.5 good units; total spikes 687,732 (KS2.5) vs ~1.5M (KS4); after tuning, ~20 good units with spike count similar to KS2.5. — [#964](https://github.com/MouseLand/Kilosort/issues/964)
- #968: KS2.5 30–55 high-quality units/session; KS4 "poor" (no count). — [#968](https://github.com/MouseLand/Kilosort/issues/968)
- #634: similar good/MUA counts but ≥50% fewer spikes in KS4 good clusters (NP1.0). — [#634](https://github.com/MouseLand/Kilosort/issues/634)
- #966: v4.1.0 vs v4.0.2 on the same data: more spikes and clusters with 4.1.0 defaults but "translates to noise rather than good" units (numbers only in attached logs). — [#966](https://github.com/MouseLand/Kilosort/issues/966)
- #910 / v4.0.33: maintainers published a benchmarking tutorial (`kilosort.bench`) for ground-truth comparison, not a KS2.5 comparison. — [#910](https://github.com/MouseLand/Kilosort/issues/910)
- GitHub Discussions disabled for MouseLand/Kilosort (API `has_discussions: false`). — [repo](https://github.com/MouseLand/Kilosort)

### Inferences
- The GitHub record cannot support a general claim that KS4 is worse than KS2.5; it supports the narrower claim that out-of-the-box KS4 can underperform KS2.5 on non-Neuropixels or sparse geometries and noisy low-channel data until geometry, drift, and threshold settings are adjusted.

### Gaps
- No ground-truth or curated-unit benchmark comparing KS2.5 and KS4 was found on the repo; such evidence would have to come from papers/SpikeInterface benchmarks (out of scope here).
- Some numbers (unit counts in screenshots, log contents) were in images/attachments that I did not read.
