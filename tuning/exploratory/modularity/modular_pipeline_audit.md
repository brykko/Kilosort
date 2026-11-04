# A modular Kilosort pipeline: multi-agent audit (2026-11-04)

The question: could steps from different Kilosort versions be swapped freely? The steps considered are:

1. preliminary spike extraction
2. drift correction
3. template learning
4. final spike extraction
5. final clustering
6. merging (+ splitting)
7. cluster-amplitude thresholding

The user's example was KS4's step 1 → KS2.5's step 3 → KS4's steps 4–5. Language (MATLAB vs Python) is ignored. Two sub-questions:
- **(a) Procedure:** do the steps' inputs, outputs and hidden assumptions fit together?
- **(b) Science:** do our experiments so far argue for or against the idea?

**Agents.** Seven agents worked in parallel. Three read the code by step group (1–2, 3–4, 5–7). One integrator traced the state each pipeline carries between steps and walked four hybrid pipelines end to end. Two collected evidence: one from the frozen harness results, one from the exploratory analyses and audits. One argued both sides and searched the literature.

**Ground rules.**
- Everything was read-only. No sorts were run, and no scripts were kept.
- *(n)* marks a finding that n agents reached independently.
- I checked the claims that carry the conclusions against the code and output files. Where agents disagreed, §7 says which one was right.

**Code sources.**

| Version | Source used | Notes |
|---|---|---|
| KS4 | this checkout (`kilosort\`, branch `ks4-tuning`) | Stock behaviour unless stated. The branch changes to steps 1–6 are bit-identical or parameter-only. |
| KS2.5 | the lab's copy, `C:\code\github\neuropixels\ExternalToolboxes\KilosortToolbox\Versions\kilosort_v2.5\`, run as `+SpikeSort\+Kilosort\runKilosortClustering.m:264-306` | Stock KS2.5 is not on disk. The lab copy differs: random batch order in learning, `batchValid`, a `spikeTemplatesPath` hook, and `autocurate`. The `master_kilosort.m` in that folder is a stale KS2 script without `datashift2`. |
| KS2.1 | `...\Versions\kilosort_v2.1\` | Stands in for KS2.0, which isn't on disk. Anything said about 2.0 is from memory and labelled so. |

## 1. Short answer

- **(a) Procedurally, the idea works only in part, and not as seven interchangeable modules.**
  - The seven steps are a fair conceptual list. They are not where the code divides. In at least one version each, steps 1|2, 3|4 and 5|6 are fused into one function.
  - Step 5 is absent from KS2.x, where clusters are templates. Step 7 is absent from KS4.
  - KS4 runs the motif "spikes → clusters → templates → extract" twice. KS2.x instead runs one online learn-while-extracting loop (§2).
  - Most handoffs can be bridged with adapters. Each adapter has to rebuild hidden state, not just reformat arrays.
  - A few swaps have no receiving end at all. One of them is the user's example as stated: **KS2.x template learning takes no spike set**. It re-detects from the data itself. *(4)*
- **(b) Scientifically, the evidence goes against a best-of-breed production pipeline. It supports a few targeted swaps as a diagnostic tool.** *(all 7)*
  - **The gains are not portable.** Tuned KS4's advantage comes from steps 3/5/6. Those steps are specific to KS4 and interact strongly with each other.
  - **There is no KS2.5 step worth importing.** No KS2.5 step has been shown to beat its counterpart in tuned KS4.
  - **The one real transplant changed its meaning.** KS2.5's `set_cutoff` on KS4 ran exactly as written but did a different job.
  - **The evaluation can't test many combinations.** It is one tuning recording plus one transfer recording, with δ = 28.6 bits/s.
- **What does transfer is ideas, not modules.** The parts of KS2.5 that look valuable (purity, few duplicates) come from its step-4 amplitude prior λ, used together with its step-7 cut. They are worth testing *inside* KS4 as changes to KS4's own steps.

## 2. What the code actually does at each step

| Step | KS2.1 (≈ KS2.0) | KS2.5 (lab) | KS4 |
|---|---|---|---|
| 0 Filter + whiten (not in the list, but it sets the scale of every threshold) | separate. Drops bad channels. | separate. `batchValid`. Whitening range 64. | separate. Range 32. FFT high-pass. |
| 1 Preliminary extraction | Only inside `clusterSingleBatches` (`extractPCbatch2.m:35-41`). The output is never stored. | Three detectors: `standalone_detector` (for drift; result saved as `rez.st0`, which nothing reads), `isolated_peaks_new` (for the PC basis), and `mexGetSpikes2` (template birth, inside learning). | A real module, run **twice**: inside `datashift.run` (`datashift.py:198`), then on the corrected data (`run_kilosort.py:804`). The second output feeds step 3. |
| 2 Drift | **Absent as registration.** A batch permutation (`rez.iorig`) plus template tracking during learning. | `datashift2`. Rewrites the temp file **in place** (`datashift2.m:153`). | `datashift.run`: a near line-by-line port of `datashift2`/`align_block2`. Applied lazily on every read (`io.py:1135-1141`). |
| 3 Template learning | **Fused with 4** in `learnAndSolve8b`. Templates are updated in every batch, including the final pass (`learnAndSolve8b.m:146-149`). | `learnTemplates`: online matching pursuit plus updates, in random batch order, with templates born from the residual and triaged. | `clustering_qr.run(mode='spikes')` + `postprocess_templates`. Clusters the step-1 spikes once. |
| 4 Final extraction | Same loop; templates still tracking. | `trackAndSort` with **frozen** templates: `dWU` is never updated in the loop (`trackAndSort.m:117-182`). | `template_matching.extract` with static templates. |
| 5 Final clustering | absent | absent (template = unit) | `clustering_qr.run(mode='template')`. Contains step 6's tree merges and swarm splits. |
| 6 Merge + split | `find_merges`, then `splitAllClusters` | `autocurate`, interleaved with 7: merge → split → cut → merge → split → cut → merge | Tree merges (inside 5), then `merging_function`. Earlier merges also happen in step 3. |
| 7 Amplitude cut | `set_cutoff` | `set_cutoff`, run twice | **absent** (only `is_ref` labels and duplicate removal at save) |

Two premises in the question need correcting:

- **Template drift.** "KS2.5's templates evolve across batches and encode residual drift" is true only of KS2.0/2.1. KS2.5 handles drift with `datashift2`. Its learning sweeps batches in random order, and its final pass uses frozen templates. *(2)*
- **Preliminary extraction.** "Preliminary spike extraction" is a standalone module only in KS4. In KS2.x it is an internal sub-step of reordering, drift estimation or template birth. *(3)*

**Shared state that every swap has to carry** *(integrator; steps 1–2)*:
- the whitening matrix and filter;
- the drift field, plus the registered frame it defines;
- the channel set;
- the batch grid (NT 65600 vs 60000; KS2.1's batches overlap: `Nbatch = ceil(samps/(NT−ntbuff))`);
- the temporal basis (`wPCA`/`wTEMP`);
- KS4's universal-template grid (`ycup`, `xcup`, `dmin`), which step 5 reads;
- the per-spike score and the scale its thresholds are expressed in;
- the global RNG state.

`stage_cache.py` reproduces KS4 sorts bit-identically by saving each stage's return values, the `ops` keys it adds and the RNG states. That confirms these items are KS4's whole hidden interface. The cache's breakpoints are drift + first detection | first clustering + extraction | final clustering + merge. That is three blocks, not seven steps.

**One useful coincidence:** KS4's saved preprocessed copy is int16 × 200 (`io.py:1228`), the same convention as KS2's `scaleproc = 200`. So a drift-corrected file can be handed from one version to the other.

## 3. Compatibility of each handoff

Cost scale: **trivial** (reformat), **adapter** (rebuild derived state), **re-derivation** (new algorithm), **no slot** (the receiving step has no such input).

| Handoff | Cost | What it takes / what is lost |
|---|---|---|
| KS4 step-1 spikes → KS2.5 drift estimator (and the reverse) | trivial | (batch, y, amplitude) on the other batch grid. Trap: KS2.5 normalises amplitude bins by log10(8), so H8's Th 7 spikes give non-positive bin indices. |
| KS2.5 drift field → KS4 downstream | adapter | Resample to KS4 batches. Recompute `yblk` (not saved) and `iKxx`. Match the channel set. The shift is applied to KS4's own whitening, not KS2.5's. |
| KS4 registered file → KS2.5 learning | adapter | `save_preprocessing` output, minus non-chanMap columns. Hand-built `rez`: `Wrot` × 200, NT/Nbatch, `batchValid`. |
| Registered file → KS2.1 learning | adapter | Re-chunk into KS2.1's overlapping-batch layout and set `iorig = 1:Nbatch`. Sound in principle: this is roughly how KS2.5 came about. |
| KS2.1 `iorig` → KS2.5 / KS4 | no slot | A batch permutation contains no displacement. |
| **KS4 step-1 spikes → KS2.5 template learning** (user's example) | **no slot** | `learnTemplates` does its own detection, matching and averaging from the start. Seeding it with KS4 spikes or templates is a code change, and after the first batch KS2.5 re-detects everything anyway. |
| KS4 `wPCA`/`wTEMP` → KS2.5 learning | trivial | The `ops.spikeTemplatesPath` hook already exists (`extractTemplatesfromSnippets.m:9-13`). |
| KS2.5 templates → KS4 `extract` | adapter | Densify `mu·W·Uᵀ` and project onto KS4's 6-PC `wPCA`; trough at sample 20 in both. Must run on **the same registered file**. Skip `postprocess_templates`. KS4 step 1 must still run for its side effects (`wPCA`, `wTEMP`, the grid). |
| KS4 templates → KS2.5 `trackAndSort` | adapter | Rank-3 SVD truncates KS4's rank-≤6 templates. Rebuild `iNeigh`, `iNeighPC`, `simScore`. The λ prior then acts on KS4 cluster means, and the Th values don't carry over. |
| KS2.0/2.1 tracked templates ↔ KS4 | re-derivation | KS2.1's dictionary is a time series (`WA/UA/muA`). KS4 can take only one snapshot. KS2.1's loop has no seeding hook. |
| KS2.5 final spikes → KS4 step 5 | adapter | `cProjPC` (32 ch × 3 PCs) → `tF`. Set `n_pcs = 3` and `nearest_chans = 32`. Build `iCC`/`iU` from `iNeighPC`/`iW`, and the grid from the probe. Step 5 is scale-invariant. The cleanest swap at the clustering layer. |
| KS4 output → KS2.5 merge/split | adapter | `Nrank = 3` is hard-coded. `dWU` comes from `Wall`×`wPCA`. A KS4 unit spans many templates, and so many channel sets: 94% of H8 units mix templates. Re-express the spikes on a shared channel set first. |
| KS4 `merging_function` on KS2.5 output | adapter (small) | `Wall` from `dWU`, `clu` from `st3(:,2)`. The `is_ref` gate, size order and time shifts differ. |
| simScore ↔ KS4 similarity; refractory CCG tests | trivial | The same objects. KS4's `CCG_metrics` is KS2's `ccg_rg` with R and Q renamed (`CCG.py:73-76`). |
| KS2.5 `set_cutoff` on KS4 (**done**, `autocurate\`) | adapter for the procedure, re-derivation for the meaning | See §4. |

The user's example works only after re-reading it. The working version is:

1. **KS4 steps 0–2:** registered file, `wPCA`/`wTEMP`.
2. **KS4 step 1:** still run, for its side effects only.
3. **KS2.5 `learnTemplates`:** run on that file with its own detection.
4. **Projection:** convert the dictionary to KS4's basis.
5. **KS4 steps 4–6.**

Since KS4's step 5 reclusters, KS2.5's template identities would influence only *which* spikes are found and their features, not the final labels.

Swaps still impossible with any common interface:
- feeding a spike set into KS2.x learning;
- splitting KS2.1's fused learn-and-track loop;
- using any threshold across score kinds without recalibrating it;
- dropping KS4 step 1 without re-creating its side effects.

## 4. The one cross-version swap we have run: `set_cutoff` on KS4

*(steps 5–7, exploratory, harness, skeptic, integrator)*

**What porting it took.**
- The procedure ports exactly. The Python port keeps the same spikes as MATLAB, with 0 cut-off differences over 1057–1125 templates, both cycles, both excerpts (`autocurate\validate_port.log`).
- Making it run on KS4 needed four changes:
  - a per-spike score KS4 doesn't save by default (`save_extra_vars`);
  - an invented ceiling, `Th_learned + w`, which adds a new free parameter;
  - redefining "unit" as a KS4 cluster that mixes templates;
  - running it only as a post-hoc step.

**Why its meaning changed.**
- **The scores differ.** KS2.5's `vexp` = √(b²/(1+λ) − λμ²) with λ = 10 includes an amplitude prior, so it is partly a membership score (`mexMPnu8_rg.cu:310-312`). KS4's score is relu(B)/‖U‖ with no prior and no misfit term (`template_matching.py:232, 267-269`; `lam = 20` at `:226` is unused).
- **The units differ.** KS2.5's cut works on one-template units, where it removes foreign spikes efficiently. On KS2.5, cut1 deletes 23.4% of spikes and takes misassignment (plain estimate) from 20.2% to 7.8% on kobe. On MEC it deletes 31.3%, for 31.7% → 12.8% (`stage_breakdown.log`).
- **On KS4 it works on mixtures of templates**, where contamination is organised by template group (`placement_audit.md` §3).

**What it did** (`autocurate\summary_variants.txt`; checked):

| Sort | S uncut | S after cut | Matched random deletion | Global score floor |
|---|---|---|---|---|
| H8 kobe | 1411.3 | 1332.6 (w=2), 1234.5 (w=4) | 1244.6, 1161.2 | 1340.1 |
| H8 MEC | 2917.9 | 2749.6 (w=2), 2511.3 (w=4) | 2530.1, 2319.0 | — |
| default kobe | 1030.1 | 911.0 (w=4) | 834.8 | — |

- S falls in every case. The cut beats random deletion by +6–9%, but a single global score floor does as well (`placement_audit.md` §2.4). The global floor deletes 83% of the same spikes.
- **Most deletions are a fixed slice.** 34–44% of units fail the cut's very first level and receive a fixed slice. Those units account for 75–85% of all deletions.
- **Lesson:** a step can transfer exactly as code and still change meaning, because the quantity it consumes was produced by a different upstream step. Only its generic part survived the move: deleting low-amplitude spikes.

**The composite still beats KS2.5**, but loses to the uncut sort. On the 20-min windows (`long_data\windows\summary_long.txt`):
- **kobe:** H8 + cut w=4 gives S 1242.1 against KS2.5 standard's 1006.9, with 6.09 M vs 5.59 M correct spikes and 0.53 M vs 0.42 M misassigned. Uncut H8 gives 1423.3.
- **MEC:** 2511.3 against 2066.9.

## 5. What our experiments say

| Finding | Steps | Bearing | Source |
|---|---|---|---|
| Tuned KS4's gains are in clustering and merging (bimodality off, γ 2, final merge 0.95, downsampling 10), plus detection thresholds. Drift and preprocessing settings did nothing useful. | 3, 5, 6 (+1, 4) | Against: the gains are KS4-specific. | `round1-4_results.md`, `README.md` |
| **Strong interactions.** Bimodality 0 alone gives +136.0, but reverting it from H8 costs 323.7. With the default tree, γ 2 + Th 7/7 + first merge 0.975 is net negative. The tree "re-merges almost everything extra". | 3/5 × 1/4 × 6 | Against independent choice of steps | `round1_results.md`, `round3_results.md`/`round4_results.md` ablations, `search_round1.md` §2.3 |
| **Some effects add up.** Final merge 0.95 adds +84 to +130 on every base. `Th_universal` 7 + `Th_learned` 7: +71.1 against a summed +68.9. | 6; 1, 4 | For (narrowly) | `round2_results.md`, `round3_results.md` |
| Effect signs depend on context. `cluster_downsampling` 10 and `cluster_neighbors` 5/20 change size, or flip sign, between default and `standard`. | 3/5 | Against | `round1_results.md`, `round2_results.md`, `round4_results.md` |
| **No KS2.5 step beats tuned KS4.** H8 beats KS2.5 by ΔS +423.6 (kobe) and +878.3 (MEC). On matched cells H8 scores 885–895 against 846–854. At KS2.5's misassignment budget, H8 keeps 6.56 M correct spikes against 5.50 M (kobe). | all | Against importing a KS2.5 step | `ks25_comparison.md`; `isolation\tradeoff.log` |
| **KS2.5 vs default KS4 flips between recordings:** −30.7 on kobe, +99.0 on MEC. | all | Against expecting small module effects to transfer | `ks25_comparison.md` |
| KS2.5's distinctive asset is purity. It comes from `set_cutoff` together with the λ prior: before its cut, KS2.5 is about as contaminated as KS4. | 4 + 7 | Points to an *idea* transfer (§6) | `stage_breakdown.log`; `duplicates_audit.md` |
| **The KS4/KS2.5 differences trace to a few steps.** Of default KS4's kobe spikes that KS2.5 doesn't output, 47% were cut by `set_cutoff`, 44% were never extracted, and 10% were extracted but not output. For H8: 28 / 68 / 4%. 94% of H8 units mix extraction templates (median 9), and splitting them by template cuts misassignment from 28.2% to 20.8%. | 1/4, 5, 7 | For swaps as a diagnostic: the differences are localised | `ks25_traced\cutoff_trace.log`, `isolation\trace_units.log`, `placement_audit.md` §3 |
| **Each sorter makes and removes duplicates in a different step.** KS4 echoes come from prior-free matching (step 4), at 0.11–1.02 per 1000 spikes. KS2.5's λ prior and cut keep it at 0.03–0.08. | 4–7 | Against plug-and-play: dedup is tied to the matcher | `duplicates_audit.md` |
| **Swapping the dictionary into extraction is unpredictable.** A third KS4 matching round, i.e. a different dictionary handed to step 4: H8 kobe +29.7 ± 3.4, MEC +57.0 ± 26.7; default kobe −0.2 ± 10.5, MEC −22.4 ± 8.7. Unit agreement between rounds is a median 0.61–0.63, about the seed-to-seed scale. *Uncommitted, in-progress worktree `ks4-round3`.* | 3 → 4 → 5 | Against predictable module effects; the size scale for E1 (§6) | `iteration\summary_round3*.txt` |
| **Behaviour over recording length differs by version.** KS4 changes ≤ 1.4% between 20-min and 2-h sorts. KS2.5 kobe: S +9.5%, misassigned +54%. Not yet traced to a step. | whole | Caution | `long_data\windows\summary_long.txt` |
| Drift in both excerpts is −6.5 to +4.5 µm, and `nblocks` 5 gives −0.1. | 2 | No test possible: a step-2 swap can't show anything on these data | `scaling_audit\...` §1; `round3_results.md` |
| KS4 stages cache bit-identically. | interfaces | For the procedure (within KS4) | `stage_cache.py` |

**Implications for evaluation** *(harness agent)*:
- **Replicates.** Each hybrid needs its own seed replicates. Seed spread differs between pipelines (KS2.5 SD 5.9, `standard_low` 16.7, default KS4 about 14), and the best of ~100 runs beats default by 2–2.5σ by chance (`evaluation_spec.md` §8.1).
- **Guards.** The hybrid-truth guard is gone for new kobe sorts (from 2026-11-03), and θ_ref was built from default KS4.
- **Cost.** A configuration that uses KS2.5's learning costs about 1.5 h per excerpt over 3 seeds. Realistically, at most about ten targeted swaps are feasible.
- **Fairness.** A module dropped into a host that was tuned for a different module starts handicapped: tuned KS4 is about 38% above default KS4 on the same code (ΔS +392.9 on 1019.7). A swap measured without retuning is therefore a lower bound on that module's value, not a fair test of it.

**Prior art** *(skeptic; I checked the citations)*:
- **Modular sorting exists and has won.** Garcia et al., "Opening the black box: a modular approach to spike sorting" (bioRxiv 2026; eLife reviewed preprint RP110588 v2, 2026), builds sorters from swappable SpikeInterface `sortingcomponents`. Their Lupin combines peak detection (matched filtering), clustering (iterative ISOSPLIT) and matching (Wobble) taken from different sorters. It beat other sorters, including KS4, on simulations and was competitive with KS4 on real data. Its reviewers flagged that modules evaluated in isolation don't predict the full system, and warned of benchmark overfitting.
- **Swapping the drift step is the best-supported case.** Garcia et al., eNeuro 11(2), 2024, re-implemented KS2.5's motion correction as components and swapped them. Monopolar localisation + decentralised estimation beat the KS2.5-like chain: 118 vs 110 well-detected units on ZigZag, 107 vs 79 on Bumps. No correction recovered static-recording accuracy.
- **KS4's own attribution is a version-level swap.** The KS4 paper credits its gain over KS2.5/KS3 to clustering, since those versions share drift correction and deconvolution (Pachitariu et al., Nat Methods 2024). That attribution is under-identified: detection, interpolation and thresholds also changed.
- **Not found:** any study that swaps stages *between Kilosort versions*.

## 6. Recommendations

1. **Don't build a general modular framework across KS versions.** The receiving slots don't exist at several boundaries. Steps 3–7 are tied to each version's template parametrisation, feature basis and score semantics. A hybrid would be an unvalidated method needing its own tuning, and the evidence shows no KS2.5 step to import.
2. **If modularity is wanted, define it by data artifact, not by the seven steps.** The artifacts are: Signal → DriftField → SpikeSet → TemplateSet → SpikeSet with features → LabelledSpikeSet → CuratedLabelledSpikeSet. A "step" is any map between two of them, and may be applied more than once.
   - Invariants the artifacts must carry:
     - one channel order;
     - the drift field indexed by sample time and attached to its registered frame;
     - the temporal basis travelling with any features;
     - the per-spike score carrying a declared kind (`amp_x_norm` vs `vexp_with_prior`), shared by any threshold applied to it;
     - each module seeded explicitly.
   - This is the integrator's proposal. Mixed pipelines built this way are reproducible, but never bit-identical to native runs.
3. **As a diagnostic tool, run 2–3 targeted swaps, each in both directions, with seed replicates and the full guards.** Asymmetric or non-additive results are themselves findings. Much of the attribution is already done by tracing (`cutoff_trace`, `trace_units`, `stage_breakdown`). So spend runs only where tracing can't answer.
   - **E1 — dictionary swap.** KS2.5's learned templates go into KS4 `extract`, then KS4 steps 5–6. Run on the same registered file, under default and H8, 3 seeds each. It asks whether template learning matters once KS4 reclusters.
     - Calibration from the third round: changing the dictionary moved H8 by about one δ and default by nothing.
     - Count it as a template-learning effect only if |ΔS| ≥ δ beyond both endpoints' seed spread and unit agreement moves toward KS2.5.
     - Also check whether it reduces the 12–15% of orphan units.
   - **E2 — KS2.5's final spikes through KS4 step 5** (cProjPC adapter). It asks whether KS4's regrouping across templates is where contamination enters (placement audit §3) — the clearest test of a step-5 effect.
   - Skip the drift swap on these data. Drift is a few µm and `nblocks` made no difference. Literature says step 2 is the best case for modularity, but it needs drifting data to show anything.
4. **Prefer importing ideas into KS4's own steps over importing modules.** Each of these is a single-parameter or single-function change, and can be judged by the harness:
   - KS2.5's λ amplitude prior in KS4's matching cost (step 4; `duplicates_audit.md` rec. 5);
   - a membership-type score for any cut (placement audit P7);
   - refractoriness inside KS4's split/merge decisions (placement audit P4′);
   - the lab's `checkCcgs` (already ported as `same_shape`: +42 alone in round 1, not in H8).

## 7. Where agents disagreed

- **Does KS2.5's final pass update its templates?** The integrator said the templates are "re-fitted by SVD every batch". Two other agents said they are frozen. Checked: in the lab copy, `dWU` is never updated inside `trackAndSort`'s loop. `dWU1` only accumulates a mean (`:117, :182`), so the per-batch `mexSVDsmall2` re-factorises the same `dWU`. The final pass is frozen. KS2.1 does update in every batch (`learnAndSolve8b.m:146-149`, checked).
- **Can a registered file go straight to KS2.1?** The integrator said nothing semantic needs translating. The steps 1–2 agent said the file must be re-chunked. Checked: KS2.1's batches overlap (`preprocessDataSub.m:16`), unlike KS2.5's (`:25`). It is an adapter, not a semantic change.
- **Is the skeptic's first experiment new?** It proposed "spike-level divergence on existing outputs" as the first experiment (E0). That is already largely done (`cutoff_trace`, `trace_units`, §5).

## 8. Open questions and side findings

- **Untested swaps.** No swap of steps 3, 4 or 5 across versions has been run. Unmeasured:
  - how much energy KS2.5's rank-3 templates lose when projected onto KS4's 6-PC basis;
  - the offset between two independent drift estimates (KS2.5's and KS4's);
  - the effect of KS4 re-whitening KS2.5's shifted file.
- **KS2.5's length dependence.** The kobe 2-hour jump (misassigned +54%) is not traced to a step.
- **Stock KS2.5 is not on disk**, so the lab-copy changes can't be separated from upstream behaviour by diffing.
- **Side finding: possible stock KS4 bug in `make_pc_features`** (`postprocessing.py:115-140`; code reading, not yet checked on data).
  - For each cluster, it overwrites `tF` for *every* spike of that cluster's templates, including spikes that went to other clusters.
  - KS4 spreads one template's spikes over several units, so later clusters overwrite earlier ones. Rows of `pc_features.npy` can then sit on another unit's channels (`pc_feature_ind.npy`).
  - This matters for Phy and for any KS2-style splitter fed KS4 features. It does not touch the harness's scores, which don't read the saved PC features.
