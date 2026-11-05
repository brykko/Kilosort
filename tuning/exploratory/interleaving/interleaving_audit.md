# Interleaving extraction, clustering and template refinement in KS4 (2026-11-04)

**Question.** KS4 learns templates by clustering spikes extracted with universal templates. Waveforms those fit
poorly may never get a template. Extra whole-recording rounds (`ks4-round3`) appear to converge on better templates.
Would interleaving extraction, clustering and template refinement in small batches, as KS2.5 interleaves extraction and
template learning, reach that faster?

**How this was done.** Eight agents worked independently: KS4 code, KS2.5 code, theory, literature, measurements on the
saved sorts, compute, a ground-truth simulation, and a skeptic. Their notes are in [`notes\`](notes). No KS4 or KS2.5 sort
was run, nothing in `kilosort\` changed, and no hybrid data was used. I rechecked the claims this report rests on against
the code and logs. Tags: **[M]** measured here, **[E]** estimated or modelled, **[I]** inferred. There is no ground truth on
the real recordings, so "accuracy" there is always a proxy.

## Bottom line

**The idea survives only in a narrower form than proposed, and the evidence does not justify building a batch-level
interleaver yet.** Three claims carry the proposal:

| Claim | Finding |
|---|---|
| Rounds converge on more accurate templates | **Not shown.** Accuracy has never been measured. For the same cell, templates barely move, and two seeds don't get closer. What moves is spike labels and unit identity. A gain in template accuracy can't be excluded, but nothing measurable shows one. |
| Waveforms poorly fit by universal templates go without a template | **Mostly a different problem.** The first pass misses 54% of final spikes, but mainly through its local-maximum rule, not fit. Almost no unit is invisible. Units without a template are mostly lumped with a same-channel neighbour rather than missed. Units never found by any sorter can't be measured here. |
| Interleaving converges much faster | **True for refining template means, not shown for the structure.** Compute doesn't favour it either: a batch costs the same whether interleaved or not, so the saving comes from visiting fewer batches. Subsampled rounds get that with about 40 lines of code. |

Two follow-up checks on saved sorts (section 9) narrow what is open. **Template labels can't replace a clustering step: using them
alone costs about 13-15% of S at H8. That doesn't show the clustering must see the whole recording at once; chunked and subsampled
clustering are untested. Template labels also improve from round to round, but through the dictionary, so that gain comes from the
previous round's clustering.** Residual-based birth would see only 5-15% of orphan spikes at H8. What is still open
is decisive: **whether that better dictionary comes from refined shapes or from the structure of the previous round's clusters.** If
shapes carry it, a coarse interleaving is cheap and worthwhile. If structure does, a dictionary still has to come from clustering, and
subsampled rounds are the simple way to try that. The pilot in section 9 settles this, with decision rules.

The strongest result in the proposal's favour is theoretical: refinement really is much faster online, and one toy
matched 3-5 rounds in a single pass once splits ran inside the loop. That scheme is not KS2.5's, and it is untested under
realistic conditions.

## 1. What the rounds do

**The gain is real.** Repeat rounds raise S, S_track and the trade-off curves at H8, at every seed, on both excerpts and at
2 hours (`round3_results.md`). Round 3 gains +29.7 ± 3.4 on kobe S and round 4 a further +4.9. δ is 28.6.

**Templates.** Same cell, round to round, H8 **[M]**:

| | kobe | MEC |
|---|---|---|
| Median cosine between rounds (mutual pairs) | 0.9994-0.9996 | 0.9993-0.9995 |
| Step \|dT\|/\|T\|, r2→r3, r3→r4 (mutual pairs) | 3.7%, 3.1% | 4.1%, 3.4% |
| The same, all eligible units | 7.3%, 4.8% | 9.7%, 6.3% |
| Step against the template's own sampling noise (median, r2→r3, r3→r4, mutual) | 1.43, 1.18 | 1.52, 1.21 |
| Same cell across seeds, round 2 → round 4 | .0248 → .0247 | .0307 → .0313 |
| Spikes relabelled r3→r4, against between seeds at r4 | 10-11%, 13% | 14%, 18% |

- The step shrinks by about 0.7 per round and is real: it exceeds a random-swap null in 89-97% of units. Both seeds also
  move the same way, so the shift is systematic. A shared shift shows a shared cause, not a truer end point.
- Seed distance for fixed cells doesn't fall, which is what convergence on a common solution would predict **[M]**. Seed
  agreement misses any shared bias, so this bounds the claim and doesn't refute it.
- The same spike's fit to its template changes by −0.05% then +0.05% (control ±0.002%) **[M]**. Score is second order in
  template error, so this too is a bound.
- Default settings show the same systematic shift with no gain in S (−0.2 on kobe, −22.4 on MEC). Moving templates this
  way is not sufficient for S **[M]**.
- ΔS barely tracks template change (Spearman .03-.09, descriptive, survivors only) **[M]**. Template change tracks label
  change at .72-.76.
- The one sharp convergence is self-consistency. Round 2's extraction templates are off from the second-pass cluster
  means by 1−r ≈ 0.01 (spike-weighted), and later steps are about 0.0005 **[M]**. So extraction templates catch up with the
  clusters in one step.

**Different quantities settle at different rates** (ratio of successive changes) **[M]**: dictionary size 0.05-0.11, S 0.17 ±
0.15, curves 0.12-0.36, unit identity 0.67-0.76. S is about finished at round 4, while the partition still moves by nearly
what a change of seed moves it. Part of that may be restart noise, because `clustering_qr.py:531` never passes the `iclust`
that `cluster()` accepts, so each round starts clustering afresh.

**Mechanism.** The skeptic's breakdown, from saved data:

| Candidate | Status |
|---|---|
| (a) Better shapes, better subtraction | Plausible, size unknown. Same-unit short ISIs fall 24% (kobe) and 12% (MEC) at round 3, then stop. But cross-unit duplicates rise 37% and 56%. |
| (b) Re-clustering into a different partition | Unresolved. A seed change moves the partition as far with ΣΔS ≈ 0, so movement alone isn't gain. |
| (c) A smaller dictionary, less competition | Unlikely as such. C1 → A3 shrinks the dictionary 30% from identical first clusters, changing S by 2.0 (kobe hybrid excerpt, S only). Larger dictionaries tend to score higher. |
| (d) Better features or spike sets alone | Weak. After cleaning, spikes rise only 1.6% and 0.6%. |
| (e) Re-roll with selection | Rejected for the gain. ΣΔS on stable pairs is about ±7 across seeds against +19 to +61 from r2 to r4. |

Theory leans structural: the dictionary drops about 30% in one step, then stops, and augment gains nothing on the curves.
Against that, most of ΔS sits in plain matched pairs, which fits refinement. **I don't think the data can rank these.**

One specific hypothesis is worth testing **[I]**. At H8 the first pass has no refractory evidence (`clustering_qr.py:525-528`
passes spike times only in template mode) and bimodality 0, so the tree merges nothing: 3933 first clusters, against 1106 at
default. Round 3's dictionary may simply be the first to contain refractory merges. That would make the rounds partly a
repair of H8's first pass, which would explain why the gain is specific to H8 in S.

## 2. The premise

**Rounds aren't evidence for it.** From round 2 on the dictionary contains learned templates only (`_next_templates`,
`run_kilosort.py:967`). A waveform truly missed by the universal pass gets no second chance from the universal bank, and
extra rounds add few spikes: 11.82 → 12.00 → 12.09 M on kobe seed 1 **[M]**.

**What the first pass does.**

- It sees 44-46% of the spikes that round 2 matches.
- Detection keeps only local maxima over about 100 neighbouring templates and ±`nt0` samples, with no subtraction
  (`spikedetect.py:176,181-182`). That rule alone would hide about 44% of matched spikes, or 71-81% of the shortfall **[E]**.
  At most about 11% of matched spikes (16% at default) are left to blame on poor fit, and that remainder also holds false
  positives and echoes.
- `nearest_templates`, which sets the spatial reach of that rule, has never been tuned.

**A poorly fit spike that does cross threshold isn't damaged.** Round 1 features are read straight from the data
(`spikedetect.py:285-287`), and later features add the template back to the residual (`template_matching.py:90-91`). So `tF`
carries the spike's own waveform. A unit can reach clustering while sharing another unit's template, and that is how orphans
arise.

**Who is actually affected** **[M]**:

- Poor universal fit (rho median .85, 1st percentile .63) lowers a unit's seen fraction, but doesn't predict owning a
  template (|Spearman| ≤ .11).
- At most 0.1% of units are ≥ 95% unseen. This counts only units that were found.
- Of about 900 KS2.5 units per excerpt, 7 and 4 are absent from KS4 (S 3.1 and 1.4), and 97% of KS2.5's spikes are present.
  This says nothing below KS2.5's sensitivity.
- After round 4, 100-118 isolated events/s/mm still cross Th_universal in the residual: 3.6-4.4% of found spikes, all near
  threshold, no fewer than after round 2. Whether they form cells needs a full residual run.
- 12-15% of units own no extraction template, but they are lumped with a same-channel neighbour, and borrowing costs little
  (`borrowed_templates.md`).

**This matters for the design.** The units without templates are mostly *absorbed*, not *left over*. An absorbed spike leaves
a residual of s√(1−ρ²), so residual-based birth, KS2.5's mechanism, can't see it when ρ > 1/√2 and s lies in a band. 50-56%
of H8 units would be invisible this way if orphaned **[E]**. Finding them means looking inside templates, by splitting on
add-back features, not in the residual. Published work mostly looks in the residual; Marre et al. 2012 searches within
templates.

The premise as first stated therefore divides into three problems, and only the second is one interleaving could address:

| Problem | Remedy |
|---|---|
| Spikes hidden by the local-maximum rule | First-pass settings (`nearest_templates`), or a residual pass |
| Units lumped in another template | A split on add-back features, at any schedule |
| Below threshold | Neither rounds nor interleaving; Th sets this |

## 3. KS2.5 as a model

**It isn't interleaved clustering.** Learning is online hard EM: match, average into the matched template, and birth new
templates from single residual snippets. There are no features, no neighbours and no split in the loop
(`learnTemplates.m`, `mexGetSpikes2`). Frozen extraction and post-hoc merge/split/cut follow. This also holds for what the lab
runs, which differs from stock: random batch order, spike-count momentum m = clip(0.1·nspTot, 5, 400), and auto-stop are lab changes.

| | Finding **[M from 18 logged sorts]** |
|---|---|
| Fast | Residual reaches about 98% of its final drop by batch 300 |
| Slow | Template count reaches 90% of final at 700-1900 batches and is still rising at stop |
| Stop | At about 3000 batches, but that is filter lag: a simulated residual that converges instantly stops at a median 2923 |
| Cost | 4.4-6.2 passes of a 20-min recording, about 1056-1170 s, against about 215 s per KS4 round |
| Churn | About 35-39 births per batch throughout, about 1% survive |
| Mixtures | The first curation cycle makes 203-335 splits (17-32% of templates), since the loop can't split |
| Order | Only batch order differs between seeds; 69-75% of units reproduce. KS4 default is 84-87%, H8 about 58% |
| Late learning | Per 100 extra batches, S +3.2 ± 1.5 (p 0.08, post hoc, n = 12) |

**The pass count is the wrong unit.** Settling takes the same number of *batches* at 20 min and 2 hours, so a fixed amount of
data does the learning. That favours any scheme that doesn't visit every batch, subsampled rounds included.

**Why upstream left it.** They never say online learning failed. KS3 and KS4 moved to clustering deconvolved features, which
removed the need for one template per neuron. My reading, not theirs: the online loop can't split, and that became the
bottleneck. No one compared the two designs with the same clusterer.

**Worth keeping:** a count-based running mean, random order, and a guaranteed full pass. **Not worth keeping:** single-snippet
birth, the rate gate (a unit at 0.05 Hz has only a 42% chance of a spike in 5 batches), 400-spike forgetting, and EMA auto-stop.

## 4. Theory

Full derivations are in [`notes\theory.md`](notes/theory.md). The proposal splits into two problems that theory treats differently.

**Refining existing templates** is continuous estimation, and online methods have a real advantage here. With forgetting, the
speed-up is about N/n\* in passes, for n\* ≈ 300-1000: 3-11× for the median unit, 13-45× weighted by spikes, and 1-2× for the
slowest decile **[E]**. In a toy, 0.05 of a pass matched five rounds of means-only refinement.

**Finding templates that don't exist** is model selection. Small batches lose evidence there, and the online literature's
remedy is to accumulate across batches. kNN clustering at about 10 spikes per unit is near chance on close pairs **[E]**.

**KS4 has no single objective.** Extraction plus means is hard EM on ‖X−Σa T‖² + Th²|S|. The graph step breaks that: labels
aren't minimisers of that cost, the graph changes each round, and merges are hypothesis tests. So rounds can only converge as
a fixed point of a map, with no guarantee against cycles, and seeds appear to settle in different basins.

**Prediction by scheme:**

| Scheme | Expected against repeat rounds |
|---|---|
| A. Online means only, then a frozen pass | Faster, and worse |
| A′. A, then one global clustering | Worse. Refining a lump erodes the bimodality a split needs |
| B. A + KS2.5-style residual birth | Compact dictionary, misses the blind band |
| C. Window splits and merges in the loop from the start | Comparable if regrouping is local |
| D. C, then one global extraction and clustering | About round 4's quality. No gain beyond it |

The prediction is **same quality at lower cost at best**, not better quality. The case for interleaving is wall time on long
recordings. Time order also lost accuracy and epochal units against shuffling, so batches should be shuffled.

## 5. What small batches can see

A 2-s batch holds few spikes of a typical unit. The median unit fires at 2.4-2.8 Hz, about 5 spikes per batch, not the 10 often
assumed; the mean-rate unit gives about 11. **[M]**, H8, all eligible units, correct labels assumed:

| | Median-rate unit | 10th-percentile rate |
|---|---|---|
| Batches to cosine .95 | 3-4 | 10-13 |
| Batches to cosine .99 | 17-20 | 53-67 |

For sampling noise to fall to the size of a round's r2→r3 step takes a median of 260-290 of the excerpt's 600 batches for matched
units, and 80-100 over all eligible units.

- Telling a unit from its neighbour is cheap, at about 21-24 correct spikes. A round's step, at cosine .999, is not.
- KS4's own overlap-subtracted features need about 0.6× these spikes, but only once templates are already good.
- Labels set a floor. With round-2 labels, 43-52% of units never reach .99 at any n. An online learner labels with its own
  noisier templates, so the true floor is worse, and it comes with a feedback these numbers don't include.
- Rate isn't steady. The median unit has ≥ 10 spikes in only 15-18% of batches, and half its spikes sit in about 21% of them.
- Templates are nearly stationary: debiased cosine .999 (kobe) and .996 (MEC) across 20 min, .990 across 2 hours.

Existing clustering can't run on a batch (from the code, with arithmetic on these rates). The graph floor needs 11 batches for a 5 Hz unit at H8, 22 at default, and
25-50 in round 1. A pair of 1 Hz units needs 125 batches to leave the tree's sparse shortcut (`swarmsplitter.py:70-72`), which
otherwise merges without a real test. On small clusters the stock refractory test over-merges (3933 → 909, upper bound **[E]**).
So "interleaving with this clustering" means rounds over tens to hundreds of batches, which is close to subsampled rounds, and
much coarser than KS2.5.

The scale of a useful window agrees across three sources: the 260-290 batches above, KS2.5's 300 batches to residual plateau,
and the 10 s to 5 min chunks of published clustering. That's hundreds of batches, not one.

## 6. Simulations

Two toys reached opposite-looking conclusions, because they tested different schemes.

**Ground-truth simulation** ([`notes\simulation.md`](notes/simulation.md)): preregistered, 48 units on 32 channels, equal tuning
budgets, 10 held-out seeds per variant. Its online arm is a KS2.5-style learner, with birth from a buffer and merge and drop but
**no split in the loop**. Paired 95% CIs:

| Hypothesis | Verdict | Result |
|---|---|---|
| H1: I reaches R's template error in fewer passes | Refuted | After 1 pass, +0.057 [+0.032, +0.086] stationary, +0.078 [+0.064, +0.093] nonstationary |
| H2: I recovers poorly matched units | Refuted | −0.025 [−0.100, +0.049]. Extra rounds don't recover them either. Residual discovery adds only +7.8 false units |
| H3: I costs merges, false units, sparse units | Supported | Merged +0.135 [+0.075, +0.198] (T labels); sparse found −0.11 [−0.19, −0.03] |
| H4: subsampled rounds match I at less complexity | Supported | Accuracy +0.027 [+0.008, +0.049] stationary at 1.75 vs 2 passes and a third of the operations |
| H5: drift favours one side | Refuted at 30 µm, unclear at 10 µm | Even the oracle reaches only 0.386 at 30 µm |
| H6: global clustering separates what I can't | Supported | R3 − I (T labels) +0.10 [+0.05, +0.15] on pairs |

The best-case online variant, chosen post hoc in its favour, ties rounds at best. The deficit tracks firing rate: template error
0.433 vs 0.192 at 1 Hz, 0.219 vs 0.204 at 15 Hz. Tripling the recording doesn't rescue it.

**Theory's toy** ([`notes\theory.md`](notes/theory.md) §3.5) adds a per-template split on add-back features every 50 batches. **One
online pass then equals 3-5 rounds with splits**, on all spikes; slow units lag a little. Refine-first-then-split is worse than
either. So the structure moves must be in the loop from the start.

**Taken together.** KS2.5-style interleaving loses, and interleaving with splits may tie rounds in fewer passes. Neither toy settles
it for real data:

- Theory's toy uses two seeds, no overlaps and no hash, easy pairs (ρ ≈ 0.85), and local splitters on both sides. It can't show
  whether KS4's graph does more than local splits.
- The simulation's dictionary hardly changes across rounds (54 → 52), unlike H8 (2450 → 1760), so it misses whatever makes rounds
  help on real data. Its clustering stands in for KS4's, and it fails at realistic density.

## 7. Literature

Full table in [`notes\literature.md`](notes/literature.md); many entries were read through a summariser, as marked there.

- **Nobody has shown that fine interleaving beats rounds.** There is no head-to-head on any data, and no benchmark varies iteration
  count. No study measures first-pass missing waveforms in KS4. The table in `round3_results.md` is the largest effect I can
  find for a further whole-recording round.
- **The closest precedent is coarse.** DARTsort learns on a subset (about 5,000 spikes per channel), with intermediate rounds
  subsampled and the final round full. Its current default is one matching step, and its "batches" only split the E-step, so it isn't
  minibatch EM.
- **The trend runs the other way.** KS3/4 left interleaved learning, and Lupin, SpyKING CIRCUS 2 and Tridesclous 2 are single-pass.
  Mini-batch k-means ends slightly worse than batch. OSort warns that online merges are irreversible. Pachitariu reported that
  the number of passes didn't matter on his data ([MouseLand/Kilosort issue 84](https://github.com/MouseLand/Kilosort/issues/84)); KS1 found full matching unnecessary during learning.
- **Coarse rounds do have support**, usually about two: Marre 2012, YASS, Combinato (hits fell in all 24 comparisons going from 2
  rounds to 1).
- **Scale:** published clustering chunks are 10 s to 5 min, or 1,000-20,000 spikes.

## 8. Compute

Full model in [`notes\compute.md`](notes/compute.md). The expected bottleneck isn't one, and the real ones are elsewhere.

**[M]** at about 1800 templates:

| Item | Cost |
|---|---|
| Full `prepare_matching` rebuild | 15 ms at 1767 templates, 65 ms at 3933 |
| Refreshing `ctc` every batch / every 10 | +11% / +2-3% |
| One batch of matching | 172 ms: `colmax` 79, I/O and preprocessing 18.5, the `ctc` step 9 |
| Clustering | About 10 µs/spike (115 s per 11.8 M); linear, with no small-buffer economy |
| Local cost of one sweep | About 0.09 s per centre on tiny buffers, so 8-10 s per sweep, about 50× a batch of matching |

The dictionary can change as often as anyone wants, and the kernels need no change. The cost is matching and clustering, and
both scale with what you visit. A stale `ctc` is tolerable for a while, then the peel loop diverges (residual up to 1e30), so
`U` and `ctc` must update together.

**Whole-sort time, minutes:**

| Design | 20 min | 2 h | Status |
|---|---|---|---|
| Four rounds | 14.8 | 100 | M |
| Stock two rounds | 6.5 | 48 | M |
| Floor: a dictionary for free | 4.7 | 32 | Sum of M |
| D3, subsampled rounds, f = 1/4 | 8.7 | 56 | E |
| D3, f = 1/6 | 8.0 | 51 | E |
| D3s, f = 1/6, first pass subsampled too | 6.6 | 42 | E |
| D2, online warm-up (1 pass) then the rest | 8.2 | 54 | E |
| D1, online learner, 1 pass | 6.6 | 44 | E |
| D1, 5 passes (near KS2.5's cold start) | 14.3 | 93 | E |

The ceiling is about 3× against four rounds, because drift, the final extraction, clustering, merge and save stay. Against the
stock two-round sort the ceiling is only 27-34%. **The number of passes dominates D1 and is unknown.** Peak RAM (47 GB at 2 h)
comes from the final stages and doesn't change in any design.

| Design | Code | Risk |
|---|---|---|
| D3 | ~40 lines | Low. Rare units; the `dshift` trap below |
| D5, polish by means (no clustering) | ~70 lines | Low |
| D2 | 200-300 lines | Medium |
| D1 | 500-800 lines, a second learner beside clustering | High |

Online also gives up batch independence: each batch then depends on all before it, so results depend on order and update
interval. GPU `index_add_` gave 30 distinct results in 30 repeats, which an online loop would feed forward; a sorted sum fixes it.

## 9. What to do

### Checks on saved sorts (run 2026-11-04)

No sort was run and no unit selected; the same rule applied to every round, seed and setting. Scripts: [`checks\`](checks).
Outputs: `C:\data\ks4_tuning\exploratory\interleaving\checks\` (`t1_summary.txt`, `t2_orphans.json`).

**T1: relabel each spike by the extraction template that caught it** instead of its graph cluster, then score with the harness.
**T** is those labels, **TM** the same after KS4's final merge, **G** the sort's own graph labels. Round k+1's templates are built
from round k's clusters, so T at round 3 keeps round 2's partition and redoes only the extraction. Control: the identity
variant reproduced the saved kobe and MEC scores exactly. H8, means over seeds 1-3:

| S | Round 2 | Round 3 | Round 4 |
|---|---|---|---|
| kobe, G (graph labels) | 1427.4 | 1457.0 | 1462.0 |
| kobe, T / TM | 1245.0 / 1237.4 | 1265.2 / 1262.4 | 1279.5 / 1276.5 |
| MEC, G | 2932.3 | 2989.2 | 3010.7 |
| MEC, T / TM | 2529.9 / 2457.8 | 2560.2 / 2541.3 | 2585.0 / 2566.0 |

| Gain, r2 → r3 | kobe G | kobe T | MEC G | MEC T |
|---|---|---|---|---|
| S | +29.7 | +20.2 (68%) | +57.0 | +30.3 (53%) |
| Correct at 2%, plain (M) | +0.73 | +0.60 (82%) | +0.46 | +0.35 (75%) |
| Correct at 2%, local (M) | +0.72 | +0.65 (90%) | +0.54 | +0.38 (70%) |
| Correct at 0.42 M misassigned, plain (M) | +0.53 | +0.57 (108%) | +0.34 | +0.35 (103%) |

Four readings:

1. **Clustering adds a lot beyond template labels, and about the same every round.** G − TM is 190, 195, 186 on kobe and 474, 448, 445 on MEC,
   about 13-15% of S at rounds 3 and 4. It isn't a one-off. Default loses less (kobe 46-48, MEC 243-255). So a design needs a clustering
   step, or something that does its job: the KS2.5-shaped option in section 8 (D1 with no clustering after the loop, 4.5 min) is out
   unless something else replaces it. T1 compares clustering with no clustering. It says nothing about how many spikes or how long a
   window the clustering needs. KS4's already works around one x-y centre at a time, and each spike searches for neighbours only among
   every `cluster_downsampling`-th spike, capped by `max_cluster_subset`; every spike stays in the graph and gets a label. Random subsamples
   (D3) keep the whole recording in view at lower cost, with rare units the main risk. Temporal chunks must also be matched across
   chunks, and need minutes of data: at 2.5 Hz a 2 s batch holds about 5 spikes. Both are untested on real data.
2. **Template labels rise round to round because the previous round's clustering rebuilt the dictionary.** In `repeat` mode, new
   information reaches the dictionary only through the previous round's clusters and their merged means, so T3 − T2 credits round 2's
   clustering while G3 − G2 credits round 3's. The percentages in the table above therefore compare gains offset by a round. They
   don't show how much of a round's gain would survive without clustering. Two observations fit this lag without proving it. Eligible units
   in T match the previous round's G: 1678 against 1722 on kobe and 1799 against 1848 on MEC at round 3, 1634 against 1665 and 1714
   against 1777 at round 4. And from r3 → r4, G's gain falls to +4.9 and +21.5 while T's holds up better at +14.3 and +24.8.
3. **The gain isn't refinement of shapes alone, as far as this shows.** The round-3 dictionary is rebuilt from round 2's graph
   clusters, so these labels inherit that structure. T1 removes the *last* clustering, not the structure. It can't
   separate shapes from structure; only the Refit arm below can.
4. **The first-pass refractory hypothesis (section 1) survives** but isn't tested here.

**Theory's prediction remains untested.** It said template labels would gain under half of repeat's gain, and that ≥ 80% would break the
structural reading. Under half held on none of the eight T measures and ≥ 80% on four, but the comparison is offset by a round (reading 2),
so these counts neither confirm nor fail the prediction.

The alternative baseline gives the opposite-looking number. As first framed in the skeptic's "no recluster" arm, TM(r3) − G(r2) is −165 on kobe
S and −391 on MEC (−556% and −687% of the gain), because it counts the whole clustering bonus of reading 1 against the template
labels. Neither baseline measures what a round's clustering adds to later rounds: this one measures dropping clustering labels altogether, and
the T gains above measure the previous round's clustering arriving through the dictionary.

Corrected 2026-11-05. This section first said that one global clustering was indispensable, that the gain was mostly independent of that
round's clustering, and that theory's prediction had failed. T1 supports none of the three.

Caveats:

- TM uses a proxy template (spike-weighted mean of final units) for the merge's shape test. Its round-2 values look harmed: correct at 2%
  falls from 4.41 to 3.66 M on kobe. So TM's gains on the curves are inflated (+1.22 M), and I've used T there. TM's shares of the S gain are 84%
  and 146%.
- T has no final merge, while G does, though the G − T gap is also steady: 182, 192, 183 on kobe.
- At 2% on the plain estimate T sits level with G at round 3, while the local estimate and the 0.42 M point both favour G by about
  0.4-0.6 M. The plain estimate may flatter labels that mix units.
- Default gains nothing in S under any labelling (kobe −0.2 G, −5.5 T; MEC −22.4 G, −10.6 T).

**T2: could a residual learner see absorbed units?** A spike of signal s caught by a template at cosine c leaves s√(1−c²). Share of
round-2 orphan spikes with that above Th, assuming a filter matched to the residual:

| | Orphans of eligible | Median c | Signal per spike | Signal per unit | With universal fit 0.85 |
|---|---|---|---|---|---|
| kobe H8, seed 1, rebuilt `U` | 217 of 1721 | 0.958 | 15.1% | 7.1% | 2.7% |
| MEC H8, seed 1, rebuilt `U` | 274 of 1798 | 0.964 | 10.6% | 5.2% | 2.0% |
| H8, all seeds, proxy shapes | | 0.964-0.977 | 5.9-10.0% | 3.1-4.7% | 1.1-2.4% |
| Default, all seeds, proxy shapes | | 0.955-0.976 | 7.6-19.6% | 4.7-16.0% | 2.0-9.8% |

**Theory's prediction of under 20% holds at H8**, by a clear margin. Per-spike figures are biased up, since score is conditioned on
passing Th. All borrowed spikes behave similarly (17-18% with `U`). Default is proxy only, and the proxy reads about 1.5-2× low at H8, so
default can't be claimed. So at H8 a residual-based birth would miss about 85-95% of these spikes. Finding such units means splitting on add-back
features, as section 2 says.

**Still to do: save `U` at every round.** Nothing saves extraction templates, so `U_r3` and `U_r4` can't be compared directly.

### Pilot

H8, both excerpts, seeds 1-3, each arm paired with its own seed's round 2 from the stage cache. About half a day of code on a
branch and about 3 hours of machine time. T1 has already given the **No recluster** arm's labels from saved files, so it needn't be run,
but its gain is offset by a round (reading 2) and can't decide what clustering contributes. **Refit** is now the decisive arm.

| Arm | Change | Isolates |
|---|---|---|
| Refit | Same templates, each refit to its own spikes' mean; extract, cluster | Shapes alone |
| No recluster | Repeat's templates; extract; labels by template | Shapes + dictionary, no re-partition |
| Subsampled | Rounds on a uniform 25% of batches, then full final rounds (also 1/8) | Speed. **Most likely outcome worth adopting** |
| First pass | Default bimodality in the first pass only, and refractory `meta` requiring significance; no extra round | Repair of H8 |
| On W | One extra round on A3 | Shrink against rebuild |
| Online updates | One pass from round 2's dictionary, updates only, no birth, shuffled | The cheapest honest interleaving (predicted: like augment) |

**Method.** All arms go through the same scoring, with all eligible units, S and S_track and the curves, and no unit subsetting. Report
per-arm eligible units, low-S units, reproducibility, KS2.5-matched units and duplicates removed. Biases to keep in view: H8 was tuned at
two rounds; S can't tell whether a lost low-S unit was a real cell; refractory contamination misses contamination that respects the
refractory period; and three seeds are few. **G** is repeat's round-3 gain: kobe S +29.7, MEC +57.0, correct at 2% about +0.73 M and
+0.45 M.

**Stop interleaving if any holds** (rules as set before T1 ran; see the note after this list):

- Refit and No recluster are each under ⅓ G in S and on the curves on both excerpts: the gain needs re-clustering.
- Subsampled keeps ≥ 70% of G at ≤ ⅓ of a round's cost: adopt it. What's left for interleaving is about 15% of runtime.
- First pass reaches δ on kobe and 2 SD on MEC with no extra round: fix the first pass instead.

Note on the first rule: it didn't say which baseline No recluster takes. Against round 2's graph labels it is far under ⅓ in S; against
round 2's template labels it is well over, but that gain comes from round 2's clustering, so it doesn't count as a gain without clustering.
I have not changed the rule after seeing the result. In practice it now turns on Refit, and a
Refit arm must be scored by graph labels after a normal clustering, so that this ambiguity doesn't arise.

**Continue only if all hold:**

- Refit reaches ≥ ⅔ G on both excerpts, so shape refit alone carries it.
- Subsampled fails, or a shape update from under 60 s of data per centre reproduces Refit. Units under 1 Hz must not lose S or agreement.
- Cross-unit duplicates and low-S unit loss rise no more than under repeat.

### If it continues

Build Scheme D, not KS2.5's loop: shuffled batches, count-based means with memoised per-batch statistics, a per-template split on
add-back features from the first batch with evidence over a window of hundreds of spikes, merges in the loop, then one global
clustering and one full extraction. Held-out checks on splits guard against path dependence. Its bar: at most 50% of round 4's
runtime at 3 seeds on both excerpts, with reproducibility and KS2.5-matched units not lower. Theory expects it within 14 of round 4 on
kobe and 24 on MEC, with no gain beyond. Its value would be wall time on long recordings, so test at 2 hours.

### Separate branch: the original premise

Interleaving is not the tool for it. One residual pass with the universal bank after final extraction tests it directly: continue only if
new units add ≥ δ/2, reproduce across seeds and aren't duplicates. `nearest_templates` is a cheap, unsearched first-pass knob.
**Reaching units below KS2.5's sensitivity needs ground truth, so it would mean reopening the hybrid-data question for this one
purpose.** That is your decision.

## 10. Where the agents differed

| Apparent conflict | Resolution |
|---|---|
| Speed-up ≤ 3×, ≤ 2×, or 1.3-1.8× | Different baselines. 3× is the floor against four rounds; 2× is subsampling against four rounds; 1.3-1.8× is a warm-started online stage replacing two rounds |
| Template step "2-4× seed distance" vs "about equal" | Units. The first is 1−r for r2→r4 (0.77 against 0.24 ×10⁻³). As \|dT\|/\|T\| that is 3.9% against 2.2% (my conversion), and per round 3.1-3.7% against 2.5% |
| Contraction 0.17 vs 0.7 | Different quantities: S, against template step and unit identity |
| KS2.5 count "settles" vs "never plateaus" | Both. It reaches 90% at 700-1900 batches, by slightly different counts, and still gains 2-9 units per 100 batches at stop |
| Local-maximum rule explains 71-81% vs "about half" of misses | Different denominators: the shortfall from round 2's matches, against above-threshold misses seen next to another detection. Both are estimates |
| Clustering 154 µs/spike vs 10 µs/spike | The benchmark ran under load from other agents (about 15× slower). Use 10 µs/spike from the sort logs, and only the linearity from the benchmark |
| Toys for and against | Section 6: only one has splits in the loop |
| No-recluster share −556% vs 53-108% | Different baselines, section 9: dropping clustering labels altogether, against the previous round's clustering arriving through the dictionary. Neither measures a round's clustering directly |

## 11. Limits and by-products

**Limits.**

- No ground truth on real data. Raw-data measurements are one seed; the 2-hour result is one seed.
- Survivor sets: matched-pair measures favour stability. The all-eligible chain is reported beside them for that reason.
- The cost figures for D1-D3 are estimates; quality for all of them is untested. Some timings ran on a loaded machine.
- The mechanism table reads saved data. Only the pilot can separate (a) to (e).
- Rounds themselves still have open questions: the default-settings MEC drop, cross-unit duplicates rising 37-56%, and 75% of
  MEC's added spikes being removed as duplicates.

**By-products, useful whatever happens to interleaving.**

- `colmax` is 79 of 172 ms in every extraction. A sparse rescan could cut much of it, and with `fast_matching` perhaps 20-30% off every
  extraction **[E, not measured]**. That is a larger sure saving than most interleaving designs against the stock sort.
- Drift and universal extraction are two passes of the same detector (33 + 32 s at 20 min).
- `cluster()`'s unused `iclust` allows warm-started clustering, which may reduce churn between rounds.

**Traps for any subsampling code.** `bfile.set_downsampling` after drift correction applies the wrong batch's `dshift`
(`io.py:893,1137,1159-1160`), an error small enough to miss. New settings go in `LATE`, not `PASS2`, in `stage_cache.py`. Scale
`cluster_downsampling` with f to keep the landmark count.

**Corrections.** Stock KS2.5 *is* on disk, as git tags `v2.0`-`v3.0` in this repository, contrary to `modular_pipeline_audit.md`. My
agents' brief wrongly credited OPASS to Pachitariu (Carlson 2013) and binary pursuit to Ekanadham (Pillow 2013).

## Files

- Notes: [`ks4_flow`](notes/ks4_flow.md), [`ks25_learning`](notes/ks25_learning.md), [`theory`](notes/theory.md),
  [`literature`](notes/literature.md), [`empirics`](notes/empirics.md), [`compute`](notes/compute.md),
  [`simulation`](notes/simulation.md), [`skeptic`](notes/skeptic.md).
- Scripts: `empirics\`, `compute\`, `simulation\`, `skeptic\`, `theory\`, `checks\` in this folder.
- Outputs: `C:\data\ks4_tuning\exploratory\interleaving\` (`empirics\` is 7.7 GB, and its `*_res.npy` reservoirs can go; `checks\variants\`
  is 25 GB of relabelled sorts, which can go once the numbers are no longer being rechecked).
