# Where should a set_cutoff-like step live in KS4? Brainstorming audit (2026-11-03)

Five independent agents (pipeline anatomist, statistical theorist, skeptic, alternatives scout, pragmatist)
read the KS4 code, the KS4 paper (code taken as authoritative where they differ) and the `autocurate\`
results, and ran read-only CPU checks on the existing scored sorts (`C:\data\ks4_tuning\exploratory\autocurate\sorts\`,
mostly kobe H8 seed 1, several repeated on mec H8 and default). Their check scripts are in the session
scratchpad, not in the repo; nothing here has been through the harness's frozen rules. Numbers marked
*(n agents)* were found independently by that many agents.

## 1. Short answer

**Don't move the deletion earlier. Move the refractory evidence earlier.** The checks show that most of KS4's
contamination is a *grouping* error made in the final reclustering. A score cut cannot repair that; it can
only thin the affected units. The theoretically sound earlier use of set_cutoff's idea is in the split/merge
*decisions* (refractoriness judged on each side's high-score core, and template-group structure respected),
not as spike deletion inside extraction or clustering. Any deletion that remains belongs at the end, preferably
as a label (core vs tail / MUA) rather than an irreversible cut.

## 2. What the checks found about the current cut

1. **The score is amplitude, not goodness of fit** *(5)*. `th_amps = relu(B)/|U|` and `amps = B/|U|²`
   (`template_matching.py:267-269`, same in the fused kernel), so score = fitted amplitude × template norm,
   in whitened-noise units. There is no misfit term and no amplitude prior (`lam = 20` at `:226` is unused).
   KS2.5's `vexp` had an amplitude prior, so it was partly a membership score; KS4's is not, and KS2.5's
   cut-off levels don't transfer. The paper's fixed-norm cost (ks4_paper.txt ~1404-1414) is not what the code does.

2. **Most of the cut is not dynamic** *(3)*. 34% (kobe) to 44% (mec) of H8 units fail the very first level
   (spikes above Th+w already non-refractory) and get a fixed slice at Th+w+0.5. Those units produce
   **75% (kobe) to 85% (mec) of all deletions** at w=2. Another ~40% of units lose nothing. After the cut the
   failing units stay heavily contaminated (median Q 0.42-0.72).

3. **On KS4 the cut acts as a between-unit amplitude filter** *(3)*. Units with median score < 9 lose 64-67% of
   their spikes; units above 14 lose ~0%. Within units, deleted spikes sit at 0.93× the unit's median score.
   On near-threshold units this largely undoes H8's lower `Th_learned`.

4. **Per S, the refractory logic adds nothing over a plain score floor**. A single global cut at score 8.78
   deleting the same 30% gives S 1340.1 against cut2's 1332.6 (σ_S 14.3), and deletes 83% of the same spikes.
   The +6-13% over the matched random control therefore mostly says that low-amplitude spikes carry less
   heading information, which any score cut captures. The per-unit refractory depth shows up instead in
   eligible units (+72 over the global cut) and in hybrid selectivity.

5. **The post-cut contamination figures are optimistic by roughly 2× on the kept side** *(2, independently)*.
   With cut-offs chosen on one time half and contamination counted on the other, held-out Q roughly doubles,
   the pass rate falls (0.54 → 0.41 at w=2), and in 26% (w=2) to 51% (w=4) of cut units the two halves' cut-offs
   differ by ≥ 1.0. The misassigned shares in `apply_cut.log` (e.g. H8 kobe 13.5% at w=2) are in-sample.

6. **Hybrid ground truth supports w ≈ 2, not w = 4** *(2)*. Both agents find w=2 deletes few true spikes
   (2.0-3.3%) and many more of the rest. From w=2 to w=4 the extra deletions hit true spikes at twice the rate
   of others, and 2 recovered units are lost. The two agents' false-spike rates for cut2 disagree (38% vs
   13.3%; one corrected an overlap-labelling bug in its own first pass), so treat the exact ratio as
   unresolved. Caveat: the hybrid sources are well-isolated units, the easy case.

7. **Within a unit, the score ranks spikes better than `|tF|` amplitude does**. In hybrid units, matched-count
   deletion by `amplitudes.npy` rank is no better than random; by score it is clearly better.

8. **Objections that did not hold up** (skeptic). Burst spikes are not deleted preferentially (kobe, mec;
   no CA1 complex-spike cells here). Deletion is not biased by heading, speed or time half. Within-unit
   score-time drift is small over 20 min (median |ρ| 0.03-0.05). This needs rechecking on the 2-hour segments.

## 3. Where the contamination actually comes from

- **Final units are mixtures of extraction templates** *(4)*. In H8, 94% of units hold spikes from more than one
  detection template (median 9); 18-23% of spikes come from a non-dominant template; 98% of templates feed more
  than one unit. Default is milder (median 4, 12-14% non-dominant).
- **Cross-template pairs carry most refractory violations** (theorist). Within final units, violation ratio
  for same-template pairs vs cross-template pairs: class A (median score < Th+3) 0.14 vs 0.62 (mec 0.24 vs 0.70);
  class B 0.03 vs 0.15. A cross-template ratio of 0.6-0.8 means the groups behave almost like separate neurons.
  About half of the testable H8 units look like more than one neuron on this measure.
- **In failing units the dominant template's train is much cleaner than the same number of top-score spikes**
  (anatomist): median Q 0.38 vs 0.52 (kobe), 0.49 vs 0.69 (mec). Dominant-vs-rest is cross-refractory in only
  6% of failing mixed units (different neurons) against 45% of passing ones (one neuron over several templates).
- **Splitting units back into template groups, deleting nothing, cuts misassignment about as much as the w=1
  cut** (H8 kobe Hill plain 28.2 → 20.8%, local 17.0 → 12.0%), though this full split also fragments real
  neurons (1904 → 6677 groups) and is an upper bound, not a proposal.
- **Reconciling an apparent conflict.** The skeptic found template identity explains only ~2% of the
  within-unit variance of *what the cut deletes*; the others found it explains much of *where the violations
  are*. Both hold, and together they say the cut works on the wrong axis: it slices by amplitude, while the
  contamination is organised by template group.
- **Likely mechanisms (reasoning).** The 60-D PC features at SNR 7-10 separate neurons less well than matching
  pursuit's template contrast; `get_data_cpu` zero-pads outside each spike's template channels
  (`clustering_qr.py:636-639`); and under H8 (`merge_bimodality_threshold` 0) the tree merges *only* through
  cross-refractoriness, whose "sparse" shortcut (`swarmsplitter.py:70`, Q00 < 0.25) merges small, low-rate
  leaves with no significance test, and a top-down merge absorbs its whole subtree untested.

## 4. Candidate insertion points

| # | Where | What "unit" means there | Verdict |
|---|---|---|---|
| P1 | Universal detection / first clustering (`run_kilosort.py:804-830`) | First-pass cluster; no deconvolution, no refractory test (`meta=None`) | No value |
| P2 | Per-template threshold in matching pursuit (`template_matching.py:250`, fused kernel) | Extraction template (≠ neuron) | **Worst** (unanimous): refused spikes stay in the residual, get claimed by another template (contamination migrates), and corrupt overlapping spikes' features; needs re-extraction and kernel changes; per S it's just raising `Th_learned`, which tuning rejected |
| P3 | After extraction, before reclustering, per template | Extraction template | Weak: same mismatch; thinner trains trigger the sparse shortcut more |
| P4 | Inside the merge tree, as **deletion** (`swarmsplitter.split`) | Tree node / leaf | Bad: refractory signal used twice, leaves too small for power, more sparse-shortcut merges, H8 optimum moves |
| P4′ | Inside the merge tree, as **decision**: refractoriness tested on each side's high-score core, plus significance and union rules | Tree node | **The defensible "inside clustering" form.** Nothing deleted; targets the grouping error |
| P5 | Between tree and `merging_function` (KS2.5's cut1 → merge2 position) | Pre-merge clusters | Collapses to post hoc under H8 (0.95 merge threshold); matters only if that threshold is relaxed. Trap: empty clusters mis-index `ns`/`feature_ind` |
| P6 | After `merging_function`, inside KS4 | Final unit | Same result as post hoc; a packaging choice (~½ day); write `spike_scores.npy` regardless |
| P7 | Re-score against the final unit (tF projection on final `Wall`, or misfit χ² = \|tF − a·U\|²) | Final unit | Gives the cut a membership meaning; needs tF (deleted by `ks4_sort_scored.py`) |
| P8 | Template-aware split of final units (dominant template refractory and clearly not cross-refractory with the rest) | Final unit | Cheap, testable now; targets the failing units that the cut only thins. Risk: oversplits drifting neurons |
| P9 | Second matching pass with final templates, then cut (KS2.5 semantics) | Final unit | 2-3 days + ~2 min GPU/sort; reverts KS4's clustering-based assignment; only if P7 shows own-unit scores are far more selective |
| P10 | Joint mixture with background component and refractory point-process prior | — | Principled endpoint (Pouzat et al. 2004 style); research-scale |

## 5. Theory points worth keeping

- **set_cutoff is a cumulative-FDR rule** (theorist): it lowers the threshold while the *whole kept set*
  stays ≤ ~5% contaminated. For an additive objective, the optimal rule thresholds the *local* contamination
  probability per spike (delete if lfdr > p*). The harness audit's ~1:1 exchange (5% thinning −51.6 bits/s,
  5% contamination −55.7) gives p* ≈ 0.5 for S; set_cutoff implicitly assumes contamination costs 2-3× more
  than a lost true spike, which suits CCG/timing work, not rate coding.
- **Relabelling cannot lose expected S; deleting can** (scout). Per-unit Poisson information is subadditive
  over superposition (log-sum inequality), so splitting a unit into core + tail never lowers expected S, while
  deleting lowers it whenever the deleted spikes carry any tuning. That explains "beats random but still loses
  S". The flip side: S is nearly blind to purity, so purity must be judged by cross-fitted contamination and
  hybrid truth, and relabelling can be metric-gaming (~1000 extra MUA "units" would pass eligibility).
- **Low-score spikes are mostly true** (theorist): within units, score predicts contamination only in
  mid-size units (class B low/mid/high third 19/7/6%); in class A it is flat or reversed. Deletion is justified
  only where a spike is foreign and unattributable.

## 6. Other findings

- **Duplicate window too narrow** *(2, independently)*. Within-unit violations pile up at 0.3-0.5 ms, just
  past `duplicate_spike_ms` 0.25 (~10k pairs on kobe, ~¼ of violations in the Hill window); neighbouring units
  show +35% coincidences at 0.27-0.4 ms. This inflates every refractory-based estimate, KS2.5's too. Try
  0.5 ms. **Withdrawn after the follow-up audit** ([duplicates_audit.md](../duplicates/duplicates_audit.md)):
  in the raw data 80-95% of these pairs are two separate spikes (a near-synchronous neighbour mixed into the
  unit), so a 0.5 ms window would delete real spikes and hide contamination. True duplicates are rarer
  same-template echoes, better caught by a score-order rule.
- **`spike_templates.npy` == `spike_clusters.npy`** *(2)* (`io.py:370`); the extraction template is
  `spike_detection_templates.npy`.
- **Paper vs code** *(4)*: a refractory CCG makes the tree *merge* (paper ~1606 says "split"); the score uses a
  free amplitude, not the paper's fixed norm; only `is_ref` units (computed once, before any merge) can start a
  final merge; `Pk` normalisation differs, so set_cutoff's Q/R thresholds are not KS4's R12/Q12; `max_peels`
  100 vs the paper's 50.
- **Two `check_CCG`s**: the tree's (`swarmsplitter.py:53`, sparse shortcut, hard-coded 0.1/0.25) and the final
  merge's (`CCG.py:81`, no shortcut, `acg_threshold` 0.2). A cut interacts with each differently.
- **Runtime is a non-issue**: the port runs on all of kobe H8 (11.9 M spikes) in 12-16 s single-threaded.
  KS4's own dedup removes 0.04% of spikes, so whether the cut runs before or after it doesn't matter.

## 7. Proposed tests, cheapest first

**Stage 0: existing sorts, CPU only, scored with the harness's `evaluate.py`/`compare.py`/`transfer.py`
outside `run_candidate.py`.** Report S against matched random, cross-fitted contamination, and hybrid
true/false deletion for every variant.

1. *Decompose the cut*: (a) good-units-only (units passing at Th+w; the dynamic part), (b) failing-units-only
   slice, (c) two-class (keep passing units whole, slice failing ones; already scored on kobe H8: S 1351.2 vs
   cut2 1332.6, hybrid true spikes deleted 0.9%), (d) global floor and per-unit quantile at matched counts. The
   theorist predicts (a) is close to S-neutral.
2. *Relabel instead of delete*: cut2's tail as a companion MUA unit per parent. Expect S ≈ uncut and core
   purity = cut2.
3. *Template-group split* guarded by a significant, non-refractory cross-CCG (KS4's `check_CCG` with
   significance required), relabelling only.
4. *Duplicate window* 0.5 ms post hoc, then rerun the cut.
5. *KS2.5 order emulation*: cut → greedy CCG merge at `merge_correlation_threshold_2` 0.5-0.9 using
   `similar_templates.npy` → cut. Tests whether a cut lets the final merge be relaxed safely.

**Stage 1: new sorts, existing switches, no code** (~7 min GPU each): `tree_refractory_significance=True`;
`tree_walk='bottom_up'`; both. Measure the failing-unit share, S, and the cut's effect on top.

**Stage 2: small default-off code changes** (needs spec v0.21 + refreeze if it enters the search):
score-gated refractoriness in `swarmsplitter.refractoriness` and `merging_function`; the commented-out
union-refractoriness rule (`swarmsplitter.py:89-95`); always write `spike_scores.npy`; per-spike own-unit
projection or misfit χ² (one sort keeping tF).

**Stage 3, only if stages 0-2 justify it:** a second matching pass with final templates (P9).

## 8. Strongest objections to this direction

- All the evidence for "grouping error dominates" is refractory-based, the same circular currency, and the
  0.3-0.5 ms excess may inflate cross-template ratios.
- Template groups often split one drifting neuron; significance-gated tests on low-rate groups have little
  power and risk fragments and seed instability; S barely rewards splitting co-tuned neighbours.
- Every cut tested so far loses S by more than δ against uncut (two-class w=2: −60 bits/s), and round 3 shows
  S favours admitting low-score spikes. A built-in cut ships something the harness says loses information;
  its case rests on single-cell identity, which only cross-fitted contamination and hybrid truth can judge.
- Hybrid recovery barely moves between variants (sources are clean), so the true/false deletion ratio is the
  usable hybrid measure, and it rests on a few units.
