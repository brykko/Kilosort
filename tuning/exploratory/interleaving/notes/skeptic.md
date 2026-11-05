# Skeptic's review: fine interleaving of extraction, clustering and template refinement (2026-11-04)

Claim reviewed: the 4-round KS4 "appears to converge on more accurate templates", so a fine interleaving of
extraction, clustering and refinement in small batches might converge much faster than whole-recording rounds.
Premise: waveforms poorly matched by the universal templates are not extracted well in the first pass and so get no
learned template.

Everything here is read-only, from saved sorts, logs and the stage cache. Scripts: `interleaving\skeptic\`; small
outputs: `interleaving\skeptic\out\`. No sort was run, and `kilosort\` and the worktree were not touched. Tags:
**[V]** verified on files or code here; **[I]** inferred; **[R]** read from an earlier result document, not rechecked.

## Verdict

**The gain from extra rounds is real. "Converges on more accurate templates" is not what the data show, and the data do
not support building fine interleaving now.** Template accuracy has never been measured. The nearest measurements I
could make show the dictionary becoming consistent with the second-pass clusters in one step, not a walk towards a
better template. The first-pass-miss premise is close to absent for every unit KS2.5 can see, and the rounds do nothing
for those units anyway. The part of the rounds most likely to matter involves refractory merging, which small batches
serve worst. And KS2.5, the fine interleaver on this machine, took 5.9 passes of the recording to learn its templates.

I can't say which of (a)-(e) carries the gain; no existing data separates them, and any ranking I gave would be a
guess. A pilot of about half a day (below) settles it. If shape refit alone carries the gain, the case for interleaving
reopens, in the much smaller form of one early subsampled refresh.

## 1. What the extra rounds do

### What is and isn't a measurement of template accuracy

`round3_results.md` contains no measurement of template accuracy. "More accurate templates" is inferred from S, the
trade-off curves and reproducibility, all of which are measures of units. With hybrid data retired there is no truth to
measure against. So I measured what can be measured without truth, all in a fixed rule applied to every comparison
(same spike: within 6 samples and 40 µm, mutual nearest; stable pair: mutual best, agreement ≥ 0.5; seed-to-seed
comparisons as the null).

| Quantity (median unless stated) | r2→r3 | r3→r4 | default r2→r3 | same round, across seeds (control) |
|---|---|---|---|---|
| Same spike's score (projection on unit-norm template), kobe, % | −0.052 | +0.049 | +0.023 | 0.000 |
| The same, MEC, % | −0.065 | +0.052 | +0.005 | −0.002 |
| Same spike's feature norm, mean, kobe / MEC, % | −0.11 / −0.15 | −0.07 / −0.11 | −0.02 / −0.13 | 0.00 / 0.00 |
| Spikes staying in a stable pair, kobe / MEC | 0.68 / 0.56 | 0.78 / 0.67 | 0.88 / 0.83 | 0.68 / 0.55 (r2) |
| Wall step, 1000(1 − r), kobe / MEC | 0.59 / 0.70 | 0.44 / 0.59 | 0.49 / 0.65 | 0.37 / 0.58 (r2) |

(`s2_pairs.py`, seeds 1-3, stable pairs; units ≥ 1000 spikes for Wall.) **[V]**

- **Fit to the same spike does not improve.** A template that points closer to a spike raises that spike's score. The
  change is −0.05% then +0.05%, against ±0.002% for the control and a 5-7% penalty for borrowed spikes. Round 2→3 has a
  reason to go down (a maximum over 30% fewer templates), so flat is not "no change in shapes": the same fit with fewer
  templates is modestly favourable. But score moves only at second order in template error, so this bounds the
  improvement rather than excluding one. It covers about 55-70% of spikes, those in stable pairs; units that change
  could behave differently. **[V for the numbers; the reading is I]**
- **The partition moves as far as a change of seed.** Spikes staying in a stable pair r2→r3: 0.68, the same as across
  seeds (0.68). r2→r4 is further than a seed change (0.63; MEC 0.51 against 0.55). And r3→r4 still loses 22-33% of spikes
  from stable pairs. There is no sign of a fixed point in the partition. **[V]**
- **Same cells, followed through all four sorts** (quartets: 2 seeds × rounds 2 and 4, `s3_quartets.py`): cross-seed
  distance does not fall. Kobe 0.21-0.24 → 0.22-0.27; MEC 0.35 → 0.35-0.40. It fell in 53-58% of quartets. The low-amplitude
  half does fall by 8-26% (kobe 0.51-0.56 → 0.43-0.48; MEC 0.63-0.78 → 0.50-0.65), but that half also falls at default on kobe, and
  splitting by round-2 amplitude invites regression to the mean. **[V]** Quartets are cells stable by selection, so they are the
  cells with least to gain.
- **But the step is systematic.** The r2→r4 step is about 0.77, 2-4 times the cross-seed distance, and the distance
  doesn't grow, so two seeds move the same way. This is the best evidence for the proposer: templates shift in a repeatable
  direction. A shared shift shows a shared cause, though, not that the end point is truer. **[V; I for meaning]**
- **Default shows the same systematic shift and no gain in S.** Default's step is 0.39-0.54 against a cross-seed distance of
  0.06-0.19, the same pattern, yet ΔS is −0.2 on kobe and −22.4 on MEC. Moving templates this way is not sufficient for
  S. Its curves do rise 0.15-0.25 M **[R]**, so it isn't nothing.
- **Extraction templates do jump once.** Round 2's dictionary U, rebuilt from the cached first clusters (counts match, 2504
  and 2526), lies far from the final cluster means: to its owner, median 1 − r = 0.027-0.029, spike-weighted 0.010-0.013, 90th
  percentile 0.10-0.11. Later steps are about 0.0005 (`s11_u_vs_wall.py`). **[V]** So in the sense of self-consistency,
  "extraction template ≈ mean of what it caught", the process does converge, in one step. Part of this distance isn't error:
  several U templates may fit their subgroups better than the unit mean does, which would explain why the score didn't rise.
  Residual if it is error: a·√(2(1−r)), so 2.2-2.5 for a median spike and about 7-8 for a norm-50 spike, about Th_learned
  (7). So a real subtraction effect at r2 is physically possible. **[I]**

**Answer: the data show a dictionary becoming self-consistent with second-pass clusters in one step, then small systematic
drift, with a partition that keeps moving by seed-sized amounts. They don't show convergence on accurate templates.**

### Disentangling (a)-(e)

| Mechanism | For | Against | Status |
|---|---|---|---|
| (a) better shapes, better subtraction | U is off by 1 − r ≈ 0.01 spike-weighted; same-unit short ISIs fall at r3, 24% on kobe and 12% on MEC, then nearly stop (0.57-1.5 ms, per 1000: 3.9 → 3.0 → 2.9; MEC 5.0 → 4.4 → 4.2); feature norm falls 0.11-0.15% | Default gets the same shift, −8 to −9% in those ISIs and no S gain; cross-unit duplicates *rise* 37% (kobe), 56% (MEC), when better subtraction should lower them | Plausible, size unknown |
| (b) re-clustering, different partition | Partition moves by seed-sized amounts | Moving isn't gaining: a seed change moves it as far with ΣΔS ≈ 0 | Unresolved |
| (c) dictionary shrink, less competition | Round 3 gain coincides with templates per pre-merge cluster going 1.34 → 0.98 (kobe; MEC 1.18 → 0.94) while default stays near 1; violations fall where the dictionary shrinks most, 6 of 6 depth-bin correlations (−0.19 to −0.61) | **Natural experiment below**; the same bins show no clear link with ΔS (−0.14 to +0.34); MEC has the smaller ratio and about the same relative gain | Shrink as such: unlikely. See below |
| (d) better features or spikes alone | Features change on 35-38% of spikes by > 1% | The same as a seed change (29-34%), and timing differs on 15-17% of spikes; after cleaning, spikes rise only 1.6% (kobe) and 0.6% (MEC) | Weak |
| (e) re-roll with selection | Lost/gained counts match the seed control **[R]** | Nothing selects between rounds; ΣΔS over stable pairs is −7.7 to +4.7 across seeds (MEC −2.7 to +2.5) against +18.6 to +27.9 (MEC +43.8 to +61.2) for r2→r4 | Rejected for the gain; true of the turnover |

**The natural experiment on (c), and the main thing I'd add to the discussion.** `merge_correlation_threshold_1` changes
the dictionary with no extra round, from identical first clusters **[V, kobe logs: 'iU shape' lines]**:

| | first clusters | templates | final clusters | ΔS against default |
|---|---|---|---|---|
| C1, first merge 0.975 | 2983-3026 | 2693-2766 | 1766-1782 | +370.6 |
| A3, first merge 0.9 (= W) | 2983-3026 | 1863-1885 | 1696-1762 | +368.6 |

That is the same 30% shrink as round 3, to about round 3's size, and it changes S by 2.0. Across configurations, larger
dictionaries go with higher S: cds 5, 3315 templates, +399.7 (one seed); H8, ~2350, +392.9; W, ~1870, +368.6. **[V]**
Caveats: kobe hybrid excerpt, S only (those folders have no spikes, so no curves), and at `cluster_downsampling` 20, not
H8's 10.

The dictionary is also not full of near-duplicates. In U, the nearest neighbour's median r is 0.85. Only 13-16% of templates
have a neighbour at r ≥ 0.9, and 2-5% at r ≥ 0.95. All such pairs differ in norm by ≥ 20%, which is why the first merge left
them. Collapsing shapes greedily at 0.9 removes only 7-8% (2504 → 2331, 2526 → 2318). Spikes aren't spread widely either: the
median unit has 1.2 effective templates at r2 and 1.13 at r3, with 11% → 9% borrowed spikes on kobe. **[V, `s1_dictionary.py`]**

So what shrinks at round 3 isn't a set of near-duplicates. The templates that go have r ≈ 0.85 to a neighbour and are joined
by the second pass's tree. There is a code reason for that **[V]**. `clustering_qr.run` passes spike times to the split only in
template mode, and at H8 (bimodality 0) a node with no refractory evidence is never merged, so the first pass merges no tree
node at all: 3933 first clusters at H8 against 1106 at default. **The dictionary at r3 is the first to contain the refractory
information.** I think this, rather than count or competition, is the difference between A3's 1870 templates and r3's 1767.
That is a hypothesis **[I]**. It predicts the gain in default's case would be small (its bimodality merges work in the first
pass), which is what happens in S, but this is not a test.

I also tested and discarded one mechanism of my own: clustering zero-pads feature channels outside a template's window, so
a cell spread over peak channels could look split. Only 0.1-0.3% of window channels are affected (H8), less than at default
(0.6-1.0%), which gains nothing. **[V]**

### Implications

- Augment gains half in S and nothing on curves. It keeps every old shape and removes nothing, so it switches off (a) and the
  dictionary rebuild together. It can't separate them, and "rebuilding is what matters" is as far as it goes.
- H8 only, in S. Default gains nothing on kobe and loses on MEC, unexplained. A mechanism described as general (better
  templates) ought to help default, and it doesn't. A mechanism tied to H8's first pass fits.
- One round does most of it; round 4 is within noise on kobe S (+4.9 ± 7.3). Short ISIs stop falling after one step on kobe.

## 2. What interleaving could deliver

| | Can small batches deliver it? | Why |
|---|---|---|
| (a) | **Yes**, and it's the one clear yes | Running means need few spikes. One early update would do it: the jump is one step, later steps are 20 times smaller |
| (b) | No, not in small batches | Graph clustering by centre. 20,000 spikes in a 2-s batch, 11 per unit. Expected ≥ 30 spikes per unit: 8% of units in 2 s, 49% in 10 s, 87% in 30 s, 95% in 60 s (`s7_rates.py`; ≥ 100 spikes needs 120 s for 91%) |
| (c) as refractory merging | **No** | First clusters have a median of 460 spikes, and on them the stock test joins too much: union gives 909 clusters (MEC 914), half of the second pass's 1874. With significance required, 2196 (2354). Upper-bound estimate, ignoring the tree and centres (`s10_first_refractory.py`), but the direction is clear: the refractory test gets less safe as batches shrink |
| (d) | Partly | Features are residual + template (`template_matching.py:90-91`). For an isolated spike the template cancels, so only timing, window and neighbours' subtraction matter. Keeping old templates (augment) leaves 90-95% of feature norms within 1% |
| (e) | Not relevant | |

If the gain is mainly global re-clustering with refractory merging, interleaving keeps the expensive step and loses what
mattered. That is the likely failure mode, and it is a question a pilot answers, not one I can close from here.

What interleaving could give that rounds can't, and how much each is worth here:
- **Tracking drift within the recording.** Nothing can be shown on these data: drift is −6.5 to +4.5 µm and `nblocks` gives
  −0.1 **[R]**. KS2.5's own final pass freezes its templates **[R, modular audit]**.
- **Finding units in the residual as it goes.** KS2.5 does this (`learnTemplates.m`, `mexGetSpikes2` on `drez`). See §3 for
  how much there is to find. One residual pass would do the same.
- **Early subtraction for detection.** The universal pass subtracts nothing. See §3: the spike-level loss is huge and seems
  harmless.
- **Risks it adds.** Self-training (a template shaped by what it has already caught), dependence on batch order, stale `ctc`
  or a costly rebuild, and a new family of tuning parameters on an evaluation that can barely afford them.

## 3. The first-pass-miss premise

**What it would take.** To show a unit was missed for poor universal match rather than low amplitude, you need: a unit known
to exist from outside KS4, enough amplitude that KS4's threshold wasn't the reason, poor universal match, and its spikes
absent from the output. KS4's own files can't supply this, since a spike never extracted leaves no trace. **`borrowed_templates.md`
is about a different problem:** orphans *were* extracted, on a neighbour's template, median 0 µm away, at 5-7% fit cost.
Everything there is conditional on having been found. It is survivorship, and neither supports nor refutes the premise.

**Do rounds address it? No, verified.** `spikedetect.run` is called once. Extra rounds call only `_next_templates`,
`_extract_learned` and `_cluster_templates` (`run_kilosort.py:354-392`); augment builds from `Wall` too. A waveform that matches
no learned template stays unseen forever, so the rounds' gain can't be evidence for solving this. What rounds do fix is the
orphan route: found by a neighbour's template, split out in clustering, given a template next round.

**A new measure: cos_u**, the best normalised response of the universal bank to a unit's template (`s6_premise.py`), which
says how much of a spike's energy the universal templates can see, free of amplitude. Median 0.855, 5th percentile 0.70-0.72.

| Result (H8, seed 1) | Kobe | MEC |
|---|---|---|
| Final spikes with a universal detection nearby, net of chance | 41% | 40% |
| Rank correlation of unit's coverage with amplitude score / with cos_u | 0.78 / 0.30 | 0.84 / 0.20 |
| Coverage across cos_u quartiles, lowest score tertile | 0.16 → 0.30 | 0.18 → 0.27 |
| Units with score × cos_u below Th_universal, share of units / S | 21% / 4.7% | 18% / 3.1% |
| KS2.5 eligible units: spikes present in KS4's output (chance 16%) | 97% | 97% |
| KS2.5 units matched / elsewhere / **absent** | 500 / 397 / **7** | 472 / 411 / **4** |
| S of absent units | **3.1** | **1.4** |
| Absent after two extra rounds | 6 (S 2.2) | 4 (S 1.4) |
| KS2.5 units matched, r2 → r4 | 500 → 502 | 472 → 468 |

**[V]**

- Poor universal match is real at spike level: at the same amplitude, a unit with low cos_u is detected up to half as often.
- It does not cost units that way. Most final spikes were never universally detected and are recovered anyway, because a cell
  needs only enough first-pass spikes to seed a cluster. Orphan rate has little or no trend with cos_u (kobe low tertile
  0.18 → 0.07; MEC and the high tertile no trend).
- From outside, almost nothing is missing. Absent units carry 0.07-0.3% of KS2.5's S, at most a tenth of δ. 3 of MEC's 4 have
  the signature (low cos_u, predicted score ≥ 12), so the effect exists, at about 1.3 bits/s.
- KS2.5's 400 'elsewhere' units (S 140 and 447) have their spikes in KS4's output under other labels. That is a partition
  question, and the larger of the two.
- Earlier experiments agree **[R]**: more universal shapes +1.9, wider footprints −2, between-column templates +13.4 (short of
  δ/2), `Th_universal` 7 +44 but saturating at 7 (6 gives the same). The first-pass lever was amplitude.

**What this does not show.** KS2.5 only vouches for what KS2.5 finds. Its own template birth uses `wTEMP` on the residual and its
thresholds are higher (Th 10/6), so a low-amplitude unit that universal shapes miss could be invisible to both. Without
hybrid data, recall below KS2.5's sensitivity can't be measured. So I don't conclude no unit is missed. I conclude that
among units either sorter sees, the premise is worth about 1-3 bits/s, and that the rounds' gain is evidence for something else.

## 4. The speed claim

**Unsound as stated.**
- **The lab's fine interleaver isn't fast.** KS2.5's learning took 3251 batch iterations over a 549-batch recording, **5.9
  passes in 1170 s**, and was still adding units when it stopped (1005 → 1054 over the last 900 batches) (`ks25_sort.log`, kobe
  seed 1 **[V]**). One KS4 extra round costs 215 s. KS2.5 anneals slowly and its cost isn't intrinsic, but it is the only
  measured instance, and it does not show fast convergence.
- **The bound on the saving is modest.** A sort needs one final extraction and clustering in any scheme. Two extra rounds cost
  about 7.5 of 14 min at 20 min of data (48-50 → 100 min at 2 h **[R]**). So the prize is at most a factor of 2, and at most the extra
  rounds' cost.
- **Simpler routes reach most of that.** In order:
  1. **One subsampled refresh.** Build U_next from a uniform quarter of batches, then final extraction and clustering. Saves
     about 3/4 of a round. 300 s of data gives ≥ 30 spikes to every scored unit and ≥ 100 to 97-98%, but no one has
     checked that low-rate units keep their templates.
  2. **Refresh once only.** Round 4 adds a third at best, at the same cost.
  3. **Move refractory information into the first pass**, with significance required (stock over-merges, §2), or use default
     bimodality in the first pass only. No extra round at all, if the hypothesis in §1 is right.
  4. **A single residual pass** for the premise, if it turns out to matter.
  Fewer templates alone is not on this list: A3 says it does nothing, and more universal templates: +1.9 **[R]**.

**Does H8's gain compensate for its own over-splitting?** Partly possible, not shown. For: the first pass has no tree merges at
H8 (§1), and default doesn't gain. Against: the simple version is refuted by A3, a 30% shrink for free with no gain, and
across configurations larger dictionaries score higher.

**Cheap test** (6 sorts per arm, about 6.5 min each, stage A cached): (i) H8 with default bimodality in the first pass only; (ii) H8
with `meta` passed in the first pass and significance required; (iii) one extra round on A3/W. If (i) or (ii) reach most of
round 3's gain with no extra round, the rounds were repairing H8. If (iii) gains as much as H8's round 3 from a dictionary already
at 1870, shrink is not the mechanism and rebuilding from second-pass clusters is.

## 5. Risks to the evaluation

- **H8 was tuned at two rounds.** Bimodality and resolution interact strongly (bimodality alone +136, reverting it from H8
  −324 **[R]**). Round gains are specific to that point; retuning at four rounds could move them either way. Direction unknown.
- **S and unit loss.** All the net loss of eligible units is in low-S units. Kobe s1: eligible −103, units below default's 25th
  percentile of S_i −113; MEC s2: −146 and −159 (`s9_pair_null.py`). **[V]** S can't tell junk fragments cleared from weakly tuned
  cells lost. The guard is passed by a mile (about 770 against about 190) and tests nothing here. S is blind to merges of
  similarly tuned cells **[R, spec §12]**, and the hybrid guard is gone. The curves see merges only through refractory
  violations, weak for low-rate units. KS2.5 finds almost none of the small units in question, so it can't arbitrate **[R]**.
- **Refractory contamination.** Misses contamination that respects the refractory period, and `s4_dose.py` shows strong regression
  to the mean: units clean at r2 gain violations (+85% to +830% from a tiny base), dirty ones lose them. "Contamination fell in
  36-41% of pairs" means it mostly didn't. Dose-response by template spread shows the same shape at default at half the size, so
  I would not lean on it.
- **Duplicates.** Spikes extracted rise 2.3% and 2.2%, but flagged duplicates rise 37% and 56%. On MEC about 75% of the added spikes
  are removed as duplicates. **[V]** Cleaning handles it, but the rising rate fits the opposite of better subtraction.
- **Reproducibility** rises 0.581 → 0.642 partly through the denominator: reproduced units are about 1000 → 1040 (+4%; MEC about 900
  → 925) while eligible units fall. **[I]** Still far under default's 0.855.
- **One seed at 2 hours; three at 20 min; two recordings.** MEC's scale is a spread of 24 across 3 seeds. δ comes from default on the
  hybrid excerpt.
- **Harness freeze.** Scores are exploratory, outside `run_candidate.py`; formal scoring would need the freeze to cover branch code
  **[R]**, and a pilot's arms would each add more branch code.
- **Forking paths.** Kobe is the excerpt H8 was selected on, from 236 runs. Repeat against augment, the curve operating points (2%, KS2.5's
  count), which estimate (plain, local), and post-hoc cuts were all chosen with the results in view. "Round 4 adds on the curves"
  rests on a chosen readout while S on kobe says noise. None of this removes a +30 at every seed, but it should limit how fine a
  conclusion is drawn. My own analyses here are post hoc too, and the natural experiment was found by looking.

## 6. The two cases

**Against the proposal.** The only evidence offered, convergence, wasn't measured, and what can be measured shows fit flat, cross-seed
distance flat and the partition moving like a re-seed. The gain is H8-specific in S, comes in one step, and probably depends on
refractory merging and graph clustering, neither of which works in small batches. The dictionary-shrink account fails a natural
experiment. The premise is worth about 1-3 bits/s among visible units, and rounds don't touch it. The speed prize is at most ×2, most of
it is available by subsampling, and the measured fine interleaver took 5.9 passes.

**For the proposal.** Extraction templates at r2 really are off, by enough to matter for big spikes. Template moves are systematic, not
noise. Most of the template change happens in one step, suggesting few spikes would be needed. Rounds waste work: they re-extract
everything to update what could be estimated from a sample. The low-amplitude half does get more consistent across seeds, with the caveat in §1, and that is where the
hard cases are. Nothing rules out shape refit carrying the gain, and if it does, even a coarse interleaving is cheap. The KS2.5 timing
is a weak comparison, since its annealing is a choice.

**Which I believe.** The first, on the claim as stated, with moderate confidence on mechanism and high confidence on the two factual
points: accuracy is unmeasured and the premise is small among visible units. The second case has one thing right, that rounds spend
far more than template estimation needs, and that argues for subsampling first. Unknown and decisive: whether (a) alone carries the
gain. If so, a coarse version is worth building. If not, stop.

## 7. Pilot and decision rules

Set in advance. H8, both excerpts, seeds 1-3, round 2 from the stage cache, each arm paired with its seed's round 2. Each arm is about
3.6 min per sort, so about 22 min per arm; about half a day with the small code changes on a branch. G is repeat's round-3 gain: S +29.7
(kobe), +57.0 (MEC); correct at 2%, +0.73 M and about +0.45 M. Report S, S_track, curves, eligible, low-S units, reproducibility,
KS2.5-matched, duplicates removed.

| Arm | What changes | Isolates |
|---|---|---|
| F1 refit | Same templates, same count, each refit to the mean of its own spikes; extract, cluster | (a) alone |
| F2 owner-collapse | Keep each owner's largest template, old shapes; extract, cluster | (c) alone, with no refractory-merged shapes |
| F3 no recluster | Repeat's U_next; extract; labels by template, no clustering | (a)+(c) without (b) |
| F4 recluster only | Round 2's spikes reclustered with a new seed, no extraction | (b)/(e) |
| F5 subsampled | U_next from a uniform 25% of batches, then full extraction and clustering | Speed |
| F6 first pass | Bimodality default in first pass only, and significance-only `meta`; no extra round | Over-split repair |
| F7 on W | One extra round on A3 | Shrink against rebuild |

**Stop interleaving if any holds:**
- F1 and F3 are each under ⅓ G in S and on curves on both excerpts: the gain needs re-clustering.
- F5 keeps ≥ 70% of G on both excerpts in S and curves at ≤ ⅓ of a round's cost: adopt it; what's left for interleaving is under about 15% of
  runtime.
- F6 reaches ≥ δ on kobe and ≥ 2 SD on MEC with no extra round: fix the first pass instead.
- A later prototype fails to beat repeat round 4 at ≤ 50% of its runtime, at 3 seeds on both excerpts, with reproducibility and KS2.5-matched
  units not lower. Prototypes get one tuning pass, since each knob spends evaluation the project lacks.

**Continue if all hold:**
- F1 ≥ ⅔ G on both excerpts in S and curves, so shape refit alone carries it.
- F5 fails, or a shape update from under 60 s of data per centre reproduces F1 within noise. Low-rate units are the check: units under 1 Hz must
  not lose S or agreement against F1.
- No increase in cross-unit duplicates or in the low-S unit loss beyond repeat's.

**Premise branch, separately.** One residual pass with the universal bank after final extraction. Continue only if the new units add ≥ δ/2,
reproduce across seeds and aren't duplicates. From §3 I expect under 3 bits/s among units KS2.5 sees; only a surprise would reopen it. To
reach units below KS2.5's sensitivity needs ground truth, and that means deciding whether to bring hybrid data back for this one
question.

Also do first, whatever else: save U at each round. Nothing today saves extraction templates, which is why a direct comparison of U_r3 and
U_r4 isn't possible here.

## Files

`interleaving\skeptic\`: `s1_dictionary.py` (near-duplicates, spread over templates), `s2_pairs.py`, `s2b_features.py` (same-spike and
same-unit comparison), `s3_quartets.py`, `s4_dose.py`, `s5_cleaning.py`, `s6_premise.py` (cos_u, coverage, KS2.5 outside view), `s7_rates.py`,
`s8_search_log.py`, `s9_pair_null.py`, `s10_first_refractory.py`, `s11_u_vs_wall.py`; logs in `out\`.
