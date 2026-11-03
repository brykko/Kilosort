# Duplicate spikes in KS4 and KS2.5: multi-agent audit (2026-11-03)

The trigger was a side-note in [placement_audit.md](../autocurate/placement_audit.md) §6: within KS4 units,
refractory violations pile up at 0.3-0.5 ms, just past KS4's 0.25 ms duplicate window. Five agents
(mechanisms, empirics, methods, threshold, skeptic) each took a different part of two questions:

1. Does KS4 suffer from duplication more than KS2.5, in theory and in the saved sorts?
2. How do the two sorters handle duplicates? Is KS4's method sound, and is there a principled way to choose
   the lag threshold?

Their scripts are in this folder by lens (`empirics\`, `mechanisms\`, `methods\`, `skeptic\`,
`threshold\`). Their outputs as run are in `C:\data\ks4_tuning\exploratory\duplicates\<lens>\` (the
empirics agent's 400 MB histogram cache was not kept). The scripts write their outputs next to themselves,
so rerunning one writes into this folder. *(n)* marks a finding that n agents reached independently.
Lags are in samples at 30 kHz (30 samples = 1 ms).

## 1. Short answer

- **The 0.3-0.5 ms pile-up is mostly not duplication** *(4: methods, skeptic, threshold, mechanisms)*. In the
  raw data, 80-95% of the excess within-unit pairs are two separate spike-shaped events. Since one neuron
  can't fire twice within 0.5 ms, one spike of each such pair belongs to another neuron: it is contamination
  by a near-synchronous neighbour, not a double count. The suggestion in `placement_audit.md` §6 to widen
  the window to 0.5 ms is withdrawn. It would delete about 10 real spikes for every duplicate it caught, and
  it would hide contamination from every refractory metric.
- **True duplicates exist, but they are rare and look different** *(3: mechanisms, threshold, methods)*.
  Most are "echoes": the same template re-fits the residual of a large spike 9-16 samples (0.3-0.53 ms)
  later, at about a quarter to a third of the first spike's score. They number about 0.1-1 per 1000 spikes,
  concentrated in a few large, clean units. They hurt those units badly: the Hill estimate puts four of KS4's
  best units (~100 µV) at c = 1.00, against 0.00-0.02 once the echoes are censored.
- **Q1: does KS4 duplicate more than KS2.5? Yes, in three specific ways, and much less than the raw lag
  profile suggests:**
  - within-unit echoes (about 3-6× KS2.5 in default, close to KS2.5 in H8);
  - cross-unit duplicates at |lag| ≤ 0.2 ms (about half of KS4's §6.3-flagged coincidences, against ~10%
    for KS2.5);
  - a small batch-edge artefact.
  Separately, KS4 units hold far more near-synchronous spikes of *other* neurons (about 10-30× KS2.5 per
  spike). That is a mixing problem, the same grouping error the placement audit found, not duplication.
- **Q2: KS4's `remove_duplicates` is not ideal**, but mostly for reasons other than its window:
  - its window, which in practice removes lags of 6 samples or fewer, catches almost none of the echoes;
  - roughly 60% of what it does remove are real overlapping spikes;
  - it never looks across units, which is where KS4's real duplicates are.
- **Threshold: no lag-only threshold is principled.** Pooled over units, true duplicates are at most about
  10% of the pairs at any lag. The principled rule is a physical bound (only test pairs within about 0.7 ms),
  then classify each pair by evidence that one spike was detected twice. The best such evidence is a weaker,
  later re-fit (score-order asymmetry) or the absence of a second spike in the residual.

## 2. The lag profile, per sorter

The ratio is the pair count at that lag divided by each unit's own 5-20 ms level, pooled over units. The
"floor" is the level at 1.0-1.5 ms. *(5 agents measured this; the numbers agree within their
normalisations.)*

| Sort (kobe) | 0.27-0.5 ms | floor | excess at 0.25-1 ms, per 1000 spikes |
|---|---|---|---|
| KS4 default, before its own removal (seed 1) | .30 | .16 | ~3.0 |
| KS4 default, final (5 seeds) | .27-.30 | .14-.16 | 2.96 |
| KS4 H8, final (3 seeds) | .16-.17 | .09-.10 | 1.01 |
| KS2.5 standard, final (3 reps) | .021-.025 | .020-.022 | ≈0 |
| KS2.5 standard_low, final (3 reps) | .026-.028 | .022-.024 | ≈0 |
| KS2.5 before its own curation | .10-.11 | .09 | ~0.1 |

- **MEC** shows the same picture, about 1.4× higher.
- **KS4's removal** takes 0.04% (H8) to 0.2% (MEC default) of spikes.
- **The KS4 excess:**
  - starts at 5-7 samples, peaks at 10-13 (0.33-0.43 ms) and reaches the floor by about 25-30 samples;
  - is spread over most units, but 100 units carry 75-98% of the net excess;
  - in KS4's top-amplitude third of units it is 50-83% of Hill-window violations.
- **On cells matched one-to-one across sorters, the same neurons give:**
  - in KS4, an excess equal to 11-26% of violations;
  - in KS2.5, none.
- **The dead time below about 0.2 ms is not KS4's 0.25 ms window.** It appears before KS4's removal and in
  KS2.5, which removes nothing. Template matching cannot resolve two similar spikes that close together.
- The agents disagree on how much of a bump KS2.5 has *before* its curation. The methods agent, labelling by
  template, finds a small one that `set_cutoff` mostly removes. The others find KS2.5's extracted stage
  close to flat. All agree it is far smaller than KS4's.

## 3. What the excess is made of

**Raw-waveform tests** were run by three agents with different designs:
- *methods:* joint refit of a1·W(t1) + a2·W(t2);
- *skeptic:* one-spike vs two-spike least-squares fit, with control pairs at 2-3 ms and phantom pairs;
- *threshold:* trough at the second time vs the same lag after isolated spikes.

| Agent and sort | Pairs at 0.3-0.5 ms judged two spikes | Controls: two real spikes / a phantom second spike |
|---|---|---|
| skeptic, KS4 H8 kobe | 89% | 88% / 21% |
| skeptic, KS4 default kobe | 84% | 89% / 17% |
| skeptic, KS4 H8 MEC | 86% | 85% / 24% |
| skeptic, KS2.5 kobe | 91% | 91% / 15% |
| methods, KS4 H8 kobe (a2 > 0.5) | 89% (median a2 1.10) | 87% / 19% |
| threshold, top-40 excess units | one-spike share of the excess 10-20% | false one-spike rate 12-18% |

- The mean peak-channel trace shows a second trough at exactly t1 + lag in every lag group, for both sorters.
- The second spike's spatial footprint matches the unit's template (cosine 0.91-0.92, against 0.94 for
  control pairs).
- **When KS2.5 resolves both spikes of a KS4 bump pair, it puts them in different units 88-91% of the
  time.** Those units are a median 18-26 µm apart.
- Other units within 100 µm are 36-107% more active around bump pairs than around control pairs.

**Explanations ruled out** (skeptic):
- an artefact of the analysis or of the removal step (simulated);
- harness artefact events (0.03% of bump spikes);
- propagation (the footprint is the unit's own);
- monosynaptic drive (wrong lag; that would peak at 1-3 ms).

What remains is near-synchronous firing of distinct, nearby neurons, with the second neuron's spikes mixed
into the KS4 unit. Whether that synchrony is physiological (gap junctions, tight local bursts) is
unresolved, and it doesn't change the answer.

**The true duplicates** *(mechanisms, threshold)*:
- **Same-template echoes.**
  - They sit at a sharp 9-16 samples, with nothing beyond about 20.
  - The later member is the weaker one in 97-99% of pairs, at a score ratio of 0.24-0.38.
  - They are concentrated in large, clean units with a single template: kobe default units 765 and 750,
    MEC default 822 and 779.
  - Half the spikes in those units' lowest-score fifth are echoes. KS4's window caught 3-13 of their
    800-1700 duplicate pairs.
  - Estimated from score-order asymmetry, P(dup | lag) ≈ 2f − 1, where f is the fraction of pairs whose
    later member scores lower. Per 1000 spikes this gives:

    | Sort | Duplicates per 1000 spikes |
    |---|---|
    | KS4 kobe default | 0.45 |
    | KS4 kobe H8 | 0.11 |
    | KS4 MEC default | 1.02 |
    | KS4 MEC H8 | 0.19 |
    | KS2.5 final | 0.03-0.08 |

    This is a lower bound, because it misses duplicates whose two scores are equal.
- **Lag 0-2 within-unit doublets** (MEC default, about 1 per 1000 spikes). These are produced when
  `merging_function` shifts a merged cluster's spike times by dt, which brings a cross-unit duplicate into
  exact coincidence. KS4's removal catches them, and this is its main useful job.
- **Cross-unit pairs at |lag| ≤ 0.2 ms** (skeptic, `raw_cross.py`). Among coincidences flagged by §6.3 at
  150 µm or less, about 45-50% are one spike reported by two units in KS4, against about 10% in KS2.5. This
  supports Phase 0's §8.10 explanation that a second template fits the first one's residual.
- **Batch edges** (methods). Removed pairs are 16-45× enriched within ±2 ms of the 60,000-sample batch
  boundaries, because batch i+1 never subtracts batch i's spikes lying in its padding. This is a real bug
  class, but at ≤ 0.01% of spikes.
- **Pairs KS4 removes now (lags 2-6):** roughly 20-40% are one spike, depending on the test. The rest are
  overlapping real spikes.
- **Hybrid units can't calibrate any of this** *(4)*. The inserted spikes are exact, scaled copies at integer
  samples, so the template fits them perfectly and they show essentially no echoes. They confirm only that a
  single detection jitters by ≤ 6 samples (KS4 puts 98.6% within −2 to +1).

## 4. Mechanisms: why KS4 makes more

- **No amplitude prior.**
  - KS4 scores a match by `relu(B)²/|U|²` with a free amplitude. `lam = 20` is defined at
    `template_matching.py:226` but never used, so any residual whose projection exceeds `Th_learned` is
    detected again, however small it is relative to the template.
  - KS2.5's cost, `(max(0,b)+λμ)²/(1+λ) − λμ²` with λ = 10 (`mexMPnu8.cu:288-333`), effectively needs a
    projection above about half the template's mean amplitude, so it rejects echoes at a third of that size.
  - The paper's fixed-norm cost (ks4_paper.txt ~1403-1407) would also have suppressed them; the code differs.
- **Low-rank templates.** Templates are limited to 6 temporal PCs and subtraction uses the least-squares
  amplitude, so a large spike leaves a structured residual. The echoes sit in the negative lobe of the
  template's autocorrelation in the clearest units, though that relationship doesn't hold up across units.
- **Peak picking can't produce these on its own.** Both sorters lock out other peaks within ±60/61 samples
  in each pass, across all templates. Close pairs therefore come from re-detecting the residual in a later
  pass: up to `max_peels` = 100 in KS4, and up to `nInnerIter` = 500 with an early break in KS2.5.
- **Reclustering turns cross-template pairs into within-unit pairs.** A final unit pools a median of 9
  extraction templates, so pairs detected by different templates end up in one unit.
- **Both sorters' correlogram tests are blind to these lags.** They zero the central bin, ignoring |Δt| under
  0.5 ms (`CCG.py:56-57`; KS2.5's `ccg.m`, `calcCcg`). So clusters that share near-synchronous spikes look
  refractory and get merged. This links the full-size bump to the grouping error in the placement audit.
- **The time shift on merge** brings some cross-unit duplicates to lag 0 (§3).

## 5. How duplicates are handled

| Handler | Within units | Across units | Which spike survives | Status |
|---|---|---|---|---|
| KS4 `remove_duplicates` (`postprocessing.py:9-29`, after merging) | lags ≤ 6 samples, since `dt = int(0.25×30) = 7`; window chained from the last kept spike | none | the earliest | [checked] |
| Lab KS2.5 pipeline | nothing explicit; the λ prior and per-pass lockout prevent many; `set_cutoff` removes most short-lag pairs | none | n/a | [checked] |
| bombcell `removeDuplicateSpikes` | ≤ ~1 sample, only templates sharing a max channel | same | the higher amplitude | [checked] off by default in the pipeline, and only changes the quality-metric inputs, never `spike_times.npy` |
| SpikeInterface `remove_duplicated_spikes` | censored period 0.3 ms | — | `keep_first` and other options | [checked, v0.103.2] |
| SpikeInterface `remove_redundant_units` | — | whole units, Δt 0.4 ms | `minimum_shift`, `highest_amplitude`, `max_spikes` | [checked] |
| Allen ecephys | within-unit window 0.166 ms | between-unit 0.166 ms, within 5 channels | the earlier spike within a unit | [from memory] |
| Harness §6.3 (`tuning/cleaning.py`) | none | 5-sample windows at lag L within ±1 ms, against flanks 10-30 samples away; significance test plus a 3% minimum | smaller unit (≤ 150 µm) or both units (> 150 µm) | [checked] |

**History.** KS4's within-unit step was added in Oct 2023 with a 15-bin (0.5 ms) default. That was lowered
to 7 bins in Apr 2024 and replaced by `duplicate_spike_ms` in Jun 2024. `docs/parameters.rst` warns
against going above 0.5 ms because KS4's own correlogram tests ignore the central 1 ms.

**Is KS4's method sound?**
- *Fine:*
  - Running after merging is right, because it catches the merge-shifted lag-0 pairs.
  - Chaining from the last kept spike changes removals by under 2%.
  - Keeping the earlier spike instead of the higher-scoring one changes the remaining short-lag pairs by
    under 0.3%. For echoes, the earlier member is almost always the real one anyway.
- *Not sound:*
  - It treats every close pair as one event, though most are two.
  - Its window misses the echoes.
  - It does nothing across units.
- *Removing the excess with a wider window* would delete real spikes of another neuron, and about half the
  time it would keep the wrong one (the foreign spike is the earlier one about half the time).

**The harness's §6.3 cleaning** removes almost nothing within units (by design). Across units:
- *Its flanks overlap the hump.* The flanks sit 0.33-1 ms from the tested lag, inside both the hump and the
  roughly 2 ms-wide synchrony peak that every sorter shows, so it catches only strong, sharp pairs.
- *Its "≤ 150 µm = same neuron detected twice" label is about half right for KS4 and mostly wrong for
  KS2.5.* About 90% of KS2.5's flagged coincidences are real synchronous spikes. §6.3 removes 2.0-2.9% of
  KS4 H8's spikes against 0.6-0.8% of KS2.5's.

## 6. Choosing the threshold

- **The three components are mixed at every lag.**
  - True duplicates are rare and lie within about 0.7 ms.
  - Real coincident spikes from mixed units are common and peak at 0.3-0.5 ms.
  - Genuine intervals of one neuron are absent below about 0.8-1 ms: the pooled refractory ramp's midpoint
    is 1.75-1.9 ms.
- **Formal lag-only rules give unstable, wide answers.** A three-component mixture with equal costs puts
  the threshold at 0.23-0.63 ms (95% intervals 0.23-1.1 ms). The window with 10% false discoveries is the
  current one. Even those "duplicates" are mostly two-spike events, which overstates true duplicates 5-10×.
- **Per-unit, empirical-Bayes windows** target better than one global window (55-63% duplicates among
  removed pairs, against 37-48%). But the units they select agree only about 50% between time halves.
- **What is principled:**
  - **For a lag-only within-unit rule:** set the window to a single detection's timing jitter, measured on
    hybrid spikes. That is ≤ 6 samples for KS4, which the current window already matches.
  - **For the echoes:** apply a physical bound (≤ ~0.7 ms), then a score-order test. Drop the later spike if
    its score is below 0.5 × the earlier one's. This flags 600-4,300 pairs per sort (0.005-0.05% of spikes),
    with a raw one-spike rate of 0.85-0.86 in default sorts and 0.61-0.70 in H8. It brings units 765 and 822
    from c = 1.00 to 0.13-0.14. A per-unit asymmetry test (2f − 1, binomial) flags units that produce echoes.
  - **For the contamination estimate:** the Hill censored period τ_c must be at least the removal window.
    Raising τ_c alone (to about 0.7 ms) fixes the echo units without touching any spike, at the cost of
    variance. It also censors the real synchronous-neighbour contamination, so report both values.
- **The costs are asymmetric across uses.**
  - **S:** a duplicate or a deleted real spike moves S about equally, and every candidate rule touches
    0.005-0.6% of spikes, far below δ.
  - **Spike timing:** a kept echo creates a fake 0.3-0.5 ms autocorrelogram peak and fake bursts, while a
    deleted real spike only thins the train.
  - **Hill estimate:** a kept duplicate pair inflates c by about p / (2(τ_r − τ_c)·r), for duplicate rate p
    per spike and firing rate r. With p = 10⁻³ that is +0.4 at 1 Hz and +0.04 at 10 Hz. The estimate
    saturates in the echo units.

## 7. Consequences for earlier conclusions

- **KS4 vs KS2.5 contamination.** Leaving the 0.25-0.5 ms band out of the Hill count lowers each sorter's
  misassigned total by 5-12%, and the KS4/KS2.5 ratio barely moves (kobe H8/KS2.5: 7.5 → 7.9; MEC
  default/KS2.5: 5.1 → 5.0). On matched cells the gap is small to begin with (2-6 points), and the bump
  accounts for 30-55% of it. But the bump is mostly real contamination from a neighbouring neuron, so it is a
  true quality difference, not a counting artefact. The headline "28-31% vs 7%" comes mostly from which units
  and spikes each sorter keeps.
- **`set_cutoff`.** The port zeroes the central bin, |lag| < 0.5 ms (`set_cutoff.py:52`), so it never sees
  the bump's peak; only the 0.5-1 ms tail enters its first counted window. The effect is modest. The echoes,
  however, can make a clean large unit fail the refractory test outright.
- **Placement audit.** §6's "try 0.5 ms" is withdrawn. Its §3 grouping-error finding is reinforced: the
  full-size bump is the same mixing of neighbouring neurons, made worse by correlogram tests that can't see
  |Δt| below 0.5 ms.
- **Harness.** §6.3 removes some real synchronous spikes (about half of KS4's flagged ones, about 90% of
  KS2.5's) and misses diffuse cross-unit excess. The effect on S is small, so this is a documentation issue
  rather than a rule change, and the harness is frozen.

## 8. Recommendations, ranked

1. **Keep `duplicate_spike_ms` at 0.25 ms.** It matches a single detection's jitter, and widening it deletes
   real spikes and hides contamination *(4 agents; the empirics agent proposed 0.5-0.75 ms before the raw
   tests were in)*.
2. **Add an echo rule post hoc: same unit, lag ≤ 0.7 ms, later score < 0.5 × earlier → drop the later spike.**
   Report the per-unit asymmetry flag. It is cheap, touches about 0.05% of spikes at most, and repairs the
   largest clean units' contamination estimate. Then test it against S, the guards and the hybrid data.
3. **Report Hill at τ_c = 0.25 ms and 0.7 ms in isolation analyses.** Read a unit whose c collapses
   (e.g. 1.00 → 0.00) as an echo unit, and a broad short-lag excess as contamination by a neighbouring neuron.
4. **Treat the full-size short-lag pairs as a mixing signal for clustering.** Correlogram tests that count
   0.25-0.5 ms coincidences between candidate clusters could block merges of near-synchronous neighbours.
   This feeds the placement audit's P4′ (refractoriness used for split/merge decisions).
5. **At the source, test an amplitude prior in matching** (KS2.5-style λ, the unused `lam`) or a joint
   re-fit of a close same-unit detection against the residual. This would prevent echoes rather than
   removing them, but it changes detection everywhere and needs sorts and kernel changes.
6. **Small fixes:**
   - batch-edge double detection in `extract`;
   - save the peel index and per-spike residual alongside the scores;
   - a cross-unit duplicate step aimed at |lag| ≤ 0.2 ms with an amplitude or raw check;
   - document §6.3's mixed "same neuron" class;
   - hybrid inserts with real waveform variability and sub-sample jitter, so echoes can be measured.

## 9. Open points

- Whether the near-synchrony is physiological. The 25-55 µV, 20-70 Hz units that carry most of it look like
  multi-unit activity.
- How many duplicates have equal scores. The asymmetry estimate misses them, and the lag 2-6 raw tests
  differ between agents (one-spike share 20-40%).
- The KS2.5 pre-curation bump (see §2).
- None of the recommendations has been run through the harness.
