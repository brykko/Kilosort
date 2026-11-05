# Interleaved template learning: empirics on the saved sorts

Lens: what the existing sorts say about the proposal (extract -> cluster -> rebuild, interleaved in small batches, for
faster convergence on more accurate templates). No new sorts, no hybrid data, CPU only. Scripts:
`tuning\exploratory\interleaving\empirics\`; outputs: `C:\data\ks4_tuning\exploratory\interleaving\empirics\` (the
snippet passes' `*_res.npy` reservoirs are 1.1 GB each and can go). Seeds: H8 and default s1-3, 20-min kobe and MEC
excerpts; every raw-data measurement is seed 1. Random draws use seed 0.

## Bottom line

- **"Converges on more accurate templates" is not what the data show.** For the same cell, successive rounds' templates are
  almost identical (median cosine 0.9993-0.9996) and a round's step is about the size of the template's own sampling
  error and of the difference between two seeds. The step does shrink (about 0.7 per round) and it isn't noise, but
  seed-to-seed distance for fixed cells does not fall from round 2 to 4. What moves is labels and unit identity.
- **Labels are not converged.** 10-16% of shared spikes still change unit from round 3 to 4, close to the 13-18% between two
  seeds at round 4. Default: 6-8% between rounds, 5-7% between seeds. That is near the floor of re-clustering noise.
- **Small batches can't see a round's step.** A correctly labelled template reaches cosine 0.95 in 3-4 batches and 0.99 in
  17-20 for the median-rate unit (10-13 and 53-67 at the 10th-percentile rate), but sampling noise only falls to the size
  of a round's step after about 260-290 batches for a matched unit (half the excerpt), 80-100 over all units.
- **The premise is half right.** 54% of final spikes have no universal-pass detection, but no unit is invisible
  (at most 0.1% of units are >=95% unseen, 1-2% have under 100 seen spikes), and about half of the misses of spikes that should
  have cleared the threshold sit next to another detection. Poor universal fit lowers a unit's seen fraction but doesn't
  predict whether it owns a template. Units missed altogether can't be measured here.
- **Little sign of large missed waveforms.** After matching, 100-118 isolated events per second per mm of probe still cross Th_universal in the residual (3.6-4.4% of
  found spikes, 6-22 times a phase-randomised null), all near threshold (99th percentile 8.7-8.9 against 30-38 in the raw batch), and round 4's dictionary leaves as many as round 2's.

## Selection rules (written before running), and their bias

- **Eligible** = keys of the harness's `S_i` (spike counts, waveform screen). No amplitude, rate or quality cut beyond
  that. Spikes are cleaned spikes (`cleaning.py`'s steps) unless stated.
- **All-eligible chain**: every eligible unit of the middle round, followed to the unit of the neighbouring round
  (eligible or not) holding most of its spikes. Nothing dropped, no threshold. Bias: towards more change (a split unit's
  successor is another template).
- **Mutual chain**: mutual best at agreement >= 0.5 among eligible units on both links, the same middle unit. Bias: towards
  less change (survivors). The truth for a typical unit lies between; both are reported, same rule for both rounds.
- **Strata**: terciles of rate and of amplitude over all eligible units of that sort; 9 cells, nothing excluded.
- **Same-event link**: nearest spike within 5 samples and 40 um; time alone is useless at 10 spikes per ms across the
  probe. Chance is measured by shifting one side 1000 samples (8% of spikes) and corrected: (s - c) / (1 - c).
- **Sample complexity**: all eligible units of the final round. Measured curves need N >= 2n (biased to high rate; unit
  counts given), the closed form covers all.

## 1. Do templates converge?

Pooled over seeds 1-3. |dT|/|T| is normalised L2 at lag 0, full templates from `Wall.npy` (`m1_convergence.py`).
Quantiles are .1 / .5 / .9. Contraction is |T4 - T3| / |T3 - T2| per unit, median (.1-.9), and sum-of-squares ratio.

| H8 | chain | units with both links | cosine r2-r3 (median) | cosine r3-r4 | \|dT\|/\|T\| r2-r3 | \|dT\|/\|T\| r3-r4 | contraction | sum ratio | persistence cosine |
|---|---|---|---|---|---|---|---|---|---|
| kobe | all | 4994 | 0.9978 | 0.9990 | .014 / .073 / .421 | .009 / .048 / .347 | 0.71 (0.25-1.98) | 0.90 | -.71 / .21 / .83 |
| kobe | mutual | 2928 | 0.9994 | 0.9996 | .010 / .037 / .130 | .007 / .031 / .093 | 0.70 (0.28-1.41) | 0.77 | -.59 / .33 / .86 |
| MEC | all | 5332 | 0.9960 | 0.9983 | .018 / .097 / .454 | .013 / .063 / .376 | 0.71 (0.25-1.96) | 0.75 | -.75 / .11 / .78 |
| MEC | mutual | 2590 | 0.9993 | 0.9995 | .013 / .041 / .154 | .009 / .034 / .094 | 0.69 (0.27-1.43) | 0.84 | -.63 / .23 / .82 |

Amplitude barely moves: |T(r+1)|/|T(r)| median 0.996-1.000 (.1-.9: 0.93-1.04 all, 0.98-1.02 mutual). Persistence is the
cosine between the steps T3 - T2 and T4 - T3: -0.5 for independent jitter around a fixed template, +1 for a straight line.
It is positive at the median, so the motion is partly systematic, but wide. In the all-eligible chain, 14% (kobe) and 18%
(MEC) of units have cosine under 0.95 with their round-2 predecessor, 11% and 13% with their round-4 successor (seed 1):
these are identity changes, and they are what makes the all-eligible tail.

By stratum (mutual chain, kobe; median contraction, |T3-T2|/|T|, |T4-T3|/|T|, units; MEC is the same pattern):

| | low amp | mid amp | high amp |
|---|---|---|---|
| low rate | 0.70, .067, .043 (88) | 0.78, .054, .043 (133) | 0.74, .019, .015 (586) |
| mid rate | 0.60, .065, .040 (147) | 0.71, .050, .034 (323) | 0.72, .019, .014 (557) |
| high rate | 0.57, .061, .034 (368) | 0.68, .044, .029 (425) | 0.71, .018, .012 (301) |

Contraction hardly depends on stratum; the step depends on amplitude (2% against 6%), not on rate. Note how few low-rate,
low-amplitude units survive the mutual rule (88 of 487): those are the units the rule hides. In the all-eligible chain their
step is 22%.

**Noise floors**, seed 1, each round's template re-estimated identically as the mean batch feature over its spikes, on
the channels shared by the two units' 32-channel sets (`m1_raw.py`). Medians, as a fraction of |T|, with .1-.9.

| | step | swap null | own sampling noise | seed s1-s2, same cell |
|---|---|---|---|---|
| kobe H8, r2-r3, mutual (1015) | .046 (.011-.125) | .014 | .030 | r2: .039 (.008-.130) |
| kobe H8, r3-r4, mutual (1198) | .040 (.008-.108) | .014 | .032 | r4: .042 (.007-.168) |
| kobe H8, r2-r3, all (1644) | .086 (.015-.415) | .025 | .036 | |
| kobe H8, r3-r4, all (1644) | .060 (.011-.346) | .021 | .036 | |
| MEC H8, r2-r3, mutual (909) | .051 (.015-.146) | .015 | .031 | r2: .050 (.013-.154) |
| MEC H8, r3-r4, mutual (1089) | .041 (.011-.108) | .016 | .033 | r4: .052 (.012-.166) |
| MEC H8, r2-r3, all (1759) | .118 (.021-.445) | .032 | .040 | |
| MEC H8, r3-r4, all (1759) | .081 (.015-.391) | .028 | .040 | |
| kobe default, r2-r3, mutual (657) | .035 (.006-.214) | .009 | .024 | r3: .019 (.003-.172) |
| MEC default, r2-r3, mutual (607) | .038 (.009-.177) | .010 | .024 | r3: .028 (.005-.173) |
| kobe default, r2-r3, all (764) | .045 (.007-.365) | .010 | .026 | |
| MEC default, r2-r3, all (743) | .055 (.010-.384) | .012 | .026 | |

- *Sampling noise* is sqrt(s2/n)/|T|. The split-half estimate agrees with it to within 1% at every quantile, so the
  scatter is right.
- *Swap null* is the step expected if the spikes unique to each round were random draws of the unit:
  s2 (1/na + 1/nb - 2m/(na nb)), with m shared. Rounds share most spikes, so it is well below the sampling noise. The
  step exceeds twice the null (in square) in 89-97% of units, so **the step is real: reassigned spikes aren't a random sample
  of the unit.** Partly that may be selection: spikes near a boundary are unlike the unit's average spike whichever way they go.
- But in size the step is 1.2-1.5 times the template's own sampling error for matched units, 1.6-2.8 for all units (medians
  of step/noise: mutual 1.43, 1.18 kobe; 1.52, 1.21 MEC; all 2.40, 1.56; 2.76, 1.89), and about equal to the
  seed-to-seed difference.

**Fixed cells across seeds.** Cells linked r2<->r4 in both seeds and across seeds at both rounds (`m1_seedfixed`, pairs
s1-s2, s2-s3, s3-s1; a survivors' set, same units at both rounds):

| | pairs | r2 \|dT\|/\|T\| | r4 \|dT\|/\|T\| | r2 cosine (lag) | r4 cosine (lag) | pairs closer at r4 |
|---|---|---|---|---|---|---|
| kobe H8 | 2484 | .006 / .0248 / .117 | .005 / .0247 / .123 | .9998 | .9997 | 56.5% |
| MEC H8 | 2147 | .008 / .0307 / .132 | .008 / .0313 / .144 | .9996 | .9996 | 53.4% |

Over each round's own mutual pairs the seed distance rises slightly, because new, less stable pairs join: kobe .0303 ->
.0328 -> .0345, MEC .0380 -> .0400 -> .0427 (pairs 3002 -> 3127, 2701 -> 2771).

**So:** the claim isn't supported as a claim about the templates of matched cells. Two independent runs don't approach
each other, and a cell's template changes by no more than it would between seeds. Seed agreement is not truth (a shared bias
would stay), and there is no ground truth here, so "more accurate" can't be excluded, only that nothing measurable shows it.
The rounds' documented gains, fewer misassigned spikes in 36-41% of pairs and rising reproducibility, are claims about
clusters: labels and which units exist.

**Assignments.** Spikes of eligible units; ranges over seeds (forward link; the backward link is within 0.02).

| | not shared, corrected | relabelled, of shared | seed s1-s2 at last round: not shared, relabelled |
|---|---|---|---|
| kobe H8 r2->r3 | 6.2-6.5% | 14.2-14.5% | at r2: 5.9%, 15.6% |
| kobe H8 r3->r4 | 3.9-4.1% | 10.4-11.0% | at r4: 5.4%, 13.4% |
| MEC H8 r2->r3 | 7.5-7.7% | 18.1-18.8% | at r2: 7.6%, 21.3% |
| MEC H8 r3->r4 | 4.9-5.2% | 14.3-14.5% | at r4: 6.8%, 18.4% |
| kobe default r2->r3 | 3.5-3.6% | 5.8-6.3% | at r2: 2.2%, 5.7%; r3: 2.8%, 5.3% |
| MEC default r2->r3 | 4.7-4.8% | 6.7-7.1% | at r2: 3.9%, 6.7%; r3: 4.0%, 7.1% |

Relabelled: the spike's unit in the next round isn't the successor of its unit. Contraction per round in these counts is
about 0.65 (not shared) and 0.76 (relabelled). The between-round relabelling is close to the between-seed one, and at default
it equals it, so a large part may be the noise of re-clustering rather than progress. Seeds differ from the first clustering
on, so this floor is an upper bound.

## 2. Does the change matter? (descriptive)

Mutual r2<->r3 pairs, seeds 1-3 (`summary.py m2`, `m2_assign.py`; 900-1040 pairs per sort, H8). dS is S_i(r3) - S_i(r2).

| | Spearman dS ~ \|dT\| | partial on rate, amp, S_i | dS ~ label change | partial | \|dS\| ~ label change | label change ~ \|dT\| |
|---|---|---|---|---|---|---|
| kobe H8 | .05, .05, .07 | .12, .07, .09 | .06, .08, .10 | .12, .11, .12 | .39, .38, .34 | .76, .75, .75 |
| MEC H8 | .09, .03, .05 | .07, .06, .02 | .10, -.01, .09 | .07, .02, .07 | .27, .27, .26 | .72, .73, .73 |
| kobe default | .02, .03, .02 | .00, -.03, -.02 | not computed | | | |
| MEC default | .07, .10, .11 | .00, .03, .04 | not computed | | | |

Summed dS by quartile of label change (lowest to highest): kobe s1 2.6, 3.6, 4.8, 12.6; s2 2.2, 3.5, 6.1, 9.8; s3 1.3, 3.5, 4.7, 6.9;
MEC s1 3.7, 12.1, 9.1, 24.1; s2 0.6, 19.3, 15.8, 2.0; s3 4.2, 4.9, 13.5, 14.8.

Template change follows label change (rank correlation 0.72-0.76). Units that change more change more in S in either
direction (|dS|), but the direction is almost unrelated (0.0-0.12), though the sum leans to the more-changed quartiles at most
seeds. Limits: survivors only, which leaves out the unmatched units' roughly one third of the gain; S_i per unit is noisy; a unit's S also
changes with what its neighbours lose; no causal reading.

## 3. Sample complexity

Batch features (`snippets.py`): 6 PCs on the 10 channels nearest the peak, without subtraction of overlaps. All eligible units,
final round (H8 r4, default r3), seed 1 (`m34_samples.py`).

Measured median cosine to the unit's held-out mean (20 draws per unit; .1-.9 in brackets; units with N >= 2n):

| n | kobe H8 | MEC H8 | kobe default | MEC default |
|---|---|---|---|---|
| 10 | .924 (.877-.980) | .916 (.868-.978) | .947 (.896-.986) | .943 (.888-.985) |
| 30 | .972 (.952-.993) | .969 (.947-.993) | .981 (.959-.995) | .979 (.956-.995) |
| 100 | .991 (.984-.998) | .990 (.982-.998) | .994 (.987-.998) | .993 (.986-.998) |
| 300 | .997 (.994-.999), 1530 units | .996 (.993-.999), 1595 | .998 (.995-.999), 744 | .998 (.995-.999), 691 |
| 1000 | .999 (.998-1.000), 1181 units | .999 (.998-1.000), 1164 | .999 (.998-1.000), 609 | .999 (.998-1.000), 566 |

The closed form, cos = 1/sqrt(1 + s2/(n|T|^2)), reproduces the measured median at n = 100 to three decimals in all four, so
n*(c) = (s2/|T|^2) / (1/c^2 - 1) is used for all units. Scatter over signal, s2/|T|^2: .4 / 1.9 / 3.3 (kobe H8), .5 / 2.1 / 3.6 (MEC H8),
.3 / 1.3 / 2.8, .3 / 1.4 / 3.1 (default).

| | rate (Hz) | n*(.95) | batches to .95 | real train, median start / 90th pct start | n*(.99) | batches to .99 | real train |
|---|---|---|---|---|---|---|---|
| kobe H8, median-rate units (324) | 2.82 | 19 | 3.2 | 4 / 7 | 100 | 16.9 | 18 / 24 |
| kobe H8, 10th-pct-rate units (162) | 0.79 | 15 | 10.0 | 12 / 18 | 81 | 53 | 54 / 70 |
| MEC H8, median-rate (348) | 2.63 | 20 | 3.8 | 5 / 7 | 108 | 20.1 | 21 / 27 |
| MEC H8, 10th-pct-rate (174) | 0.56 | 13 | 12.7 | 16 / 23 | 70 | 67 | 68 / 89 |
| kobe default, median-rate (154) | 4.06 | 11 | 1.3 | 2 / 5 | 57 | 7.1 | 8 / 13 |
| kobe default, 10th-pct-rate (77) | 1.01 | 8 | 3.9 | 6 / 13 | 43 | 20.9 | 23 / 38 |
| MEC default, median-rate (145) | 4.07 | 11 | 1.3 | 2 / 5 | 61 | 7.1 | 8 / 14 |
| MEC default, 10th-pct-rate (72) | 0.85 | 6 | 3.7 | 6 / 11 | 32 | 19.8 | 22 / 35 |

(Median-rate: 40-60th percentile of rate; 10th-percentile rate: 5-15th. Real train: consecutive 2-s batches from 20 random starts to
collect n*, medians over units.) Median batches to .95 / .99 by stratum, H8:

| | kobe low amp | mid | high | MEC low amp | mid | high |
|---|---|---|---|---|---|---|
| low rate | 11.2 / 60 | 9.8 / 52 | 2.9 / 15 | 17.4 / 93 | 13.2 / 70 | 3.8 / 21 |
| mid rate | 4.3 / 23 | 3.6 / 19 | 1.2 / 7 | 5.4 / 29 | 3.9 / 21 | 1.2 / 7 |
| high rate | 1.4 / 8 | 1.1 / 6 | 0.2 / 1 | 2.0 / 11 | 1.4 / 8 | 0.4 / 2 |

**KS4's own features** (`m3_tf_check.py`; round-2 tF from the stage cache, overlaps subtracted, each unit's most common channel set,
99.6% of its spikes): s2/|T|^2 is 0.59-0.61 of the batch features' (kobe 1.23 against 1.93, MEC 1.33 against 2.15). So
the n* above are about 1.6 times too high for a learner with subtraction, which an online learner has only as far as its templates
are already good.

**What cosine is the right target?** 0.95 and 0.99 are loose. Nearest-neighbour template cosine in H8 is .61 / .83 / .95 (kobe), .66 /
.86 / .96 (MEC), so telling a unit from its neighbour takes about 21-24 correctly labelled spikes (n = 8 s2/|Ta - Tb|^2; 3.5-4.6
batches; 10-13 at the low-rate tercile; `m3_neighbour.py`), which is cheap. But a round's step is at cosine 0.999:

| spikes for sampling noise to equal a round's step, n s2 / \|dT\|^2 | r2-r3 mutual | r3-r4 mutual | r2-r3 all | r3-r4 all |
|---|---|---|---|---|
| kobe H8, spikes (.1 / .5 / .9) | 261 / 2045 / 13455 | 470 / 3035 / 26919 | 33 / 826 / 8368 | 50 / 1572 / 18470 |
| MEC H8, spikes | 142 / 1606 / 10154 | 378 / 2428 / 18334 | 25 / 424 / 5424 | 45 / 920 / 11145 |
| as a share of the unit's own spikes, median | .49, .43 | .71, .68 | .17, .13 | .41, .28 |
| 2-s batches, r2-r3, median unit (low, mid, high rate) | kobe 292 (899, 369, 102); MEC 259 (523, 316, 132) | | kobe 104 (143, 163, 62); MEC 79 (103, 80, 60) | |

Against 600 batches in the excerpt. A loop of tens of batches moves a matched cell's template by sampling noise, not by a round's
worth of real change. Units that truly change identity (the 10th percentile: 25-50 spikes) are visible within a few batches.

**Noisy labels.** All the above assumes correct labels; labels add a bias no n removes (`m3_floor.py`). All eligible round-4 units, the
round-2 unit holding most of their spikes, as a stand-in for an online learner's labels. This is generous: round 2 has had a full-data
clustering. Cosine of the two full-data means, on shared channels:

| | plateau .1 / .5 / .9 | units under .99 | under .95 | spikes at which noise = bias, median | by rate tercile, spikes (batches) |
|---|---|---|---|---|---|
| kobe H8 | .880 / .994 / .9998 | 43% | 19% | 421 | 164 (84), 511 (85), 632 (32) |
| MEC H8 | .851 / .989 / .9997 | 52% | 25% | 240 | 83 (53), 275 (58), 466 (28) |

So cosine 0.99 is unreachable for 43-52% of units with round-2-quality labels at any n. With worse early labels, it is worse: the
first clustering's template has cosine .73-.76 / .96-.98 / .996 with its final unit's (10th, 50th, 90th percentiles; section 5). A learner
that also labels with its own noisy templates has a feedback the numbers here don't include. In practice n* is the optimistic bound;
the floor is what will bind beyond about 100 batches for most units.

## 4. Non-stationarity

First against last epoch; control: random halves of the two epochs' union, same spike numbers (`m34_samples.py`). 'Debiased' is shape
similarity with sampling noise removed. All eligible units had at least 30 spikes in both epochs at 20 min.

| | units | cosine | control | debiased cosine | control | amp last/first | \|F-L\|/\|F\| | control | \|F-L\| > 1.5 x control |
|---|---|---|---|---|---|---|---|---|---|
| kobe H8, 5 min each end | 1618 | .990 / .997 / .999 | .992 / .998 / 1.000 | .997 / .999 / 1.000 | .999 / 1.000 / 1.001 | .962 / .990 / 1.017 | .053 / .081 / .148 | .025 / .059 / .125 | 36% |
| MEC H8 | 1742 | .977 / .993 / .997 | .989 / .998 / 1.000 | .986 / .996 / .999 | .999 / 1.000 / 1.002 | .947 / 1.001 / 1.044 | .084 / .127 / .230 | .028 / .066 / .152 | 67% |
| kobe default | 770 | .993 / .998 / .999 | .996 / .999 / 1.000 | .996 / .999 / 1.000 | 1.000 | .963 / .993 / 1.019 | .046 / .073 / .126 | .018 / .042 / .087 | 58% |
| MEC default | 725 | .980 / .994 / .997 | .996 / .999 / 1.000 | .984 / .995 / .998 | 1.000 | .941 / .996 / 1.040 | .079 / .119 / .219 | .020 / .042 / .090 | 87% |
| kobe 2 h, round 4, 20 min each end | 1750 | .970 / .989 / .995 | .998 / .999 / 1.000 | .972 / .990 / .996 | 1.000 | .880 / .952 / 1.005 | .104 / .157 / .261 | .014 / .032 / .071 | 98% |
| MEC 2 h, round 4 | 1908 of 1916 | .968 / .989 / .996 | .997 / .999 / 1.000 | .970 / .991 / .997 | 1.000 | .869 / .982 / 1.041 | .099 / .158 / .279 | .015 / .036 / .079 | 96% |

(2-hour units: those eligible in the sort's 20-min scoring window; nearly all are measurable. Seed 1. The 20-min epochs are 100 minutes apart.)

Waveforms change over 20 minutes by about as much as a round's step (kobe roughly 5.5% after removing noise, MEC about 11%, from the medians): real, small, larger on
MEC. Over 2 hours it is 15-16%, with kobe's amplitude falling 5% at the median (12% at the 10th percentile), so a template from the first 20 minutes has cosine 0.99 with the last 20. A rolling online template could follow that where one global template can't. It is also a reason why a window's template differs from a
full-data one, so small-batch estimates have a further floor.

**Clustering in time.** Fraction of 2-s batches with at least 10 spikes, all eligible units, against Poisson at the unit's rate:

| | >= 1 spike | >= 3 | >= 10 | Poisson >= 10 | >= 10 by rate tercile (real / Poisson) | batches holding half the spikes |
|---|---|---|---|---|---|---|
| kobe H8 | .47 / .92 / 1.00 | .20 / .64 / 1.00 | .003 / .18 / .98 | 0 / .06 / 1.00 | .02 / 0; .18 / .06; .79 / .96 | .09 / .21 / .38 |
| MEC H8 | .45 / .92 / 1.00 | .13 / .63 / 1.00 | 0 / .15 / .86 | 0 / .04 / .99 | .005 / 0; .15 / .04; .64 / .82 | .09 / .22 / .36 |
| kobe default | .44 / .92 / 1.00 | .24 / .70 / 1.00 | .03 / .31 / 1.00 | 0 / .30 / 1.00 | .08 / 0; .31 / .30; .95 / 1.00 | .08 / .20 / .38 |
| MEC default | .47 / .94 / 1.00 | .22 / .75 / 1.00 | .01 / .31 / .99 | 0 / .30 / 1.00 | .05 / 0; .31 / .30; .95 / 1.00 | .09 / .22 / .37 |
| kobe 2 h (all spikes) | .45 / .88 / 1.00 | | .004 / .16 / .86 | 0 / .04 / 1.00 | | |
| MEC 2 h | .45 / .90 / 1.00 | | .002 / .13 / .84 | 0 / .03 / .98 | | |

Half the spikes of the median unit sit in about a fifth of its batches, so a unit is bursty on 2-s scales. Most batches hold fewer than 10 spikes of a
given unit: only 15-18% (H8) for the median unit, and for the slowest tercile nearly none. Over a run of many batches the real-train numbers in section 3
are close to n*/(2r) at the median start and 1.3-2.2 times it at the 90th-percentile start, so clustering in time delays things, but not by an order of magnitude.

## 5. The premise: do waveforms that fit the universal templates poorly go without a template?

`m5_universal.py`, on st0 from the stage cache (H8; default has no cached st0, so none of this is measured for default). Universal-pass times
sit 10 samples after KS4's spike times (the conv1d's offset; `lag_hist` peaks sharply there, 65% of pairs within 3 samples). Seen: a
detection within 10 samples and 50 um; chance from a 1000-sample shift (8.8%). Tolerances (5, 40) and (20, 60) give 0.44 and 0.51 after correction
(kobe), against 0.46 at (10, 50). Seeds 1-3 pooled, all eligible units.

| | kobe r2 | kobe r4 | MEC r2 | MEC r4 |
|---|---|---|---|---|
| universal detections | 5.32 M | | 4.89 M | |
| final spikes seen, spike-weighted, corrected | 46.4% | 46.1% | 45.8% | 46.2% |
| per unit f_seen .1 / .5 / .9 | .20 / .41 / .87 | .20 / .41 / .87 | .22 / .39 / .86 | .21 / .40 / .87 |
| units >= 50% unseen (their share of spikes) | 62% (62%) | 61% (62%) | 62% (64%) | 61% (62%) |
| units >= 80% unseen | 9.6% (12%) | 10.8% (13%) | 6.5% (7%) | 7.7% (9%) |
| units >= 95% unseen | 0.00% | 0.00% | 0.00% | 0.1% |
| units with under 100 seen spikes | 1.6% | 0.9% | 1.3% | 1.0% |
| fit rho .01 / .1 / .5 / .9 / .99 | .65 / .75 / .86 / .92 / .96 | .64 / .74 / .85 / .92 / .95 | .64 / .75 / .86 / .92 / .96 | .62 / .73 / .85 / .92 / .96 |
| units whose template's universal response is under Th_universal | 16% | 18% | 13% | 15% |
| own a template | 87% | 91% | 83% | 86% |
| their spikes in the best first-round cluster | .19 / .44 / .79 | .18 / .43 / .79 | .18 / .43 / .77 | .18 / .41 / .77 |
| unit's share of that cluster | .05 / .65 / .87 | .06 / .64 / .87 | .03 / .59 / .84 | .04 / .57 / .84 |
| cosine (first-round cluster's template, final) | .76 / .98 / 1.00 | .74 / .97 / 1.00 | .73 / .98 / 1.00 | .68 / .96 / 1.00 |

Rho is the unit's own template's universal response over its norm: a spike clears Th_universal only if it is 1/rho times larger than it needs to
clear the same Th_learned. So median 1.18 times, 1.36 at the 10th percentile, 1.6 at the 1st. Those are real penalties, in a pass that also
doesn't subtract.

Seen fraction against learned score (seed 1, corrected):

| score | 7-8 | 8-9 | 9-10 | 10-12 | 12-15 | 15-20 | 20-30 | >= 30 |
|---|---|---|---|---|---|---|---|---|
| kobe r4 | .17 | .28 | .37 | .46 | .58 | .71 | .88 | .97 |
| MEC r4 | .19 | .28 | .34 | .42 | .58 | .73 | .88 | .97 |

Why spikes are unseen (`m5_unseen.py`, r4, seed 1): 59% (kobe) and 64% (MEC) of the unseen would be predicted to clear the threshold (score x rho >= 7);
of those, 73% and 72% have a detection within the pass's suppression window (20 samples, 150 um), against 42% by chance, so about half of them are
masked by a neighbouring event: no subtraction in that pass, rather than a poor fit. The other 36-41% are simply small.

Does poor fit decide the outcome? Within template-norm terciles (so amplitude is held; rho terciles, r4 pooled):

| \|T\| tercile | kobe f_seen at rho ~.75, .83, .90 | own template | Spearman f~rho, own~rho | MEC f_seen | own template | Spearman |
|---|---|---|---|---|---|---|
| low | .18, .26, .33 | .90, .92, .93 | .67, .03 | .22, .27, .33 | .84, .87, .91 | .55, .08 |
| mid | .31, .36, .43 | .92, .92, .93 | .35, .03 | .32, .36, .43 | .89, .87, .88 | .35, -.02 |
| high | .65, .72, .87 | .88, .87, .89 | .40, -.03 | .71, .77, .81 | .84, .85, .80 | .28, -.05 |

Rho does what the mechanism says: it lowers a unit's seen fraction at fixed amplitude. But it says nothing about whether the unit
gets a template (|Spearman| <= 0.11 at r2 and r4). Orphans have the same rho as owners (.844 against .846; MEC .846, .846 at r4).

Do extra rounds rescue poorly fit units? (`m5_turnover.py`; all eligible units, same rule on both sides, means over seeds of tercile medians of rho.)

| \|T\| tercile: partnered vs not | kobe r4 (gained) | kobe r2 (lost) | kobe r2 vs other seed (control) | MEC r4 (gained) | MEC r2 (lost) | MEC control |
|---|---|---|---|---|---|---|
| low | .851 / .817 | .868 / .854 | .862 / .855 | .871 / .827 | .890 / .857 | .889 / .856 |
| mid | .850 / .839 | .854 / .831 | .852 / .835 | .855 / .835 | .864 / .839 | .856 / .840 |
| high | .865 / .835 | .868 / .816 | .866 / .825 | .867 / .839 | .868 / .826 | .867 / .833 |

Unstable units have lower rho whichever way they go, including between seeds, so low rho goes with instability rather than with rescue. Only among the
smallest templates are gained units below lost ones, by .03-.04 at both excerpts; rho's low tail also falls slightly from r2 to r4 (10th percentile .75 to .74).
That's the most one can read as rescue, and it's small.

**Limit.** All of this conditions on units that exist in the output. A cell that never got a template doesn't appear, or appears only as spikes in a neighbour.
The direct test would be a run. Either (a) rerun the universal pass with Th_universal lowered (say 5, 6) and everything else as H8, with `save_extra_vars`, then ask
whether new final units appear with low rho; or (b) seed the dictionary with templates learned elsewhere (KS2.5's units or a previous sort's round 4) in place of round
1's, and count eligible units whose rho is low and that don't otherwise appear. Section 6 is a partial proxy.

## 6. Residual energy

`m6_residual.py`: five evenly spaced batches (10 s; batches 100-500), both excerpts, seed 1, H8. The extraction dictionary isn't saved, so matching was rerun on the CPU with the saved final
units' aligned templates (no first-pass merge): close to one more extraction, not the same. A 1000 um band around the median unit depth (templates within 150 um of it). KS4's universal detector then ran
on the batch (it reproduces st0's count exactly, every batch), on the residual, and on a null: the residual with the same random phases on all channels. 'Isolated': no found spike within 10 samples and
50 um, and spike-signed. An isolated event fails Th_learned on every template and passes Th_universal on a generic shape.

| events/s in the band | found spikes | universal on batch | on residual | spike-signed | near a found spike (chance) | isolated | null isolated | excess, share of found | isolated amp .5 / .9 / .99 |
|---|---|---|---|---|---|---|---|---|---|
| kobe, r2 dictionary | 3161 | 1116 | 283 | 199 | 122 (68) | 120.6 | 5.8 | 115, 3.6% | 7.34 / 8.01 / 8.78 |
| kobe, r4 dictionary | 3175 | 1100 | 282 | 197 | 118 (75) | 123.0 | 5.5 | 118, 3.7% | 7.35 / 8.01 / 8.70 |
| MEC, r2 dictionary | 2317 | 919 | 328 | 173 | 112 (67) | 120.4 | 18.2 | 102, 4.4% | 7.37 / 8.05 / 8.86 |
| MEC, r4 dictionary | 2374 | 919 | 324 | 170 | 109 (71) | 116.3 | 15.9 | 100, 4.2% | 7.38 / 8.17 / 8.87 |

(The r4 kobe band is 40 um deeper than r2's because the median unit depth moved; MEC's band is the same.) Residual variance falls from 0.96 to 0.80 (kobe), 0.95 to 0.82 (MEC). The batch's own events reach
30-38 at the 99th percentile, the residual's 9.3-9.5, and the threshold is 7.

- A rough count, not a census: about 100-118 isolated events per second per mm. That is 3.6-4.4% of the found spikes, 6-22 times the null, spread evenly over depth (10-bin counts 64-202).
- They are all near threshold, so no large unexplained waveform remains in these 10 s. If every one were a real spike they would be about 40 units' worth per mm at the median rate, some 8-9% of the units there; that's an upper
  bound, since some are mismatch of templates to spikes, and a phase-randomised null only keeps second-order statistics.
- **Round 4's dictionary leaves as many as round 2's**, and finds about as many spikes: this short stretch shows no reduction in unexplained events from more rounds, though these aren't the true extraction dictionaries.
- Ten seconds can't say whether the events cluster into cells. That takes a run: extraction with the final dictionary, saving the residual's universal detections and features across the whole excerpt, then the first clustering on those
  alone. On the GPU it costs about one extra extraction and clustering. That would be the direct test of the proposal's premise for waveforms the dictionary lacks.

## Caveats

- No ground truth: 'accurate' can only be proxied. Seed agreement misses shared bias.
- Raw measurements use one seed's preprocessing, batch features, with no overlap subtraction; `Wall` differences are lag 0 on 32 channels, and at most 5 samples of lag
  change nothing (cosine at lag equals cosine at lag 0 at the median).
- Spikes: cleaned, but mean-template scatter includes amplitude variability and overlaps; that is what a learner would face.
- Round 4 is the last available; nothing here says where the steps would stop.
- Saved H8 sorts started from H8's 2400-template dictionary. Default's numbers are one round.
- The 2-hour sorts' `Sort` cleaning applies the 20-min excerpt's artefact mask to 2-hour times, which removes a few spikes at the wrong time; harmless for means.
- Memory: I overshot the 15 GB limit in committed memory at times. The CPU residual run committed about 15 GB of private memory (4.5 GB resident, even after the depth-band cut) and each snippet pass about 11 GB (5 GB resident). With other
  agents' jobs running, free memory fell to about 1.1-1.6 GB twice. My own residual run was stopped each time and restarted later (a `Stop-Process` on it was refused and not retried; it was stopped through the task tool). m5 was
  also stopped once, for thread contention, and resumed.
