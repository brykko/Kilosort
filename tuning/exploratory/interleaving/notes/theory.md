# Interleaving extraction, clustering and template refinement: mathematics and theory (2026-11-04)

Lens: what the mathematics says about replacing KS4's whole-recording rounds with small-batch interleaving, treated as a
hypothesis. Scripts are in `..\theory\`: `contraction.py` (closed forms, checked by Monte Carlo), `empirics.py` (numbers
from existing sorts and KS2.5 logs), `toy_online.py` and `toy_order.py` (mixture toys), `knn_purity.py`. No sorts, no
edits to `kilosort\`. Round evidence: `.claude\worktrees\ks4-round3\tuning\exploratory\iteration\round3_results.md`.

## 0. Conclusions

1. **The proposal is two problems, and theory treats them differently.** Refining templates that exist is continuous
   estimation; there online methods have a real, large advantage. Finding templates that do not exist is model
   selection; there small batches lose evidence, and the online literature's answer is to accumulate across batches.
2. **KS4's pipeline has no shared objective.** Extraction plus plain means is hard EM on a convolutive L0 cost. The
   graph step breaks this: labels are not argmins of that cost, and the graph changes each round. "Convergence" can only
   mean a fixed point of a map, with no guarantee against cycles, and the seeds sit in different basins.
3. **Refinement speed-up is real.** Forgetting-based online updates beat rounds by about N/n\* in passes: about 3-11×
   for the median unit, 13-45× weighted by spikes, about 1-2× for the slowest decile. In the toy, 0.05 of a pass of
   KS2.5-style updates matches five rounds of mean-only refinement.
   KS2.5's logs agree: settling takes the same number of batches at 20 min and 2 hours.
4. **The rounds' gain looks mostly structural, and refinement alone does not reach it.** The dictionary falls by about
   30% in one round and then stops (ratio of changes 0.05-0.11); augment, which only adds, gains half in S and nothing on
   the curves. In the toy, means-only online settles at a worse fixed point than rounds with split/merge and never repairs
   orphans; in KS2.5's logs, template count takes 1100-1900 batches to settle where spike yield takes a few hundred.
5. **Residual-based birth, KS2.5's mechanism, is blind to the units that matter.** A spike absorbed by a template at
   cosine ρ leaves a residual of norm s√(1−ρ²). 50-56% of H8 units would be absorbed and invisible if they lost their
   template. Discovery has to look inside templates (split on add-back features), with evidence accumulated over a window of
   hundreds of spikes per template. Graph clustering inside a batch is hopeless on close pairs (§3.3).
6. **With that in the loop, interleaving can work.** In the toy, one pass with window splits equals 3-5 rounds with splits;
   the speed-up is 3-5×, set by structure and not the 100× of refinement. Refine-first-then-split is worse than either, so
   the structure moves must be inside the loop from the start.
7. **Prediction.** Plain interleaving (KS2.5-style): faster to a worse solution. Interleaving with window splits, merges and
   one final global clustering: about the same quality as round 4, saving about 1.3× of the extra rounds' time on 20-min
   excerpts and about 1.8× at 2 hours. Interleaving is not predicted to beat the rounds' fixed point. Its case is wall time
   on long recordings. Two cheap tests (§6, T1 and T2) should come before any build.

## 1. Formalising it

### 1.1 The objective behind extraction

Whitened data X (noise variance 1), spikes S = {(t_k, c_k, a_k)}, dictionary D = {T_c}:

    L(D, S) = ‖X − Σ_k a_k T_{c_k}(· − t_k)‖² + Th² |S|,   a_k ≥ 0.

For a candidate (t, c), with B = ⟨r, T_c(·−t)⟩ and nm = ‖T_c‖², the best amplitude is a = B/nm and the energy removed is
2aB − a²nm = B²/nm, which is exactly `Cf` in `run_matching` (`template_matching.py:232`); a spike is accepted if
√Cf > Th. So matching pursuit with threshold Th (Mallat & Zhang 1993) is greedy descent of L, and Th² is an L0 penalty,
i.e. a Gaussian-noise MAP with a per-sample spike prior of about e^(−Th²/2). Parallel peeling of non-overlapping peaks
is the same descent, and amplitudes aren't re-fitted, so this is MP and not OMP. It is a good approximation only because
spikes are sparse, which the KS paper says too.

KS1 stated this kind of objective: reconstruction error of the raw voltage plus a prior on amplitude, minimised by
alternating template matching and template optimisation (Pachitariu et al. 2016). The KS4 supplement calls the same loop
a standard EM-style algorithm. Two details:

- `lam = 20` is unused in KS4, so amplitudes are free. KS2.5's score, `Cf = b²/(1+λ) − λμ²` with
  `b = relu(B) + λμ` (`mexMPnu8_rg.cu:310-312`), is the same cost with a Gaussian amplitude prior around μ. It equals
  s²(ρ² + 2λρ − λ)/(1+λ) for a spike at cosine ρ to a template of its own norm. At λ = 10 that is 0.70 at ρ = 0.85 against
  KS4's 0.72, so the prior barely rejects mismatched spikes until ρ < 0.7 (`contraction.py`).
- The KS4 paper's Methods say learned-template matches use the template's average norm; the code here has a free scale
  (`amps = B/nm`). I go by the code. The paper's rule is the λ→∞ limit, 2ρ−1 in the same units.

### 1.2 Extraction plus means is hard EM

- **E-step:** S(D) ≈ argmin_S L(D, S). Approximate: greedy, and restarted from empty each round, so L need not fall
  even with D unchanged.
- **M-step:** for fixed S, minimising L over T_c gives T_c = Σ a_k (r_k + a_k T_c) / Σ a_k², where r_k is the residual
  around spike k. r_k + a_k T_c is KS4's feature `xfeat += amps*Ucc` (`template_matching.py:86`): the add-back. It is the
  dictionary-learning atom update on the residual with the atom's own contribution restored (Aharon et al. 2006, K-SVD,
  is the same structure with coefficients updated too). KS2.5 averages the same thing (`average_snips` after adding
  subtracted spikes back). KS4 and KS2 use a plain mean, without the amplitude weights; the KS4 supplement says this is
  for robustness to outlier amplitudes. The cost is a variance factor 1 + CV², a few percent, and a template scale equal to
  the mean amplitude.
- With D held in a sparse-coding view this is **convolutive dictionary learning with a sparsity-1-per-event code**, i.e.
  scaled k-means on events; with non-overlapping events it reduces to k-means under the metric max_c (x·T_c)²/‖T_c‖².
  Bottou & Bengio (1994) show k-means is Newton's method on quantisation error, which matters in §2.
- Neither KS4 nor KS2 has a rate prior: no log π_c term in the assignment. That has a consequence (§2.6).

### 1.3 What the graph step does

KS4's actual map is

    F: D → S(D), features tF(D) → kNN graph G(tF) → labels z → merges → D′ = {mean of tF over z = c}.

Inside one spatial centre with G fixed, the two alternating assignments (`assign_isub`, `assign_iclust`) are exact block
coordinate ascent on Barber's bipartite modularity (Barber 2007; the KS4 Methods give the reassignment rule), since each
side is separable given the other:

    Q(z, w) = Σ_ij [M_ij − γ k_i k_j/m] δ(z_i, w_j).

That is monotone and finite, ending at a stationary point that the paper uses deliberately as an over-split. Then the
tree and swarm splitter apply merge tests that are not derivatives of Q. So the step does have an objective, locally,
but it does not share one with L:

1. z is not argmin L. Relabelling can raise L, so "extract, cluster, mean" is not coordinate descent on L.
2. Q is defined on G, and G depends on D through tF. Each round optimises a different function.
3. Merges are hypothesis tests (bimodality, refractoriness), not moves on either objective.
4. Augment makes the point empirically: adding atoms can only lower L for fixed labels, yet it does less for quality
   than rebuilding, which removes 30% of atoms. L is not the quantity that predicts the score. (L per round has not
   been measured. Logging ‖Xres‖² in `extract` would measure it, with no change to the spikes.)

The best description I can give: F is a **self-consistency map**. In the population limit, graph clusters approximate
density modes of p(tF | D), so a fixed point D\* satisfies "the modes of the add-back feature density under peeling with
D\* are D\*". That is a well-posed definition, close to mean-shift, and it is not variational.

### 1.4 In what sense can rounds converge?

Three senses, in decreasing strength:

- **Monotone descent of a Lyapunov function:** not available, by §1.3.
- **Fixed point of a deterministic map.** With the seed fixed, F is deterministic on a finite label space, so orbits
  are eventually periodic. They reach a fixed point or a limit cycle; nothing excludes the cycle. Near a fixed point, if
  the template part is smooth, linearisation gives ‖D_{k+1} − D\*‖ ≤ κ‖D_k − D\*‖ with κ the spectral radius of F′.
  That is the EM form: a linear rate given by the fraction of missing information, with slow components eventually
  dominating (Dempster et al. 1977; Meng & Rubin 1994). Any score then follows S_k = S\* − cκ^k.
- **Stationary distribution of a random map.** Across seeds F is a Markov chain with a noise floor: S_k = S\* − cκ^k + η_k.

Which applies is testable: label churn between rounds falling to zero means a fixed point; a floor means a cycle or
floor noise.

### 1.5 The evidence read as rates

Ratio of successive changes, (x₄ − x₃)/(x₃ − x₂), from `round3_results.md`:

| Quantity | Ratio | Note |
|---|---|---|
| Dictionary size, kobe / MEC | 0.05 / 0.11 | one-shot |
| kobe S | 0.17 ± 0.15 | per seed 0.47, 0.01, 0.04 |
| kobe S_track | 0.18 ± 0.12 | |
| MEC position S | 0.38 ± 0.41 | per seed −0.05, 1.66, 0.27 |
| Curves: kobe at 2%, at 0.42 M; MEC at 2% | 0.12, 0.34, 0.36 | |
| Whole output: kobe correct, misassigned | 0.35, 0.30 | |
| Orphan units, kobe | 0.35 | |
| Reproducibility, MEC / kobe | 0.39 / 0.61 | |
| Units changed (1 − fraction matched), kobe / MEC | 0.67 / 0.76 | |

(± is a delta-method SE over three paired seeds; two steps and three seeds can't separate κ = 0 from κ = 0.5 on S.)

- Taken as a contraction rate, +4.9/+29.7 = 0.17 gives an asymptote of +35.6 above round 2, of which +34.6 is already
  in, and round 5 worth +0.8. On the curves, 0.34 gives a further +0.06 M at round 5. Neither is detectable with three
  seeds (seed SD of ΔS is about 3-10). So the S gain is effectively finished at round 4 if this reading holds.
- **The quantities do not share one rate, and that is informative.** There is a fast one-shot mode (dictionary size), a
  middle mode (κ ≈ 0.2-0.4, score and curves), and a slow one (units changed, ratio about 0.7). The slow one is either
  a true slow mode, which would keep 18% of units moving at round 5, or a floor from clustering noise. Round 3 replaces
  about as many units as a change of seed does, so a floor is plausible.
- **Seeds do not converge to one solution.** Linear extrapolation of reproducibility gives asymptotes of about 0.68 (kobe)
  and 0.54 (MEC), far from 1, against 0.86 at KS4 default. These are two-point extrapolations and only a sketch, but each
  seed seems to settle in its own basin. "Better solution" here does not mean "the solution".

**A toy for scale.** Two Gaussians at separation d′ in noise SDs, hard assignment. At the symmetric fixed point
(closed forms, checked by Monte Carlo in `contraction.py`):

    μ̂ = a(2Φ(a) − 1) + 2φ(a), a = d′/2      centroid; bias outward μ̂ − a = 2[φ(a) − aΦ(−a)]
    κ_b = 2φ(a)μ̂                             boundary shift
    κ_rot = 2φ(a)/μ̂                          rotation of the normal

| d′ | Bayes error | centroid bias | κ_b | κ_rot | noise gain 1/(1−κ_rot) |
|---|---|---|---|---|---|
| 1 | 31% | 0.40 | 0.63 | 0.79 | 4.7 |
| 2 | 16% | 0.17 | 0.57 | 0.41 | 1.7 |
| 3 | 6.7% | 0.06 | 0.40 | 0.17 | 1.2 |
| 4 | 2.3% | 0.02 | 0.22 | 0.05 | 1.06 |
| 5 | 0.6% | 0.004 | 0.09 | 0.01 | 1.01 |

Measured κ = 0.17-0.35 corresponds to d′ ≈ 3.4-4.3, or Bayes error 1.5-4.5%. Those are the pairs at ρ ≈ 0.9-0.95 and
s ≈ 10 (separation D = s√(2(1−ρ)) = 3.2-4.5). On kobe 26.5% of units have a neighbour above ρ = 0.9, and the median best
neighbour is 0.845. So the observed rate is of the size hard EM would give on the close pairs that limit sorting. That
is suggestive and no more: the map isn't this toy.

## 2. Incremental versus batch

### 2.1 What is established

| Result | Statement | Bears on |
|---|---|---|
| Neal & Hinton 1998 | EM is coordinate ascent on a free energy F(q, θ); updating q for one item at a time keeps ascent, justifying incremental EM | Needs a shared F. Applies to extraction + means, not to the graph step. |
| Gunawardana & Byrne 2005 | The free-energy argument alone doesn't establish convergence to likelihood maxima; they supply conditions | Limits the guarantee |
| Robbins & Monro 1951; Polyak & Juditsky 1992; Ruppert 1988 | Decreasing steps converge under noise; averaging iterates with slower steps gives the optimal rate | KS2.5 ends on a constant step; §2.5 |
| Cappé & Moulines 2009 | Online EM as stochastic approximation in sufficient-statistic space; independent observations; step ∝ i^(−α), α ∈ (½, 1]; Polyak-Ruppert averaging recommended | KS2.5's update is this form, hard-assigned |
| Bottou & Bengio 1994 | Batch k-means is Newton on quantisation error | Rounds are second-order per step |
| Sculley 2010 | Mini-batch k-means with a per-centre rate shrinking with that centre's count: much faster than batch, better than plain online SGD | Direct analogue |
| Tang & Monteleoni 2017 | Stochastic k-means converges to a local optimum at O(1/t) | Rate for the 1/n schedule |
| Schwartzman 2023 | Local progress on a batch implies global progress when batch size is of order (d/ε)²; terminates in O(d/ε) iterations | Guarantee needs a large batch: not 10 spikes per template |
| Mairal et al. 2009, 2010 | Online dictionary learning: stationary points, i.i.d. draws; cycling a permuted finite set is an in-practice substitute; replaces unused atoms; notes that correlated atoms change how the sufficient statistics behave | Closest analogue; correlated atoms are our case |
| Liang & Klein 2009 | Stepwise EM reaches batch quality much sooner and sometimes better optima; remembering old statistics (incremental EM) helped less than forgetting | §2.2 |
| Hughes & Sudderth 2013 | Stochastic online inference is sensitive to the learning rate and often lands in poor optima; memoized statistics avoid that | The counter-case |
| Bottou & Bousquet 2007 | Under a time budget, optimisation error need only match estimation error, so crude fast optimisers win on large data | Why n\* governs, §2.3 |

Note the disagreement between the last entries. Whether noisy online steps find better or worse optima than batch is
problem-dependent, and no theorem settles it.

### 2.2 The speed-up, derived

Write h(D) = M(D) − D for the hard-EM update. For a continuous density, the population Lloyd map is smooth even though
each assignment is hard, so stochastic-approximation arguments apply.

- **Rounds:** D ← D + h(D). Error × κ per pass.
- **Forgetful online** (KS2.5): per template, γ_j = 1 − exp(−n_jb/m_j) ≈ n_jb/m_j, so D_j ← D_j + γ_j(h_j + ξ). This is the
  Euler discretisation of Ḋ = h(D) in time V_j = Σγ_j ≈ ∫dn/m(n), "virtual rounds". Error falls as exp(−(1−κ)V_j), to a
  noise floor var ≈ σ²/(2m(1−κ)).
- **Memoized** (Neal-Hinton, every batch's statistics replaced in turn, no forgetting): on a scalar linear model the
  per-pass rate μ solves μ = exp(κ(1 − 1/μ)). Result: μ = 0.05, 0.16, 0.51 at κ = 0.15, 0.35, 0.7, which is close to κ².
  So memoization saves **a factor below 2 in passes, whatever the number of batches** (1.6-1.9). It is the Jacobi to
  Gauss-Seidel gain. Larger speed-ups need forgetting, which is what discards stale assignments. This agrees with Liang
  & Klein's finding that stepwise beat incremental.

Rounds are Newton steps; online is Euler. To match R rounds, online needs V = R·ln(1/κ)/(1−κ), so 1.2-2.2 virtual rounds
per round for κ from 0.7 down to 0.15.

The usual reason for the speed-up is redundancy: N data points contain many copies of the information needed. It is
limited by a per-template n\*: forgetting memory m must satisfy 2m(1−κ) ≥ n\*, so V_j ≈ 2N_j/n\* per pass.

### 2.3 How many spikes does a template need?

Isotropic noise on d = 60 features (KS4's 10 channels × 6 PCs). A template from n spikes has error norm² d/n. Against
its own norm that is harmless: s = 10, n = 10 gives cosine 0.97. What matters is the discriminant of a close pair,
separation D: cos² = D²/(D² + 2d/n), error Φ(−D cos/2).

| Pair (s, ρ) | D | n=10 | n=30 | n=100 | n=300 | n=1000 | no noise |
|---|---|---|---|---|---|---|---|
| 10, 0.80 | 6.3 | 0.3% | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| 10, 0.85 | 5.5 | 1.0 | 0.5 | 0.4 | 0.3 | 0.3 | 0.3 |
| 10, 0.90 | 4.5 | 3.9 | 2.1 | 1.5 | 1.3 | 1.3 | 1.3 |
| 8, 0.90 | 3.6 | 9.9 | 5.9 | 4.4 | 3.9 | 3.7 | 3.7 |
| 10, 0.95 | 3.2 | 14.3 | 9.1 | 6.8 | 6.1 | 5.8 | 5.7 |

The rounds' gain is an 8-13% fall in misassigned spikes in plain pairs, so template noise must add well under 10% to the
error of a close pair: **n\* ≈ 300-1000**. At 10 spikes per batch per template, the 10-spike templates of a single batch
are far short of it.

### 2.4 The rate distribution

The user's "10 per batch" is the mean. kobe H8 round 3 (`empirics.py`): 1795 units, 549 batches.

| Unit percentile | 5 | 10 | 25 | 50 | 75 | 90 | 99 |
|---|---|---|---|---|---|---|---|
| Spikes in 20 min | 356 | 742 | 1673 | 3219 | 6518 | 13292 | 63291 |
| Per batch | 0.65 | 1.35 | 3.05 | 5.9 | 11.9 | 24 | 115 |

Batches to 300 spikes: 12 for the 90th percentile unit, 51 for the median, 222 for the 10th. Fraction of a pass to reach
1000 spikes: median unit 0.31; 90th percentile 1.35, i.e. never in one pass.

Speed-up in passes, N_j/n\*:

| | median unit | 10th percentile | spike-weighted median |
|---|---|---|---|
| n\* = 300 | 11× | 2.5× | 45× |
| n\* = 1000 | 3.2× | 0.7× | 13× |

So the first-order advantage survives for the bulk, and for the score, which is rate-weighted. It does not survive for
slow units. KS2.5's standard schedule (m = clip(0.1n, 5, 400)) shows the same thing another way. Nominal virtual rounds
per pass are 37 / 52 / 77 (10th / median / 90th), but steps taken at m < 50 are too noisy to count. Counting from n = 500,
that is about 4 / 19 / 44. And the final n_eff = 2m is 148 / 644 / 800 where a round has 742 / 3219 / 13292, so a whole-
round mean is less noisy for every unit.

A round's strength is noise: it gets all N spikes per template. Online's is the number of updates. A final frozen pass
with plain means gets both, and is the limiting case of Polyak-Ruppert tail averaging (§2.5).

Dictionary size is not itself the obstacle: 2000 templates each see their own spikes, so the update is per template.
It matters through coupling: a slow template's neighbours move fast, and it must track them (§2.6).

**Toy** (`toy_online.py`; two seeds; about 180 units, 1.1-1.4 M spikes, 549 batches; start dictionary with 33-38% of units
orphaned and some duplicates; assignment only, no convolution). Spike accuracy, seed 1 / seed 2:

| Scheme | Passes | All spikes | Slowest third of units | Units owning a template |
|---|---|---|---|---|
| True templates | | 0.900 / 0.886 | 0.83 / 0.88 | 0.99 / 1.00 |
| Start | 0 | 0.768 / 0.772 | 0.41 / 0.40 | 0.66 / 0.70 |
| Rounds, means only, round 5 | 5 | 0.845 / 0.822 | 0.30 / 0.40 | 0.69 / 0.74 |
| Online forgetful (`ks`) | 0.05 | 0.842 / 0.818 | 0.30 / 0.39 | 0.69 / 0.74 |
| Online forgetful | 1 | 0.854 / 0.827 | 0.36 / 0.38 | 0.71 / 0.74 |
| Online forgetful | 3 | 0.854 / 0.828 | 0.37 / 0.37 | 0.72 / 0.74 |
| Online 1/n or memoized (identical in pass 1) | 1 | 0.841 / 0.818 | 0.35 / 0.43 | 0.70 |
| Memoized | 3 | 0.850 / 0.823 | 0.35 / 0.43 | 0.71 |

Means-only online matches five rounds at **0.05 of a pass, about 27 batches**: a speed-up near 100× for this refinement, and
it then settles slightly above them. Forgetful beats 1/n and memoized, as §2.2 predicts. Nothing here repairs orphans
(§3.5 has the rest of this table).

**Real evidence, KS2.5** (20 logs parsed; standard and standard_low; auto-stop). From cold, on the lab's copy, which
reshuffles batches:

| | 20 min (549 batches) | 2 hours (3293 batches) |
|---|---|---|
| Batches at auto-stop | 2431-3419 (4.4-6.2 passes) | 2509-3648 (0.76-1.11 passes) |
| Batch by which spikes per batch reach 90% of final | mostly 200-700, a few to 2300 | 200-600 |
| Batch by which template count reaches 90% of final | 1100-1900 | 1100-1800 |
| Learning wall time | 800-1860 s, 0.32-0.61 s/batch | 880-1820 s |

Three readings:

- The work scales with updates and not with data: 2 hours settles in the same number of batches as 20 minutes, under one
  pass. That is the redundancy argument in real data. Caution: the stop rule compares a 500-batch-smoothed residual over a
  500-batch lookback, so the stop itself can't come early. The count of templates is the better measure, and it agrees.
- **Refinement settles 3× or more before structure does.** Spike yield flattens by a few hundred batches; the number of
  templates takes 1100-1900. So in the one real online learner available, structure is the slow part.
- **Measured in batches, online from cold isn't obviously ahead of rounds.** KS4 to round 4 processes 4 × 549 ≈ 2200
  batches of extraction, plus four clusterings. KS2.5 processes about 1100-1900 to settle its template count, and 3000 by
  its stop rule. On a 20-min excerpt that's parity. Online's advantage in passes grows with length: at 2 hours, rounds
  process 13,000 batches and online still about 3000. This favours interleaving on long recordings more than on the
  excerpts, the same wall-time argument as for any sublinear method (Bottou & Bousquet 2007). The caveat is that KS2.5's
  quality isn't length-invariant (2-hour misassigned +54%, not traced to a step), so learning on a fraction of a long recording
  has a cost still to be understood.

### 2.5 Step sizes

- Constant m never converges; it tracks, with noise floor n_eff = 2m(1−κ). KS2.5 standard is close to 1/n up to n = 4000
  (m = n/10, which is a Robbins-Monro step of 10/n) and constant after.
- m = n/10 forgets all but the latest 10-20% of spikes. Its error falls as (n₀/N)^(10(1−κ)), much faster than the
  (n₀/N)^(1−κ) of a true 1/n average, which keeps its stale early assignments. This is why `ks` beats `inv` and `mem` in
  the toy.
- Recommended form: forget during learning, then one frozen pass of plain means. That costs one extraction and removes
  the constant-step noise without losing the transient speed.

### 2.6 Bias from hard assignment

These hold for rounds and for online alike. They are properties of the fixed point.

- **Centroid bias.** Hard-assigned centroids move apart by 2[φ(a) − aΦ(−a)] each (table above): 0.06 noise SD at
  d′ = 3. Kearns, Mansour & Ng (1997) give the general form: k-means finds densities with less overlap than EM.
- **Confirmation mode.** A boundary selects noise along its own normal, which re-sharpens that normal. κ_rot → 1 as
  d′ → 0: splitting one Gaussian unit is neutrally stable, in any direction. Separation 1.6σ is self-consistent from
  nothing. This is a population property; it stays at N = ∞ and has nothing to do with spikes re-entering their own
  template. It is the same phenomenon as "Einstein from noise" in cryo-EM (Henderson 2013): aligned noise averages to the
  reference. For a thresholded template, E[x | x·t̂ > Th] = t̂(Th + 1/Th approximately), so any direction with a local
  supply of hash is a fixed point with norm just above Th. Gaussian noise alone gives negligible rates at Th = 7, but
  hash isn't Gaussian. Lowering Th makes more of these.
- **Rate imbalance.** No log π term means a rare unit next to a common one loses. Lloyd on d′ = 4: at π = 0.2 the rare
  side is 90% pure; at π = 0.05 the fixed point has the rare template on the common unit's tail, 16% pure. At d′ = 6,
  π = 0.05 stays 97% pure and π = 0.01 collapses to 2%. There are two regimes with a sharp switch. In the toy, mean-only
  refinement lowers the slow tercile's accuracy (rounds 0.41 → 0.30; online 0.41 → 0.36). A rate prior in assignment
  would change this for both schemes. Neither rounds nor interleaving addresses it.

### 2.7 Self-reinforcement: rounds against interleaved

Two things get called confirmation bias, and only one depends on the scheme.

**Self-inclusion.** Template T̂ contains spike i with weight 1/n. The spike's score under its own template is inflated by
about d/(ns) noise SDs: 0.006 at n = 1000, s = 10; 0.06 at n = 100; 0.2 at n = 30. So it matters only below about 100
spikes.

- Whole round: always present, small except in tiny clusters.
- Online, one pass or m ≪ N: absent. The template used on a spike was built from other spikes. Forgetful online is
  cross-fitted by construction, a modest point in its favour.
- Memoized: present, as in rounds, unless a spike's own batch is subtracted before assignment. Hartigan's method does
  this for k-means, and its fixed points are a subset of Lloyd's (Telgarsky & Vattani 2010).
- Rounds can have this for free: build templates on odd batches, extract even, and swap.

**Mean-field reinforcement.** The same κ_rot in both schemes. The difference is noise. Online adds a forcing term of
variance 1/(2m), amplified by 1/(1−κ_rot): 4.7× at d′ = 1, 1.2× at d′ = 3. So online directions wander for near-unresolved
pairs where rounds, with noise n = N, do not. On pairs of d′ ≈ 1-2, online needs m several times larger for the same
stability, which with m bounded by rate means slow, close pairs are the worst case.

There is one further hazard for online only. In a round all assignments share one D; in online, fast templates move
before slow ones have adapted, so at equal Euler time they are not synchronised. Per-spike forgetting gives a Jacobi-like
update with unit-dependent clocks. Nothing guarantees the fixed point is reached in the same order, and order is what
selects the basin.

## 3. Structure changes

### 3.1 The two problems are different

- **Refinement:** fixed K, continuous parameters; estimation error scales as 1/n; local moves are enough; §2 applies.
- **Discovery:** K changes, a discrete step on evidence that may be spread thinly across a recording; local moves can't
  get there (there is no gradient toward a component that does not exist). Mairal's atom replacement, DP-means and split-merge
  all exist because of this. Interleaving helps the first. It helps the second only if it collects evidence over batches.

### 3.2 Birth from the residual is blind

Let a unit of norm s have no template, and let the best remaining template have cosine ρ. Three outcomes for a spike:

1. ρs < Th: not absorbed, stays whole in the residual, and is visible if s > Th.
2. ρs ≥ Th: absorbed, at a = ρs/‖T‖. Residual norm s√(1−ρ²).
   - If that is ≥ Th: visible.
   - If it is below Th: **blind**. The spike has been explained away. It is in a template, its residual is below
     threshold, and no residual-based detector can see it.

The blind band is Th/ρ < s < Th/√(1−ρ²). It is non-empty iff ρ > 1/√2. For Th = 7: at ρ = 0.85, 8.2-13.3; at ρ = 0.9, 7.8-16.1.
Typical s on kobe is 10. Orphans' borrowed templates have cosine 0.80-0.86 on the same channel
(`borrowed_templates.md`), inside the band.

Applied to real dictionaries, removing each unit's template in turn and using its best lag-maximised neighbour (units
with ≥ 300 spikes; `empirics.py`):

| Dictionary, Th | Below Th | Left whole in residual | Absorbed, residual still crosses | Absorbed and blind |
|---|---|---|---|---|
| kobe H8 round 3, 7 | 0.3% | 22% | 27% | **50%** |
| MEC H8 round 3, 7 | 0.5% | 18% | 26% | **56%** |
| kobe default round 2, 8 | 0% | 30% | 40% | **31%** |
| kobe H8 round 3, 10 | 49% | 14% | 13% | 23% |
| MEC H8 round 3, 10 | 51% | 12% | 11% | 26% |

So if a kobe H8 unit lost its template, the odds are even that nothing in the residual would ever show it. KS4's H8 settings
worsen this: a lower Th means more absorption.

This is an upper bound on what a residual detector sees: s√(1−ρ²) is the score of a perfectly matched new template, and
multi-atom peeling reduces the residual further. Also, single-spike amplitudes vary, which blurs the band.

KS2.5's birth (`mexGetSpikes2` on `drez`, at Th(1)) is the DP-means rule (Kulis & Jordan 2012): open a new centre when the
squared distance to every centre exceeds λ, here λ = Th². Its retention test is a firing-rate floor on an EMA of counts. So
**λ = Th² is a resolution limit**: nothing within √λ of an existing atom can be born from a single spike. KS2.5 works
better than this suggests for one reason: learning uses Th(1) = 10, not 6. That keeps spikes of s < 10/ρ un-absorbed and
visible. The cost is that units of s below 10 aren't learned at all, half of H8's units (above). That also explains the
finding that KS2.5 matches almost none of KS4's units below median spikes and amplitude (`ks25_found_rate.py`). It looks
like a threshold effect more than an online-learning effect.

The λ-prior doesn't help here (§1.1). What would: a fit gate at matching time, rejecting a peel when the residual within
the template's support remains large. That is a new mechanism, and it would leave orphans' spikes whole for a birth to
use, at the cost of missing true spikes in overlaps.

### 3.3 What can see them: evidence summed across spikes

Each blind spike still carries s²(1−ρ²) of unexplained energy. A new template is justified, BIC-style, when
n·s²(1−ρ²) > d ln n: at s = 10, ρ = 0.85 that is about 13 spikes **if you know which spikes**. So the evidence is
strong in aggregate, and invisible per spike. The way to aggregate is within a template, on add-back features, which is
what graph clustering does globally.

**Split detection.** Among m spikes of one template, a minority π at separation D gives a covariance spike
ℓ = π(1−π)D². Detectable above noise when ℓ > √(d/m) (Baik, Ben Arous & Péché 2005; Paul 2007 for the eigenvector); usable
at about ten times that m (cos² ≈ 0.8).

| Pair | π | m usable | minority spikes |
|---|---|---|---|
| ρ 0.85, s 10 | 0.2 | 26 | 5 |
| ρ 0.85, s 10 | 0.05 | 291 | 15 |
| ρ 0.95, s 10 | 0.2 | 224 | 45 |
| ρ 0.95, s 10 | 0.05 | 2536 | 127 |
| ρ 0.98, s 10 | 0.2 | 1465 | 293 |

These assume isotropic noise. Real add-back features have an amplitude mode with ℓ ≈ (CV·s)² ≈ 4, jitter and overlaps,
so the true figures are likely several times larger: **m of order 10³ for pairs that matter.** One batch has about 6
spikes per median template, so a split needs about 50-200 batches of accumulation.

**Undifferentiated density.** For a cluster of n spikes among M events, the threshold scales as n > √(dM)/s². So a window
holding a fraction w of the data has threshold N/√w: a 2 s batch is √549 ≈ 23× less sensitive than the whole recording.
Better than w⁻¹, but fatal for rare units unless evidence accumulates.

**kNN graphs in small windows** (`knn_purity.py`; d = 60, k = 10, fraction of a spike's neighbours from its own unit, two
equal units):

| D | n=10 | n=30 | n=100 | n=300 | n=1000 | n=10000 |
|---|---|---|---|---|---|---|
| 3.2 | 0.58 | 0.67 | 0.74 | 0.77 | 0.79 | 0.82 |
| 4.5 | 0.64 | 0.81 | 0.88 | 0.89 | 0.92 | 0.93 |
| 5.5 | 0.73 | 0.88 | 0.94 | 0.96 | 0.97 | 0.98 |

(0.9 is the ceiling at n = 10.) A graph needs hundreds of spikes per unit, and nodes count, not spikes: with H8's
`cluster_downsampling` of 10, a 3000-spike unit is 300 nodes. So re-clustering inside a batch at about 10 spikes per unit
is near chance on the close pairs that the gains come from. A window would need about 300 spikes per unit, which is 10%
of the pass for the median unit, and modularity's resolution limit (Fortunato & Barthélemy 2007) also shifts with
graph size. So: no small-batch graph clustering. Windows of 10% or more, or reservoirs.

### 3.4 Guarantees

- **DP-means** (Kulis & Jordan 2012): monotone descent of k-means cost + λK to a local optimum, in batch form.
- **Online:** Meyerson 2001 and Liberty et al. 2016 give competitive-ratio bounds on cost (O(log n)-type) with
  irreversible centres; Pan et al. 2013 validate proposals serially in epochs. Bounds are on cost, and none says that a
  particular cluster gets found.
- **Memoized** (Hughes & Sudderth 2013): statistics per batch give an exact full-data objective, so a merge is accepted
  only if the full ELBO rises; merges based on single-batch estimates can lower the full objective. Births are always
  accepted and pruned later. A birth has phases of collection, creation and adoption over two passes, targeting one
  component's data. Their reason is exactly ours: "Each small batch may not have enough examples of a missing component".
- **Split-merge MCMC** (Jain & Neal 2004): valid Metropolis-Hastings moves between modes that Gibbs can't cross, with
  asymptotic guarantees for a posterior, and none for a time-limited online run.

The theory's lesson: propose locally, validate globally, and accumulate before proposing. KS4 has no full-data objective
to validate against, which is the one guarantee this literature leans on. Its substitute would be refractoriness and
bimodality tests.

### 3.5 Does interleaving help the core premise?

Only partly, and not directly. Units never given a template divide into:

1. **Absorbed inside another template** (orphans: 12-15% of units; 50% of units by the band test would be if orphaned).
   Residual birth can't help. A per-template split on add-back features can, and it doesn't need global density, so this
   is compatible with interleaving given a reservoir.
2. **Not absorbed, above threshold, rare.** Stays in the residual; birth sees it, one spike at a time; retention by rate.
   Online is fine if the proposals survive until the template has enough spikes. With KS2.5's floor (0.05 Hz, about 0.1 per
   batch) rare units are fragile.
3. **Below threshold, or absorbed nowhere.** Neither scheme helps. Th sets this.

The toy on class 1, continuing the table of §2.4. "Rounds + split/merge" adds, per round, a per-template split on PC1 off the
template axis (accept if d′ > 4 and ≥ 30 spikes a side) and a shape merge. "Online + window split" runs that same test every
50 batches (9% of a pass, so about 300 spikes for a median unit; a template needs 60 spikes in the window to be tested) on the
spikes collected since the last test:

| Scheme | Passes | All spikes | Slowest third | Units owning a template |
|---|---|---|---|---|
| Rounds + split/merge, round 1 | 1 | 0.865 / 0.850 | 0.55 / 0.59 | 0.80 / 0.85 |
| ... round 3 | 3 | 0.886 / 0.870 | 0.61 / 0.68 | 0.87 / 0.91 |
| ... round 5 | 5 | 0.887 / 0.878 | 0.66 / 0.75 | 0.89 / 0.94 |
| Online + window split, cap 400 | 1 | 0.884 / 0.877 | 0.57 / 0.70 | 0.85 / 0.93 |
| ... cap 400 | 3 | 0.888 / 0.879 | 0.59 / 0.73 | 0.88 / 0.94 |
| ... cap 2000 | 3 | 0.890 / 0.879 | 0.63 / 0.73 | 0.90 / 0.94 |
| Online 1 pass, **then** one global split/merge | 2 | 0.876 / 0.855 | 0.49 / 0.54 | 0.80 / 0.83 |
| ... then a second | 3 | 0.880 / 0.866 | 0.52 / 0.60 | 0.83 / 0.87 |

Three results, the first of which goes against the sceptical reading:

- **One online pass with splits inside the loop equals 3-5 rounds with splits**, on all spikes; slow units lag a little.
  So when structure moves are interleaved and evidence accumulated over about 10% of a pass, the online speed-up survives at
  about 3-5× in passes. That is far below the 100× of bare refinement. Structure sets the rate, as in the KS2.5 logs.
- **Refine first, split afterwards is worse than either**: 15-18 splits are accepted, against 33-36 from the start, and the
  units recovered stay about 0.06-0.10 lower. Mean-only refinement of a lump moves it toward its major component and erodes
  the bimodality that a split test needs. So order matters, and the structure moves must be in the loop from the start. A plain
  online warm-up before a global clustering may even hurt.
- Slow units gain least in every scheme. Below 60 spikes per window the online test never runs, so they depend on their
  lump-partner's split.

Limits: both splitters here are local, by construction. The toy can't show whether KS4's graph does something beyond local
splitting and merging, and its pairs are easy (ρ ≈ 0.85 only, D ≈ 5.5).

## 4. Non-stationarity

The theory assumes i.i.d. draws (Cappé & Moulines; Mairal), or a mixing process with cost √(τ_mix/T) (Duchi et al. 2012;
Lyu et al. 2020 for Markov data: online matrix factorisation still reaches critical points of the expected loss). A
recording is neither unless batches are shuffled.

**Time-ordered.** An exponential filter of m spikes has window τ = m/r: 80 s at 5 Hz and m = 400, 800 s at 0.5 Hz, 20 s at
20 Hz. If a state has correlation time τ_s and shifts a unit's mean by δ, the bias at the end of the pass has variance about
δ²π(1−π)·τ_s/(τ_s + τ) (for an AR(1)-like state), the full between-state variance when τ ≪ τ_s. So a fast unit's
template is its latest state's template. Rounds and shuffling both remove this, the first exactly.

Epochal units: per-spike forgetting freezes a silent template, which is right, but triage counts per batch. nsp decays as
p1^L, dropped at nsp < minFR·NT/fs. For a 5 Hz unit, that takes ln(r_b/m0)/(−ln p1) batches: 458 (1000 s) at p1 = 0.99 and
minFR 0.05 (standard), but 107 (235 s) at p1 = 0.95 and 1/50 (original). KS2 chose time order to track drift. Its
authors note that a 400-spike window can't track fast drift, and that slow or bursty units are the problem.

**Shuffled.** Batches become exchangeable draws without replacement. Reshuffling beats replacement sampling (Gürbüzbalaban
et al. 2021). KS1 and the lab's KS2.5 do this (`learnAndSolve8b.m`: `randperm` until `initNBatch`). What is lost:

- tracking, which doesn't matter on these excerpts (drift −6.5 to +4.5 µm) after registration;
- contiguity: a unit's spike times arrive in 2 s pieces, so any correlogram test run during learning sees only
  within-batch lags, which is enough for refractory checks and not for slow structure;
- disk locality, a mild cost.

Toy (`toy_order.py`, three seeds, warm start at the truth, so order is the only cause; 60% stationary units, 20% epochal
in 25% of the recording, 20% state-dependent):

| | shuffled | time order | time order, p1 .95 minFR 1/50 |
|---|---|---|---|
| All spikes correct | 0.80 / 0.81 / 0.74 | 0.77 / 0.73 / 0.69 | 0.61 / 0.65 / 0.64 |
| Epochal units, accuracy | 0.76 / 0.83 / 0.82 | 0.61 / 0.66 / 0.71 | 0.12 / 0.28 / 0.39 |
| Epochal units dropped | 0 | 0 | 26/43, 24/49, 14/36 |
| Stationary units, accuracy | 0.86 / 0.85 / 0.84 | 0.83 / 0.84 / 0.81 | 0.79 / 0.82 / 0.80 |

Standard triage (p1 .99, minFR .05) drops at most 1 under time order, none shuffled. Shuffled lost no unit under any
triage. The stationary units also suffer in time order, since their neighbours' templates go stale. Time order would
help only if drift remained after registration, and even then rounds or registration handle it more cleanly.

## 5. What changes at the fixed point?

Two methods with a shared objective and the same moves may reach the same stationary points at different speeds; which
of many they reach depends on order and noise. Here the objectives aren't shared.

| Scheme | Objective | Moves | Expected relation to rounds |
|---|---|---|---|
| A. Online refinement, then frozen pass | L, hard | means only | **Faster, different, worse.** Fixed points of L at fixed K. Cannot change K. Orphans stay. |
| B. A + KS2.5-style residual birth and triage | L + rate floor | birth by λ = Th², merges at 0.9-0.975 | Reaches a compact dictionary. Misses the blind band and everything under its learning threshold. Quality depends on that threshold. |
| A′. A, then a global clustering | L, then F | means, then F once | **Worse than repeat.** Refinement erodes lumps' bimodality before anything splits them (toy, §3.5). |
| C. A with window splits + merges in the loop from the start | L + local tests | local split/merge on add-back features | **Comparable.** Close to rounds if regrouping is local. Worse if it needs cross-template structure. |
| D. C, then one global extract + graph cluster | as C, then F | adds F once | **Same family as rounds.** F is contractive here (κ ≈ 0.2-0.4), so one application from a good D gets most of the way. Best prediction: within δ/2 of round 4. |
| E. Rounds (repeat) | F | global | Reference. |

Are the rounds' gains structural? Evidence:

- **For structural:** dictionary drops 28% in one step and stops. Rebuild beats augment, and only rebuild gains on the curves.
  Round 3 replaces as many units as a seed change does. KS4 labels are graph labels: 94% of H8 units mix templates. Gain
  is large at H8 (2450 → 1760), zero in S at default (900 → 870), which fits a gain proportional to the distance from the
  fixed point.
- **For refinement:** most of ΔS is in plain pairs, same unit before and after, 8-13% cleaner, spread thinly over most
  units. That looks like better templates and features. But a plain pair can be cleaner because a neighbour's redundant
  templates went, which is also structural. The data so far can't separate them. T1 below can.

Three mechanisms by which a rebuild could matter, none of which refinement of the same atoms reproduces:

1. **Coherence.** Greedy recovery degrades with mutual coherence (Tropp 2004 gives the condition). A dictionary with many
   atoms at ρ > 0.9 assigns near-duplicates by noise. Removing atoms makes extraction more stable, so features are cleaner.
2. **Features.** Round 1 clusters features from universal templates. Later rounds cluster background-corrected features
   from learned ones, so first-round errors are made on the worst features and rebuilt on better ones.
3. **Global re-partition.** Graph clustering can reassign a lump's spikes across template boundaries at once.

Statement of what theory predicts:

- **Same quality, faster:** only for refinement-limited gains, with same model order. Expected for scheme A against rounds
  *with graph held fixed*. Condition: N_j ≫ n\*.
- **Better quality:** no general reason. Possible by noise escaping shallow optima (Liang & Klein), mainly on fast units,
  and by cross-fitting. Rounds are not shown to be at a global optimum: basins differ by seed. A modest gain is possible, not
  predicted.
- **Worse quality:** expected for A and B, by the structure gap, plus more noise for slow units, close pairs (§2.7) and
  rate imbalance. Also expected for any time-ordered scheme, or for graph clustering within batches.

The gain has an upper bound worth stating. Interleaving replaces rounds' mechanism; it doesn't obviously add a new one.
Its gain over rounds is limited to wall time, unless the slow mode (κ ≈ 0.7 in unit identity) is real. Round 4 → ∞ is
worth about +1 in S and +0.1 M on the curves by the extrapolation of §1.5. The budget being saved is two rounds: 14 min
against 6.5 for a 20-min excerpt, 100 against 50 for 2 hours.

Costs for scale: a round is about 215 s per 20 min (extract 100 s, 0.18 s/batch; cluster 115 s). Online from cold at
1100-3000 batches is 200-540 s of matching at that speed, against about 860 s for the rounds to round 4. From a warm
round-2 dictionary, online plus one global round is 1 pass (100-150 s) + 215 s, against 430 s for two rounds: about 1.3×,
less any cost of the reservoirs. At 2 hours, two rounds cost about 50 min; about 1000 online batches (3 min) plus one global
round (25 min) is about 1.8×. Those are arithmetic on the throughput, not measurements.

A cheaper alternative follows from the same reading. If H8's gain is proportional to first-round redundancy, then
cleaning the round-1 dictionary (stronger shape merge, or a better first clustering) may get much of round 3 with no
extra extraction. The prior results (ablation) show this interacts with the tree, so it needs testing.

## 6. Falsifiable predictions

Rules for all comparisons, fixed in advance: H8, kobe and MEC 20-min excerpts, seeds 1-3, paired by seed against the same
round-2 sort; S, S_track and curves as in `round3_results.md`; hyperparameters (m schedule, reservoir, split thresholds)
fixed from kobe seed 1 and then applied unchanged to seeds 2-3 and MEC, since H8 itself was not retuned for rounds.

**Zero-cost tests**

- **T1, refinement or structure.** Score `round3` with each spike labelled by its extraction template
  (`spike_detection_templates.npy`) instead of the graph label, then the same with the final merge. This is the hard-EM
  output of one round. Compare with round 2's labels by template, and with repeat. Prediction: template labels gain less than
  half of repeat's +29.7 on kobe. If they reach ≥ 80%, the gain is refinement, online inherits it, and this note's
  structural reading is wrong.
- **T2, blind band.** For round-2 orphans (12-15% of units), compute the residual score of their spikes after the owning
  template's peel, from `full_st`, amplitudes and features. Prediction: below 20% of orphan spikes have a residual score above
  Th; so a residual-birth learner recovers under about a fifth of them.

**Rounds**

- **P1.** Round 5 from round 4, H8: kobe ΔS within ±8 (mean +1 expected), MEC mean below +15, curves +0.03-0.08 M at
  0.42 M misassigned. A mean over +8 on kobe breaks the κ ≈ 0.2-0.4 reading.
- **P2.** Dictionary changes under 1% at round 5. Units changed stay at ≥ 15%: floor or slow mode. Below 10% would support
  a single fixed point.
- **P3.** Cross-fitted round (templates from odd batches extract even, and swap): |ΔS| < 7 (δ/4) against a plain round. A
  larger effect means self-inclusion matters and §2.7 is wrong.

**Online, from the same round-2 dictionary**

- **P4a.** Interleaved splits give a speed-up of about 3-5× in passes against rounds with the same split rule, not 10× or
  more: passes to reach 90% of the final template count, compared directly.
- **P4.** Template change per batch under forgetful updates falls as 1/√n to a plateau set by m. Refinement (score of a
  frozen-dictionary template-labelled pass) saturates by 0.1-0.3 of a pass for units over 3000 spikes, and doesn't for
  units under 300. Saturation in S by 0.3 pass is the speed claim. If it takes ≥ 2 passes, the speed-up is not there.
- **P5.** Scheme A′ (one online pass from round 2's dictionary, then one global graph round): ΔS +10 to +20 on kobe, within
  ±0.15 M of round 2 on the curves at 2%, i.e. like augment, below repeat's +29.7 and +0.7 M. Reaching ≥ +25 and ≥ +0.4 M would
  refute the structural reading. The same run with window splits from the first batch should beat A′ at every seed.
- **P6.** Scheme D with reservoirs of about 1000 spikes: within δ/2 (14) of round 4 on kobe, within 24 on MEC, curves within
  0.15 M, reproducibility at least 0.60, wall time 70-90% of two rounds at 20 min and about 60% at 2 hours. Beating round 4 by
  more than δ on either excerpt is not expected.
- **P7.** Scheme B at Th 7: dictionary grows with hash-seeded templates clustered just above Th (norm under 8) unless the
  rate floor removes them; at Th 10 it loses units below norm 10 (as KS2.5 does). Neither reaches repeat's gain.
- **P8.** Slow units gain least: units under 1000 spikes show no gain over round 2 under A at one pass, and in the
  template-labelled score lose some, from rate imbalance. A rate prior in assignment reduces the loss under both schemes.
- **P9.** Shuffled is no worse than time order on any measure here, by at least the seed SD on S, and with time order
  losing units present in only part of the excerpt. On a recording with strong residual drift this could reverse.
- **P10.** Default settings: all schemes |ΔS| ≤ δ/2, as with rounds: at default the dictionary is already near its fixed point.
- **P11.** E1 of `modular_pipeline_audit.md` (KS2.5's dictionary into KS4 extract + cluster) is a ready test of an online
  dictionary. Prediction: units below norm 10 are lost and the sort sits below H8 round 2 at S unless their spikes are absorbed
  by neighbours and regrouped; ΔS negative, by more than δ.

**Order of work.** T1 and T2, then P1 if a round 5 is cheap, then E1. A prototype is worth building if wall time on long
recordings is the goal, and then with window splits inside the loop, shuffled batches and a final global clustering. It is not
worth building to improve quality on the excerpts: nothing here predicts a gain beyond round 4.

## 7. Caveats

- The toys have no overlaps, hash, jitter or drift, and their splitter is a stand-in, not KS4's graph. In `toy_online.py` both the
  rounds' splitter and the window's are per-template, by construction, so the near-tie can't show whether KS4's global
  regrouping does more than local splitting. That is the main open point, and why D keeps a final global round. The toy's
  `res` argument caps a window that is reset every 50 batches, so it isn't a long-memory reservoir; caps of 400 and 2000 differ
  little for that reason. Two seeds only.
- The KS2.5 "settling" figures come from a heuristic (first batch at 90% of the final value) on noisy per-100-batch log
  values, from a cold start, and with auto-stop in force.
- κ estimates come from two differences and three seeds. All rate readings are order-of-magnitude.
- n\* assumes isotropic noise in 60 dims. Rank-limited templates and the PCA basis denoise, which lowers n\*. Real
  non-Gaussian background raises it.
- The blind-band calculation uses template norms and final-unit templates as stand-ins; spike amplitudes vary, and templates of
  orphan units specifically were not separated out.
- The population self-consistency reading of §1.3 is a heuristic. I haven't proved F is locally a contraction.
- Literature here covers mixtures and dictionaries. I found none treating a graph step inside an EM-like loop, and no
  analysis of online spike-sorter fixed points.

## 8. Sources and how each was checked

Read in full text (PDF extracted locally): Pachitariu et al. 2016 (NIPS); Pachitariu et al. 2024 (Nat Methods) and its
supplement; Mairal et al. 2010 (JMLR); Hughes & Sudderth 2013; Cappé & Moulines 2009 (arXiv version); Liang & Klein 2009.
Abstract or proceedings page: Bottou & Bengio 1994; Neal & Hinton 1998; Gunawardana & Byrne 2005; Kulis & Jordan 2012; Telgarsky & Vattani 2010; Kearns
et al. 1997; Tropp 2004; Jain & Neal 2004; Pan et al. 2013; Lyu et al. 2020; Duchi et al. 2012; Bottou & Bousquet 2007;
Polyak & Juditsky 1992; Robbins & Monro 1951; Mallat & Zhang 1993; Schwartzman 2023; Tang & Monteleoni 2017; Gürbüzbalaban et
al. 2021; Fortunato & Barthélemy 2007; Baik et al. 2005; Paul 2007; Aharon et al. 2006; Henderson 2013.
Secondary only (bibliographic record or another paper's description): Sculley 2010 (a copy of the paper via a course site);
Dempster et al. 1977 and Meng & Rubin 1994 (via tutorials and a summary; volume and pages of Meng & Rubin, and of Paul 2007, are
from memory); Barber 2007 (citation in a later paper; KS4 Methods describe the same bipartite modularity); Meyerson 2001 and Liberty et al. 2016
(via later papers); Ruppert 1988 (technical report, reference lists only).

- Aharon, Elad, Bruckstein 2006. K-SVD. IEEE TSP 54(11):4311-4322.
- Baik, Ben Arous, Péché 2005. Ann Probab 33(5):1643-1697.
- Barber 2007. Modularity and community detection in bipartite networks. Phys Rev E 76:066102.
- Bottou, Bengio 1994 (NIPS 7). Convergence properties of the k-means algorithms.
- Bottou, Bousquet 2007 (NIPS 20). The tradeoffs of large scale learning.
- Cappé, Moulines 2009. On-line EM for latent data models. JRSS B 71(3):593-613.
- Dempster, Laird, Rubin 1977. JRSS B 39:1-38.
- Duchi, Agarwal, Johansson, Jordan 2012. Ergodic mirror descent. SIAM J Optim 22(4):1549-1578.
- Fortunato, Barthélemy 2007. Resolution limit in community detection. PNAS 104(1):36-41.
- Gunawardana, Byrne 2005. Convergence theorems for generalized alternating minimization procedures. JMLR 6:2049-2073.
- Gürbüzbalaban, Ozdaglar, Parrilo 2021. Why random reshuffling beats SGD. Math Program, doi 10.1007/s10107-019-01440-w.
- Henderson 2013. Einstein from noise. PNAS 110(45):18037-18041.
- Hughes, Sudderth 2013 (NIPS 26). Memoized online variational inference for DP mixture models.
- Jain, Neal 2004. JCGS 13(1):158-182.
- Kearns, Mansour, Ng 1997 (UAI). Hard and soft assignment methods for clustering.
- Kulis, Jordan 2012 (ICML). Revisiting k-means.
- Liang, Klein 2009 (NAACL). Online EM for unsupervised models.
- Liberty, Sriharsha, Sviridenko 2016 (ALENEX). An algorithm for online k-means clustering.
- Lyu, Needell, Balzano 2020. JMLR 21(251):1-49.
- Mairal, Bach, Ponce, Sapiro 2010. JMLR 11:19-60 (and ICML 2009).
- Mallat, Zhang 1993. IEEE TSP 41(12):3397-3415.
- Meng, Rubin 1994. Linear Algebra Appl 199:413-425.
- Meyerson 2001 (FOCS). Online facility location.
- Neal, Hinton 1998. In Jordan (ed), Learning in Graphical Models, 355-368.
- Pachitariu, Steinmetz, Kadir, Carandini, Harris 2016 (NIPS 29).
- Pachitariu, Sridhar, Pennington, Stringer 2024. Nat Methods 21(5):914-921.
- Pan, Gonzalez, Jegelka, Broderick, Jordan 2013 (NIPS 26).
- Paul 2007. Stat Sinica 17:1617-1642.
- Polyak, Juditsky 1992. SIAM J Control Optim 30(4):838-855.
- Robbins, Monro 1951. Ann Math Stat 22(3):400-407.
- Ruppert 1988. Tech Rep 781, Cornell.
- Schwartzman 2023 (ICLR). Mini-batch k-means terminates within O(d/ε) iterations.
- Sculley 2010 (WWW). Web-scale k-means clustering.
- Tang, Monteleoni 2017 (AISTATS). Convergence rate of stochastic k-means.
- Telgarsky, Vattani 2010 (AISTATS). Hartigan's method. PMLR 9:820-827.
- Tropp 2004. Greed is good. IEEE TIT 50(10):2231-2242.
