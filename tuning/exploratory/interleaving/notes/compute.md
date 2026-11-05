# Interleaving extraction and clustering in KS4: compute and engineering audit (2026-11-04)

The proposal, as a hypothesis: interleaving extraction, clustering and template refinement over small batches
might be much faster than repeating whole-recording rounds, and the template cross-correlations (`ctc`) would be
the bottleneck of doing so.

Data: kobe H8, 20-min excerpt (seed 1) and the 2-hour segment. Code: `ks4-round3` worktree, read-only. Scripts:
`tuning/exploratory/interleaving/compute/`. Outputs: `C:/data/ks4_tuning/exploratory/interleaving/compute/`. No full
sorts, nothing edited in `kilosort/` or the worktree.

**[M]** = measured here or read from a sort log. **[E]** = estimated or extrapolated. Quality is not assessed here: every
"saving" below is a cost figure and says nothing about whether the design keeps the rounds' gain.

## 0. Findings

1. **The hypothesised bottleneck is not one.** A full `prepare_matching` takes 15 ms at 1767 templates, 28-50 ms at
   2504 and 65 ms at 3933 **[M]**, against 172-210 ms to match one batch. Refreshing every batch adds about 10%; every
   10 batches about 2%. The dictionary can change as often as anyone wants.
2. **The real per-batch cost is the matching itself:** 172 ms per batch at 1795 templates, of which 18.5 ms (11%) is
   reading and preprocessing **[M]**. Nothing is saved on the 89% unless fewer batches are visited. An online step
   costs about 190-210 ms per batch, against 390 ms per batch for a whole round (extraction plus clustering).
3. **Clustering has no economy at small scale.** A subset costs about 3 s + 10.5 µs per spike down to 0.4M spikes
   (linear from 1/4 to all spikes at steady load; ±50% in absolute terms below that) **[M, noisy]**. A rolling buffer of L batches clustered every N batches costs L/N of a
   full clustering per pass.
4. **So the saving comes from visiting fewer batches, not from interleaving.** D3 (ordinary rounds on a subset) gets
   there with about 40 lines of change. Its cost at 2 h with f = 1/6 is about 51 min against 100 **[E]**, close to the
   48-min conventional sort.
5. **The ceiling is low relative to the conventional sort.** With a free dictionary, a sort still needs drift, one
   full extraction, one full clustering, merge and export: 283 s at 20 min and 1911 s (32 min) at 2 h **[M sum]**.
   That is 68% below the 4-round sort but only 27% (20 min) to 34% (2 h) below the 2-round one. D1 at 1 pass is
   6.6 and 44 min **[E]**: about the 2-round sort's cost. At KS2.5's cold-start 6 passes, D1 costs as much as 4 rounds.
6. **Three traps a cost table would hide:** a stale `ctc` makes the peel loop diverge (residual up by 1e20-1e30);
   `index_add_` on the GPU is non-deterministic (30 distinct results in 30 repeats), and greedy matching then amplifies
   the difference; and `bfile.set_downsampling` after drift correction reads the wrong batch's drift shift. All three have
   simple fixes (sections 2.5, 5, 6).

## 1. Per-stage cost

### 1.1 Whole sort [M]

Seconds. 20 min: `iteration/memtest/nosave` (no stage cache, no round saves, one run, idle machine). 2 h:
`kobe_2h_repeat_s1`. Round-2 dictionary 2504 / 3003 templates, later rounds 1767, 1716 / 1969, 1924.

| stage | 20 min | 2 h | ratio | scales with |
|---|---|---|---|---|
| preprocessing filters | 1.0 | 3.1 | 3.2 | sampled batches |
| drift (one universal pass + `bin_spikes`) | 33.1 | 220.4 | 6.7 | batches, plus Nb x N in `bin_spikes` |
| universal extraction (second universal pass) | 32.0 | 173.6 | 5.4 | batches (53 ms per batch at 20 min, 48 ms at 2 h) |
| first clustering + `postprocess_templates` | 73.8 | 498.4 | 6.8 | universal spikes (5.3M, 31.2M; 13.9, 16.0 µs/spike) |
| learned extraction, round 2 | 133.6 | 872.2 | 6.5 | batches x (a + b x templates) |
| clustering, round 2 | 124.2 | 924.3 | 7.4 | spikes (11.8M, 68.5M; 10.5, 13.5 µs/spike) |
| learned extraction, round 3 | 108.4 | 659.7 | 6.1 | as above, fewer templates |
| clustering, round 3 | 122.9 | 901.7 | 7.3 | |
| learned extraction, round 4 | 106.2 | 649.8 | 6.1 | |
| clustering, round 4 | 127.8 | 925.9 | 7.2 | |
| final merge | 8.2 | 64.0 | 7.8 | spikes (correlograms) and merge pairs |
| export | 7.0 | 47.9 | 6.8 | spikes |
| **total, 4 rounds** | **888** | **6006** | 6.8 | |
| total, conventional 2 rounds | 390 | 2883 | 7.4 | (the 2 h figure includes pc_features, 110 s export) |

A round costs 234 s at 20 min and 1576 s at 2 h. `prepare_matching` is under 0.1% of a round. `postprocess_templates`
is 3.4 s with 3933 clusters **[M]**.

### 1.2 One batch of learned extraction [M]

60 consecutive batches, about 19k spikes per batch, ms per batch, synchronised sections (`batch_breakdown.py`). The
unsynchronised pass gives 171.6 and 210.2 ms, and the log gives 170 ms, so the instrumentation is not distorting this.

| section | 1795 templates | 2504 templates | scales with |
|---|---|---|---|
| read (prefetched) | 1.9 | 2.0 | fixed |
| filter, CAR, whitening with drift | 16.5 | 16.5 | fixed |
| **I/O + preprocessing** | **18.5 (11%)** | **18.5 (9%)** | **any design pays this** |
| temporal conv | 6.4 | 6.4 | fixed |
| projection `U x B` | 17.6 | 22.9 | templates |
| initial `Cf` | 4.4 | 5.9 | templates |
| peel loop | 123.8 | 156.2 | see below |
| feature extraction + copy to host | 2.4 | 2.3 | spikes |

Inside the peel loop (`peel_breakdown.py`; each kernel synchronised, inflating the total by about 8%):

| kernel | 1795 exact | 1795 `fast_matching` | 2504 exact | scales with |
|---|---|---|---|---|
| `colmax` | 79.4 | 79.2 | 109.3 | templates x spikes x window |
| `xres` (residual) | 30.1 | 14.8 | 34.6 | spikes x channels x nt |
| `bupdate` (uses ctc) | 9.3 | 8.6 | 14.4 | spikes x overlapping templates |
| `peaks` | 9.3 | 9.2 | 10.5 | peels x batch length |
| `record` | 3.0 | 2.9 | 3.4 | spikes |

- **Every batch hits `max_peels` = 100**, at about 195 spikes per peel; the first three peels find 7% of the spikes.
- The one step that uses `ctc` takes 5% of the batch. The largest single cost, `colmax`, doesn't use it at all.
- Fit: **e(n) = 74 ms + 54 µs x n** per batch. Fitted at 20 min, it predicts the 2-hour logs: 236 vs 242 ms (n = 3003),
  180 vs 183 (1969), 178 vs 181 (1924). So extraction per batch does not depend on recording length.

Of a round's extraction, then, I/O and preprocessing are 11 s of 102 s at 20 min and about 67 s of 650 s at 2 h.
An interleaved design pays that on every batch it visits, and a stored preprocessed copy would not reduce it by much,
since reading is the small part of it.

### 1.3 Read order [M]

`io_order.py`, with a 170 ms sleep standing in for GPU work so the prefetch has time to run:

| order | read, 20-min file | read, 2-h file |
|---|---|---|
| sequential | 3.2 ms | 3.1 ms |
| strided (every 6th, 15th) | 3.2 ms | 2.9 ms |
| random | 30.4 ms | 33.9 ms |

Strided reads keep the prefetch, because `_BatchReader` guesses the step. A random order loses about 30 ms per batch,
which is what a KS2.5-style random learning order would pay unless it passes the next batch to the reader (a small
change). Filter time here was 28 ms rather than 16.5 ms, because other jobs were then using the GPU.

## 2. The cross-correlations

### 2.1 Cost and sparsity of `prepare_matching` [M]

`ctc_bench.py`, real templates (random subsets of the aligned first-pass clusters), median of 7 repeats.

| templates | full build | nnz pairs | density | kmax | stored | dense equivalent | transient peak |
|---|---|---|---|---|---|---|---|
| 500 | 1.8 ms | 17.8k | 7.1% | 52 | 8 MB | 117 MB | 104 MB |
| 1000 | 5.4 ms | 70k | 7.0% | 93 | 34 MB | 469 MB | 242 MB |
| 1767 | 14.6 ms | 219k | 7.0% | 169 | 105 MB | 1.5 GB | 519 MB |
| 2504 | 28-51 ms | 439k | 7.0% | 247 | 209 MB | 2.9 GB | 863 MB |
| 3933 | 65 ms | 1.08M | 7.0% | 383 | 515 MB | 7.3 GB | 1.75 GB |

Cost grows about as n^1.8. The non-zero pattern is exactly the channel-overlap pattern (median support 13 channels). It
doesn't change while templates keep their channels, so in-place updates need no restructuring.

### 2.2 Updating k templates [M]

Rows and columns of k templates, written in place into the sparse form, ms:

| k | n = 1767, as `prepare_matching` | restricted to overlapping channels and templates | n = 3933, as stock | restricted |
|---|---|---|---|---|
| 1 | 0.5 | 1.05 | 0.8 | 0.8 |
| 10 | 0.5 | 1.1 | 0.7 | 1.0 |
| 100 | 1.0 | 2.0 | 2.1 | 2.4 |
| 300 | 2.6 | 4.4 | 5.6 | 3.4 |
| 1000 | 8.4 | 14.2 | 17 | 9.8 |
| n | 14.6 | 25.3 | 66 | 38.6 |

- Cost is linear in k: about 8 µs per template at n = 1767.
- The restricted version only helps above 3000 templates. The dense einsum at this size is too fast to be worth beating.
- The symmetry `ctc[u,y,l] = ctc[y,u,-l]` holds to 1e-7 of the maximum, so columns can be filled by flipping rows, though
  not bit-identically with `prepare_matching`.
- **Partial updates save little in practice.** In a 2 s batch, 1534 of 1716 templates have at least one spike and 560
  have at least ten; over 10 batches, 1713 and 1652 **[M]**. Updating "only what fired" is close to a full rebuild.

### 2.3 What KS2.5 does [read from code]

KS2.5 has no `ctc` in its matching at all. `mexMPnu8.cu` subtracts each spike from the data and re-filters only around
it (`spaceFilterUpdate`, `timeFilterUpdate`), with `UtU` a boolean channel-overlap mask rebuilt every batch
(`getMeUtU.m`). This is matrix-free, which is why it tolerates an every-batch dictionary. Per batch it also runs an SVD
of each template (`mexSVDsmall2`), and every 5 batches `triageTemplates2` compares all pairs (`getMeWtW2`). The full
`getMeWtW` runs once, at the end of learning.

The measured rate on kobe's 20-min excerpt is 0.39-0.42 s per 2.19 s batch at 900-1070 units, and learning ran 3270
batches out of 549 distinct ones, about **6 passes**, 1290 s of a 1963 s sort **[M, log]**. KS4's online step here is
0.21 s at 1795 templates. So a KS4-native online loop would be about twice as fast per batch as KS2.5's, but that isn't
so much faster per batch than a whole KS4 round: a round is 0.39 s per batch, an online step 0.19-0.21.

### 2.4 Affordable update frequency [M for the parts, E for the sum]

Overhead on 172 ms per batch at 1767 templates, for updating dictionary and ctc together every R batches: ctc 16 ms,
running-mean update 1.8 ms (4 ms with a deterministic sum, section 5).

| R | n = 1767 | n = 2504 (ctc 28-50 ms) | n = 3933 (ctc 65 ms) |
|---|---|---|---|
| 1 | +18-20 ms, +11% | +14-26% | +24% |
| 10 | +3.4-5.6 ms, +2-3% | +2-4% | +3-4% |
| 100 | +2-4 ms, +1-2% | +1-2% | +1-2% |

Measured end to end: the online path with a per-batch update and rebuild ran at 0.212 s per batch, including feature
extraction and logging of the path, against 0.172 plain. **Every batch is affordable; every 10 is nearly free.**

### 2.5 Staleness [M]

`staleness.py`: a reference path evolves the dictionary over 120 consecutive batches (running mean of each template
towards its spikes' features with momentum m spikes, as KS2.5's `updateTemplates`, ctc rebuilt every batch). Test batches
are then rematched with the same `U_t` but the `ctc` of `U_(t-R)`. `sensitivity.py` is the yardstick: the same comparison
with both dictionaries fresh and consistent. Mean over 11-14 test batches, about 20,000 spikes per batch.

Spikes (of about 20,000) that differ from the fresh match, lost / extra, and residual power relative to fresh:

| R | m = 400, stale ctc | residual | m = 100, stale ctc | residual | yardstick: older dictionary, consistent ctc (m = 400) |
|---|---|---|---|---|---|
| 1 | 214 / 209 | 1.0000 | 539 / 537 | 1.0001 | 337 / 314 |
| 5 | 376 / 372 | 1.0000 | 865 / 892 | 1.0003 | |
| 10 | 494 / 507 | 1.0001 | 1154 / 1240 | 1.0007 | 796 / 817 |
| 20 | 619 / 695 | 1.0003 | 1752 / 1817 | **2.3e4** | |
| 50 | 1214 / 1557 | 1.022 | 5142 / 4063 | **6e29** | 1943 / 1872 |
| 100 | 7395 / 5383 | **7e19** | 13719 / 7153 | **3e30** | 3331 / 3085 |

Same-template ghosts among the extras stay small while stable: 6-21 per batch at m = 400 up to R = 20, 15-100 at m = 100.

Template change (median, 95th percentile, over the template): 2.3%, 5.8% per batch at m = 100; 0.6%, 1.6% at m = 400;
over 100 batches 20%, 48% and 10%, 25%. Self-subtraction error, |stale norm² / fresh norm² - 1|, 95th percentile:
3%, 12%, 24%, 61% at R = 1, 10, 50, 100 (m = 100); 0.9%, 3.8%, 15%, 35% (m = 400).

Reading this:

- **Small staleness is harmless.** Up to about R = 20 at m = 400, or R = 5 at m = 100, the error is smaller than the effect
  of a dictionary that old with a consistent `ctc`. Matching's output is sensitive to any change: 1.7% of spikes differ
  after one batch of updating, 4% after ten, 17% after a hundred.
- **Beyond that it does not degrade, it diverges.** After a subtraction the self-term is
  `B(1 - ctc_stale / nm_fresh)`. Once a template has shrunk enough for that factor to reach 1 in size, peels amplify
  rather than remove, and 100 peels at a factor of 2 give 2^100 = 1e30, which is the size measured. It takes only one such
  template, a rare one with a noisy running mean, to wreck a batch. Spike counts fall by 30% at R = 100, m = 100.
- **It is entirely avoidable.** If `U`, `nm` and `ctc` change together every R batches there is no inconsistency, only lag,
  and section 2.4 shows that costs 1-3%. Stale `ctc` should not be a design option. A debug assertion on residual
  growth would catch a violation.

A caveat on magnitude: the running mean is noisy (11 spikes per template per batch) and this path is a plausible update
rule, not a tuned one.

## 3. Clustering

### 3.1 Scaling from the sort logs [M]

`center_times.py`, from the timestamps of `Center i | Xd shape`. 96 centres in every pass.

| pass | spikes | median centre | largest | median s per centre | µs/spike overall | log-log slope |
|---|---|---|---|---|---|---|
| 20 min, first | 5.3M | 52k | 124k | 0.64 | 13.9 | 0.79 |
| 20 min, round 2 | 11.8M | 118k | 248k | 1.10 | 10.5 | 1.22 |
| 20 min, round 4 | 12.1M | 124k | 260k | 1.22 | 10.6 | 1.29 |
| 2 h, first | 31.2M | 311k | 697k | 5.17 | 16.0 | 1.07 |
| 2 h, round 2 | 68.5M | 707k | 1.38M | 9.45 | 13.5 | 0.94 |
| 2 h, round 4 | 69.9M | 710k | 1.52M | 9.50 | 13.3 | 0.97 |

At H8, `cluster_downsampling` = 10: at 20 min landmarks (about 12k per centre) stay below the 25k cap, so the kNN search
grows as n x n/10, a slope above 1. At 2 h every centre is capped, the search is linear, and the slope falls to 1 at a
higher rate. The top 10% of centres take 17-22% of the time, so no few centres dominate.

GPU allocated peak **[M]**: 4.4 GB (20 min) and 5.3 GB (2 h) in the first clustering, 3.7-4.2 GB at 2 h in later ones;
reserved 5.6 and 9.2 GB. Not a constraint.

### 3.2 Subsets [M, noisy]

Every k-th batch (D3) or a contiguous buffer (D1, D2), with the 2504-template round-2 spikes, `cluster_downsampling` 10
unless stated. "Idle equivalent" scales process CPU time by the calibration set's CPU time and idle time, and the range
comes from the calibration's own spread; the model is 3 s + 10.5 µs per spike.

| subset | spikes | clusters | wall under load, s | process CPU, s | idle equivalent, s | model, s |
|---|---|---|---|---|---|---|
| every 30th batch | 0.40M | 1226 | 60 | 51 | 4-7 | 7 |
| every 16th | 0.73M | 1409 | 36 | 55 | 4-8 | 11 |
| 30 batches, contiguous | 0.57M | 1341 | 76 | 69 | 5-10 | 9 |
| 60 batches, contiguous | 1.14M | 1498 | 55 | 78 | 6-11 | 15 |
| every 8th | 1.47M | 1600 | 146 | 149 | 11-21 | 18 |
| 120 batches, contiguous | 2.32M | 1690 | 183 | 196 | 15-27 | 27 |
| every 4th | 2.94M | 1745 | 203 | 237 | 18-33 | 34 |
| every 4th, `cluster_downsampling` 20 | 2.94M | 1634 | 137 | 191 | 14-26 | |
| all spikes (sort log, idle) | 11.8M | 1874 | 116-124 | | 120 | 127 |

The cleanest evidence of linearity is `cluster_bench.py`, which ran at a steady heavy load: 154, 151, 161 µs per spike at
all spikes, every 2nd and every 4th batch (1821, 891, 474 s). Scaled to the idle 120 s, every 4th batch is 31 s, in line
with the model, which sits at the upper end of the range above and so is on the conservative side for D3.

**`cluster_downsampling`.** At f = 1/4, going from 10 to 20 cuts process CPU by 20% (kNN 23 to 13 s, split 149 to 93 s
under load) and clusters by 6%. So clustering time isn't very sensitive to it. H8's 10 raises the kNN term's share only
while landmarks are under the cap; at 2 h every centre is capped at 25k and it matters less. At f < 1 the landmark count
falls with the subset, so holding it fixed would mean `cluster_downsampling` x f, which raises the cost (kNN grows as n x n/cds). GPU
peak in these subsets is under 0.2 GB. Memory doesn't limit a rolling buffer.

**How far to trust this.** The machine was saturated during this benchmark (CPU at 100%). Other jobs, and my own two
benchmarks running side by side, are the cause. The default-thread run of all spikes took 1821 s against 116-124 s idle,
15 times longer, with faiss's kNN search taking 1231 s of it (OpenMP threads spinning at barriers), though it reproduced
the sort's 1874 clusters. `cluster_bench2.py` limits threads and repeats a calibration set between runs, which still ranged
from 2.6 to 11.4 times idle. So the idle equivalents carry about ±50%, and wall times can't be compared across rows. The
stable result is the shape: linear in spikes with a floor of a few seconds, and no economy from small buffers. **This
should be re-measured on a quiet machine before any fine decision rests on it.**

`kmeans_plusplus` is about flat at 11-20 s across subsets under load: 96 centres x 200 iterations x a GPU round trip.
This is the floor. The tree split, on the CPU, took the largest share of every subset run under load (50-75%). That is
contention as well as work, so the shares shouldn't be carried over to an idle machine.

### 3.3 Rolling buffer [E, from 3.2]

Clustering L batches every N batches, over a whole pass, costs about (3 s + 10.5 µs x spikes in L) x (batches / N):

| buffer, period | at 20 min (600 batches) | relative to one full clustering, 120 s |
|---|---|---|
| 30 batches, every 30 | 20 x 9 s = 180 s | 1.5 |
| 30 batches, every 60 | 10 x 9 s = 90 s | 0.75 |
| 30 batches, every 120 | 5 x 9 s = 45 s | 0.4 |
| 120 batches, every 120 | 5 x 28 s = 140 s | 1.2 |

Clustering "every N batches" with L ≈ N is a full clustering per pass plus the floor, in a design that was meant to avoid
that. The buffer also sees less: cluster counts fall with the subset (1874 at all spikes, 1690 at 4 min of data, 1341 at
1 min), so a rolling clustering doesn't find what a global one does. Memory is not the issue: a 120-batch buffer is 570
MB of features.

## 4. Cost model

`cost_model.py`. Stage times are from section 1.1 **[M]**; online-step and subset costs are from the benchmarks; every
total is **[E]**. Uncertainties: extraction ±5%, subset clustering ±50% on its own term (±30 s per D3 run at 20 min),
birth and merge terms within a factor of 3, and **P, the number of passes, is not known**.

Model:
- e(n) = 74 ms + 54 µs n per batch.
- Sub-round at fraction f: f x batches x e(n) + 3 s + 10.5 µs x f N + 3.5 s for `postprocess_templates`.
- Online step: e(n) + 1.8 ms update + ctc/R + 7 ms birth detection (35 ms every 5th batch, from the universal detector) +
  15 ms for one merge per 100 batches. 0.191-0.206 s per batch at 20 min; 0.202-0.221 at 2 h.

| design | 20 min | below 4-round | 2 h | below 4-round |
|---|---|---|---|---|
| **baseline, 4 rounds** [M] | **14.8 min** | | **100.1 min** | |
| conventional, 2 rounds [M] | 6.5 | 56% | 48.0 | 52% |
| floor: dictionary for free [M sum] | 4.7 | 68% | 31.9 | 68% |
| D3, f = 1/2 | 10.7 | 28% | 67.8 | 32% |
| D3, f = 1/4 | 8.7 | 41% | 55.6 | 44% |
| D3, f = 1/6 | 8.0 | 46% | 51.5 | 49% |
| D3, f = 1/8 | 7.7 | 48% | 49.4 | 51% |
| D3s, f = 1/4, universal + first clustering subsampled too | 7.4 | 50% | 46.9 | 53% |
| D3s, f = 1/6 | 6.6 | 56% | 41.9 | 58% |
| D2, full seed + 1/2 online pass + final | 7.3 | 51% | 48.5 | 52% |
| D2, full seed + 1 online pass + final | 8.2 | 45% | 53.9 | 46% |
| D1, 1/2 pass, then final extraction + clustering | 5.7 | 62% | 38.0 | 62% |
| D1, 1 pass | 6.6 | 55% | 44.1 | 56% |
| D1, 2 passes | 8.6 | 42% | 56.3 | 44% |
| D1, 5 passes (near KS2.5's cold start) | 14.3 | 3% | 92.9 | 7% |
| D1, 1 pass, no final clustering (KS2.5's shape) | 4.5 | 70% | 28.6 | 71% |

Sensitivities:
- **Update interval** barely matters: R = 1, 10, 100 give 0.206, 0.192, 0.191 s per batch at 20 min.
- **Clustering rate** barely matters for D3: at 16 µs/spike, D3 f = 1/4 is 9.2 and 58.7 min, f = 1/6 is 8.4 and 53.6.
- **The number of passes dominates D1.** Between P = 1/2 and 5 it spans 5.7 to 14.3 min.

D3 at 2 h with f = 1/6 sees the same volume of data as the 20-min excerpt: about 600 batches and 11.5M spikes per
sub-round. That makes it the best-supported extrapolation here, since the sub-round cost is a 20-min round's cost.

**The most an interleaved design can save.**
- Against 4 rounds: 68%, with a dictionary that costs nothing.
- Against the conventional 2-round sort: 27% at 20 min, 34% at 2 h, under the same assumption. Of that, part is simply
  that the final extraction uses a better, smaller dictionary (133 to 106 s, 872 to 650 s).
- Realistically D1 at one pass costs the same as a 2-round sort. It removes universal extraction and first clustering (105 s,
  672 s) and adds a pass of about 115 s, 730 s, so it is not faster than today's 2-round sort unless a partial pass suffices.
- The peak RAM is set by the final extraction, merge and save, about 47 GB at 2 h **[M]**, and no design here changes
  it. GPU memory stays within today's range: `ctc` under 0.5 GB, peaks 4-5 GB.

**Does D3 get most of it?** On cost, yes: at f = 1/6 it is 49% below 4 rounds against a 68% ceiling, within 3.5 min of the
2-round sort at 2 h. On quality this audit has no evidence; see section 7. A specific risk is that a subset's dictionary
loses units too rare to form a cluster, and that subset clusters number fewer (1600 at f = 1/8, 1226 at f = 1/30, 1874 at
all spikes). 'augment' mode for the last dictionary would be a cheap insurance.

**Other designs worth having.**
- **D4, cheaper rounds, no design change.** `fast_matching` halves `xres`: -15 ms per batch, about 9% **[M]** (parity was
  checked in AUDIT_2). `colmax` is 79 of 172 ms: it rescans every template per column, though only overlapping templates
  changed (7% density). A kernel that only rescans when the column's current best is among them could cut much of this
  **[E, not measured, possibly 2x on that term; ties need care to remain bit-identical]**. Combined potentially 20-30% off
  every extraction, in every design, including the final one that sets the floor.
- **D5, polish instead of cluster.** A sub-round that re-estimates each template as the mean of its spikes, skipping
  clustering: f x batches x e(n), about 25 s at 20 min for f = 1/4. It is a batched version of KS2.5's update, without
  ordering, staleness or determinism issues. Whether a polish gives the gain depends on whether the gain comes from
  better means or from re-splitting; section 7 tests this.
- **Skip the first-pass duplication.** Drift and universal extraction are two passes of the same detector (33 + 32 s,
  220 + 174 s). Not an interleaving question, but as large as any term here.

## 5. Parallelism, determinism, resilience

- **Batch independence today.** `extract` holds `U` and `ctc` fixed, uses no RNG, and treats batches independently, so a
  batch's result does not depend on the order. Repeating a match with identical inputs gives identical spikes **[M]**
  (12 batches, 0 differences, residual ratio 1.0). KS4 doesn't use this for parallel batches on the GPU: there is one
  stream, with one prefetch thread. D3 keeps all of it.
- **Online loses it.** Each batch then depends on all before it. The read prefetch survives, since reading doesn't depend
  on the dictionary, but nothing can run out of order, and a result depends on order and update interval. KS2.5 shuffles
  batches to avoid tracking the end of a recording; doing that here costs 30 ms per batch unless the reader is told the
  next batch.
- **`index_add_` is not deterministic on the GPU [M].** 30 repeats of the per-template sum over one real batch gave 30
  distinct results, differing by 2e-7. torch documents this. Alone that is rounding, but section 2.5 shows a 1% template
  change alters 1-2% of spikes, and an online loop feeds each result to the next batch. A sum over spikes sorted by template
  (in double, by cumulative sum) gave 1 distinct result in 30, at 4 ms rather than 0.14 ms. Any per-batch update must use
  something like this, and should be tested by repeating a run. Today's rounds have no such exposure on this path: Wall is
  computed on the CPU, per centre.
- **Seeds.** `kmeans_plusplus` reseeds per centre with `cluster_init_seed`, and `get_nearest_centers` consumes torch RNG. A
  rolling clustering is reproducible if run at the same points, but it changes the global RNG state compared with the
  stock path, so seeding must be confined to the new code.
- **Bit-identical when off.** D3: with `batches=None` the loop in `extract` is unchanged. D1/D2: a separate function, which
  doesn't touch `run_matching` or the kernels. Check as for rounds: `same_sort.py` plus exact `score.npy` and `Wall.npy`
  against `kobe_H8_20min_s1` and `kobe_repeat_s1`.
- **Stage cache.** A new setting must be in `LATE` and **not** in `PASS2` in `stage_cache.py`, or a sort with a different
  fraction silently hits another's first full `extract`. Only the first `extract` is cached; with D3 that call is the
  subset. An online stage needs its own cached outputs (`U`, RNG state) and hooks; today's `install` wraps four functions.
  Editing `kilosort/*.py` invalidates all caches through `code_hash` (27 GB on disk), as for any code change.
- **Crash and resume.** KS4 has no resume, and the stage cache is tuning-only. D3 is as today. An online loop has state that
  could be checkpointed every 100 batches (`U`, counts, ages, batch index, RNG: about 100 MB), but this is new code to
  write and test, and a resumed run must match an unbroken one.
- **Windows allocator.** `alloc_test.py` **[M]**: 3000 repeated allocations of 8 or 1-30 MB left private bytes at 1.5 GB and
  flat; 2000 of 10-120 MB, 2.3 GB; 100 of 600 MB, 2.8 GB; 100 varied of 300-900 MB, 3.0 GB; and one growing 300 MB to 2 GB,
  4.1 GB, against numpy which returned to 3.0 GB. So repeated same-size buffers don't grow without limit: the allocator
  reuses them. Retention is a concern for growing sizes. A rolling buffer should be numpy or a fixed preallocation, which
  is what `extract` now does. D3 sub-rounds allocate less than today's rounds.

## 6. Engineering scope

**Kernels.** No kernel needs changing or rebuilding for any design. `ks_matching_*` hold no state: `U`, `nm` and the `ctc`
pointers are passed per call, and `nm` is recomputed per batch in `run_matching`. Limit: `kmax` under 65535, far above the
383 at 3933 templates. What must be kept in step with the dictionary in the Python: `iU`, `Ucc` (computed once in `extract`,
and `ops['iU']`, `ops['iCC']`, which clustering reads through `xy_templates`). A template's peak channel can move.

| design | files | lines (est.) | new behaviour | risk | tests |
|---|---|---|---|---|---|
| D3 | `template_matching.py`, `run_kilosort.py`, `parameters.py`, `stage_cache.py` | ~40 | `extract(..., batches=)`, a fraction per round | low | off is bit-identical; subset equals those batches of a full extraction exactly; drift index; cache key |
| D5 | D3 + one function | ~70 | template means by spike | low | as D3; means match `Wall` on a toy case |
| D2 | new `online.py`, `run_kilosort.py`, `parameters.py`, `stage_cache.py` | 200-300 | update loop, no birth | medium | off bit-identical; repeat-run identical; no divergence guard trips; checkpoint round trip |
| D1 | as D2 plus birth, death, merge | 500-800 | a second template learner beside clustering | high | as D2; births on toy data; merge consistency of `iU`, `Ucc`; sequence-order dependence; long-run memory |

**Traps in D3.**
- `bfile.set_downsampling(k)` must not be used after drift correction: `BinaryRWFile.padded_batch_to_torch` multiplies
  `ibatch` by the downsampling factor in its own local variable, but `BinaryFiltered.padded_batch_to_torch` then calls
  `filter(X, ops, ibatch)` with the original value, so `dshift[ibatch]` belongs to another batch (`io.py:893`, `io.py:1153-1160`).
  Stock is safe because `dshift` is then computed on the downsampled sequence. Pass an explicit batch list instead, or slice
  `dshift`.
- In `extract`, `t_shift` must be the true batch index, and the negative-time discard at `ibatch == 0` should stay on the
  true first batch.
- Correlograms with gaps are fine: in `CCG_metrics` T cancels in R12, and the lags are under 0.5 s of 2 s batches. The 250-500
  ms shoulder loses some pairs across batch edges, and tests lose power with fewer spikes.
- `cluster_downsampling` at f < 1 halves the landmarks again, so it may need to be scaled by f to mean the same.

**Smallest experiments that distinguish the designs.** Each uses the same tail (final extraction, clustering, merge), so
they differ only in the dictionary reaching it. About 25 lines, in a copy of the worktree, on kobe 20 min with H8, using the
stage cache for rounds 1-2:

| arm | dictionary into the final round | added cost |
|---|---|---|
| B, D3 | two rounds at f = 1/4 | ~2 min |
| C, D3 | two rounds at f = 1/8 | ~1 min |
| D, D5 | two polishes at f = 1/4, no clustering | ~1 min |
| E, online | one pass, updates only, no birth or death, R = 10, sorted sum | ~2.5 min |
| reference | existing rounds 2, 3 and 4 | |

Each sort is about 6-8 min, and scoring reuses `score_round3.py`, `tradeoff_round3.py` and `matched_units_round3.py`. With
3 seeds: about 1.5 h unattended. If B matches round 4 on S and the trade-off curves, D3 is the design and D1 has no case on
speed. If only D or E gains, the gain lies in refined means and clustering isn't needed mid-way. If neither D nor E gains,
updates alone don't do what the rounds do, and D1 would be relying on birth, death and merge, the costly parts, to do it.

## 7. Open, and what could change these numbers

- **Quality:** none of this says D3, D2 or D1 preserves the rounds' gain.
- **Clustering at small subsets** was measured on a saturated machine (section 3.2).
- **P**, the number of online passes, is unmeasured. KS2.5's 6 passes are a cold start with annealed momentum. The number
  needed for a warm start could be well under 1 for common units (400 spikes in about 40 batches at 5.5 Hz) but it stays
  above any 20-minute pass for units of 0.1 Hz, which is also where D3's subset is weakest.
- **Birth, merge costs in D1** are estimates by analogy with the universal detector and `postprocess_templates`.
- `colmax` savings are unmeasured.
- One recording, one seed for the timings. MEC's 2-hour logs look the same (rounds 722-746 s extraction, 872-907 s
  clustering) with more clusters.

Scripts: `common.py`, `batch_breakdown.py`, `peel_breakdown.py`, `ctc_bench.py`, `staleness.py`, `sensitivity.py`,
`center_times.py`, `cluster_bench.py` (default threads, slowed about 15 times by load), `cluster_bench2.py`, `io_order.py`, `alloc_test.py`,
`update_determinism.py`, `cost_model.py`.
