# Search round 2: early splitting, kept clean

Plan for round 2, frozen before its first run. Builds on [evaluation_spec.md](evaluation_spec.md), whose score, guards,
information barrier and budget rules apply unchanged, on [round1_results.md](round1_results.md), and
on the static audit `resolution_audit/KS4_clustering_resolution_audit.md` (2026-10-31).

## 1. Aim

Round 1's largest gains came from splitting earlier in the pipeline: the `standard` preset
(+178.6 bits/s over three seeds), bimodality 0 with γ 2.0 (+181.8) and with γ 1.0 (+155.7), about
three times the leader's +55.5 (final merge 0.7). They also produce more duplicate spikes, which
cleaning removes before scoring.

Round 2 looks for configurations that keep the early-split gain and pull those quantities back
inside their limits with a margin. Round 1 shows two levers:

- **A stricter first merge lowers duplicates.** First merge 0.95–0.975 removes 1.08–1.10% of spikes
  in cleaning, against 1.18% for default and 1.34% for bimodality 0. The first merge decides how
  many distinct templates KS4 matches with; with more of them, fewer residuals are fitted twice.
  This is why `standard` (which includes first merge 0.975) removes 1.18–1.27% where bimodality 0
  alone removes 1.34%.
- **A stricter final merge adds clean gain.** Final merge 0.85–0.95 gains +102 to +110 alone and
  adds no duplicates.

It also tests:
- **the graph-clustering settings** (neighbours, downsampling) with the merge tree relaxed, since
  in round 1 the tree re-merged most of what they produced;
- **three fixes to the merge tree** found by the audit (§2.1). With bimodality 0, as in all three
  high-gain bases, the tree in pass 2 merges only through its refractoriness test, so how that test
  and the tree walk behave now decides most of what is merged before the final merge.

## 2. Preparation

### 2.1 New KS4 settings

All default to stock behaviour.

| Setting | Default | What it changes |
|---|---|---|
| `tree_refractory_significance` | False | The tree's refractoriness test (pass 2) counts a pair as cross-refractory if R12 < 0.25 and either its dip is significant (Q12 < 0.05) or the pair is sparse (Q00 < 0.25, an expected count under 0.25 per 1 ms bin, true whenever n1·n2 < 250·T). The sparse shortcut skips the significance test, so small sub-cluster pairs deep in the tree are declared refractory by chance and collapsed. True requires the significance test. (Audit A2. The test is live since 549ec32; the `extra-settings` work predates it.) |
| `tree_walk` | `'top_down'` | Stock: the tree is walked from the root, and a node that merges takes its whole subtree with it, so no split below it is tested. The tree is ordered by connectivity, not size, so a node can join a fragment of one unit with another unit before the first unit's fragments are joined, and a merge there swallows everything beneath. `'bottom_up'`: nodes are decided in the order the tree formed them, and a node may merge only if each of its two children is by then a single cluster; the tests at each node are unchanged. (Audit A4.) |
| `cluster_min_group_spikes` | 1000 | A clustering group with fewer spikes than this becomes one cluster, in both passes, without clustering. (Audit C1.) |

### 2.2 Bit-identity check (one GPU sort)

A seed-1 default sort with the three settings passed explicitly at their defaults must reproduce
`hybrid_default_seed1` exactly, as in round 1.

**Result:** identical (`runs/check_round2_seed1`). The diagnostic sort (§2.3, `standard` preset)
is identical to round 1's seed-1 `standard` sort as well.

### 2.3 Diagnostic (one GPU sort, behaviour-blind)

A hook in a seed-1 sort with the `standard` preset records, per clustering group and pass:
- spikes in the group, including groups below the 1000-spike floor (how many groups and spikes the
  floor turns into single clusters);
- per tree node: the test that decided it (connectivity, refractoriness with a significant dip,
  refractoriness through the sparse shortcut only, bimodality), the two sides' spike counts, and
  for merges, the number of leaves the collapse removed without testing;
- clusters after the tree under each fix, re-run on the same clustering: significance only,
  bottom-up only, both, and floor 200.

**Rule, set before any run is scored:** a fix whose diagnostic changes post-tree clusters by less
than 2% in both passes is dropped from the screen (as `cluster_init_clusters` was in round 1).

**Results** (`tuning/round2_diagnostic.py`; `runs/round2_tree_diagnostic.json`; 96 groups per pass,
`standard` preset, seed 1):

| | Pass 1 | Pass 2 |
|---|---|---|
| Leaves into the tree | 1591 | 1493 |
| Clusters after the stock tree | 1576 | 1254 |
| Node decisions | 487 connectivity (kept apart), 993 bimodality kept apart, 6 merged | 608 connectivity, 550 bimodality kept apart, 4 merged; 157 refractory merges |
| Refractory merges through the sparse shortcut only | — | 4 of 157 (median sides 471 and 214 spikes) |
| Nodes removed untested by collapses | 9 | 78 |
| Refractory significance | +0.0% | +0.3% |
| Bottom-up walk | +0.8% | **+3.5%** |
| Both | +0.8% | +3.8% |
| Groups below 1000 spikes | 0 | 0 |

**Decisions:** `tree_walk` `'bottom_up'` stays (R14; R16 pairs it with the best γ configuration).
`tree_refractory_significance` (R13) and `cluster_min_group_spikes` (R15) are dropped: on this
data the sparse shortcut decides 4 of 157 merges, and no group falls under the floor.

### 2.4 Running faster: stage cache and overlap

- **Stage cache** (`tuning/stage_cache.py`, used by `run_sort.py --cache`; KS4 itself is not
  changed). Stage A (drift correction and the first detection pass, ~1 min) depends on no round 2
  setting and is reused by every run with the same seed. Stage B (first clustering pass and template
  matching, ~2 min) is reused by runs that differ only in second-pass or final-merge settings. Keys
  hash the KS4 code, data, probe and relevant settings. Checked: five cached sorts (cold, warm,
  stage A only, and second-pass-only changes, for default and `standard` settings) are bit-identical
  to full sorts; a fully cached sort takes ~1.1 min instead of ~4.2.
- **Overlap:** the next run's sort starts as soon as the previous run's sort finishes, while its
  harness evaluation (CPU only) completes.

### 2.5 Freezing

Commit the code, this plan and its diagnostic results, then re-freeze the harness.

## 3. Screen (seeds 1–3)

Every configuration runs on seeds 1, 2 and 3, so each gets a three-seed answer and no selection
rests on a single seed.

Bases: **STD** = `standard` preset (bimodality 0, shape criterion, first merge 0.975);
**G2** = bimodality 0 + γ 2.0; **G1** = bimodality 0 + γ 1.0. **F1** = first merge 0.975;
**F2(x)** = final merge x.

| Run | Configuration | Question |
|---|---|---|
| R1 | STD + F2(0.85) | Does a stricter final merge add to `standard`? |
| R2 | STD + F2(0.95) | Same, further |
| R3 | G2 + F1 | Does the stricter first merge reduce γ 2.0's duplicates? |
| R4 | G2 + F1 + F2(0.95) | Both levers on γ 2.0 |
| R5 | G2 + shape criterion + F1 (= STD + γ 2.0) | γ on top of `standard` |
| R6 | G1 + F1 | As R3, for γ 1.0 |
| R7 | G1 + F1 + F2(0.95) | As R4, for γ 1.0 |
| R8 | STD + γ 1.0 | As R5, for γ 1.0 |
| R9 | Tree connectivity 1.45 + F1 + F2(0.95) | A gentler early split (+93 alone, cleanly) with both levers |
| R10 | STD + `cluster_neighbors` 5 | Graph settings with the tree relaxed |
| R11 | STD + `cluster_neighbors` 20 | |
| R12 | STD + `cluster_downsampling` 10 | |
| R13 | (dropped by §2.3) STD + `tree_refractory_significance` | |
| R14 | STD + `tree_walk` `'bottom_up'` | Does deciding the tree bottom-up keep more real units? |
| R15 | (dropped by §2.3) STD + `cluster_min_group_spikes` 200 | |
| R16 | Best γ configuration of R3–R8 + `tree_walk` `'bottom_up'` | Run after R1–R14 |
| | **Total** | **14 configurations × 3 seeds** |

R1–R14 run first, seed by seed; R16 is chosen from their results by mean ΔS among configurations passing every guard on every seed,
and logged with its rationale before it runs.

## 4. Leader and refinement

- **Leader:** starts as final merge 0.7 (mean ΔS +55.5). A configuration becomes leader if its
  3-seed mean ΔS exceeds the current leader's by ≥ δ/2 (14.3), it passes every guard on every
  seed, its mean ΔS is ≥ δ (28.6) and its mean ΔS_track is ≥ δ_track/2 (13.6).
- **Near misses:** a configuration whose mean ΔS beats the leader but fails a guard on one seed is
  reported with the guard's values on all three seeds, so it is clear whether it fails by a margin
  or by seed-to-seed scatter at the limit.
- **Refinement:** further configurations that adjust one setting of the best ones (for example the
  final-merge level, or γ between 1.0 and 2.0), as many as needed to settle each question the screen
  leaves open, each on three seeds and logged with its rationale before it runs.

## 5. Budget

No fixed cap: runs are spent until each question is settled. Expected: 42 screen runs plus
refinement, about 3 GPU-hours with the stage cache and overlap. Runs start only when the GPU is
idle.

## 6. End of round 2

Report the leader with its replicates, a table of every run (ΔS, ΔS_track, eligible units, hybrid
recovery, cleaning statistics, guard failures), and proposals for the
next step. Then stop for discussion.

Candidates from the audit left for a later round: feature width (`nearest_chans`), explicit
`x_centers` and `dmin`, the residual used for features after re-peeling, and re-checking the
autocorrelogram after chained final merges.
