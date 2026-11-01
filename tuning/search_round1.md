# Search round 1: clustering resolution

Plan for round 1, frozen with spec v0.7. Builds on [evaluation_spec.md](evaluation_spec.md), whose
score, guards, information barrier and budget rules all apply unchanged except where this document
says otherwise. §2 records the preparation and its results; the levels in §3 were set from them
before any run was scored.

## 1. Aim

Out-of-the-box KS4 tends to under-separate MEC units. Round 1 asks which clustering and merging
settings raise resolution, i.e. separate units that default merges, without over-splitting. The
expected signature of a real gain is S rising (separating differently tuned cells is worth
−5.6σ_S when undone, §8.6) while G3 (suspected splits) and G4 (hybrid recovery, HD pairs merged)
stay within bounds. More units is the expected outcome, not a warning sign: G1 only guards against
losing units (spec v0.7). Over-splitting barely moves S (§8.6: −2.1 bits/s for 5% splits), so
G3 is expected to be the binding guard, not S.

Out of scope for round 1: detection thresholds, drift correction, template settings, KS4 code
changes beyond those in §2. These are candidates for round 2, to be discussed after round 1.

## 2. Preparation (not counted in the budget)

### 2.1 New settings in KS4

All default to stock behaviour. Ported from `extra-settings` by hand onto the current base, since
`merging_function`, `swarmsplitter` and `check_CCG` have changed since that branch (549ec32,
5e05b53).

| Setting | Default | Where | What it does |
|---|---|---|---|
| `correlogram_shape_criterion` | False | tree (template pass), global merges, `check_CCG` | Vetoes a refractoriness-based merge unless the pair's CCG has the shape of both ACGs (smoothed, ±50 ms, Pearson r > 0.5 on the better side, minimum over the two ACGs). Guards against anti-phase units whose broad CCG trough looks refractory. Threshold 0.5 fixed in round 1. From `extra-settings`. |
| `merge_bimodality_threshold` | 0.6 | `swarmsplitter.split` | The tree merges a pair unless its features project bimodally (score ≥ threshold). Lower = fewer merges; 0 disables bimodality merges. From `extra-settings`. |
| `merge_correlation_threshold_1` | 0.9 | `postprocess_templates` | First-pass template merge: waveform correlation above this and norms within 20%. Higher = fewer merges. From `extra-settings`. |
| `merge_correlation_threshold_2` | 0.5 | final `merging_function` | Final global merge: candidates need correlation above this and a cross-refractory CCG. Higher = fewer merges. Declared on `extra-settings` but not wired; wired here. |
| `tree_connectivity_threshold` | 0.2 | `swarmsplitter.split` | New. The tree keeps a pair apart without further tests when its graph connectivity is below this multiple of chance (`tstat[:,0]`, hard-coded 0.2). Higher = more pairs kept apart. |
| `cluster_init_clusters` | 200 | `clustering_qr.cluster` | New. Number of k-means++ seeds per clustering centre. Label propagation can only merge clusters, so this is a ceiling on clusters per centre (in practice it doesn't bind, §2.3). |
| `cluster_resolution` | None | `clustering_qr.cluster` | New. None keeps the stock penalty (λ = 1, counting subset nodes: a fixed tie-breaker, not a dial, §2.3). A number γ switches both assignment steps to the degree-weighted modularity penalty γ·kᵢ·K_c/m, where K_c is the summed degree of the cluster's nodes on the other side of the graph. γ = 1 is standard modularity; larger γ penalises big clusters more. |

The GUI names copied wrongly on `extra-settings` are corrected in the port. The `standard` preset
(`merge_bimodality_threshold` 0, `correlogram_shape_criterion` True,
`merge_correlation_threshold_1` 0.975) is ported with `get_preset`.

### 2.2 Bit-identity check (one GPU sort)

A seed-1 sort of the hybrid excerpt with every new setting passed explicitly at its default must
reproduce `hybrid_default_seed1` exactly: spike times, cluster labels, templates, amplitudes. This
also covers the commits since the default replicates were made (`e28143c` → current HEAD, each
claimed bit-identical). If it fails, the default replicates are re-run before freezing.

**Result:** identical (`runs/check_port_seed1`, 3.9 min): spike times, clusters, templates,
amplitudes, positions, whitening and KS labels all equal `hybrid_default_seed1`, so the five
default replicates stand. Both diagnostic sorts (§2.3) are identical to it as well.

`fast_matching` is fixed at False (exact path) and added to the fixed settings.

### 2.3 Clustering diagnostic (one GPU sort, behaviour-blind)

A hook in a single default sort (seed 1) re-runs the clustering of every fourth clustering centre,
in both passes, under alternative settings, and records results without changing the sort itself.
It reads no tracking data or scores, so it sits outside the information barrier.

Per centre and variant: clusters after `cluster()`, clusters after the tree, and agreement with
stock labels (adjusted Rand index). Variants:

- stock λ = 0 and λ = 1 (is the stock penalty inert?);
- `cluster_resolution` γ ∈ {0.25, 0.5, 1, 2, 4, 8};
- `cluster_init_clusters` 400 at stock λ and at γ = 1;
- the distribution of tree connectivity (`tstat[:,0]`) at nodes the default tree merges.

The diagnostic fixes the round 1 levels before any run is scored:

- **γ:** the three values giving about +15%, +30% and +60% clusters after the tree, relative to
  stock, interpolated on the grid. If no γ raises the post-tree count by 15% (the tree re-merges
  what γ separates), γ is screened with `merge_bimodality_threshold` 0 instead of at defaults.
- **`tree_connectivity_threshold`:** the 25th and 50th percentiles of connectivity among nodes the
  default tree merges.

**Results** (`tuning/cluster_diagnostic.py`; `runs/round1_cluster_diagnostic.json` and
`..._bimod0.json`; 48 centres, 24 per pass). Clusters summed over centres:

| Variant | Pass 1 (spikes): after `cluster()` | after tree | Pass 2 (templates): after `cluster()` | after tree |
|---|---|---|---|---|
| stock (λ = 1) | 394 | 292 | 313 | 241 |
| λ = 0 | −12% | −5% | −9% | −4% |
| γ = 1 | +24% | −1% | +28% | +0% |
| γ = 2 | +118% | −1% | +245% | −0% |
| γ = 8 | +655% | +13% | +1030% | −3% |
| `cluster_init_clusters` 400 | +1% | +0% | +2% | +0% |
| `merge_bimodality_threshold` 0, stock clustering | | +35% | | +13% |
| bimodality 0, γ = 1 | | +58% | | +22% |
| bimodality 0, γ = 2 | | +147% | | +42% |
| bimodality 0, γ = 4 | | +313% | | +109% |

- **The merge tree sets the resolution, not the graph clustering.** At default bimodality the
  tree re-merges almost everything extra the clustering produces, at any γ. With bimodality
  merges off, γ passes through.
- **The stock λ is a tie-breaker:** λ = 0 vs 1 changes post-tree counts by 4–5%; on synthetic
  blobs λ = 1, 3 and 10 gave the same or nearly the same clustering.
- **The 200 seeds never bind:** at most 29 clusters per centre survive `cluster()`, and 400
  seeds change post-tree counts by 0–2%, also with γ = 1 (−3% to +3% against γ = 1 alone).
- **Tree merges are few and large:** 17 merge decisions across the 48 centres (in pass 1 they
  remove a quarter of the clusters). Connectivity at those decisions: 0.21–13.5, 25th/50th
  percentiles 0.98 and 1.44. Of all tree nodes, 35% (pass 1) and 49% (pass 2) are below the stock
  0.2 and are kept apart untested.

**Levels set by the rules above:**
- No γ reaches +15% after the tree at default bimodality, so γ is screened with
  `merge_bimodality_threshold` 0, and its targets are relative to bimodality 0 alone (pooled over
  both passes: γ = 1 gives +14%, γ = 2 +60%, γ = 4 +157%). Interpolated: **γ = 1.0, 1.35, 2.0**.
- **`tree_connectivity_threshold` = 1.0 and 1.45.**
- `cluster_init_clusters` is dropped from the screen: it has no effect to measure.

### 2.4 Spec v0.7 and freezing

- §4 lists the new settings and `fast_matching = False`; §9 points to this document.
- `freeze` also hashes the KS4 package sources (`kilosort/*.py` and the kernel library), so KS4
  code can't change during the round unnoticed. This document joins the hashed files.
- Commit, then `run_candidate.py freeze`.

## 3. Screen (seed 1, one setting at a time from default)

| Run | Setting | Default | Level(s) | Runs |
|---|---|---|---|---|
| S1 | `standard` preset | — | as §2.1 | 1 |
| S2 | `correlogram_shape_criterion` | False | True | 1 |
| S3 | `merge_bimodality_threshold` | 0.6 | 0.4, 0.2, 0 | 3 |
| S4 | `tree_connectivity_threshold` | 0.2 | 1.0, 1.45 (§2.3) | 2 |
| S5 | `merge_correlation_threshold_1` | 0.9 | 0.95, 0.975 | 2 |
| S6 | `merge_correlation_threshold_2` | 0.5 | 0.7, 0.85 | 2 |
| S7 | `ccg_threshold` | 0.25 | 0.15 | 1 |
| S8 | `acg_threshold` | 0.2 | 0.1 (fewer units may seed global merges; KS4's own note suggests 0.1) | 1 |
| S9 | `cluster_resolution`, with `merge_bimodality_threshold` 0 | None | 1.0, 1.35, 2.0 (§2.3); compared with S3's bimodality-0 run | 3 |
| S10 | `cluster_neighbors` | 10 | 5, 20 | 2 |
| S11 | `cluster_downsampling` (+ `max_cluster_subset`) | 20 (25,000) | 10; 10 with 50,000 | 2 |
| | | | **Total** | **20** |

`x_centers` is left out: the probe's columns span about 50 µm, so splitting clustering centres
horizontally isn't expected to change resolution.

**Range extension.** If a setting's best level is the end of its range, gains ≥ δ/2 and passes
every guard, one further step in the same direction is run. At most 4 such runs.

## 4. Combination

- **Leader:** the best configuration replicated on 3 seeds (seed 1 plus 2 and 3), starting with
  default. A configuration becomes leader if its 3-seed mean ΔS exceeds the current leader's by
  ≥ δ/2, it passes every guard on every replicate, and its mean ΔS over default is ≥ δ (§5.4).
- **Replication trigger:** a single run is replicated when its ΔS exceeds the leader's mean by
  ≥ δ/2 and it passes every guard. This applies §9.1 to round 1: single runs never become leader.
- **Carried forward:** each setting's best screen level, if it gained ≥ δ/2 over default and
  passed every guard.
- **Greedy addition:** starting from the best replicated screen result, add the carried-forward
  settings one at a time, in order of screen ΔS. Keep an addition if it becomes leader (above);
  otherwise drop it. Settings that act on the same step (S3/S4, S5/S6, S7/S8) are tried as the
  pair's better member only, unless both were carried forward.
- **Cap:** at most 10 combination runs, excluding replicates.

## 5. Budget and running

- Round 1 uses at most **44** of the 85 search runs: 20 screen, ≤ 4 extensions, ≤ 10
  combinations, and replicates (2 per trigger, ≤ 10). The rest stay for round 2. The 15 reserved
  runs (§9.2–9.3) are untouched.
- Runs go one at a time, about 5.5 minutes each (some clustering settings will be slower), so
  about 4 hours of GPU for the whole round. Each run starts only when the GPU is idle
  (`nvidia-smi`), so it doesn't compete with other KS4 work.
- Every run goes through `run_candidate.py run` with a one-line rationale that names its row in
  §3 or step in §4. The optimising agent reads only `summary.json` and the log's result records.

## 6. End of round 1

Report to you, then stop for discussion before round 2:

- a table of every run: settings, ΔS, S_track, eligible units, suspected splits, hybrid recovery
  and merged pairs, guard failures;
- the leader, with its replicates;
- which settings moved unit counts without moving S (over-splitting caught by G3, or not);
- proposals for round 2.

Selection (§9.2) and ablation (§9.3) happen only after the last round.

## 7. Points for you

- **Budget split:** 44 runs for round 1, 41 for later rounds.

Settled: G1's upper bound (1.25 × default) is dropped (spec v0.7), since more units is the aim.
