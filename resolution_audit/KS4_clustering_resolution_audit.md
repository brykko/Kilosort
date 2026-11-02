# KS4 clustering resolution: static audit (2026-10-31)

Code audited: branch `ks4-tuning` at `c7ceaca`. Static read only; nothing was run. Five parallel audits (core graph
clustering, split/merge decisions, features feeding clustering, settings wiring, failure modes), with the main claims
re-checked against the code. Line numbers are for `c7ceaca`.

## How the cluster count is decided

Each spatial group (a 2·`dmin` slab in y × an `x_centers` column; one column on NP1) is clustered on its own, in
both passes (`clustering_qr.run`):

1. **Seeds.** kmeans++ picks 200 seeds (`nclust`, hardcoded).
2. **Graph stage.** Alternating label propagation on a bipartite kNN graph runs between spikes and landmark spikes
   (every `cluster_downsampling`-th spike, capped at `max_cluster_subset`). A cluster label can disappear here but is
   never created, so the clusters that survive ("leaves") are a **hard ceiling** for the group.
3. **Tree.** `hierarchical.maketree` merges the leaves greedily, ranked by observed/expected edge ratio (`crat`).
4. **Splitter.** `swarmsplitter.split` walks the tree top-down. At each node it decides "keep split" (−1) or
   "collapse" (+1). A collapsed node takes its whole subtree with it.
5. **Final merge.** After pass 2, `merging_function` runs a CCG-based merge across groups.

Pass 1 (`mode='spikes'`) groups spikes by universal-template position. Pass 2 (`mode='template'`) groups them by the
learned template's peak channel and re-clusters from scratch. The two passes take the same settings. The one
difference in tests: **the refractory veto in step 4 runs only in pass 2.**

On your data, units are lost in pass 2 and in the final merge, not in pass 1. In `hybrid_default_seed1`:

| stage | units |
|---|---|
| Pass 1 | 1137 |
| Pass 2 | 932 |
| Final merge | 849 |

## Why no exposed setting reliably raises resolution

The decisions that set the count are almost all hardcoded:
- the graph-stage resolution (`lam`)
- every splitter test (0.2 / refractory veto / 0.6)
- the small-group floor
- both merge-similarity thresholds

The exposed clustering settings (`cluster_neighbors`, `cluster_downsampling`, `max_cluster_subset`) mainly change
how many leaves reach the tree. The splitter's tests then merge them back. Two exposed settings (`acg_threshold`,
`cluster_neighbors`) work against intuition: raising them **reduces** resolution.

---

## Tier A: most promising (code changes, not reachable through settings)

### A1. Graph-stage resolution `lam`: hardcoded at `lam=1`
**Location:** `clustering_qr.py:511`.

**Mechanism.** The null-model penalty in `assign_iclust` (:69-86) and `assign_isub` (:89-105) uses unweighted
cluster *counts* (`tones = ones`), not degree sums. Compared with standard bipartite modularity (γ = 1):
- In `assign_isub` the effective γ is `lam/n_neigh` = **0.1**.
- In `assign_iclust` it is `lam/(n_neigh·nskip_eff)` ≈ **0.005**. The penalty there is at most `lam/nskip` = 0.05,
  against integer neighbour counts of 0–10. That step is in practice a pure majority vote.

So label propagation lets big clusters swallow neighbours joined by a modest number of bridge edges. Once two units
share a leaf, nothing downstream can separate them.

**Direction.** Raise it:
- `lam ≈ n_neigh` (10) gives γ ≈ 1 in the isub step.
- `lam ≳ n_neigh·nskip_eff` (~200 at defaults) is needed before the iclust step stops being a majority vote.
- A cleaner fix replaces the counts with degree sums (true modularity) plus an explicit γ.

**Cost.** More fragments of single units. The tree and final merge can recombine those, but only if the splitter
lets them (see A2–A4).

**Confidence: high** that this is the main resolution parameter of the objective and that it is set very low.

Upstream's `hierarchical.prepare(..., lam=1)` takes `lam` and ignores it. The earlier audit on
`origin/claude/audit-clustering-algorithms-Om7Re` says "lam ↑ slightly" and "lam=0 = pure modularity". Both are
wrong: `lam=0` removes the null model entirely.

### A2. Refractory veto on splits (pass 2 only): hardcoded
**Location:** `swarmsplitter.py:66-67, 113`.
```
cross_refractory = R12<.25 and (Q12<.05 or Q00<.25)
```
When this is true the node collapses.

**The `Q00<.25` clause skips the significance test for sparse pairs.** Q00 is the baseline CCG count per 1 ms bin
≈ n1·n2·0.001/T, so the clause is true whenever n1·n2 < 250·T[s]:
- That is 300k for your 20-min excerpt, e.g. 550 × 550 spikes, or 1000 × 300.
- At such counts R12 (a minimum over ±1…10 ms windows) is often exactly 0.
- So sparse sub-cluster pairs are declared "cross-refractory" by chance and never split.

This bites deep in the tree, where sub-clusters are small. A collapse there also takes the subtree with it (A4).

**History.** This veto was dead code in released 4.1.5–4.1.7 (`len(st2 == 0)` bug). Commit `549ec32`, which is
in upstream/main and this branch, re-enabled it. **This checkout under-splits more than 4.1.7 does.** Any
experience from mid-2025, such as the `extra-settings` work, came from a code base without this veto.

**Settings have no effect here.** The splitter ignores the exposed `ccg_threshold` (it reaches only the final merge).
The `Q00` clause is commented out in `CCG.check_CCG` but is still live in this copy of the function.

**Direction:** any of the following.
- Drop `or Q00<.25`.
- Require a minimum expected count before the veto can fire.
- Pass `meta=None` at `clustering_qr.py:525` to turn the veto off.

**Confidence: high** that it collapses sparse pairs. How often it fires is unmeasured.

### A3. Bimodality test: hardcoded threshold 0.6, plus a small-unit bias
**Location:** `swarmsplitter.py:120`; `check_split` :11-29; `bimod_score` :40-51.

A node splits only if the dip in the regression projection is below 40% of the *smaller* peak. The trough search
is limited to bins 175–225 (projection −0.25…0.25), on a histogram of 400 bins over [−2, 2] smoothed with σ = 4 bins.

**Small-unit bias.** The regression is class-balanced, but the histogram is not. For a size ratio r, the big unit's
tail at the dip must fall below ≈ 0.4·r of its own peak, so the separation needed grows with imbalance:

| r | separation needed |
|---|---|
| 1 | ≈ 1.35σ |
| 0.1 | ≈ 2.2σ |
| 0.01 | ≈ 3.2σ |

These are rough analytic figures. The result: small or low-rate units next to big ones are merged.

**Direction:**
- Lower the threshold (0.4–0.5).
- Weight the histogram by class, or normalise each class's peak.
- Widen the trough window.

The `extra-settings` branch exposed this threshold as `merge_bimodality_threshold`; it is not merged into
`ks4-tuning`.

**Minor:** `Xs[:,-1] = 1` (:17) overwrites one real feature with the intercept.

**Confidence: high.** This is the gate for every node with crat ≥ 0.2.

### A4. Top-down subtree collapse
**Location:** `swarmsplitter.py:133-135`, `clean_tree` :31-38.

A "collapse" verdict at a node removes every node below it, so no lower split is ever tested. The tree is ordered
by a ratio that ignores size (`crat`). It can therefore pair a fragment of unit A with unit B before joining A's
fragments, and a failed test at that mixed node merges everything beneath it. If the root fails, the whole group
becomes one cluster (`new_clusters`, :145-146).

**Direction:** test children of a collapsed node independently, or decide bottom-up.

**Confidence: medium-high** (structural). Its size depends on how often high nodes fail A2/A3.

### A5. Auto-split ratio 0.2: hardcoded
**Location:** `swarmsplitter.py:107`.

`tstat[kk,0] < 0.2` (observed/expected edges) splits unconditionally and skips A2 and A3.

**Direction:** raise it (0.3–0.5).
- **Cost:** fragmenting drifting or bursting units.
- **Note:** the earlier audit doc gets the direction of this one backwards in its prose.

**Confidence: medium.**

## Tier B: exposed settings that really act on resolution

| setting (default) | what it actually does | direction for more units | notes |
|---|---|---|---|
| `cluster_downsampling` (20) | Landmark stride. Landmark floor ≈ `n_neigh × nskip_eff` spikes per unit per group (~200); smaller units can't form their own neighbourhood and are absorbed in the graph stage. Also dilutes the iclust penalty (A1) | Lower (5, or 1) | Memory/time. Description is stale ("all spikes are used" — default is 20) |
| `max_cluster_subset` (25000) | Caps landmarks per group. Once it binds, `nskip_eff` grows with recording length, so **long recordings have lower resolution** | Raise / None | Memory. Description says it adjusts `cluster_downsampling`; code drops evenly spaced landmarks (same effect) |
| `cluster_neighbors` (10) | kNN graph degree. Raising it *lowers* effective γ (`lam/n_neigh`) and adds bridge edges | **Lower** (5–8) | Noisier graph, low-SNR fragmentation. Described as resource-only |
| `acg_threshold` (0.2) | Only units passing the ACG test can act as merge seeds in the final merge (`template_matching.py:336-339, 361`). Raising it creates more seeds, so more merges | **Lower** (e.g. 0.1) | Also sets saved "good" labels. **Works against intuition.** Can't disable merging (`exclude: [0]`) |
| `ccg_threshold` (0.25) | Final-merge cross-CCG test only (`CCG.py:91`, with hardcoded `Q12<.05`). Not used in splits, despite the description | Lower | Merge stage only |
| `nearest_chans` (10) | Feature width: each spike's features are the 10 channels around its template peak × `n_pcs`. Also the detection neighbourhood | Raise (14–20)? | Uncertain direction: extra channels add noise dimensions to L2 |
| `n_pcs` (6) | Temporal basis size. **Effectively fixed at 6 unless `templates_from_data=True`** (must match stored `wPCA`) | Raise (8–12) with `templates_from_data` | Basis learned from isolated peak-channel clips only |
| `dmin` (auto, pitch) | Also sets the clustering group height (2·dmin, `clustering_qr.py:407`), not just the detection grid | Smaller | More groups hit the 1000-spike floor (C1); boundary splits |
| `x_centers` (auto) | Auto mode uses a hardcoded 50 µm bin → one column on NP1 | Set explicitly | As `dmin` |
| `nblocks` (1) | Rigid drift correction; non-rigid residual stretches units into chains that the kNN graph bridges | Raise | Only if drift is non-rigid |
| `Th_universal` (9) | Units not detected in pass 1 get no template | Lower | Noise spikes; indirect |

## Tier C: secondary hardcoded items

| item | location | value | direction | confidence |
|---|---|---|---|---|
| Small-group collapse: group with < 1000 spikes becomes **one cluster**, both passes | `clustering_qr.py:500` | 1000 | Lower (100–300) | High that it happens; impact depends on spikes per group (log at `:494`) |
| Final-merge similarity gate; with no amplitude check in `'ccg'` mode, the CCG test is in practice the only gate | `template_matching.py:321`, call `run_kilosort.py:946` | `r_thresh=0.5` | Raise (0.7–0.9) | Medium |
| Pre-matching template merge (`'mu'`): cosine > 0.9 and norm within 20%, no CCG check | `template_matching.py:147-150, :396` | 0.9 / 0.2 | Raise / tighten / skip | Low-medium (pass 2 re-clusters from features, but channel set and group are inherited) |
| Merge chaining: seed retries after each merge; seed ACG never re-checked | `template_matching.py:398-421` | — | Re-check merged ACG | Medium |
| Re-peel residual: a spike's features use the residual after *all* peels, so a second peel at the same time can subtract the part that distinguishes an absorbed unit | `template_matching.py:84-86` | design | Add back overlapping peels when extracting features | Medium mechanism, low size |
| kmeans++ seeds / iterations / candidates | `clustering_qr.py:121-122, 217` | 200 / 200 / 100 | Raise `nclust` | Low (rarely binds) |
| Drift-interpolation ridge: spatial low-pass even at zero shift (~20% cut on highest spatial mode) | `datashift.py:223` | 0.01 | Smaller (1e-3) | Low-medium |
| CCG windows: baseline from 10–50 / 250–500 ms; central bin only zeroed | `CCG.py:40-57` | — | — | Low |
| No feature normalisation: plain L2 on raw PC features; peak-channel PCs dominate | `get_data_cpu`, `neigh_mat` | — | Per-channel scaling | Speculative |

## Misdescribed or dead settings (worth knowing before testing)

- **`dminx`:** no effect on clustering. `run()` always passes `ix`, so `get_data_cpu` never reads `dmin`/`dminx`
  (`clustering_qr.py:591-595`). It affects only the detection grid.
- **`max_channel_distance`:** its description says it limits channel distance during clustering. No code does this.
- **`ccg_threshold`:** described as used for splits and merges. It is used for merges only.
- **`acg_threshold`:** its description omits that it gates merge seeds.
- **`n_pcs`:** described as used only with `templates_from_data`. In fact it is always used, and must equal 6
  without that setting.
- **docs:** say correlograms ignore "the central 1 ms". The code zeroes only the 0-lag bin.
- **Unreachable code:** the 4th splitter test ("local modularity > .15", `swarmsplitter.py:122-125`), `lam = 20`
  in `run_matching`, and the `lam` argument of `hierarchical.prepare`.

## Context found along the way

- **Local `extra-settings` branch** (6d15002, June 2025). It exposes `merge_bimodality_threshold`,
  `merge_correlation_threshold_1/2` and `correlogram_shape_criterion`, and is not merged into `ks4-tuning`.
  It was written when the A2 veto was dead code.
  - **If that branch's bimodality = 0 didn't raise unit counts**, every node with crat ≥ 0.2 was kept as a split
    back then. That points the bottleneck at the **graph stage** (A1, leaf count) or the feature space, not the
    splitter.
- **`runs/search_log.jsonl` doesn't exist yet.** No non-default setting has been scored in the harness.
  - `evaluation_spec.md` §4 allows code changes under `kilosort/`; they are logged as diffs.
  - The Phase 0 perturbation shows the score can detect artificial merges (S ≈ −44.6).

## Recommended next steps (not done; outside this static audit)

### Diagnostics (one instrumented default sort)

Log per spatial group, per pass:
1. spikes in the group (to see whether the 1000 floor binds);
2. leaves out of `cluster()` vs clusters after `new_clusters` (graph-stage vs tree-stage loss);
3. per tree node: n1, n2, `tstat[kk,0]`, the veto result with Q00/R12/Q12, bimodality score, and whether the
   node was decided or wiped by `clean_tree`.

These logs show which of A1–A4 removes the most units.

### Suggested test order (after diagnostics)

1. `lam` ∈ {3, 10, 30}. Optionally degree-weighted null + γ.
2. Veto: drop `Q00<.25` (or `meta=None`).
3. Bimodality threshold ∈ {0.5, 0.4}, then the class-weighted histogram.
4. Non-collapsing `clean_tree`.
5. Auto-split ratio 0.2 → 0.35.
6. `cluster_downsampling` 5, `cluster_neighbors` 6, `acg_threshold` 0.1, `r_thresh` 0.8 (settings / one-liners).
7. Small-group floor 1000 → 200.

Interactions matter. Items 1 and 6 add leaves, and items 2–5 decide whether they survive. So a graph-stage change
alone may show little unless it is paired with a splitter change, which would also explain past null results.
