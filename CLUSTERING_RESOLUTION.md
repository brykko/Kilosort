# Tuning Clustering Resolution in Kilosort4

This document audits the mechanics of the clustering algorithms in
Kilosort4 and explains, in detail, every knob that influences the
*resolution* of the final clustering, i.e. how aggressively spikes are
split into separate units versus collapsed into one. It is intended as a
practical guide for users who want to push KS4 toward more or fewer
units, and as an internal reference for developers.

The relevant source files are:

| File | Role |
|---|---|
| `kilosort/spikedetect.py` | Universal-template spike detection (first pass) |
| `kilosort/clustering_qr.py` | Bipartite graph clustering, called twice |
| `kilosort/hierarchical.py` | Builds the binary merge tree from the graph |
| `kilosort/swarmsplitter.py` | Walks the tree to decide which merges to keep |
| `kilosort/template_matching.py` | Template-matching extract + global merging |
| `kilosort/CCG.py` | Cross/auto-correlogram metrics used for merge gating |
| `kilosort/parameters.py` | All user-facing settings + presets |
| `kilosort/run_kilosort.py` | Orchestration of the pipeline stages |

Default values cited below come from `kilosort/parameters.py`.

---

## 1. Pipeline overview

The clustering output is shaped by **six** sequential stages. Each stage
adds, splits, or merges units, so the resolution of the final result is
the cumulative effect of all of them. The stages, in order, are:

1. **Universal-template spike detection**
   (`spikedetect.run`, called from `run_kilosort.detect_spikes`,
   `run_kilosort.py:768`).
2. **First clustering** of those detections via the bipartite/QR
   algorithm (`clustering_qr.run` with `mode='spikes'`,
   `run_kilosort.py:793`).
3. **First global merge** on the resulting templates
   (`template_matching.postprocess_templates` →
   `template_matching.merging_function` in `'mu'` mode,
   `run_kilosort.py:798`).
4. **Learned-template spike re-detection** (matching pursuit) with the
   merged templates (`template_matching.extract`,
   `run_kilosort.py:819`).
5. **Final clustering** (`clustering_qr.run` with `mode='template'`,
   `run_kilosort.py:885`).
6. **Final global merge** on the final templates
   (`template_matching.merging_function` in `'ccg'` mode,
   `run_kilosort.py:906`).

A given spike's cluster identity at the end is therefore the product of
two clustering passes and two global merge passes, which interact in
non-trivial ways. The single biggest take-away is that there is **no
single "resolution" knob** — there is a *clustering knob*, a *splitter
knob*, and several *merge knobs*, and the right one to turn depends on
what you want to change.

---

## 2. Stage-by-stage mechanics

### 2.1 Universal-template detection

`spikedetect.run` (`spikedetect.py:196`) lays a regular grid of "universal"
template centers across the probe, controlled by `dmin` and `dminx`
(`template_centers`, `spikedetect.py:97`). At each location it convolves
the data with a small bank of `template_sizes` Gaussian-weighted spatial
kernels of standard deviation `min_template_size, 2·min_template_size,
3·min_template_size, …` and a set of `n_templates` short temporal
templates (`wTEMP`, learned from data via k-means in
`extract_wPCA_wTEMP`, `spikedetect.py:50`). Local maxima above
`Th_universal` become detections, and `n_pcs` PC features per
`nearest_chans` neighboring channels are saved to `tF`.

This stage does not assign cluster IDs, but it gates how many spikes
ever reach the clusterer and how spatially fine-grained the input is.
Resolution-relevant settings:

- `dmin` — vertical spacing of universal templates. Defaults to the
  median row spacing of the probe. Smaller `dmin` → more candidate
  detection sites → potentially more, smaller clusters downstream.
- `dminx` — horizontal spacing of universal templates (default 32 μm).
- `min_template_size`, `template_sizes` — sets the spatial scale and
  number of scales of the Gaussian envelopes. More/smaller templates
  pick up tighter spike footprints.
- `Th_universal` (default 9), `Th_single_ch` (default 6) — detection
  thresholds. Lower thresholds detect more spikes (and more noise).
- `nearest_chans` (default 10) — how many channels of features each
  spike carries. Affects feature dimensionality used for clustering.
- `max_channel_distance` (default 32 μm) — universal templates whose
  nearest channel is farther than this are dropped, which limits how
  far the clusterer can reach.

### 2.2 First clustering (`clustering_qr.run`, mode `'spikes'`)

`clustering_qr.run` (`clustering_qr.py:409`) is the heart of KS4
clustering. It is run **twice** with the same code path; only the input
spike features and the centering scheme differ.

The probe is tiled with a coarse grid of cluster centers
(`x_centers(ops)` and `y_centers(ops)`, `clustering_qr.py:331`/`380`).
For each x-y center, every spike whose universal template is closest to
that center is fed into `cluster()` as a feature matrix `Xd`
(`get_data_cpu`, `clustering_qr.py:555`). This is the spatial
"divide-and-conquer" — clustering never crosses center boundaries.

Inside `cluster()` (`clustering_qr.py:115`):

1. **Build a sparse neighbor graph** (`neigh_mat`, line 20). Take every
   `nskip`-th spike as a "landmark" subset. For every spike, find its
   `n_neigh` nearest landmarks via FAISS L2 search. The result is a
   `n_spikes × n_landmarks` adjacency matrix `M`.
2. **Initialize with kmeans++** (`kmeans_plusplus`, line 177). Seed
   `nclust=200` centroids by sampling spikes proportional to residual
   variance.
3. **Iterate bipartite re-assignment** for `niter=200` iterations
   (lines 155–161): given `iclust`, recompute landmark labels `isub`;
   given `isub`, recompute `iclust`. The penalty term `lam=1` discourages
   clusters from absorbing too much mass (it's a softmax-like
   regularizer expressed via `Mstats`). After convergence, every spike
   has one of up to 200 cluster IDs within that center.

The four hard-coded values inside `cluster()` —
`nclust=200`, `niter=200`, `lam=1`, and the call signature
`max_sub=np.inf` — are *the* most powerful resolution levers in the
codebase, but they are not currently exposed as user settings. See
§3.4 below for how to override them safely.

After convergence `cluster()` returns the per-spike `iclust`, the
landmark-level `isub`, and the adjacency matrix `M`. These are then
passed to the merge tree builder.

### 2.3 Building the binary merge tree (`hierarchical.maketree`)

`hierarchical.maketree` (`hierarchical.py:93`) takes `M` and
`iclust`/`iclust0` and computes a "modularity matrix"

```
crat = (cc + cc^T) / (cneg + cneg^T)
```

where `cc` is the within/between-cluster co-occurrence count and
`cneg` is its expected value under a null model
(`prepare`, `hierarchical.py:22`; `Mstats`, line 14). It then greedily
merges the pair of remaining clusters with the highest `crat` until
only one cluster remains, recording at each step:

- `xtree[t] = (child_x, child_y, parent_id)` — the merge.
- `tstat[t] = (lam, ntot, M)` — the global modularity score, total mass
  of the merged group, and a "local modularity" `M = cpos_l/cneg_l`
  used as a final tie-breaker.

This is purely deterministic given `M` and `iclust`; it produces an
*over-split* tree of `nclust` leaves and `nclust-1` internal nodes that
the splitter then prunes.

### 2.4 Pruning the tree (`swarmsplitter.split`)

`swarmsplitter.split` (`swarmsplitter.py:96`) is the function that
controls **how aggressively the over-split tree is collapsed back into
final clusters**. For each internal node it answers a single question:
*should this merge be kept (combine the two children) or rejected
(leave them separate)?* The decision is made by trying four criteria,
in order, and stopping at the first that returns a non-zero verdict:

1. **Global modularity rejection** (line 115).
   `if tstat[kk,0] < 0.2: criterion = -1` → reject the merge. This is
   a hard gate: anything with global modularity below 0.2 is **never**
   merged. Tightens (`>0.2`) → fewer, larger clusters; loosens (`<0.2`)
   → more, smaller clusters.
2. **CCG refractoriness override** (line 121, `refractoriness` →
   `check_CCG`). When meta data (spike times) is supplied — i.e. on the
   second clustering pass with `mode='template'` — a *cross-refractory*
   CCG between the two children forces merging (`criterion = 1`).
   Thresholds here (`R12<.25`, `Q12<.05`, `Q00<.25`) are **hard-coded**
   in `swarmsplitter.check_CCG` (line 57).
3. **Bimodality of regression projection** (line 124,
   `check_split` → `bimod_score`). A weighted least-squares projection
   onto a 1-D axis that best separates the two sub-clusters is
   computed; the histogram dip is scored from 0 (clean separation) to
   1 (no separation). The cluster pair is merged iff
   `score < merge_bimodality_threshold`. **This is the primary
   user-facing splitter knob.** Setting it to `0` rejects all
   splits-via-bimodality (everything stays merged unless 1 or 2 fired);
   setting it to `1` accepts every merge.
4. **Local modularity tie-break** (line 134). If none of the above
   fired (which is rare in practice — the comment notes "not
   reachable"), `criterion = score > 0.15` on the local modularity.

Important: after a merge is rejected, `clean_tree` walks down the
sub-tree of *both* children and rejects every merge below them too.
This is irreversible — once two leaves are decided to live in separate
clusters, the splitter never re-merges them in this pass.

### 2.5 First global merge (`merging_function` in `'mu'` mode)

`template_matching.postprocess_templates` (`template_matching.py:143`)
runs after the first clustering. It calls
`merging_function(..., r_thresh=merge_correlation_threshold_1,
mode='mu', check_dt=False)`.

In `'mu'` mode the merge condition is:

```
templates A and B are merged if
    waveform_correlation(A, B) > merge_correlation_threshold_1   and
    abs(2 * (||A|| - ||B||) / (||A|| + ||B||)) < 0.2
```

i.e. very-similar normalized waveforms whose template norms differ by
less than 20%. This step is *not* informed by the CCG, so the 20% norm
gate is the only protection against over-merging. The default
`merge_correlation_threshold_1 = 0.9` is permissive and yields many
merges; raising it to ~0.97 (as in the `"standard"` preset) makes this
step very strict and preserves more units.

### 2.6 Second clustering (`clustering_qr.run`, mode `'template'`)

Identical machinery to §2.2, but the input spikes come from
`template_matching.extract` (matching pursuit on the merged
templates from the first pass). The grouping is now by the *learned*
templates' nearest center rather than universal-template centers, so
spike-to-center membership is more accurate. Spike times are passed
through to the splitter as `meta`, which enables the CCG-based
override in the splitter (criterion 2 above).

### 2.7 Final global merge (`merging_function` in `'ccg'` mode)

Run from `cluster_spikes` (`run_kilosort.py:906`) with
`r_thresh=merge_correlation_threshold_2, mode='ccg', check_dt=True`.

In `'ccg'` mode the merge condition is:

```
templates A and B are merged if
    waveform_correlation(A, B) > merge_correlation_threshold_2   and
    cross-CCG of (A, B) is "cross-refractory" per CCG.check_CCG
```

`CCG.check_CCG` (`CCG.py:81`) compares the central CCG dip to a
baseline using `acg_threshold` and `ccg_threshold`:

```
cross_refractory = (R12 < ccg_threshold) and (Q12 < 0.05)
```

So this stage will only merge pairs that *look like one neuron* both in
waveform and in refractory dip. The default
`merge_correlation_threshold_2 = 0.5` casts a wide net for *candidate*
pairs, but the CCG gate is strict. If `correlogram_shape_criterion` is
True, an additional Pearson correlation of the cross-CCG against each
auto-CCG must exceed 0.5 (`CCG.CCG_ACG_similarity`); this prevents
merging anti-phase-tuned pairs (e.g. grid cells) that happen to look
refractory in the CCG.

`check_dt=True` also realigns spike times by the inter-template lag
(`roll_features`, line 391) before merging, so two copies of the same
unit at different latencies still combine cleanly.

---

## 3. Adjusting clustering resolution: a practical guide

The settings below are listed in order of **expected impact on cluster
count**, largest first. Most users should start at the top.

### 3.1 The two global-merge correlation thresholds

These are the most direct user-level knobs.

#### `merge_correlation_threshold_1` (default 0.9, `'mu'` mode)

This is the threshold used by the first global merge step on
*templates only*, with a 20% norm gate but **no CCG check**. Because
the CCG isn't consulted, the acceptance rate of candidate merges is
high once they pass the correlation+norm gates, so this knob has an
outsized effect on the final unit count.

- **More units** (finer resolution): raise to 0.95–0.98. The
  `"standard"` preset uses 0.975 (`parameters.py:615`).
- **Fewer units** (coarser): lower toward 0.85.

#### `merge_correlation_threshold_2` (default 0.5, `'ccg'` mode)

The threshold for the *final* CCG-gated merge. Because the CCG check
is strict, this threshold mostly controls how many *candidate* pairs
are inspected, not how many actually merge.

- **More units**: raise toward 0.7. Effect is moderate.
- **Fewer units**: lower toward 0.3. Pairs in different anatomical
  regions can sneak in if their templates happen to correlate, so be
  cautious lowering this.

### 3.2 `merge_bimodality_threshold` (default 0.6)

Controls the splitter's behavior on the *over-split* tree from
`hierarchical.maketree` (§2.4, criterion 3). The bimodality score is
0 for clean separation and 1 for total overlap; the merge happens when
`score < threshold`.

- `0.0` → reject all bimodality-based merges. Used by the `"standard"`
  preset (`parameters.py:596`); leaves the raw KS4 over-split intact
  and relies on the global merge stages to find true duplicates. Use
  this when neural activity is dense (e.g. MEC) and you suspect the
  default merge step is fusing real neighboring units.
- `1.0` → accept all bimodality-based merges. Effectively disables
  this guard; gives the coarsest splitter output.
- Defaults around `0.6` are intermediate.

This knob only affects what comes out of *each* `clustering_qr.run`
call, before the global merge stages, so its effect is partly
attenuated by the subsequent merging.

### 3.3 `correlogram_shape_criterion` (default False)

When True, both `swarmsplitter.check_CCG` and
`template_matching.merging_function` add an extra requirement: the
cross-CCG of a candidate pair must look like the auto-CCGs of *both*
units (Pearson `r > 0.5` over ±50 ms). This:

- **Vetoes** merges in the splitter (criterion 2 in §2.4) when shapes
  don't match — i.e. produces *more* units in the second clustering.
- **Vetoes** merges in the final global merge — also produces *more*
  units.

Recommended for grid cells, head-direction cells, and any data with
genuinely anti-phase pairs. The `"standard"` preset enables it.

### 3.4 Hard-coded splitter thresholds (developer territory)

The splitter's first and fourth criteria use literal values that are
not exposed as settings:

- `swarmsplitter.split:115` — `tstat[kk,0] < 0.2` for global modularity.
  Increasing toward 0.3 reject more merges → finer clusters.
  Decreasing toward 0.1 accepts more merges → coarser clusters.
- `swarmsplitter.split:135` — `tstat[kk,-1] > 0.15` (rarely reached).
- `swarmsplitter.check_CCG:64` — `R12<.1`, `Q12<.2`, `Q00<.25`,
  `R12<.25`, `Q12<.05` for the CCG-forced-merge override. These are
  **stricter** than the user-tunable `acg_threshold`/`ccg_threshold`
  used elsewhere.

Editing these requires a code change but is the only way to tune the
splitter's first/fourth criteria.

### 3.5 Hard-coded clusterer parameters (developer territory)

Inside `clustering_qr.run` the call to `cluster()` (line 490) uses:

```python
iclust, iclust0, M, _ = cluster(
    Xd, nskip=nskip, n_neigh=n_neigh, max_sub=max_sub,
    lam=1, seed=5, device=device, verbose=v
)
```

`cluster()` itself defaults `nclust=200, niter=200, lam=0` (`clustering_qr.py:115`).
So at run time the per-center clusterer is hard-wired to:

- `nclust = 200` initial cluster centroids per center,
- `niter = 200` bipartite assignment iterations,
- `lam = 1` regularization weight.

Effects of changing these:

- **`nclust` ↑** → finer over-split → more leaves in the tree → more
  candidate splits for the splitter. Has a sub-linear effect on the
  final count because the splitter and global merge will collapse
  most of the extra clusters; but in dense regions (where the splitter
  refuses bimodality merges) it can substantially increase the number
  of detected units.
- **`niter` ↑** → more iterations of bipartite assignment. Past ~200
  iterations the assignments are essentially stable, so increasing
  beyond 200 has diminishing returns; *decreasing* below ~100 risks
  unstable assignments (clusters never settle).
- **`lam` ↑** → stronger penalty on cluster size; produces more, more
  evenly-sized clusters. Setting `lam = 0` reverts to a pure modularity
  criterion (more imbalanced clusters).
- **`seed`** is fixed at 5 for reproducibility; changing it just
  shuffles cluster IDs.

To experiment, the safest pattern is to add a setting (e.g.
`init_clusters_per_center`) in `parameters.py`, plumb it through
`ops['settings']` to the `cluster()` call at `clustering_qr.py:490`,
and leave the defaults backward-compatible.

### 3.6 Center-grid spacing: `dmin`, `dminx`, `x_centers`

The `(xcent, ycent)` grid in `clustering_qr.run` partitions templates
into independent clustering problems. Centers are placed at
`y = ymin+dmin-1, ymin+3*dmin-1, …` (`y_centers`,
`clustering_qr.py:380`) and at automatically-detected x-density peaks
(`x_centers`, line 331). Effects:

- **`dmin` ↓** → centers placed every `2*dmin` along y, so finer
  spacing → more, smaller groups → potentially more units (each group
  produces independent clusters, then global merge stitches them). For
  dense probes (NP1/NP2 the default `median(diff(yc))` is fine; for
  sparse arrays you may want to set `dmin` manually to roughly the
  vertical reach of a unit (~30–40 μm).
- **`dminx`** plays the same role on the x-axis but is also used by
  spike detection (§2.1) and in `get_data_cpu` to gate which templates
  feed the clusterer (`clustering_qr.py:570`). A larger `dminx` pulls
  in more templates per center, giving the clusterer more context but
  larger feature matrices.
- **`x_centers`** (default `None` → auto-detected from the channel
  density). For 2D arrays, set this manually to ~one center per
  ~150–300 μm of x extent.

Changing these does not directly change the merge rules, but it
changes how the clusterer carves the probe into sub-problems.

### 3.7 Graph-construction parameters

These shape the FAISS neighbor graph that drives the bipartite
assignment.

- **`cluster_neighbors`** (`n_neigh`, default 10). Each spike is
  connected to its `n_neigh` nearest landmarks. More neighbors →
  smoother graph → typically slightly fewer, more stable clusters.
  Affects runtime and memory roughly linearly.
- **`cluster_downsampling`** (`nskip`, default 20). Every `nskip`-th
  spike becomes a landmark. Smaller `nskip` → more landmarks → finer
  granularity, but much slower. Dropping below ~10 rarely helps.
- **`max_cluster_subset`** (default `None`). Caps the landmark count
  for very long recordings. Lower → fewer landmarks → coarser graph;
  higher (or `None`) → finer graph at the cost of memory.

### 3.8 Refractory thresholds: `acg_threshold`, `ccg_threshold`

Used by `CCG.refract` and `CCG.check_CCG` during the *final* global
merge. These do not affect cluster count directly via the splitter
(which uses its own hard-coded constants), but they influence merging
in `'ccg'` mode and the "good"/"refractory" labeling at save time.

- **Stricter merging** (more units): lower both thresholds toward 0.1
  / 0.15. The KS4 paper used 0.1 for `acg_threshold`.
- **Looser merging** (fewer units): raise toward 0.3 / 0.4.

---

## 4. Cookbook recipes

### 4.1 "I want more units" (e.g. dense cortical or MEC recordings)

Strongest to weakest effect, applied cumulatively:

1. `merge_correlation_threshold_1 = 0.975` — strictly limit the first
   merge stage (this single change is the bulk of the `"standard"`
   preset's effect).
2. `merge_bimodality_threshold = 0` — disable the splitter's
   bimodality-based merging entirely. Lets the over-split tree pass
   through to the global merge stages.
3. `correlogram_shape_criterion = True` — veto false-positive CCG
   merges from anti-phase pairs.
4. `merge_correlation_threshold_2 = 0.6` (mild) — narrows the candidate
   set in the final merge.
5. (Code change) Bump `nclust=200` → `400` in `clustering_qr.py:490`
   if you suspect the initial over-split is still under-resolving.

The first three are exactly what `parameters.get_preset("standard")`
does (`parameters.py:582`), so the easiest entry point is

```python
from kilosort.parameters import get_preset
settings = {**get_preset("standard"), 'n_chan_bin': N}
```

### 4.2 "I want fewer units" (e.g. high-noise or duplicates)

1. `merge_correlation_threshold_1 = 0.85` — let the first merge stage
   absorb more near-duplicates.
2. `merge_bimodality_threshold = 0.8` — splitter accepts most merges.
3. `acg_threshold = 0.3, ccg_threshold = 0.4` — be more permissive
   about calling a CCG "refractory", so more pairs merge in the final
   stage.
4. `merge_correlation_threshold_2 = 0.4` — opens the final-merge
   candidate net.

### 4.3 "I want to keep KS4 defaults but split anti-phase pairs"

Just set `correlogram_shape_criterion = True`. This is a safe,
near-zero-risk change for grid/HD-cell recordings.

### 4.4 "I want reproducible behavior across runs"

The clusterer already seeds `torch.manual_seed(seed=1)` and
`np.random.seed(1)` inside `kmeans_plusplus`, plus
`torch.cuda.manual_seed_all(1)` in `_sort` (`run_kilosort.py:269`).
The only non-determinism left is FAISS' brute-force search (which is
deterministic for `IndexFlatL2`) and CUDA reduction order. Changing
`x_centers`/`y_centers`-relevant parameters between runs is the most
common source of perceived non-determinism.

---

## 5. Quick reference table

| Setting | File / Line | Default | Direction for *more* units |
|---|---|---|---|
| `merge_correlation_threshold_1` | `parameters.py:443` | 0.9 | ↑ (e.g. 0.975) |
| `merge_correlation_threshold_2` | `parameters.py:466` | 0.5 | ↑ (e.g. 0.6) |
| `merge_bimodality_threshold` | `parameters.py:429` | 0.6 | ↓ (e.g. 0.0) |
| `correlogram_shape_criterion` | `parameters.py:477` | False | True |
| `acg_threshold` | `parameters.py:353` | 0.2 | ↓ (stricter) |
| `ccg_threshold` | `parameters.py:363` | 0.25 | ↓ (stricter) |
| `cluster_neighbors` | `parameters.py:373` | 10 | ↓ (modest effect) |
| `cluster_downsampling` | `parameters.py:385` | 20 | ↓ (modest, slower) |
| `max_cluster_subset` | `parameters.py:395` | None | ↑ |
| `dmin` | `parameters.py:228` | auto | ↓ (smaller groups) |
| `dminx` | `parameters.py:238` | 32 | ↓ (smaller groups) |
| `x_centers` | `parameters.py:416` | auto | ↑ (more centers) |
| `Th_universal` | `parameters.py:61` | 9 | ↓ (more spikes in) |
| `Th_learned` | `parameters.py:71` | 8 | ↓ (more spikes in) |
| `nclust` (hard-coded) | `clustering_qr.py:490` (call) / `:116` (default) | 200 | ↑ (code change) |
| `niter` (hard-coded) | same | 200 | ≥200 (no benefit ↑↑) |
| `lam` (hard-coded) | `clustering_qr.py:490` | 1 | ↑ slightly |
| Splitter modularity gate | `swarmsplitter.py:115` | `<0.2` | ↑ threshold |
| Splitter CCG refractory | `swarmsplitter.py:64` | `R12<.1`/`<.25`, etc. | ↓ thresholds |

---

## 6. Suggested workflow

1. Run KS4 with defaults.
2. Inspect the output in Phy. If you see *over-merging* (units that
   look like obvious mixtures), apply recipe 4.1 — try the
   `"standard"` preset first.
3. If you still see over-merging in specific cell types known to be
   anti-phase (grid, HD), enable `correlogram_shape_criterion`.
4. If you see *over-splitting* (one neuron broken into multiple
   clusters), apply recipe 4.2 incrementally: lower
   `merge_correlation_threshold_1` first, then
   `merge_bimodality_threshold`, then
   `acg_threshold`/`ccg_threshold`.
5. Only resort to changing the hard-coded `nclust`/`niter`/`lam` (or
   the splitter modularity constants) once the user-facing settings
   have plateaued. Those changes require editing
   `clustering_qr.py` / `swarmsplitter.py` and re-running the full
   pipeline.

A useful diagnostic: after a run, check the unit count after each
stage in the log file (`{results_dir}/kilosort4.log`):

- `"{N} clusters found"` after first clustering
  (`run_kilosort.py:808`)
- spikes extracted at re-detection (`run_kilosort.py:829`)
- `"{N} clusters found"` after final clustering
  (`run_kilosort.py:896`)
- `"{N} units found"` after the final merge (`run_kilosort.py:917`)

Watching how `N` changes across these four lines tells you which stage
is dominating your resolution and which knob to turn.
