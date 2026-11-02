# KS4 performance audit (2026-10-31)

Target: default KS4 sort of `ap_1668-2868s.bin` (20 min, 385 ch, 27.7 GB) on the RTX 5070 Ti, torch 2.8.0+cu128.
Code base: commit `0e07514` plus the `random_seed` patch (diff hash `af5015…`, same as the reference sorts).

**Status:** all patches except concurrency are ported into KS4 on branch `perf-optimizations` (worktree
`C:\code\github\Kilosort-perf`, uncommitted as of writing). Plots are skipped with `run_kilosort(..., make_plots=False)`;
the fused kernels load from `kilosort/cuda/kilosort_kernels.dll` (build with `kilosort/cuda/build.bat`) and fall back
to PyTorch code with identical results when it's absent. A seed-1 sort on the branch is bit-identical to the
`det,h2d,tm,race,noplots` run below (393 s). `ks_fast.py` remains for reference.

## Result

| | wall time per sort | 100-run budget |
|---|---|---|
| KS4 as is | 16.7 min (≈1000 s) | ≈28 h |
| All patches below, one sort at a time | 6.9 min (414 s) | ≈11.5 h |
| All patches, two sorts at once | 5.5 min effective (659 s per pair) | ≈9.2 h |

With `det` + `race`, a sort is **bit-for-bit reproducible**: two sorts of the excerpt with the same seed give identical
`spike_times`, `spike_clusters`, `amplitudes`, `templates`, `spike_positions` and `kept_spikes`. Stock KS4 is not: two
seed-1 sorts gave 799 vs 811 units.

## Where the time went (stock KS4, instrumented run)

| stage | s | GPU util | what dominates |
|---|---|---|---|
| drift correction | 219 | 94% | `spikedetect.template_match` (320 ms/batch) |
| spike detection (universal) | 220 | 94% | same function, second pass |
| first clustering | 53 | 32% | `kmeans_plusplus`, faiss kNN (CPU), cluster assignment loop |
| spike detection (learned) | 275 | 84% | `template_matching.run_matching` (420 ms/batch, always 100 peels) |
| final clustering | 86 | 41% | as first clustering |
| merge + Phy export | 45 | ~10% | CCGs, `make_pc_features` (writes 2 GB `pc_features.npy`) |
| plots | ≈160 | ~8% | `plot_spike_positions` 110 s, `plot_drift_scatter` 48 s |

Disk was not a factor (the excerpt sits in the OS page cache; reads ≈ 0 MB/s during passes).

## Patches (`ks_fast.py`, monkeypatches; no KS4 file is edited)

| patch | change | measured | equivalence |
|---|---|---|---|
| `tm` | fused CUDA kernel for `template_match`: no 10× gathered intermediate; neighbour max only where a detection is possible | 323 → 26 ms/batch (×12.6); both detection passes 440 s → 85 s | bit-identical, 30/30 batches |
| `rm` | `run_matching`: recompute `relu(B)²/nm` only on columns the peel changed; fused B-update and column-max kernels | 422 → 181 ms/batch (×2.3); pass 275 s → 142 s | bit-identical, 30/30 batches |
| `race` | `rm` plus: when tied peaks put two spikes of one update group within 2·nt samples, use more groups so their writes don't overlap | no measurable cost | differs from KS4 only in the 7 of 600 batches where KS4 itself races (9 overlapping writes in 58,735 peels) |
| `det` | single-threaded KMeans in `extract_wPCA_wTEMP` | +1 s per call (2 calls/sort) | multithreaded KMeans gave a different `wTEMP` in each of 3 processes; single-threaded gave identical |
| `h2d` | copy int16 batch to GPU contiguous, transpose there | ≈1.2 ms/batch (≈2 s/sort); the 15 → 3 ms microbenchmark was wrong, because `np.asarray` of a memmap slice touches no pages, so the page faults were charged to the copy being timed (see AUDIT_2.md, DET-1) | bit-identical |
| `noplots` | skip the four PNG plots | ≈160 s/sort | outputs untouched; your harness reads none of the PNGs |

Use: `import ks_fast; ks_fast.apply(['det', 'h2d', 'tm', 'race', 'noplots'])` before `run_kilosort`. The DLLs are built
for sm_120 (this GPU); rebuild with `build.bat` (VS 2019 + CUDA 12.9).

### Caveats

- **Bit-identity of `tm` is shape-dependent.** The kernel reproduces cuBLAS's summation order for KS4's default shapes
  (`nearest_chans=10`, `template_sizes=5`, `n_templates=6`, specialised code path). Other values use a generic path whose
  results can differ from KS4 by ~1 ulp in `amp`. Re-run `tools/bench_tm.py` if the search changes those settings.
- **The race fix changes KS4's output** in the racing cases (where KS4's output is itself arbitrary). Spike times were
  unchanged on the excerpt; ≈0.01% of cluster labels moved, the same size as stock KS4's run-to-run variation.
- **Provenance:** monkeypatches are invisible to `kilosort_diff_sha256` in `run_sort.py`. Either port the changes into
  `kilosort/` (the diff hash then covers them) or record `ks_fast.py`'s and the DLLs' hashes in `provenance.json`.

## Implications for the evaluation spec

- §12's "GPU runs are not bit-reproducible" stops being true with `det` + `race`. Seed replicates then measure only the
  seed's effect, and any rerun can be checked for exact agreement.
- The reference sort and Phase 0 sorts made so far used stock KS4. If the patches are adopted, do it before the spec is
  frozen and redo those sorts (7 min each now).

## Not done (measured or estimated, ranked)

1. **`run_matching` is now CPU-bound** (≈200 ms CPU vs ≈107 ms GPU per batch: 100 peels × ~20 small launches + 2 syncs).
   Moving the peel loop into one or two kernels could take the 142 s pass to ~40–60 s.
2. **Clustering (≈140 s/sort) is launch/sync-bound.** `scatter_add_` versions of `assign_iclust`/`assign_isub` are exact
   and 1.1–2× faster (≈15–20 s/sort). Dense `index_put_` and the sync-free `kmeans_plusplus` were slower or ≤15% faster
   (kept in `ks_fast` as `clu`, not recommended). faiss kNN on CPU takes 30 s; a GPU kNN would be faster but not bit-identical.
3. **Skip `pc_features.npy`** (2 GB, ≈14 s compute + write per sort) if Phy isn't used on search runs. Needs a change in
   `save_to_phy`; also saves ≈200 GB of disk over 100 runs.
4. **Reuse drift correction + universal detection across runs** that share upstream settings: now worth only ≈85 s/sort,
   so lower priority than before the kernels.
5. **Three concurrent sorts:** untested. Each sort peaks at ≈16 GB RSS on a 63 GB machine, and the 27.7 GB page cache would
   start to be evicted.

## Files

- `ks_fast.py`, `fused_tm.cu`, `fused_match.cu`, `build.bat`, `*.dll`: the patches.
- `tools/profile_run.py`: instrumented sort (per-function timers, GPU/CPU traces, `--patches`, `--snapshot`).
- `tools/compare_outputs.py RUN_A RUN_B`: bit-identity check of two result folders.
- `tools/bench_*.py`: per-batch equivalence and speed checks on a `--snapshot` run.
- Run folders from the audit (≈2.5 GB each) are in the session scratchpad, not here.
