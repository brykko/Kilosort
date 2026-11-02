# KS4 performance audit, round 2: multi-agent review (2026-10-31)

**Method.** Five area reviewers covered detection and drift (DET), learned matching (MATCH), clustering (CLU), merge and
export (POST), and run-level, environment and cross-run effects including the tuning harness (SYS). Each area had an
adversarial verifier, and a completeness critic (GAP) with its own verifier ran last: 12 agents in all. They worked from
an instrumented profile of the current code (branch `perf-optimizations`, 398 s per sort) and the stock baseline, and
were not allowed to use the GPU or edit files. Some ran small CPU-only checks.

The result is 72 findings, about 55 distinct after duplicates are merged. Verifiers confirmed 17, judged most of the
rest plausible, and refuted none outright. Full output: the session scratchpad's `audit_result.json`.

**Read the savings with care.** They are agent estimates from the profiles, not GPU measurements. Items in the same
stage overlap, so the savings don't simply add up.

Equivalence classes:
- **exact**: bit-identical output;
- **no-effect**: doesn't touch sorting output;
- **numeric**: float rounding changes;
- **behaviour**: the algorithm changes.

Effort is S (small), M (medium) or L (large).

## Implemented: Tier 1 (commit 5e05b53 on `perf-optimizations`)

CLU-1, CLU-2, DET-1 (with prefetch), SYS-2, POST-1 (`save_pc_features=False`), POST-3, POST-4, POST-5, SYS-6 and
DET-3. A full seed-1 sort is bit-identical in all 13 output files (pc_features included) to e28143c.

| | e28143c | 5e05b53 | 5e05b53, `save_pc_features=False` |
|---|---|---|---|
| wall time | 393.5 s | **284.8 s** | **273.1 s** |
| drift / universal | 42.8 / 42.1 s | 29.7 / 29.1 s | 29.6 / 29.0 s |
| first / final clustering | 49.8 / 78.6 s | 30.6 / 50.6 s | 31.3 / 47.9 s |
| learned matching | 135.7 s | 125.6 s | 125.4 s |
| merge / export | 16.3 / 26.7 s | 5.3 / 12.9 s | 5.4 / 3.6 s |

## Implemented: MATCH-1 (commit b011d7b on `perf-optimizations`)

The peel loop runs as a few fused kernels with one host sync per peel. The residual update keeps the
cuBLAS einsum, so output is bit-identical for any settings. Checked on full sorts at default,
`Th_learned=7`, `max_peels=50` and `nt=51`, and per batch at several `Th_learned` and `max_peels` values.

| setting | 5e05b53 | b011d7b |
|---|---|---|
| default | 273.1 s | **230.9 s** (learned matching 125 → 82 s) |
| `Th_learned=7` | 289.6 s | 246.8 s |
| `max_peels=50` | 218.4 s | 196.2 s |
| `nt=51` | 251.1 s | 216.3 s |

`fast_matching=True` (opt-in) computes the residual's template projection in a kernel: ≈8 s faster per sort (223 s
vs 231 s). Parity on the hybrid excerpt, seeds 1–3, scored with the current harness:
- identical: spike times, cluster labels, unit counts, `n_eligible`, S and S_track (|Δ|/seed SD = 0 on all);
- differing at float rounding (<1e-5): amplitudes, templates, spike positions, similar_templates.

The exact sorts were bit-identical to the default replicates 1–3. The kernel library now reports a version, and an
out-of-date build is refused (PyTorch fallback with a rebuild warning).

Not done in KS4: GAP-7 (recording thread counts). It belongs in `tuning/run_sort.py`'s provenance, on `ks4-tuning`.

## GPU-measured follow-up (same day, GPU otherwise idle)

| id | measurement | saving per sort | output |
|---|---|---|---|
| CLU-1 | `clustering_qr.run` on saved inputs: first 50.1 → 36.5 s, final 76.7 → 56.0 s; median 12 iterations to the fixed point (max 37) instead of 200 | **34 s** | clu and Wall bit-identical, both passes |
| CLU-2 | first 50.1 → 40.6 s, final 76.7 → 67.5 s | **19 s** | bit-identical, both passes |
| CLU-1 + CLU-2 | first 50.1 → 29.7 s, final 76.7 → 47.5 s | **50 s** | bit-identical, both passes |
| DET-1 | 600-batch read + filter + template_match loop: 34.8 s → 28.7 s with synchronous `readinto`, → 25.7 s with a one-batch-ahead prefetch thread; 3 such passes per sort | **≈27 s** with prefetch (≈18 s without) | X identical on every checked batch |
| SYS-2 | `np.save` of the real 2.06 GB pc_features: 9.5 s permuted → 1.0 s contiguous | **8.5 s** | same bytes (sha256) |

Scripts: `bench_tier1_clu.py`, `bench_det1.py` and `bench_sys2.py` in the session scratchpad's `perf` folder.

## Tier 1: exact and cheap, the best value (≈90-110 s per sort together)

| id | change | est. s/sort | class | effort | verdict |
|---|---|---|---|---|---|
| CLU-1 | Stop clustering's 200-iteration assign loop at an exact fixed point (`torch.equal(new, old)` → break). The loop is a deterministic pure function of `iclust`, and centers converge in 9-30 iterations | **33** | exact | S | confirmed |
| DET-1 (= SYS-1, MATCH-10) | Read batches with `readinto()` into pinned buffers instead of a fresh `np.memmap` per batch, and prefetch batch i+1 on a thread during GPU work on batch i. Page faults on the fresh memmap cost ~12 ms of the 18 ms/batch read, which is about the GPU's idle time in the detection passes. The no-thread variant gets about 2/3 of the saving | **28** | no-effect | M | plausible |
| POST-1 (= SYS-3) | Optional `save_pc_features=False` for search runs (the harness never reads it): skips `make_pc_features` plus a 2 GB write | **22** | no-effect | S | confirmed |
| SYS-2 (= POST-2) | If pc_features is kept: `np.save` of the permuted tensor goes through numpy's slow non-contiguous path, so save `.contiguous()` instead (same bytes, sha256-checked) | 8 | exact | S | confirmed |
| CLU-2 | Sync-free `kmeans_plusplus`: replace `multinomial` with its own internals (`exponential_` + `topk`, same RNG draws), index with tensors instead of 0-dim `.item()`, drop the dead `mu`, hoist `2*Xg`. About 7 host syncs per iteration × 38,400 iterations | 17 | exact (CUDA check pending) | S | plausible |
| POST-5 | `CCG.refract`: one stable argsort and contiguous slices instead of a full-length mask per cluster | 4.4 | exact | S | confirmed |
| SYS-6 (= CLU-6) | `@njit(cache=True)` on `CCG.compute_CCG` and `postprocessing.remove_duplicates` (1.6 s compile per process) | 2 | exact | S | confirmed |
| POST-4, POST-3 | `merging_function` and `make_pc_features`: per-cluster index arrays instead of masks over all 8.57 M spikes | 8.5 + 7.5 | exact | M | plausible |

## Tier 2: exact but more work (the GPU-loop rewrites)

| id | change | est. s/sort | class | effort |
|---|---|---|---|---|
| MATCH-1 | Device-resident peel loop: about 4 fused kernels per peel and at most 1 host sync, instead of about 55 launches and 3 syncs (GPU is only ~55% busy in this pass). MATCH-2 (Xres update, 25 s), MATCH-5 (peak find, 12 s), MATCH-3 (skip all-zero ctc rows, 6 s) and MATCH-4 (drop `torch.unique`, 6 s) are steps toward it | **≈80** | exact | L |
| CLU-4 | Pipeline the clustering CPU stages across centers: faiss kNN and `get_data_cpu` for the next centers on a thread pool while the GPU works on the current one (numpy BLAS stays single-caller) | 15 | exact | M |
| CLU-3, CLU-11 | Fused kernel for the kmeans++ elementwise ops; CUDA graphs per center | 6 + 1 | exact | M |
| DET-2 | Run sklearn KMeans's 10 Lloyd runs in parallel threads, with inits drawn sequentially (identical centres and RNG state on CPU tests) | 3 | exact | M |
| DET-3, DET-4 | `torch.empty` instead of `zeros` in `template_match` (a regression in my port), int8 `imaxs`, and a fused peak-mask kernel | 2.5 + 3.5 | exact | S/M |
| misc. | Keep extract outputs on the GPU or append in chunks (MATCH-9, GAP-5), vectorised `bin_spikes` (DET-6), direct CSR build in `neigh_mat` (CLU-7), `inference_mode` (GAP-3), stop Intel OpenMP spin-waiting (SYS-12, MATCH-11) | ≈1-3 each | exact / no-effect | S |

## Tier 3: cross-run reuse (large, but depends on which settings the search varies)

| id | change | est. saving |
|---|---|---|
| MATCH-12 | Cache the learned-pass output keyed by every upstream setting, the seeds and the code hash, and restore the RNG states (clustering calls `torch.rand` before reseeding). A search over merge or label thresholds (`acg_threshold`, `ccg_threshold`) then skips about 350 s of a ~400 s sort | up to 350 s per run |
| CLU-9, DET-9, SYS-4 | Cache at the stage boundaries (drift + universal, first clustering), keyed by exactly the settings each stage reads (`parameters.py` 'step' fields) and by package versions | 30-130 s |
| GAP-1 | NVMe cache of the CAR + high-passed batches (seed-independent) | ≈20 s |
| SYS-5 | **Behaviour change:** fixed seed for drift and universal detection, so seed replicates can share that cache. It changes what σ_S measures in spec §5.4, so it's your call | ≈30-80 s per replicate |

## Tier 4: scheduling and harness (wall time per evaluation, not sort time)

| id | change | est. |
|---|---|---|
| SYS-9 | Run the harness (evaluate/compare) for sort N while sort N+1 runs, whenever N+1 is planned in advance (Phase 0, replicates, ablations) | ≈30 s |
| SYS-8 | `cleaning.load_sort` / `load_clean_units`: one stable argsort and split instead of a per-cluster mask over 8.6 M spikes (exact) | ≈14 s |
| GAP-6 | `compare.match_units`: numba two-pointer agreement instead of 16k Python calls; cache the default sort's `load_clean_units` | ≈10 s |
| SYS-10, SYS-7 | Start sort N+1 while sort N exports (no GPU in use); one warm worker process | 15, 5 |

## Not exact: listed for completeness, not recommended under the current spec

| id | change | est. | class |
|---|---|---|---|
| GAP-8 | Filter length 60122 = 2·23·1307 forces cuFFT into Bluestein; use a 7-smooth padded length | 10 | numeric |
| CLU-13 | GPU kNN instead of faiss CPU | 25 | numeric |
| DET-10, CLU-5, MATCH-8, SYS-13 | rfft high-pass; single-threaded 90×90 solve; sparse template conv; Python 3.11/3.12 | 2-5 each | numeric |
| DET-12, DET-13, MATCH-14 | Learn wPCA/wTEMP once; subsampled drift pass; fewer peels | 1-7 | behaviour |
| SYS-14 | `cudnn.benchmark=True` | ≈0 | numeric (rejected) |

## Reproducibility findings (not speed)

- **GAP-7:** `swarmsplitter.check_split` uses numpy's OpenBLAS (24 threads here), and its results can depend on the
  thread count. Sorts are reproducible on this machine. To keep them reproducible across machines or environment
  changes, pin the BLAS, torch, faiss and numba thread counts and record them in `provenance.json`.
- **README correction:** the `h2d` change saved ≈1.2 ms/batch, not 12. The earlier microbenchmark charged the memmap's
  page faults to the copy being timed (found by the DET reviewer).

## Projection

- Tier 1 brings a sort from ≈400 s to roughly 290-310 s.
- Tier 1 plus MATCH-1 brings it to roughly 210-230 s.
- Tier 3 can cut individual runs far further, but only for searches over downstream settings.

None of these numbers are GPU-measured yet. The checks for CLU-1, DET-1, CLU-2 and SYS-2 are ready to run once the
GPU is free.
