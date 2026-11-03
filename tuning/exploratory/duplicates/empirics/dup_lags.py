"""Compute and cache, per sort and variant, (1) within-unit forward-pair lag histograms at sample resolution
(0-20 ms) and (2) cross-unit coincidence histograms (|lag| 0-10 ms) pooled by the pair's distance bin
(median spike positions) x template-similarity bin (zero-lag cosine of templates.npy).

Variants: raw (saved output), clean (after the harness's cleaning, cleaning.npz + masks), pre (KS4 seed 1,
before KS4's own 0.25 ms dedup). Also KS2.5's traced reruns, stage 'extracted' (before any of KS2.5's
post-processing) and 'merge3' (final), within-unit only. Output: cache/<excerpt>_<sorter>_<k>_<variant>.npz.
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from dup_common import _median_xy  # noqa: E402
from dup_common import (CACHE, DIST_EDGES, MAXW, MAXX, PRE, SIM_EDGES, SORTS, TR, by_unit, cross_hist,  # noqa: E402
                        load_variant, template_similarity, within_hist)


def job(args):
    ex, g, k, sort, variant = args
    out = CACHE / f'{ex}_{g}_{k}_{variant}.npz'
    if out.exists():
        return str(out)
    st, clu, ids, xy, T, copies = load_variant(sort, ex, variant, PRE.get((ex, g)))
    t_u, u = by_unit(st, clu, ids)
    Hw = within_hist(t_u, u, ids.size, MAXW)
    n = np.bincount(u, minlength=ids.size)
    amp_all = np.load(Path(sort) / 'amplitudes.npy').ravel()
    clu_saved = np.load(Path(sort) / 'spike_clusters.npy').ravel()
    amp = _median_xy(amp_all[:, None], clu_saved.astype(np.int64), ids)[:, 0]
    # cross-unit
    D = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    S = template_similarity(sort, ids)
    db = np.digitize(D, DIST_EDGES[1:-1])
    sb = np.digitize(S, SIM_EDGES[1:-1])
    B = (db * (SIM_EDGES.size - 1) + sb).astype(np.int16)
    ncat = (DIST_EDGES.size - 1) * (SIM_EDGES.size - 1)
    Hx = cross_hist(st, np.searchsorted(ids, clu), B, ncat, MAXX)
    # pairs present per category (for per-pair normalisation)
    np.savez_compressed(out, Hw=Hw, Hx=Hx, n=n, ids=ids, xy=xy, amp=amp, T=T, copies=np.array(sorted(copies)),
                        D=D.astype(np.float32), S=S.astype(np.float32))
    return str(out)


def job_traced(args):
    ex, stage = args
    out = CACHE / f'{ex}_ks25traced_1_{stage}.npz'
    if out.exists():
        return str(out)
    z = np.load(TR / f'{ex}_s1/stages.npz')
    st, clu = z[f'{stage}_t'].astype(np.int64), z[f'{stage}_clu'].astype(np.int64)
    ids = np.unique(clu)
    t_u, u = by_unit(st, clu, ids)
    Hw = within_hist(t_u, u, ids.size, MAXW)
    np.savez_compressed(out, Hw=Hw, n=np.bincount(u, minlength=ids.size), ids=ids)
    return str(out)


if __name__ == '__main__':
    CACHE.mkdir(exist_ok=True)
    jobs = []
    for ex, groups in SORTS.items():
        for g, sorts in groups.items():
            for k, s in enumerate(sorts, 1):
                jobs += [(ex, g, k, s, 'raw'), (ex, g, k, s, 'clean')]
                if k == 1 and (ex, g) in PRE:
                    jobs.append((ex, g, k, s, 'pre'))
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 6) as pool:
        for r in pool.map(job_traced, [(ex, s) for ex in ('kobe', 'mec') for s in ('extracted', 'merge3')]):
            print(r, flush=True)
        for r in pool.map(job, jobs):
            print(r, flush=True)
