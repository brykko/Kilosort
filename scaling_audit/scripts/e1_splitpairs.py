"""Are the extra units of the 20-min sort oversplits or real separations?

Pairs of 20-min units that are both sub-parts of one unit of a comparison sort (inside a 5-min window: each holds >= 20%
of the comparison unit's spikes and puts >= 50% of its own in-window spikes there). Each pair's 20-min spike trains go
through KS4's own CCG test: cross-refractory (R12 < 0.25, Q12 < 0.05) suggests one neuron split by the 20-min sort;
a flat CCG suggests two neurons merged by the comparison sort. Comparison sorts: the 5-min window sorts, and (noise
floor) another 20-min seed.
"""
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault('NUMBA_CACHE_DIR', str(Path(__file__).parent / 'numba_cache'))
sys.path.insert(0, r'C:\code\github\Kilosort')
from kilosort import CCG  # noqa: E402

FS, TOL = 30000, 12
R = Path('C:/data/ks4_tuning/26648_kobe_2020-08-21/runs')


def load(p):
    st = np.load(p / 'spike_times.npy').ravel().astype(np.int64)
    cl = np.load(p / 'spike_clusters.npy').ravel().astype(np.int64)
    o = np.argsort(st, kind='stable')
    return st[o], cl[o]


def table(a_st, a_cl, b_st, b_cl):
    ai, ac = np.unique(a_cl, return_inverse=True)
    bi, bc = np.unique(b_cl, return_inverse=True)
    lo, hi = np.searchsorted(b_st, a_st - TOL), np.searchsorted(b_st, a_st + TOL, 'right')
    pairs = []
    for d in range(int((hi - lo).max())):
        has = lo + d < hi
        pairs.append(np.stack([np.where(has)[0], bc[lo[has] + d]], 1))
    key = np.unique(np.concatenate(pairs), axis=0)
    h = np.zeros((ai.size, bi.size))
    np.add.at(h, (ac[key[:, 0]], key[:, 1]), 1)
    return ai, h, np.bincount(ac), np.bincount(bc)


L = load(R / 'scaling_audit/full_0-1200_s1')
trains = {u: L[0][L[1] == u] / FS for u in np.unique(L[1])}
comps = [('w5_0-300', R / 'scaling_audit/w5_0-300_s1', 0), ('w5_600-900', R / 'scaling_audit/w5_600-900_s1', 600),
         ('seed2 (0-300)', R / 'hybrid_default_seed2', 0), ('seed2 (600-900)', R / 'hybrid_default_seed2', 600)]
for name, p, a in comps:
    lo_, hi_ = a * FS, (a + 300) * FS
    S = load(p)
    ml, ms = (L[0] >= lo_) & (L[0] < hi_), (S[0] >= lo_) & (S[0] < hi_)
    li, h, nl, ns = table(L[0][ml], L[1][ml], S[0][ms], S[1][ms])
    sub = (h / ns[None] >= .2) & (h / nl[:, None] >= .5) & (nl[:, None] >= 30)
    pairs = []
    for j in np.where(sub.sum(0) >= 2)[0]:
        us = li[sub[:, j]]
        pairs += [(us[x], us[y]) for x in range(len(us)) for y in range(x + 1, len(us))]
    res = np.array([CCG.check_CCG(trains[u], trains[v])[1:] for u, v in pairs], float) if pairs else np.zeros((0, 2))
    n = len(pairs)
    print(f'{name}: comparison units split by the 20-min sort: {(sub.sum(0) >= 2).sum()}, sub-unit pairs {n}; '
          f'cross-refractory (same neuron) {int(res[:, 0].sum()) if n else 0} ({res[:, 0].mean() if n else 0:.0%}); '
          f'R12 median {np.median(res[:, 1]) if n else np.nan:.2f}, R12 < 0.5: {(res[:, 1] < .5).sum() if n else 0}', flush=True)
