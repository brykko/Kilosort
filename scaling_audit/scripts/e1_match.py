"""Match the 20-min sort's units to each window sort's units by shared spikes (within +-0.4 ms), inside the window.

For each window: per long unit with >= 30 spikes in the window, its best short unit and the share of its spikes there;
per short unit, how many long units each hold >= 20% of its spikes (> 1 = the long sort splits it) and vice versa.
Long units that match no short unit (best share < 0.5) are 'new at 20 min'; their temporal concentration tells whether
they are neurons active mainly elsewhere in the 20 min.
"""
from pathlib import Path

import numpy as np

FS, TOL = 30000, 12
SA = Path('C:/data/ks4_tuning/26648_kobe_2020-08-21/runs/scaling_audit')


def load(r):
    st = np.load(SA / r / 'spike_times.npy').ravel().astype(np.int64)
    cl = np.load(SA / r / 'spike_clusters.npy').ravel().astype(np.int64)
    o = np.argsort(st, kind='stable')
    return st[o], cl[o]


def table(a_st, a_cl, b_st, b_cl):
    """hits[i, j]: spikes of A-unit i with a B-spike of unit j within TOL (each A spike counted once per j)."""
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
    return ai, bi, h, np.bincount(ac), np.bincount(bc)


L_st, L_cl = load('full_0-1200_s1')
q = np.minimum(L_st // (300 * FS), 3)
lid, linv = np.unique(L_cl, return_inverse=True)
qc = np.bincount(linv * 4 + q, minlength=lid.size * 4).reshape(-1, 4)
conc = qc.max(1) / qc.sum(1)
cmap = dict(zip(lid, conc))
print(f'20-min sort: {lid.size} units; spike share in most active 5-min quarter: median {np.median(conc):.2f}, '
      f'> 0.7: {(conc > .7).sum()}, > 0.9: {(conc > .9).sum()}')
for r in ['w5_0-300_s1', 'w5_300-600_s1', 'w5_600-900_s1', 'w5_900-1200_s1', 'w10_0-600_s1', 'w10_600-1200_s1']:
    a, b = (int(x) * FS for x in r.split('_')[1].split('-'))
    m = (L_st >= a) & (L_st < b)
    S_st, S_cl = load(r)
    li, si, h, nl, ns = table(L_st[m], L_cl[m], S_st, S_cl)
    big = nl >= 30
    share = h.max(1) / nl                                   # long unit's best share in one short unit
    frag_of_short = ((h / ns[None, :]) >= 0.2).sum(0)       # long units holding >= 20% of a short unit
    frag_of_long = ((h / nl[:, None]) >= 0.2).sum(1)        # short units holding >= 20% of a long unit
    new = big & (share < 0.5)
    print(f'{r}: long units active {big.sum()}, short units {si.size}; '
          f'short units split by the long sort (>=2 long units with >=20%): {(frag_of_short >= 2).sum()}; '
          f'long units split by the short sort: {(frag_of_long[big] >= 2).sum()}; '
          f'long units with no short match (<50%): {new.sum()} '
          f'(their median quarter-concentration {np.median([cmap[x] for x in li[new]]) if new.any() else np.nan:.2f}, '
          f'median spikes in window {np.median(nl[new]) if new.any() else 0:.0f})')
