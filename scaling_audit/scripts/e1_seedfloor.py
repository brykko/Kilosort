"""Noise floor for e1_match: the same reshuffling counts between two 20-min seeds, restricted to each 5-min window."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import importlib.util
spec = importlib.util.spec_from_file_location('m', Path(__file__).parent / 'e1_match.py')
FS, TOL = 30000, 12
R = Path('C:/data/ks4_tuning/26648_kobe_2020-08-21/runs')
def load(p):
    st = np.load(p / 'spike_times.npy').ravel().astype(np.int64); cl = np.load(p / 'spike_clusters.npy').ravel().astype(np.int64)
    o = np.argsort(st, kind='stable'); return st[o], cl[o]
def table(a_st, a_cl, b_st, b_cl):
    ai, ac = np.unique(a_cl, return_inverse=True); bi, bc = np.unique(b_cl, return_inverse=True)
    lo, hi = np.searchsorted(b_st, a_st - TOL), np.searchsorted(b_st, a_st + TOL, 'right')
    pairs = []
    for d in range(int((hi - lo).max())):
        has = lo + d < hi; pairs.append(np.stack([np.where(has)[0], bc[lo[has] + d]], 1))
    key = np.unique(np.concatenate(pairs), axis=0)
    h = np.zeros((ai.size, bi.size)); np.add.at(h, (ac[key[:, 0]], key[:, 1]), 1)
    return h, np.bincount(ac), np.bincount(bc)
A = load(R / 'scaling_audit/full_0-1200_s1')
for other in ['hybrid_default_seed2', 'hybrid_default_seed3']:
    B = load(R / other)
    for a in (0, 300, 600, 900):
        lo_, hi_ = a * FS, (a + 300) * FS
        ma = (A[0] >= lo_) & (A[0] < hi_); mb = (B[0] >= lo_) & (B[0] < hi_)
        h, na, nb = table(A[0][ma], A[1][ma], B[0][mb], B[1][mb])
        big = na >= 30
        print(f'seed1 vs {other}, window {a}-{a+300}: units {len(na)} vs {len(nb)}; B units split by A: {(((h / nb[None]) >= .2).sum(0) >= 2).sum()}; '
              f'A units split by B: {(((h / na[:, None]) >= .2).sum(1)[big] >= 2).sum()}; A units with no B match: {(big & (h.max(1) / na < .5)).sum()}')
