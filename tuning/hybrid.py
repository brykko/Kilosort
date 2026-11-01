"""Hybrid recovery (spec §7.4, the hybrid-recovery guards) of a sort of the hybrid excerpt.

Uses the sorter's own output (before the harness's cleaning), so it measures the sort. For each
inserted unit and each sorted unit, a hit is an inserted spike with a sorted spike within +-0.4 ms;
agreement = hits / (n_inserted + n_sorted - hits), and each inserted unit is scored by its
best-matching sorted unit. Recovered: agreement >= 0.8. An HD pair is merged if one sorted unit
holds >= 30% of each member's spikes. Sorted units whose spikes are >= 50% hits on inserted spikes
are hybrid copies and are excluded from eligibility (§6.4) and S (§5.3).
"""
from pathlib import Path

import numpy as np

TOL = 12            # samples (0.4 ms)
RECOVERED = 0.8
MERGE = 0.3
COPY = 0.5


def recovery(sort, truth):
    sort = Path(sort)
    st = np.load(sort / 'spike_times.npy').ravel().astype(np.int64)
    cl = np.load(sort / 'spike_clusters.npy').ravel().astype(np.int64)
    o = np.argsort(st, kind='stable')
    st, cl = st[o], cl[o]
    ids, cl_idx = np.unique(cl, return_inverse=True)
    n_sorted = np.bincount(cl_idx, minlength=ids.size)

    tt, tu = truth['spike_samples'], truth['spike_unit']
    n_ins = int(tu.max()) + 1
    n_true = np.bincount(tu, minlength=n_ins)
    sorted_hit = np.zeros(st.size, bool)                    # sorted spikes that hit an inserted spike
    lo, hi = np.searchsorted(st, tt - TOL, 'left'), np.searchsorted(st, tt + TOL, 'right')
    pairs = []                                              # (inserted spike, sorted unit)
    for d in range(int((hi - lo).max()) if tt.size else 0):
        has = lo + d < hi
        j = lo[has] + d
        sorted_hit[j] = True
        pairs.append(np.stack([np.where(has)[0], cl_idx[j]], 1))
    key = np.unique(np.concatenate(pairs), axis=0)          # each inserted spike counts once per unit
    hits = np.zeros((n_ins, ids.size), np.int64)
    np.add.at(hits, (tu[key[:, 0]], key[:, 1]), 1)

    agree = hits / (n_true[:, None] + n_sorted[None, :] - hits)
    best = agree.argmax(1)
    best_agree = agree[np.arange(n_ins), best]
    frac = hits / n_true[:, None]
    merged = []
    for p in range(int(truth['pair'].max()) + 1):
        a, b = np.where(truth['pair'] == p)[0]
        both = np.where((frac[a] >= MERGE) & (frac[b] >= MERGE))[0]
        if both.size:
            merged.append({'pair': p, 'sorted_units': ids[both].tolist()})
    copy_frac = np.bincount(cl_idx, weights=sorted_hit, minlength=ids.size) / n_sorted
    return {
        'median_agreement': float(np.median(best_agree)),
        'n_recovered': int(np.sum(best_agree >= RECOVERED)),
        'n_inserted': n_ins,
        'merged_pairs': merged,
        'hybrid_copies': ids[copy_frac >= COPY].tolist(),
        'per_unit': [{'inserted': k, 'best_sorted': int(ids[best[k]]), 'agreement': float(best_agree[k])}
                     for k in range(n_ins)],
    }
