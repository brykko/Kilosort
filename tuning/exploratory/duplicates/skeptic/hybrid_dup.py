"""Ground-truth test of the duplication hypothesis on the kobe hybrid excerpt.

Question: around an inserted (hybrid) spike that has no other inserted spike within +-3 ms, does the
sorter report a SECOND spike at ~0.3-0.5 ms in the unit that captured it (within-unit duplicate) or in
another unit nearby (cross-unit duplicate)? Inserted spikes are exact scaled copies of a template at a
known time; their trains are circularly shifted, so they have no physiological synchrony with anything in
the recording. Any excess over the 2-3 ms floor at +-7..20 samples from the detection is therefore
produced by the sorter (or by chance coincidence with a real spike, which the floor measures).

For each inserted unit k: best-matching label m (agreement, +-12 samples, as hybrid.py). Lags of every
spike of m (and, separately, of every spike of OTHER labels whose median position is within 100 um of
m's) relative to each isolated inserted spike, -90..+90 samples. o = modal lag (the detection).
Reported, pooled over the 40 units: per detected inserted spike, extra same-label spikes at
|lag - o| in [7, 20] minus the floor (mean per-lag count at |lag - o| in [60, 90] x 28 lags), etc.
Also the lag profile itself.
"""
import json
import sys
from pathlib import Path

import numpy as np

KOBE = Path('C:/data/ks4_tuning/26648_kobe_2020-08-21')
EXP = Path('C:/data/ks4_tuning/exploratory')
TRUTH = KOBE / 'search/hybrid/hybrid_truth.npz'
OUT = Path(__file__).resolve().parent / 'hybrid_dup.json'
W = 90
ISO = 90


def lags_of(tt, s):
    """All lags s_j - tt_i within +-W; returns (lag, index into tt)."""
    lo = np.searchsorted(s, tt - W, 'left')
    hi = np.searchsorted(s, tt + W, 'right')
    out_l, out_i = [], []
    for d in range(int((hi - lo).max()) if tt.size else 0):
        has = lo + d < hi
        out_l.append(s[lo[has] + d] - tt[has])
        out_i.append(np.where(has)[0])
    if not out_l:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(out_l), np.concatenate(out_i)


def analyse(t, c, pos_of_label, truth):
    t = np.asarray(t, np.int64).ravel()
    c = np.asarray(c, np.int64).ravel()
    o = np.argsort(t, kind='stable')
    t, c = t[o], c[o]
    tt, tu = truth['spike_samples'].astype(np.int64), truth['spike_unit']
    ot = np.argsort(tt)
    tt, tu = tt[ot], tu[ot]
    gap = np.minimum(np.diff(tt, prepend=-10**9), np.diff(tt, append=10**12)[:])
    iso = gap > ISO
    labels, inv = np.unique(c, return_inverse=True)
    n_lab = np.bincount(inv)
    # matching (agreement within +-12 samples)
    lag, idx = lags_of(tt, t)
    jj = np.searchsorted(t, tt[idx] + lag)  # index of the sorted spike
    near = np.abs(lag) <= 12
    key = np.unique(np.stack([idx[near], inv[jj[near]]], 1), axis=0)
    hits = np.zeros((tu.max() + 1, labels.size), np.int64)
    np.add.at(hits, (tu[key[:, 0]], key[:, 1]), 1)
    n_true = np.bincount(tu)
    agree = hits / (n_true[:, None] + n_lab[None, :] - hits)
    best = agree.argmax(1)
    H_same = np.zeros(2 * W + 1)
    H_near = np.zeros(2 * W + 1)
    tot = {'iso_inserted': 0, 'detected': 0, 'same_extra_7to20': 0.0, 'same_floor_7to20': 0.0,
           'near_7to20': 0.0, 'near_floor_7to20': 0.0, 'same_0to6_extra': 0.0}
    per_unit = []
    for k in range(tu.max() + 1):
        m = best[k]
        sel = (tu == k) & iso
        tk = tt[sel]
        s_m = t[inv == m]
        l, i = lags_of(tk, s_m)
        h = np.bincount(l + W, minlength=2 * W + 1).astype(float)
        off = int(np.argmax(h[W - 12:W + 13])) - 12
        # recentre on the detection lag
        hc = np.zeros(2 * W + 1)
        src = np.arange(2 * W + 1) - off
        ok = (src >= 0) & (src < 2 * W + 1)
        hc[np.arange(2 * W + 1)[ok]] = h[src[ok]]
        det = h[W + off - 3:W + off + 4].sum()
        flo = np.r_[hc[W - 90:W - 59], hc[W + 60:W + 91]].mean()
        ex = np.r_[hc[W - 20:W - 6], hc[W + 7:W + 21]].sum()
        ex06 = np.r_[hc[W - 6:W - 3], hc[W + 4:W + 7]].sum()
        # near labels
        pm = pos_of_label.get(int(labels[m])) if pos_of_label is not None else None
        hn = np.zeros(2 * W + 1)
        if pm is not None:
            nb = [j for j in range(labels.size) if j != m and int(labels[j]) in pos_of_label
                  and np.linalg.norm(pos_of_label[int(labels[j])] - pm) <= 100]
            s_n = t[np.isin(inv, nb)]
            l2, _ = lags_of(tk, s_n)
            h2 = np.bincount(l2 + W, minlength=2 * W + 1).astype(float)
            hn[np.arange(2 * W + 1)[ok]] = h2[src[ok]]
        nflo = np.r_[hn[W - 90:W - 59], hn[W + 60:W + 91]].mean()
        nex = np.r_[hn[W - 20:W - 6], hn[W + 7:W + 21]].sum()
        H_same += hc
        H_near += hn
        tot['iso_inserted'] += int(sel.sum())
        tot['detected'] += float(det)
        tot['same_extra_7to20'] += float(ex)
        tot['same_floor_7to20'] += float(flo * 28)
        tot['same_0to6_extra'] += float(ex06 - flo * 6)
        tot['near_7to20'] += float(nex)
        tot['near_floor_7to20'] += float(nflo * 28)
        per_unit.append({'unit': k, 'label': int(labels[m]), 'agreement': float(agree[k, m]),
                         'iso': int(sel.sum()), 'detected': float(det), 'offset': off,
                         'dup_7to20_per_det': float((ex - flo * 28) / max(det, 1)),
                         'near_7to20_per_det': float((nex - nflo * 28) / max(det, 1))})
    tot['same_dup_per_detected'] = (tot['same_extra_7to20'] - tot['same_floor_7to20']) / tot['detected']
    tot['near_dup_per_detected'] = (tot['near_7to20'] - tot['near_floor_7to20']) / tot['detected']
    tot['same_0to6_per_detected'] = tot['same_0to6_extra'] / tot['detected']
    return tot, H_same.tolist(), H_near.tolist(), per_unit


def positions(sort, t_len=None):
    p = np.load(sort / 'spike_positions.npy')
    cl = np.load(sort / 'spike_clusters.npy').ravel()
    return {int(k): np.median(p[cl == k], 0) for k in np.unique(cl)}


def main():
    truth = np.load(TRUTH)
    sorts = {}
    R = KOBE / 'runs'
    for i in (1, 2, 3):
        sorts[f'ks4_default_seed{i}'] = R / f'hybrid_default_seed{i}'
        sorts[f'ks25_standard_r{i}'] = R / f'ks25_standard_r{i}'
        sorts[f'ks25_standard_low_r{i}'] = R / f'ks25_standard_low_r{i}'
    for i, n in ((1, 198), (2, 208), (3, 218)):
        sorts[f'ks4_H8_s{i}'] = R / f'search/{n}_H8_cds10_s{i}/sort'
    res = {}
    for name, sort in sorts.items():
        st = np.load(sort / 'spike_times.npy').ravel()
        cl = np.load(sort / 'spike_clusters.npy').ravel()
        tot, hs, hn, pu = analyse(st, cl, positions(sort), truth)
        res[name] = {'tot': tot, 'H_same': hs, 'H_near': hn, 'per_unit': pu}
        print(name, {k: round(v, 4) for k, v in tot.items()}, flush=True)
    # before KS4's dedup, scored sorts (positions from the kept spikes' clusters)
    for v in ('default', 'H8'):
        sort = EXP / f'autocurate/sorts/kobe_{v}_s1'
        f = np.load(sort / 'full_st.npy')
        fc = np.load(sort / 'full_clu.npy').ravel()
        tot, hs, hn, pu = analyse(f[:, 0], fc, positions(sort), truth)
        res[f'ks4_{v}_s1_before_dedup'] = {'tot': tot, 'H_same': hs, 'H_near': hn, 'per_unit': pu}
        print(f'ks4_{v}_s1_before_dedup', {k: round(v, 4) for k, v in tot.items()}, flush=True)
    s = np.load(EXP / 'ks25_traced/kobe_s1/stages.npz')
    tot, hs, hn, pu = analyse(s['extracted_t'] - 1, s['extracted_clu'], None, truth)
    res['ks25_traced_extracted'] = {'tot': tot, 'H_same': hs, 'per_unit': pu}
    print('ks25_traced_extracted', {k: round(v, 4) for k, v in tot.items()}, flush=True)
    OUT.write_text(json.dumps(res))


if __name__ == '__main__':
    main()
