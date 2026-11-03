"""Same joint refit as pair_raw_refit.py, for the pairs KS4's own dedup acts on (lags 4-6 samples) and
again for kept pairs at 8-15 and phantoms, reporting the amplitude SUM a1 + a2 (robust when the two
copies are nearly collinear): one event gives a sum ~1, two overlapping spikes ~2.

Question: are the pairs KS4 deletes (<= 0.2 ms) single events detected twice (deletion right), unlike
the 0.27-0.5 ms pairs it keeps? Uses full_st/full_clu (before KS4's dedup).
Usage: python pair_raw_refit_removed.py <autocurate sort dir> <raw binary>
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pair_raw_refit as P  # noqa: E402  (sets SORT/RAW from argv)

OUT = Path(__file__).parent / f'pair_raw_refit_removed_{P.SORT.name}.json'


def main():
    fs = np.load(P.SORT / 'full_st.npy')
    st = fs[:, 0].astype(np.int64)
    cl = np.load(P.SORT / 'full_clu.npy').astype(np.int64)
    tmpl = np.load(P.SORT / 'templates.npy')
    o = np.lexsort((st, cl)); st, cl = st[o], cl[o]
    same = cl[1:] == cl[:-1]; d = np.diff(st)
    prev_gap = np.r_[np.inf, np.where(same, d, np.inf)]; next_gap = np.r_[np.where(same, d, np.inf), np.inf]
    nmax = P.raw.shape[0]
    bands = {'lag_4_6_removed_by_ks4': (4, 6), 'lag_8_15_kept': (8, 15)}
    idx = {k: np.where(same & (d >= a) & (d <= b))[0] for k, (a, b) in bands.items()}
    units, counts = np.unique(cl[idx['lag_4_6_removed_by_ks4']], return_counts=True)
    nsp = np.bincount(cl)
    ok = nsp[units] >= 2000
    units = units[ok][np.argsort(counts[ok])[::-1][:20]]
    agg = {k: [] for k in list(bands) + ['phantom_4_6', 'phantom_8_15']}
    for u in units:
        a, b = np.searchsorted(cl, [u, u + 1])
        chans = np.argsort(np.ptp(tmpl[u], axis=0))[::-1][:P.NCH_FIT]
        iso = np.arange(a, b)[(prev_gap[a:b] > 90) & (next_gap[a:b] > 90)]
        iso = iso[(st[iso] > 2 * P.PADF) & (st[iso] < nmax - 2 * P.PADF)]
        pick = P.rng.choice(iso, min(P.N_MEAN, iso.size), replace=False)
        W = np.mean([P.snippet(st[i] - P.PRE, st[i] + P.POST, chans) for i in pick], axis=0)
        for k in bands:
            pr = idx[k][cl[idx[k]] == u]
            pr = pr[(st[pr] > 2 * P.PADF) & (st[pr + 1] < nmax - 2 * P.PADF)]
            pr = P.rng.choice(pr, min(40, pr.size), replace=False)
            for i in pr:
                L = int(st[i + 1] - st[i])
                x = P.snippet(st[i] - P.PRE - P.J, st[i] + L + P.POST + P.J, chans)
                agg[k].append(P.fit_pair(x, W, L)[:2])
            ph = 'phantom_4_6' if k.startswith('lag_4') else 'phantom_8_15'
            lo, hi = bands[k]
            for i in P.rng.choice(iso, min(40, iso.size), replace=False):
                L = int(P.rng.integers(lo, hi + 1))
                x = P.snippet(st[i] - P.PRE - P.J, st[i] + L + P.POST + P.J, chans)
                agg[ph].append(P.fit_pair(x, W, L)[:2])
        print(int(u), {k: (len(v), float(np.median(np.sum(v, 1))) if v else None) for k, v in agg.items()}, flush=True)
    summ = {}
    for k, v in agg.items():
        v = np.array(v)
        if not v.size:
            continue
        s = v.sum(1)
        summ[k] = {'n': int(len(v)), 'a1_median': float(np.median(v[:, 0])), 'a2_median': float(np.median(v[:, 1])),
                   'sum_median': float(np.median(s)), 'sum_quartiles': np.percentile(s, [10, 25, 75, 90]).tolist(),
                   'frac_sum_gt_1.5': float(np.mean(s > 1.5)), 'frac_sum_lt_1.3': float(np.mean(s < 1.3))}
    print(json.dumps(summ, indent=1))
    OUT.write_text(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
