"""Within-unit short-lag pairs at sample resolution, KS4 (before/after its 0.25 ms removal) and KS2.5
(each traced autocurate stage), normalised by the 5-20 ms level (pairs per lag sample over 150-600
samples). For KS4 full sorts, split by same vs different extraction template, and record the score
ratio (min/max) and which member was larger, per lag band. Also per-unit excess at 7-15 samples to see
whether it is concentrated in few units (and their amplitude / rate).

Answers: at which lags does each sorter put the second spike of a within-unit short-ISI pair, how much
of it is above the unit's own 5-20 ms level, and whether the KS4 excess is same-template or cross-template.
"""
import json
from pathlib import Path

import numpy as np

KS4 = Path(r'C:\data\ks4_tuning\exploratory\autocurate\sorts')
KS25T = Path(r'C:\data\ks4_tuning\exploratory\ks25_traced')
OUT = Path(__file__).parent / 'out'
MAXLAG = 600


def within_pairs(t, c, maxlag):
    o = np.lexsort((t, c))
    t, c = t[o], c[o]
    I, J = [], []
    for k in range(1, 400):
        a, b = np.arange(t.size - k), np.arange(k, t.size)
        m = (c[a] == c[b]) & (t[b] - t[a] <= maxlag)
        if not m.any():
            break
        I.append(a[m]); J.append(b[m])
    I, J = np.concatenate(I), np.concatenate(J)
    return o[I], o[J]


def profile(t, c, tmpl=None, score=None):
    i, j = within_pairs(t, c, MAXLAG)
    lag = np.abs(t[j] - t[i])
    h = np.bincount(lag, minlength=MAXLAG + 1)[:MAXLAG + 1].astype(float)
    base = h[150:601].mean()
    r = {'n_spikes': int(t.size), 'base_per_sample': base, 'norm': (h / base).tolist()}
    if tmpl is not None:
        same = tmpl[i] == tmpl[j]
        r['norm_same'] = (np.bincount(lag[same], minlength=MAXLAG + 1)[:MAXLAG + 1] / base).tolist()
        r['norm_diff'] = (np.bincount(lag[~same], minlength=MAXLAG + 1)[:MAXLAG + 1] / base).tolist()
        # same-template share at long lags (chance level of 'same')
        r['same_frac_150_600'] = float(same[(lag >= 150)].mean())
    if score is not None:
        first, second = np.where(t[i] <= t[j], i, j), np.where(t[i] <= t[j], j, i)
        mn = np.minimum(score[i], score[j]) / np.maximum(score[i], score[j])
        later_over_earlier = score[second] / score[first]
        q = {}
        for nm, (a, b) in {'0': (0, 0), '1-6': (1, 6), '7-15': (7, 15), '16-30': (16, 30),
                           '150-600': (150, 600)}.items():
            m = (lag >= a) & (lag <= b)
            q[nm] = dict(n=int(m.sum()), minmax_med=float(np.median(mn[m])) if m.any() else None,
                         later_over_earlier_med=float(np.median(later_over_earlier[m])) if m.any() else None,
                         frac_later_bigger=float((later_over_earlier[m] > 1).mean()) if m.any() else None)
        r['score'] = q
    # per-unit excess at 7-15 samples
    nu = c.max() + 1
    m7 = (lag >= 7) & (lag <= 15)
    mb = lag >= 150
    cu7 = np.bincount(c[i[m7]], minlength=nu)
    cub = np.bincount(c[i[mb]], minlength=nu) / 451 * 9
    ex = cu7 - cub
    order = np.argsort(ex)[::-1]
    tot = ex[ex > 0].sum()
    r['excess_7_15_total'] = float(ex.sum())
    r['excess_7_15_top10_units_share'] = float(ex[order[:10]].sum() / tot) if tot > 0 else None
    r['excess_7_15_units_for_half'] = int(np.searchsorted(np.cumsum(ex[order]), tot / 2) + 1)
    r['excess_7_15_per_spike'] = float(ex.sum() / t.size)
    return r


def bands(n):
    n = np.asarray(n)
    return {k: float(n[a:b + 1].mean()) for k, (a, b) in
            {'0': (0, 0), '1-3': (1, 3), '4-6': (4, 6), '7-9': (7, 9), '10-12': (10, 12), '13-15': (13, 15),
             '16-21': (16, 21), '22-30': (22, 30), '31-45': (31, 45), '60-90': (60, 90)}.items()}


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    res = {}
    for name in ['kobe_H8_s1', 'kobe_default_s1', 'mec_H8_s1', 'mec_default_s1']:
        d = KS4 / name
        st = np.load(d / 'full_st.npy')
        clu = np.load(d / 'full_clu.npy').astype(np.int64)
        kept = np.load(d / 'kept_spikes.npy')
        t = st[:, 0].astype(np.int64)
        tm = st[:, 1].astype(np.int64)
        res[name + '_full'] = profile(t, clu, tm, st[:, 2])
        res[name + '_kept'] = profile(t[kept], clu[kept], tm[kept], st[kept, 2])
    for ex in ['kobe_s1', 'mec_s1']:
        z = np.load(KS25T / ex / 'stages.npz')
        tm0 = z['extracted_clu'].astype(np.int64)
        for stg in ['extracted', 'merge1', 'split1', 'cut1', 'merge2', 'split2', 'cut2', 'merge3']:
            if stg + '_t' not in z.files:
                continue
            t = z[stg + '_t'].astype(np.int64)
            res[f'ks25_{ex}_{stg}'] = profile(t, z[stg + '_clu'].astype(np.int64), None,
                                              z[stg + '_score'].astype(float))
    for k, r in res.items():
        b = bands(r['norm'])
        print(f"{k:30s} n={r['n_spikes']:9d} " + ' '.join(f"{a}:{v:.2f}" for a, v in b.items()) +
              f" | exc7-15/spike {r['excess_7_15_per_spike']:.5f} top10 {r['excess_7_15_top10_units_share']}"
              f" half-by {r['excess_7_15_units_for_half']}")
        if 'norm_same' in r:
            bs, bd = bands(r['norm_same']), bands(r['norm_diff'])
            print('   same tmpl ' + ' '.join(f"{a}:{v:.2f}" for a, v in bs.items()) +
                  f" | same share at 5-20ms {r['same_frac_150_600']:.2f}")
            print('   diff tmpl ' + ' '.join(f"{a}:{v:.2f}" for a, v in bd.items()))
        if 'score' in r:
            print('   score', json.dumps(r['score']))
    json.dump(res, open(OUT / 'within_unit_lags.json', 'w'))
