"""In KS4 units that look refractory (1.0-1.5 ms level < 0.1 of the 5-20 ms level, >= 1000 spikes), what is
the 7-15-sample (0.23-0.5 ms) excess made of? Compares pairs at 7-15 samples with pairs at 31-45 samples
(the unit's own floor, which is contamination, not duplication) on full_st (before KS4's removal):
member scores relative to the unit median, min/max ratio, same extraction template, and the excess
(7-15 minus 31-45 scaled) per class. Excess with a small member (min/unit-median < 0.6) = residual-type
double detection; excess with both members full-size = two physical spikes (contamination by a
synchronous neighbour) or a full-size duplicate.
"""
from pathlib import Path
import numpy as np
D = Path(r'C:\data\ks4_tuning\exploratory\autocurate\sorts')

def within_pairs(t, c, maxlag):
    o = np.lexsort((t, c)); ts, cs = t[o], c[o]; I, J = [], []
    for k in range(1, 400):
        a, b = np.arange(ts.size - k), np.arange(k, ts.size)
        m = (cs[a] == cs[b]) & (ts[b] - ts[a] <= maxlag)
        if not m.any(): break
        I.append(a[m]); J.append(b[m])
    return o[np.concatenate(I)], o[np.concatenate(J)]

for name in ['kobe_H8_s1', 'kobe_default_s1', 'mec_H8_s1', 'mec_default_s1']:
    st = np.load(D / name / 'full_st.npy'); c = np.load(D / name / 'full_clu.npy').astype(np.int64)
    t, tm, sc = st[:, 0].astype(np.int64), st[:, 1].astype(np.int64), st[:, 2]
    i, j = within_pairs(t, c, 600); lag = t[j] - t[i]; nu = c.max() + 1
    n = np.bincount(c, minlength=nu)
    f31 = np.bincount(c[i[(lag >= 31) & (lag <= 45)]], minlength=nu) / 15
    base = np.bincount(c[i[lag >= 150]], minlength=nu) / 451
    good = (n >= 1000) & (f31 / np.maximum(base, 1e-9) < 0.1)
    med = np.zeros(nu)
    o = np.argsort(c, kind='stable'); b = np.concatenate(([0], np.cumsum(n)))
    for u in np.where(n > 0)[0]: med[u] = np.median(sc[o[b[u]:b[u+1]]])
    g = good[c[i]]
    rmin = np.minimum(sc[i], sc[j]) / med[c[i]]
    same = tm[i] == tm[j]
    A = g & (lag >= 7) & (lag <= 15); B = g & (lag >= 31) & (lag <= 45)
    print(f'{name}: good units {good.sum()}, spikes {n[good].sum()}; pairs 7-15: {A.sum()}, 31-45 (x9/15): {B.sum()*9/15:.0f}')
    for lab, cls in [('small member (<0.6 med)', rmin < 0.6), ('both full, same tmpl', (rmin >= 0.6) & same),
                     ('both full, diff tmpl', (rmin >= 0.6) & ~same)]:
        a, bb = (A & cls).sum(), (B & cls).sum() * 9 / 15
        print(f'   {lab:26s} 7-15: {a:6d}  floor: {bb:7.0f}  excess {a-bb:7.0f}')
    # lag profile of small-member pairs vs full pairs, 1..30
    for lab, cls in [('small', rmin < 0.6), ('full', rmin >= 0.6)]:
        h = np.bincount(lag[g & cls], minlength=31)[:31]
        print(f'   lag hist 0..30 ({lab}): {h.tolist()}')
