"""How much of the within-unit 0.23-0.5 ms bump sits in units that look refractory (would matter for
contamination estimates of usable units) vs in multi-unit-like clusters? Units with >= 1000 spikes are
binned by their 1.0-1.5 ms level relative to their 5-20 ms level ('floor'); per bin: units, spikes, bump
pairs (7-15 samples minus the 31-45-sample level scaled to 9 samples), all violations in the Hill window
(lags 8-45 samples, 0.25-1.5 ms) and the bump's share of them. KS4 on kept spikes (as saved) and KS2.5 final.
"""
from pathlib import Path
import numpy as np

SORTS = {'kobe_ks4_H8': r'C:\data\ks4_tuning\exploratory\autocurate\sorts\kobe_H8_s1',
         'kobe_ks4_default': r'C:\data\ks4_tuning\exploratory\autocurate\sorts\kobe_default_s1',
         'mec_ks4_H8': r'C:\data\ks4_tuning\exploratory\autocurate\sorts\mec_H8_s1',
         'mec_ks4_default': r'C:\data\ks4_tuning\exploratory\autocurate\sorts\mec_default_s1',
         'kobe_ks25': r'C:\data\ks4_tuning\exploratory\ks25_traced\kobe_s1',
         'mec_ks25': r'C:\data\ks4_tuning\exploratory\ks25_traced\mec_s1'}

def within_pairs(t, c, maxlag):
    o = np.lexsort((t, c)); ts, cs = t[o], c[o]; I, J = [], []
    for k in range(1, 400):
        a, b = np.arange(ts.size - k), np.arange(k, ts.size)
        m = (cs[a] == cs[b]) & (ts[b] - ts[a] <= maxlag)
        if not m.any(): break
        I.append(a[m]); J.append(b[m])
    return o[np.concatenate(I)], o[np.concatenate(J)]

for name, d in SORTS.items():
    d = Path(d)
    t = np.load(d / 'spike_times.npy').ravel().astype(np.int64)
    c = np.load(d / 'spike_clusters.npy').ravel().astype(np.int64)
    i, j = within_pairs(t, c, 600); lag = t[j] - t[i]; nu = c.max() + 1
    cnt = lambda a, b: np.bincount(c[i[(lag >= a) & (lag <= b)]], minlength=nu).astype(float)
    b7, f31, base, hill = cnt(7, 15), cnt(31, 45) * 9 / 15, cnt(150, 600) / 451, cnt(8, 45)
    bump = b7 - f31; n = np.bincount(c, minlength=nu)
    floor = f31 / 9 / np.maximum(base, 1e-9)
    ok = n >= 1000
    print(name, f'units>=1000 sp: {ok.sum()}, bump total {bump[ok].sum():.0f}')
    for lo, hi in [(0, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 9)]:
        m = ok & (floor >= lo) & (floor < hi)
        print(f'   floor {lo:.1f}-{hi:.1f}: units {m.sum():5d} spikes {n[m].sum():9d} bump {bump[m].sum():7.0f}'
              f' ({bump[m].sum()/max(n[m].sum(),1)*1e3:.2f}/1000 sp) Hill-window pairs {hill[m].sum():8.0f}'
              f' bump share {bump[m].sum()/max(hill[m].sum(),1):.3f}')
