"""Do KS4's residual-type within-unit duplicates (second member < 0.6 of the unit's median score) at
7-20 samples sit at the lag where the unit template's temporal autocorrelation (whitened, Wall @ wPCA,
summed over channels) is most negative? After a least-squares subtraction, B at lag L changes by
-amp * ACF(L); where ACF(L) < 0 this ADDS a positive projection, so any shape/amplitude mismatch there
leaves a positive residual peak that relu(B) can detect (a negative-lobe 'echo').
Per unit with >= 30 such pairs: modal duplicate lag vs ACF zero-crossing and ACF minimum lag.
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

def acf(wpc, wPCA):
    w = wpc @ wPCA; nt = w.shape[1]
    f = np.array([np.sum(w[:, :nt - L] * w[:, L:]) for L in range(31)])
    return f / f[0]

for name in ['kobe_default_s1', 'mec_default_s1', 'kobe_H8_s1', 'mec_H8_s1']:
    st = np.load(D / name / 'full_st.npy'); c = np.load(D / name / 'full_clu.npy').astype(np.int64)
    Wall = np.load(D / name / 'Wall.npy'); wPCA = np.asarray(np.load(D / name / 'ops.npy', allow_pickle=True).item()['wPCA'])
    t, sc = st[:, 0].astype(np.int64), st[:, 2]
    i, j = within_pairs(t, c, 30); lag = t[j] - t[i]; nu = c.max() + 1
    n = np.bincount(c, minlength=nu); med = np.zeros(nu)
    o = np.argsort(c, kind='stable'); b = np.concatenate(([0], np.cumsum(n)))
    for u in np.where(n > 0)[0]: med[u] = np.median(sc[o[b[u]:b[u+1]]])
    small = np.minimum(sc[i], sc[j]) / med[c[i]] < 0.6
    m = small & (lag >= 7) & (lag <= 20)
    cnt = np.bincount(c[i[m]], minlength=nu)
    rows = []
    for u in np.where(cnt >= 30)[0]:
        h = np.bincount(lag[m & (c[i] == u)], minlength=21)
        a = acf(Wall[u], wPCA)
        zc = int(np.argmax(a < 0)); amin = int(np.argmin(a[:25]))
        rows.append((u, cnt[u], int(h.argmax()), zc, amin, a[amin], med[u]))
    rows = np.array(rows)
    print(f'{name}: units with >=30 small-member pairs at 7-20 samples: {len(rows)}')
    if len(rows):
        d = rows[:, 2] - rows[:, 4]
        print(f'   modal dup lag - ACF-min lag: median {np.median(d):.1f}, IQR {np.percentile(d,25):.1f}..{np.percentile(d,75):.1f};'
              f' modal dup lag - ACF zero crossing: median {np.median(rows[:,2]-rows[:,3]):.1f};'
              f' corr(dup lag, ACF-min lag) {np.corrcoef(rows[:,2], rows[:,4])[0,1]:.2f}')
        print(f'   modal dup lags: {np.bincount(rows[:,2].astype(int), minlength=21)[7:].tolist()} (lags 7..20)')
        print(f'   ACF-min lags:  {np.bincount(rows[:,4].astype(int), minlength=25)[7:21].tolist()}; ACF min value median {np.median(rows[:,5]):.2f}; unit median score median {np.median(rows[:,6]):.1f}')
        print(f'   pairs in these units: {int(rows[:,1].sum())} of {int(m.sum())} small-member 7-20 pairs')
