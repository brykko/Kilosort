"""Why a final spike has no universal detection (round 4, seed 1, all cleaned spikes of eligible units).

    predicted below: score x rho < Th_universal: the spike's learned score, scaled by its unit's fit, is under threshold
    predicted above: it should have cleared Th_universal. Then: is a universal detection within 20 samples and 150 um,
                     the suppression window of the pass's non-maximum step (nt0min; 100 nearest centres)? Shift control.

The score carries noise and the selection of matching at threshold, so the split is approximate.

    python m5_unseen.py
"""
import numpy as np

from common import OUT, Sort, nearest_event, sort_dir

for ex in ('kobe', 'mec'):
    s = Sort(sort_dir(ex, 'H8', 1, 4), ex)
    x = np.load(OUT / f'm5_{ex}_s1_r4.npz')
    rho = np.full(s.n_units, np.nan)
    rho[x['unit']] = x['resp'] / x['norm']
    t, y, c = s.cleaned()
    score = np.load(s.d / 'score.npy')[s.order][s.keep]
    e = np.isfinite(rho[c])
    t, y, c, score = t[e], y[e], c[e], score[e]
    st0 = np.load(OUT / f'st0_{ex}_s1.npy')
    o = np.argsort(st0[:, 0], kind='stable')
    t0 = np.rint(st0[o, 0] * 30000).astype(np.int64) - 10
    y0 = st0[o, 1].astype(np.float32)
    seen = nearest_event(t, y, t0, y0, 10, 50.0) >= 0
    ch = nearest_event(t, y, t0 + 1000, y0, 10, 50.0) >= 0
    pred = score * rho[c] >= 7
    near = nearest_event(t, y, t0, y0, 20, 150.0) >= 0
    nearc = nearest_event(t, y, t0 + 1000, y0, 20, 150.0) >= 0
    u = ~seen
    print(ex, 'spikes', t.size, 'seen raw', seen.mean().round(3), 'chance', ch.mean().round(3))
    print('  unseen', u.mean().round(3), 'of which predicted below', (u & ~pred).sum() / u.sum(), 'predicted above', (u & pred).sum() / u.sum())
    print('  predicted above and unseen: detection in the suppression window', near[u & pred].mean().round(3), 'control', nearc[u & pred].mean().round(3))
    print('  predicted above: seen', seen[pred].mean().round(3), 'chance', ch[pred].mean().round(3), '; predicted below: seen', seen[~pred].mean().round(3), 'chance', ch[~pred].mean().round(3), 'share of spikes predicted above', pred.mean().round(3), flush=True)
