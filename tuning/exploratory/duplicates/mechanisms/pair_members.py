"""What are the two members of a KS4 within-unit pair at 7-15 samples (0.23-0.5 ms)? If one physical spike
were counted twice, the two detections would share its energy (at least one member well below the unit's
typical score/amplitude) and sit at the same position. If they are two physical spikes (a second neuron's
spike assigned to the unit, or real synchrony inside a merged unit), both are full-sized and the second
is displaced. Compares 7-15-sample pairs with 150-600-sample pairs (5-20 ms) of the same units, on the
kept spikes of the final KS4 output (positions and amplitudes as saved), and also for KS2.5 (final).
"""
from pathlib import Path

import numpy as np

SORTS = {
    'kobe_ks4_H8': Path(r'C:\data\ks4_tuning\exploratory\autocurate\sorts\kobe_H8_s1'),
    'kobe_ks4_default': Path(r'C:\data\ks4_tuning\exploratory\autocurate\sorts\kobe_default_s1'),
    'mec_ks4_H8': Path(r'C:\data\ks4_tuning\exploratory\autocurate\sorts\mec_H8_s1'),
    'kobe_ks25': Path(r'C:\data\ks4_tuning\exploratory\ks25_traced\kobe_s1'),
    'mec_ks25': Path(r'C:\data\ks4_tuning\exploratory\ks25_traced\mec_s1'),
}


def within_pairs(t, c, maxlag):
    o = np.lexsort((t, c))
    ts, cs = t[o], c[o]
    I, J = [], []
    for k in range(1, 400):
        a, b = np.arange(ts.size - k), np.arange(k, ts.size)
        m = (cs[a] == cs[b]) & (ts[b] - ts[a] <= maxlag)
        if not m.any():
            break
        I.append(a[m]); J.append(b[m])
    return o[np.concatenate(I)], o[np.concatenate(J)]


if __name__ == '__main__':
    for name, d in SORTS.items():
        t = np.load(d / 'spike_times.npy').ravel().astype(np.int64)
        c = np.load(d / 'spike_clusters.npy').ravel().astype(np.int64)
        pos = np.load(d / 'spike_positions.npy')
        amp = np.load(d / 'amplitudes.npy').ravel().astype(float)
        if (d / 'score.npy').exists():
            st = np.load(d / 'full_st.npy')
            kept = np.load(d / 'kept_spikes.npy')
            score = st[kept, 2]
        else:
            z = np.load(d / 'stages.npz')
            score = z['merge3_score'].astype(float)
        nu = c.max() + 1
        rel = {}
        for nm, x in [('amp', amp), ('score', score)]:
            med = np.zeros(nu)
            for u in np.unique(c):
                med[u] = np.median(x[c == u])
            rel[nm] = x / med[c]
        i, j = within_pairs(t, c, 600)
        lag = t[j] - t[i]
        dy = np.abs(pos[j, 1] - pos[i, 1])
        dxy = np.sqrt(((pos[j] - pos[i]) ** 2).sum(1))
        print(name)
        for band, (a, b) in {'7-15': (7, 15), '16-30': (16, 30), '150-600': (150, 600)}.items():
            m = (lag >= a) & (lag <= b)
            lo_s = np.minimum(rel['score'][i], rel['score'][j])[m]
            lo_a = np.minimum(rel['amp'][i], rel['amp'][j])[m]
            print(f"  lag {band:8s} n={m.sum():8d} | dist um: med {np.median(dxy[m]):5.1f} p75 {np.percentile(dxy[m],75):5.1f}"
                  f" p90 {np.percentile(dxy[m],90):5.1f} frac>10um {np.mean(dxy[m]>10):.3f}"
                  f" | min(score/unit med): med {np.median(lo_s):.2f} frac<0.6 {np.mean(lo_s<0.6):.3f}"
                  f" | min(amp/unit med): med {np.median(lo_a):.2f} frac<0.6 {np.mean(lo_a<0.6):.3f}"
                  f" | both score>0.8 med: {np.mean(lo_s>0.8):.3f}")
