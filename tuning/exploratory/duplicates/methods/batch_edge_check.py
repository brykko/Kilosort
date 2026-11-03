"""Are short-lag within-unit pairs concentrated at KS4 batch boundaries (batch_size 60000 samples)?

Question: could the 0.27-1 ms within-unit pairs be the same spike detected in two overlapping batches
(a methods artefact a post hoc rule would hide rather than fix)? Compares the fraction of pairs whose
earlier member lies within 2 ms (60 samples) of a batch edge with the uniform expectation (120/60000)
and with control pairs at 2-5 ms. Also the extraction-template distance for pairs with different
templates (template positions from the median position of each template's spikes).
"""
from pathlib import Path

import numpy as np

AUTO = Path(r'C:\data\ks4_tuning\exploratory\autocurate\sorts')
NAMES = ('kobe_default_s1', 'kobe_H8_s1', 'mec_default_s1', 'mec_H8_s1')
BATCH = 60000

for name in NAMES:
    p = AUTO / name
    fs = np.load(p / 'full_st.npy')
    t = fs[:, 0].astype(np.int64); tm = fs[:, 1].astype(np.int64)
    c = np.load(p / 'full_clu.npy').astype(np.int64)
    kept = np.load(p / 'kept_spikes.npy')
    pos = np.load(p / 'spike_positions.npy')          # kept spikes only
    tk = tm[kept]
    ny = np.zeros(tm.max() + 1); nx = np.zeros(tm.max() + 1)
    for k in np.unique(tk):
        m = tk == k
        ny[k] = np.median(pos[m, 1]); nx[k] = np.median(pos[m, 0])
    o = np.lexsort((t, c)); t, tm, c = t[o], tm[o], c[o]
    i = np.where(c[1:] == c[:-1])[0]; d = np.diff(t)[i]
    ph = t % BATCH
    edge = (ph < 60) | (ph >= BATCH - 60)
    out = [name]
    for lo, hi in ((1, 6), (7, 15), (16, 30), (60, 150)):
        j = i[(d >= lo) & (d <= hi)]
        diff = tm[j] != tm[j + 1]
        dist = np.hypot(ny[tm[j]] - ny[tm[j + 1]], nx[tm[j]] - nx[tm[j + 1]])[diff]
        out.append(f'lag {lo}-{hi}: edge {edge[j].mean():.4f} (uniform {120 / BATCH:.4f}); '
                   f'diff-template pairs {diff.mean():.2f}, their template distance median {np.median(dist):.0f} um')
    print('\n  '.join(out), flush=True)
