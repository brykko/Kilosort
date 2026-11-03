"""Q: at what lags do TRUE duplicates (raw check: one physical spike) sit? Lag histograms (pre-dedup KS4,
samples) of the units that raw_classify.py found to be ~all one-spike in the 9-15 band, plus template ACF
and extraction-template sharing and score ratio of their short-lag pairs."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from common import load_ks4_full, pairs_within, AUTOC  # noqa
for name, units in (('kobe_default_s1', [765, 750]), ('mec_default_s1', [822, 779])):
    d = load_ks4_full(name)
    W = np.load(AUTOC / name / 'templates.npy')
    for c in units:
        m = d['clu'] == c
        t = d['t'][m]; tm = d['tmpl'][m]; sc = d['score'][m]
        I, J = pairs_within(t, np.zeros(t.size, np.int64), 90)
        lag = t[J] - t[I]
        h = np.bincount(lag, minlength=91)
        base = h[60:91].mean()
        w = W[c]; acf = np.array([np.sum(w[:61 - k] * w[k:]) for k in range(25)]); acf /= acf[0]
        s = (lag >= 7) & (lag <= 25)
        print(f'{name} unit {c}: n={t.size} ({t.size/1200:.1f} Hz)  base(2-3ms)/sample={base:.1f}')
        print('  lag 0..30 counts:', ' '.join(str(x) for x in h[:31]))
        print(f'  pairs 7-25: same extraction tmpl {np.mean(tm[I[s]] == tm[J[s]]):.2f}, '
              f'score min/max median {np.median(np.minimum(sc[I[s]], sc[J[s]]) / np.maximum(sc[I[s]], sc[J[s]])):.2f}, '
              f'later has lower score {np.mean(sc[J[s]] < sc[I[s]]):.2f}, removed by KS4 (0-6): {h[:7].sum()}')
        print('  template ACF lags 0..24:', ' '.join(f'{x:.2f}' for x in acf))
