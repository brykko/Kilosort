"""Q: are the pairs KS4's own window removes (lags 0-6 samples) duplicates (one physical spike) or two
overlapping spikes? raw_classify's trough test on pairs at lags 2-6 and on control pairs (25-45)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from common import load_ks4_full, pairs_within  # noqa
from score_rule import raw_test  # noqa
for site, v in (('kobe', 'default'), ('mec', 'default'), ('kobe', 'H8')):
    name = f'{site}_{v}_s1'
    d = load_ks4_full(name)
    t, clu, sc = d['t'], d['clu'], d['score']
    I, J = pairs_within(t, clu, 45)
    lag = t[J] - t[I]
    for lo, hi in ((2, 6), (25, 45)):
        m = (lag >= lo) & (lag <= hi)
        a = raw_test(site, name, t, clu, I[m], lag[m], nmax=250)
        print(f'{name} lags {lo}-{hi}: pairs {m.sum()}, later-lower {np.mean(sc[J[m]] < sc[I[m]]):.2f}, '
              f'raw one-spike fraction {a[0]:.2f} (n={a[1]})', flush=True)
