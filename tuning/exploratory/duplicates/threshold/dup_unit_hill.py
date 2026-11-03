"""Q: what does true duplication do to the Hill contamination of the units that carry it, and what do
(a) the score rule (drop the later member if its score < 0.5 x the earlier's, lag 7-21) and (b) raising
tau_c to 0.7 ms do? Also counts how many units the score rule touches and their Hill c before/after."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from common import load_ks4_full, pairs_within  # noqa
T, TR = 1200.0, 1.5e-3
def hill(n, v, tc):
    a = v * T / (2 * (TR - tc) * n ** 2)
    return 1 - np.sqrt(max(1 - 2 * a, 0)) if a <= 0.5 else 1.0
def viol(t, tc):
    t = np.sort(t) / 30000.0
    return int(np.sum(np.searchsorted(t, t + TR, 'left') - np.searchsorted(t, t + tc, 'right')))
for name, units in (('kobe_default_s1', [765, 750]), ('mec_default_s1', [822, 779]), ('kobe_H8_s1', []), ('mec_H8_s1', [])):
    d = load_ks4_full(name)
    t, clu, sc, k = d['t'], d['clu'], d['score'], d['kept']
    I, J = pairs_within(t, clu, 21)
    lag = t[J] - t[I]
    drop = np.zeros(t.size, bool)
    m = (lag >= 7) & (sc[J] < 0.5 * sc[I])
    drop[J[m]] = True
    keep2 = k & ~drop
    ids, n = np.unique(clu[k], return_counts=True)
    touched = np.unique(clu[drop & k])
    print(f'\n== {name}: score rule drops {int((drop & k).sum())} spikes in {touched.size} units')
    rows = []
    for c in touched:
        t0 = t[k & (clu == c)]; t2 = t[keep2 & (clu == c)]
        if t0.size < 1000: continue
        rows.append((c, t0.size, hill(t0.size, viol(t0, .25e-3), .25e-3), hill(t2.size, viol(t2, .25e-3), .25e-3),
                     hill(t0.size, viol(t0, .7e-3), .7e-3), t0.size - t2.size))
    R = np.array(rows)
    if R.size:
        print(f'  touched units with >=1000 spikes: {len(R)}; c<0.1 before {np.sum(R[:,2]<.1)}, after rule {np.sum(R[:,3]<.1)}, '
              f'tau_c 0.7 ms (no removal) {np.sum(R[:,4]<.1)}')
        big = R[np.argsort(-R[:, 5])[:8]]
        for r in big:
            print(f'   unit {int(r[0])}: n={int(r[1])} dropped={int(r[5])}  Hill c: current {r[2]:.2f} -> score rule {r[3]:.2f}; tau_c 0.7 ms {r[4]:.2f}')
