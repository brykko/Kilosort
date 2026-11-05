"""Measurement 2, labels: for the mutual r2<->r3 pairs of m1_convergence.py, the change in S_i against the share of the
unit's spikes that changed (1 - shared / max(n2, n3)) and against template change, with rank partials on rate, amplitude and
S_i. Descriptive; pairs are survivors, S_i is noisy per unit, and a unit's dS depends on its neighbours' changes too.

    python m2_assign.py
"""
import gc

import numpy as np
from scipy.stats import rankdata, spearmanr

from common import OUT, SEEDS, Link, Sort, sort_dir


def part(x, y, Z):
    D = np.column_stack([np.ones(x.size)] + [rankdata(z) for z in Z])
    rx, ry = rankdata(x), rankdata(y)
    f = lambda v: v - D @ np.linalg.lstsq(D, v, rcond=None)[0]  # noqa: E731
    return np.corrcoef(f(rx), f(ry))[0, 1]


for ex in ('kobe', 'mec'):
    for s in SEEDS:
        m = np.load(OUT / f'm1_{ex}_H8_s{s}.npz')
        a, b = Sort(sort_dir(ex, 'H8', s, 3), ex), Sort(sort_dir(ex, 'H8', s, 2), ex)
        l = Link(a, b)
        sh = dict(zip(zip(l.ua.tolist(), l.ub.tolist()), l.n.tolist()))
        na, nb = (np.bincount(x.clu[x.keep], minlength=x.n_units) for x in (a, b))
        v = m['mut_2_unit']
        ok = v >= 0
        u, v = m['unit'][ok], v[ok]
        chg = np.array([1 - sh.get((int(i), int(j)), 0) / max(na[i], nb[j]) for i, j in zip(u, v)])
        dS = m['S_mid'][ok] - m['S_2_mut'][ok]
        dT = m['mut_2_l2'][ok]
        Z = (m['rate'][ok], m['amp'][ok], m['S_mid'][ok])
        print(f'{ex} s{s}: pairs {ok.sum()}; median label change {np.median(chg):.3f}; Spearman dS~change {spearmanr(dS, chg)[0]:.3f}, |dS|~change {spearmanr(np.abs(dS), chg)[0]:.3f}, '
              f'change~|dT| {spearmanr(chg, dT)[0]:.3f}; partial dS~change {part(dS, chg, Z):.3f}, |dS|~change {part(np.abs(dS), chg, Z):.3f}', flush=True)
        q = np.quantile(chg, [.25, .5, .75])
        g = np.searchsorted(q, chg)
        print('   sum dS by label-change quartile', [round(float(dS[g == i].sum()), 1) for i in range(4)], '; by template-change quartile',
              [round(float(dS[np.searchsorted(np.quantile(dT, [.25, .5, .75]), dT) == i].sum()), 1) for i in range(4)], flush=True)
        del a, b, l
        gc.collect()
