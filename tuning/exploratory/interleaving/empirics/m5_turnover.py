"""Do the extra rounds find units with worse fit to the universal templates? Same rule on both sides: all eligible units of
round 4 split into those with a mutual (>= 0.5) round-2 partner and 'gained' ones, and all eligible units of round 2 into
partnered and 'lost' ones; rho and f_seen (m5_universal.py) compared within template-norm terciles (pooled edges of that
round), since gained and lost units are smaller. Seeds 1-3. The seed control is the same split between round-2 sorts of two
seeds: gained units there are just turnover, so a real effect of extra rounds should exceed it.

    python m5_turnover.py
"""
import gc

import numpy as np

from common import OUT, SEEDS, Sort, mutual_pairs, sort_dir


def load(ex, s, r):
    x = np.load(OUT / f'm5_{ex}_s{s}_r{r}.npz')
    f = np.clip((x['seen'] - x['chance']) / np.maximum(x['n'] - x['chance'], 1), 0, 1)
    return x['unit'], x['resp'] / x['norm'], x['norm'], f


def report(lab, units, rho, nrm, f, partner):
    m = np.isin(units, partner)
    t = np.quantile(nrm, [1 / 3, 2 / 3])
    g = np.searchsorted(t, nrm)
    out = []
    for i in range(3):
        a, b = m & (g == i), ~m & (g == i)
        out.append(f'|T| t{i}: rho {np.median(rho[a]):.3f} vs {np.median(rho[b]):.3f}, f {np.median(f[a]):.3f} vs {np.median(f[b]):.3f} ({a.sum()}/{b.sum()})')
    print(lab, 'partnered vs not:', '; '.join(out), flush=True)


for ex in ('kobe', 'mec'):
    for s in SEEDS:
        a, b = (Sort(sort_dir(ex, 'H8', s, r), ex) for r in (2, 4))
        p = mutual_pairs(a, b)
        u2, r2, n2, f2 = load(ex, s, 2)
        u4, r4, n4, f4 = load(ex, s, 4)
        report(f'{ex} s{s} r4 (not = gained)', u4, r4, n4, f4, [v for v, _ in p.values()])
        report(f'{ex} s{s} r2 (not = lost)', u2, r2, n2, f2, list(p))
        s2 = SEEDS[s % 3]
        c = Sort(sort_dir(ex, 'H8', s2, 2), ex)
        q = mutual_pairs(a, c)
        report(f'{ex} s{s} r2 control vs seed {s2} r2 (not = unmatched)', u2, r2, n2, f2, list(q))
        del a, b, c
        gc.collect()
