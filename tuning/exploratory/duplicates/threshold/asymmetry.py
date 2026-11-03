"""Q: a threshold-free estimate of true duplicates at each lag. Two separate spikes in one unit are
exchangeable in score order (control lags 25-45: later member higher in 49-50% of pairs), while a
re-detection is weaker than the detection it re-fits (raw-confirmed duplicate units: later member lower
in 97-99%). So at lag k, with f(k) the fraction of within-unit pairs whose later member has the lower
score, the duplicate share is P(dup | k) = 2 f(k) - 1 and duplicates n(k) (2 f(k) - 1). Gives the
duplicate lag distribution (including lags 0-6, before KS4's removal), the window holding 90/95% of
duplicates, and P(dup|k) with binomial 95% CIs. Pre-dedup KS4 sorts and KS2.5 traced stages.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import load_ks4_full, load_traced, pairs_within  # noqa

BANDS = [(0, 3), (4, 6), (7, 8), (9, 10), (11, 12), (13, 15), (16, 18), (19, 21), (22, 25), (26, 30), (31, 45)]


def run(name, t, clu, sc):
    I, J = pairs_within(t, clu, 45)
    lag = t[J] - t[I]
    low = sc[J] < sc[I]
    tie = sc[J] == sc[I]
    n = np.bincount(lag, minlength=46).astype(float)
    nl = np.bincount(lag, weights=low + 0.5 * tie, minlength=46)
    dup = np.clip(2 * nl - n, 0, None)
    ctrl = nl[31:46].sum() / n[31:46].sum()
    print(f'\n== {name}: N={t.size}; control (31-45) later-lower {ctrl:.3f}; '
          f'est. duplicates at lags 0-45: {dup.sum():.0f} ({1000 * dup.sum() / t.size:.2f}/1000 spikes)')
    tot = dup[:31].sum()
    cum = np.cumsum(dup[:31]) / max(tot, 1)
    print(f'  share of lag 0-30 duplicates at lags 0-6 (KS4 window): {cum[6]:.2f}; window holding 90% / 95%: '
          f'lag <= {int(np.argmax(cum >= .9))} / {int(np.argmax(cum >= .95))} samples')
    for a, b in BANDS:
        nn, ll = n[a:b + 1].sum(), nl[a:b + 1].sum()
        f = ll / max(nn, 1)
        se = np.sqrt(f * (1 - f) / max(nn, 1))
        print(f'  lag {a:2d}-{b:2d}: pairs {nn:7.0f}  later-lower {f:.3f}  P(dup)=2f-1 {2 * f - 1:6.3f} '
              f'[{2 * (f - 1.96 * se) - 1:6.3f},{2 * (f + 1.96 * se) - 1:6.3f}]  est dups {max(2 * ll - nn, 0):6.0f}')


if __name__ == '__main__':
    for n in ('kobe_default_s1', 'kobe_H8_s1', 'mec_default_s1', 'mec_H8_s1'):
        d = load_ks4_full(n)
        run(f'KS4 {n} pre-dedup', d['t'], d['clu'], d['score'])
    for site in ('kobe', 'mec'):
        for stage in ('extracted', 'merge3'):
            d = load_traced(f'{site}_s1', stage)
            run(f'KS2.5 {site} traced {stage}', d['t'], d['clu'], d['score'])
