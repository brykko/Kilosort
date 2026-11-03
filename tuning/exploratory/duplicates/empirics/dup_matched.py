"""Within-unit duplicate excess on the SAME cells in KS4 and KS2.5 (controls amplitude, rate and cell identity).

Units matched one-to-one between each sorter (seed/replicate 1) and KS2.5 standard r1 with
tuning/exploratory/isolation/matched_contam.py (harness-cleaned eligible units, agreement >= 0.5, <= 100 um).
For each matched set, pooled within-unit profile (obs/exp from each unit's 5-20 ms level), duplicate excess
0.25-1 ms above the 1.0-1.5 ms floor (pairs per 1000 spikes), its share of Hill-window violations, the floor,
and the number of matched cells whose own excess is positive and significant (Poisson, p < 0.01).
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import poisson

sys.path.insert(0, str(Path(__file__).parent))
from dup_common import FS, SORTS, within_hist  # noqa: E402

sys.path.insert(0, 'C:/code/github/Kilosort/tuning/exploratory/isolation')
import matched_contam as mc  # noqa: E402


def hist(units):
    H = []
    for t in units:
        s = np.round(np.sort(t) * FS).astype(np.int64)
        H.append(within_hist(s, np.zeros(s.size, np.int64), 1, 600)[0])
    return np.array(H, float)


def stats(H):
    E = H[:, 150:601].mean(1); F = H[:, 30:45].mean(1); N = H.sum(1)  # N unused
    prof = [H[:, a:b].sum() / (E.sum() * (b - a)) for a, b in ((8, 15), (15, 30), (30, 45))]
    exc = H[:, 8:30].sum(1) - 22 * F
    V = H[:, 8:45].sum()
    p = poisson.sf(H[:, 8:30].sum(1) - 1, np.maximum(22 * F, 1e-9))
    return prof, exc, V, F.sum() / E.sum(), int(np.sum((p < 0.01) & (exc > 0)))


if __name__ == '__main__':
    for ex in ('kobe', 'mec'):
        ub, db, Tb = mc.load(SORTS[ex]['ks25_std'][0], ex)
        for g in ('ks4_default', 'ks4_H8') + (('ks25_low',) if ex == 'kobe' else ()):  # no MEC eligibility list for ks25_low
            ua, da, Ta = mc.load(SORTS[ex][g][0], ex)
            pr = mc.pairs(ua, da, ub, db)
            A = [ua[a] for a, _ in pr]; B = [ub[b] for _, b in pr]
            na = sum(t.size for t in A); nb = sum(t.size for t in B)
            for lab, U, n in ((g, A, na), ('ks25_std', B, nb)):
                prof, exc, V, fl, nsig = stats(hist(U))
                print(f'{ex:5s} {g:12s} matched ({len(pr)} cells) {lab:12s}: spikes {n:8d}; obs/exp .27-.5 {prof[0]:.3f} '
                      f'.5-1 {prof[1]:.3f} floor 1-1.5 {fl:.3f}; dup excess .25-1 ms {1000 * exc.sum() / n:5.2f}/1000 spikes '
                      f'= {exc.sum() / V:5.1%} of Hill violations; cells with significant excess {nsig}', flush=True)
