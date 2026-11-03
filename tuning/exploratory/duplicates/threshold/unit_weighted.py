"""Q: does the short-lag excess look different when every unit counts equally (rather than pooling pairs,
which high-rate units dominate)? Sum over units with >= 1000 spikes of pairs per lag band, each unit's
pairs divided by its own 5-20 ms level (so each unit contributes its obs/expected ratio), averaged.
Also the median over units of the band-to-floor ratio n(8-20)/13 : n(25-45)/21 among units with >= 20
control pairs.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import AUTOC, KOBE, MEC, load_phy, load_traced  # noqa
from mixture import unit_hist  # noqa

BANDS = [(0, 6), (7, 7), (8, 12), (13, 20), (21, 30), (31, 45), (46, 60)]


def run(name, t, clu):
    H, Lu, ns = unit_hist(t, clu)
    m = (ns >= 1000) & (Lu > 0)
    R = H[m] / Lu[m][:, None]
    prof = [R[:, a:b + 1].mean(1).mean() for a, b in BANDS]
    nD, nC = H[m, 8:21].sum(1), H[m, 25:46].sum(1)
    ok = nC >= 20
    br = (nD[ok] / 13) / (nC[ok] / 21)
    print(f'{name:30s} units {m.sum():4d}  unit-avg ratio ' + ' '.join(f'{a}-{b}:{p:.3f}' for (a, b), p in zip(BANDS, prof))
          + f'   median band/floor {np.median(br):.2f} (n={ok.sum()}), frac units >1.5: {np.mean(br > 1.5):.2f}')


if __name__ == '__main__':
    for n in ('kobe_default_s1', 'kobe_H8_s1', 'mec_default_s1', 'mec_H8_s1'):
        d = load_phy(AUTOC / n)
        run(f'KS4 {n} final', d['t'], d['clu'])
    for site, root in (('kobe', KOBE / 'runs'), ('mec', MEC / 'runs')):
        for p in ('standard', 'standard_low'):
            d = load_phy(root / f'ks25_{p}_r1')
            run(f'KS2.5 {site} {p} r1', d['t'], d['clu'])
        d = load_traced(f'{site}_s1', 'extracted')
        run(f'KS2.5 {site} traced extracted', d['t'], d['clu'])
