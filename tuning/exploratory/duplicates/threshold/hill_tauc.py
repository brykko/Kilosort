"""Q: how much does the 0.25-0.8 ms excess move the Hill contamination estimate (tau_r 1.5 ms), and what
does the censored period tau_c do about it? Per unit of each final sort (no harness cleaning; units with
>= 1000 spikes), c from V = 2 (tau_r - tau_c) N^2 c (1 - c/2) / T for tau_c = 0.25 (current), 0.5 and
0.8 ms. Window edges as isolation.py: violations are intervals in (tau_c, tau_r).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import KOBE, MEC, AUTOC, load_phy, pairs_within  # noqa

T, TR = 1200.0, 1.5e-3


def hill(n, v, tc):
    a = v * T / (2 * (TR - tc) * n ** 2)
    return np.where(a <= 0.5, 1 - np.sqrt(np.clip(1 - 2 * a, 0, None)), 1.0)


def run(name, t, clu):
    I, J = pairs_within(t, clu, 45)
    lag = (t[J] - t[I]) / 30000.0
    ids, ns = np.unique(clu, return_counts=True)
    ui = np.searchsorted(ids, clu[I])
    keep = ns >= 1000
    out = {}
    for tc in (0.25e-3, 0.5e-3, 0.8e-3):
        m = (lag > tc) & (lag < TR)
        v = np.bincount(ui[m], minlength=ids.size)
        out[tc] = hill(ns, v, tc)[keep]
    c0, c1, c2 = out[0.25e-3], out[0.5e-3], out[0.8e-3]
    print(f'{name:28s} units>=1000 sp: {keep.sum():4d}  median c: {np.median(c0):.3f} {np.median(c1):.3f} {np.median(c2):.3f}'
          f'  n(c<0.1): {np.sum(c0 < .1):4d} {np.sum(c1 < .1):4d} {np.sum(c2 < .1):4d}'
          f'  n(c<0.2): {np.sum(c0 < .2):4d} {np.sum(c1 < .2):4d} {np.sum(c2 < .2):4d}'
          f'  spikes in c<0.1 units (M): {ns[keep][c0 < .1].sum() / 1e6:.2f} {ns[keep][c1 < .1].sum() / 1e6:.2f} {ns[keep][c2 < .1].sum() / 1e6:.2f}')


if __name__ == '__main__':
    print('tau_c = 0.25 / 0.5 / 0.8 ms')
    for n in ('kobe_default_s1', 'kobe_H8_s1', 'mec_default_s1', 'mec_H8_s1'):
        d = load_phy(AUTOC / n)
        run(f'KS4 {n} final', d['t'], d['clu'])
    for site, root in (('kobe', KOBE / 'runs'), ('mec', MEC / 'runs')):
        for p in ('standard', 'standard_low'):
            d = load_phy(root / f'ks25_{p}_r1')
            run(f'KS2.5 {site} {p} r1', d['t'], d['clu'])
