"""How much of the between-sorter difference in % misassigned spikes is predicted by unit firing rates alone.

One curve, contamination against log firing rate, fitted to every unit of every sorter together (spike-
weighted least squares, cubic in log10 rate). Each sorter's % misassigned predicted from its units' rates
alone, sum n_i chat(r_i) / sum n_i, is compared with the observed sum n_i c_i / sum n_i: the difference
between two sorters' predictions is the part of their gap that firing rate accounts for. Also the share of
unit-to-unit variance in contamination explained by rate, by sorter, and by both (spike-weighted R^2).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from comod import sort_units  # noqa: E402
from rate_mix import SETS  # noqa: E402

DEG = 3


def wls(X, y, w):
    sw = np.sqrt(w)
    beta = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
    return X @ beta


def r2(pred, y, w):
    m = np.sum(w * y) / w.sum()
    return 1 - np.sum(w * (y - pred) ** 2) / np.sum(w * (y - m) ** 2)


for ex, groups in SETS.items():
    rows = {g: [r for s in sorts for r in sort_units(s, ex)] for g, sorts in groups.items()}
    names = list(groups)
    g_idx = np.concatenate([[k] * len(rows[g]) for k, g in enumerate(names)])
    n = np.array([r[0] for g in names for r in rows[g]], float)
    lr = np.log10(np.array([r[1] for g in names for r in rows[g]]))
    for m in ('plain', 'local'):
        c = np.array([r[2][m] for g in names for r in rows[g]])
        Xr = np.vander(lr, DEG + 1)
        Xs = (g_idx[:, None] == np.arange(len(names))[None, :]).astype(float)
        pred_r = wls(Xr, c, n)
        pred_s = wls(Xs, c, n)
        pred_rs = wls(np.c_[Xr, Xs[:, 1:]], c, n)
        print(f'\n{ex} ({m} estimate): spike-weighted R^2 of unit contamination: rate {r2(pred_r, c, n):.2f}, '
              f'sorter {r2(pred_s, c, n):.2f}, rate + sorter {r2(pred_rs, c, n):.2f}')
        obs = {g: np.sum((n * c)[g_idx == k]) / n[g_idx == k].sum() for k, g in enumerate(names)}
        prd = {g: np.sum((n * pred_r)[g_idx == k]) / n[g_idx == k].sum() for k, g in enumerate(names)}
        for g in names:
            print(f'  {g:8s} observed {obs[g]:6.1%}   predicted from rates alone {prd[g]:6.1%}')
        for g1, g2 in (('default', 'ks25'), ('H8', 'ks25'), ('default', 'H8')):
            d_obs, d_prd = obs[g2] - obs[g1], prd[g2] - prd[g1]
            print(f'  {g2} - {g1}: observed {d_obs:+6.1%}, predicted by rate {d_prd:+6.1%} ({d_prd / d_obs:4.0%} of the gap)')
