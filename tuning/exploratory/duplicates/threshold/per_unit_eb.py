"""Q: does a per-unit (empirical-Bayes) duplicate window beat one global window, judged on held-out time?

Per unit, with the duplicate lag shape g(k) shared (the pooled gamma from mixture.py) and a flat floor
b_u: band counts n_D (lags 8-20) and n_C (25-45) on one time half give n_D | n_D+n_C ~ Binomial with
pi_u = (13 + rho_u G13) / (34 + rho_u G13), rho_u the unit's duplicate-to-floor ratio, G13 the shape's mass
in 8-20. pi_u gets a beta prior fitted across units (beta-binomial marginal likelihood) and its
posterior mean gives rho_u; the unit's window is the last lag with P_u(dup|k) = rho g/(rho g + 1) >= q.
Global rule: one window W for every unit. Evaluation on the OTHER half (both directions summed):
pairs removed at lags 8..W_u, non-duplicates among them estimated from that half's own 25-45 level,
duplicates captured = removed - non-dup. Compared on the trade-off curve (dups captured vs non-dups
removed).
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln, gammaln
from scipy.stats import gamma as G

sys.path.insert(0, str(Path(__file__).parent))
from common import load_ks4_full, pairs_within  # noqa

HERE = Path(__file__).parent
HALF = 600 * 30000


def counts(t, clu, ids):
    I, J = pairs_within(t, clu, 45)
    lag = t[J] - t[I]
    ui = np.searchsorted(ids, clu[I])
    H = np.zeros((ids.size, 46))
    np.add.at(H, (ui, lag), 1)
    return H


def fit_beta(x, n):
    m = n > 0
    x, n = x[m], n[m]
    def nll(p):
        a, b = np.exp(p)
        return -np.sum(betaln(x + a, n - x + b) - betaln(a, b))
    r = minimize(nll, [0, 0], method='Nelder-Mead')
    return np.exp(r.x)


def windows_eb(H, g, q):
    nD, nC = H[:, 8:21].sum(1), H[:, 25:46].sum(1)
    a, b = fit_beta(nD, nD + nC)
    pi = (nD + a) / (nD + nC + a + b)
    G13 = g[8:21].sum()
    rho = np.clip((34 * pi - 13) / (1 - pi), 0, None) / G13
    k = np.arange(8, 41)
    P = rho[:, None] * g[k][None] / (rho[:, None] * g[k][None] + 1)
    W = np.full(H.shape[0], 7)
    for i in range(H.shape[0]):
        w = np.where(P[i] >= q)[0]
        if w.size:
            W[i] = k[w.max()]
    return W, (a, b)


def evaluate(H, W):
    floor = H[:, 25:46].sum(1) / 21
    rem = np.array([H[i, 8:W[i] + 1].sum() for i in range(H.shape[0])])
    nond = (W - 7) * floor
    return rem.sum(), nond.sum(), (rem - nond).sum()


if __name__ == '__main__':
    res = np.load(HERE / 'mixture_results.npy', allow_pickle=True).item()
    for name in ('kobe_default_s1', 'kobe_H8_s1', 'mec_default_s1', 'mec_H8_s1'):
        site, v = name.split('_')[:2]
        p = res[f'{site} KS4 {v} pre-dedup']['p']
        g = G.pdf(np.arange(46), p[5], scale=p[6])
        d = load_ks4_full(name)
        t, clu = d['t'], d['clu']
        ids = np.unique(clu)
        h1, h2 = t < HALF, t >= HALF
        H1, H2 = counts(t[h1], clu[h1], ids), counts(t[h2], clu[h2], ids)
        print(f'\n== {name}  (held-out evaluation, both directions summed; pairs)')
        print('  global W : removed  non-dup  dups-captured  units-touched')
        for W in (9, 12, 15, 18, 21, 24):
            r = [evaluate(Hb, np.full(ids.size, W)) for Hb in (H2, H1)]
            s = np.sum(r, 0)
            print(f'    W={W:2d}   {s[0]:8.0f} {s[1]:8.0f} {s[2]:10.0f}   all')
        print('  per-unit EB q: removed  non-dup  dups-captured  units with W>7 (half1/half2)  beta(a,b) half1')
        for q in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            W1, ab = windows_eb(H1, g, q)
            W2, _ = windows_eb(H2, g, q)
            s = np.add(evaluate(H2, W1), evaluate(H1, W2))
            print(f'    q={q:.1f}   {s[0]:8.0f} {s[1]:8.0f} {s[2]:10.0f}   {np.sum(W1 > 7)}/{np.sum(W2 > 7)}'
                  f'  median W among touched {np.median(W1[W1 > 7]) if np.any(W1 > 7) else 0:.0f}  ({ab[0]:.2f},{ab[1]:.2f})',
                  flush=True)
        # stability: agreement of the touched-unit sets between halves at q=0.5
        W1, _ = windows_eb(H1, g, 0.5); W2, _ = windows_eb(H2, g, 0.5)
        a, b = W1 > 7, W2 > 7
        print(f'  q=0.5 touched-set overlap between halves: {np.sum(a & b)} both, {np.sum(a & ~b)} only h1, {np.sum(~a & b)} only h2')
