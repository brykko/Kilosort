"""Q: what duplicate-detection lag threshold does a mixture model of the within-unit lag distribution
imply, with uncertainty, per sorter, and how many spikes would each rule remove?

Model, pooled over units at sample resolution k = 8..90 (k <= 7 is affected by the peeling dead time and
KS4's own removal), counts h(k) ~ Poisson(L * r(k)) with L the pooled 5-20 ms pair level per lag sample:
    r(k) = c                                (contamination: flat in lag)
         + rm * Phi((k - mu) / sd)          (true ISIs: refractory recovery ramp, monotone)
         + A * Gamma(k; shape, scale)       (duplicates: one bump, a density in k)
Posterior P(dup | k) = A Gamma / r. Thresholds:
  k50  : last lag with P(dup|k) >= 0.5 (equal costs);  k67/k33: costs 2:1 / 1:2 (remove-genuine : keep-dup)
  k90/k95: lag below which 90/95% of the duplicate mass lies (coverage rule)
  FDR10: largest window W (lags 8..W) whose removed pairs are <= 10% non-duplicate (in excess of the
         current window)
Uncertainty: bootstrap over units (200 resamples). Removal counts: KS4's chained rule (later spike
within W of the last kept spike of the same unit) applied to the pre-dedup spikes for each W.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import gamma as G, norm

sys.path.insert(0, str(Path(__file__).parent))
from common import KOBE, MEC, load_ks4_full, load_phy, load_traced, pairs_within  # noqa

HERE = Path(__file__).parent
K = np.arange(8, 91)
NBOOT = 200


def unit_hist(t, clu):
    I, J = pairs_within(t, clu, 600)
    lag = t[J] - t[I]
    ids, ui = np.unique(clu, return_inverse=True)
    u = ui[I]
    H = np.zeros((ids.size, 91))
    m = lag <= 90
    np.add.at(H, (u[m], lag[m]), 1)
    b = (lag >= 150) & (lag < 600)
    Lu = np.bincount(u[b], minlength=ids.size) / 450.0
    return H, Lu, np.bincount(ui, minlength=ids.size)


def rate(p, k):
    c, rm, mu, sd, A, a, s = p
    return c + rm * norm.cdf((k - mu) / sd) + A * G.pdf(k, a, scale=s)


def unpack(x):
    return np.array([np.exp(x[0]), np.exp(x[1]), x[2], np.exp(x[3]), np.exp(x[4]), np.exp(x[5]), np.exp(x[6])])


def fit(h, L, x0=None):
    def nll(x):
        mu = L * rate(unpack(x), K)
        return np.sum(mu - h[K] * np.log(mu + 1e-12))
    starts = [x0] if x0 is not None else [np.array([np.log(.15), np.log(.6), m, np.log(15), np.log(A), np.log(a), np.log(s)])
                                          for m in (60, 75) for A in (1, 3) for a, s in ((4, 3), (8, 1.5))]
    best = None
    for x in starts:
        r = minimize(nll, x, method='Nelder-Mead', options=dict(maxiter=8000, maxfev=8000, xatol=1e-5, fatol=1e-4))
        if best is None or r.fun < best.fun:
            best = r
    return best.x, best.fun


def thresholds(p):
    k = np.arange(8, 61)
    c, rm, mu, sd, A, a, s = p
    D = A * G.pdf(k, a, scale=s)
    P = D / rate(p, k)
    out = {}
    for lab, q in (('k33', 1 / 3), ('k50', .5), ('k67', 2 / 3)):
        w = np.where(P >= q)[0]
        out[lab] = int(k[w.max()]) if w.size else 7
    for lab, q in (('k90', .9), ('k95', .95)):
        out[lab] = float(G.ppf(q, a, scale=s))
    nond = rate(p, k) - D
    cumD, cumN = np.cumsum(D), np.cumsum(nond)
    ok = np.where(cumN / (cumD + cumN) <= 0.10)[0]
    out['FDR10'] = int(k[ok.max()]) if ok.size else 7
    out['mode'] = float((a - 1) * s) if a > 1 else 0.0
    out['dup_frac_below_8'] = float(G.cdf(7.5, a, scale=s))
    return out, P


def chained_removed(t, clu, W):
    """Spikes removed by KS4's rule with dt = W + 1 (removes lags <= W from the last kept spike)."""
    o = np.lexsort((t, clu))
    ts, cs = t[o], clu[o]
    removed = 0
    keep_t = ts.copy()
    # iterative: a spike is removed if within W of the last KEPT spike; resolve chains by passes
    newu = np.r_[True, cs[1:] != cs[:-1]]
    kept = np.ones(ts.size, bool)
    last = np.full(ts.size, -10 ** 12, np.int64)
    # numba-free vectorised approximation is wrong for chains; do a simple loop over candidates only
    cand = np.where(~newu & (ts - np.r_[ts[0], ts[:-1]] <= W))[0]
    lastkept = {}
    for i in cand:
        j = i - 1
        while j >= 0 and cs[j] == cs[i] and not kept[j]:
            j -= 1
        if j >= 0 and cs[j] == cs[i] and ts[i] - ts[j] <= W:
            kept[i] = False
    return int((~kept).sum())


def sources():
    for site in ('kobe', 'mec'):
        for v in ('default', 'H8'):
            d = load_ks4_full(f'{site}_{v}_s1')
            yield f'{site} KS4 {v} pre-dedup', d['t'], d['clu'], True
    for site, root in (('kobe', KOBE / 'runs'), ('mec', MEC / 'runs')):
        d = load_phy(root / 'ks25_standard_r1')
        yield f'{site} KS2.5 standard r1', d['t'], d['clu'], False
        d = load_traced(f'{site}_s1', 'extracted')
        yield f'{site} KS2.5 extracted', d['t'], d['clu'], False


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    allres = {}
    for name, t, clu, do_rm in sources():
        H, Lu, ns = unit_hist(t, clu)
        h, L = H.sum(0), Lu.sum()
        x, f = fit(h, L)
        p = unpack(x)
        th, P = thresholds(p)
        boots = []
        for b in range(NBOOT):
            idx = rng.integers(0, H.shape[0], H.shape[0])
            xb, _ = fit(H[idx].sum(0), Lu[idx].sum(), x0=x)
            boots.append(thresholds(unpack(xb))[0])
        q = lambda key: np.percentile([bb[key] for bb in boots], [2.5, 97.5])  # noqa
        exc_pairs = L * p[4] * (1 - G.cdf(7.5, p[5], scale=p[6]))
        print(f'\n== {name}: N={t.size}  units={H.shape[0]}  c={p[0]:.3f} ramp rm={p[1]:.2f} mid={p[2]:.1f} sd={p[3]:.1f} '
              f'A={p[4]:.2f} gamma(shape={p[5]:.2f}, scale={p[6]:.2f})  dup pairs at k>=8: {exc_pairs:.0f} '
              f'({1000 * exc_pairs / t.size:.2f}/1000 spikes)')
        print('  P(dup|k) k=8..30: ' + ' '.join(f'{P[k - 8]:.2f}' for k in range(8, 31)))
        for key in ('mode', 'k33', 'k50', 'k67', 'k90', 'k95', 'FDR10', 'dup_frac_below_8'):
            lo, hi = q(key)
            print(f'  {key:16s} {th[key]:7.2f}  95% CI [{lo:.2f}, {hi:.2f}]'
                  + (f'  ({th[key] / 30:.3f} ms)' if key not in ('dup_frac_below_8',) else ''))
        if do_rm:
            print('  window W (remove lags <= W): spikes removed (chained), and model split of removed pairs at 8..W')
            k = np.arange(8, 91)
            D = p[4] * G.pdf(k, p[5], scale=p[6]) * L
            N = (rate(p, k) * L) - D
            for W in (6, 9, 12, 15, 18, 21, 24, 30):
                n = chained_removed(t, clu, W)
                dd, nn = D[:max(W - 7, 0)].sum(), N[:max(W - 7, 0)].sum()
                print(f'    W={W:2d} ({(W + 1) / 30:.3f} ms window): removed {n:6d} ({100 * n / t.size:.3f}%)'
                      f'  beyond current: dup {dd:7.0f}  non-dup {nn:7.0f}  frac dup {dd / max(dd + nn, 1):.2f}', flush=True)
        allres[name] = dict(p=p, th=th, boots=boots)
    np.save(HERE / 'mixture_results.npy', allres, allow_pickle=True)
