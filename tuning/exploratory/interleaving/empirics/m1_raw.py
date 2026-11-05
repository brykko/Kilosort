"""Measurement 1, raw estimator and noise floors (seed 1 preprocessing; reads snippets.py's outputs and m1_convergence.py's chains).

Each round's template re-estimated the same way, the mean of the batch's features over the round's cleaned spikes, on
the channels shared by the two units' 32-channel sets, so that a difference between rounds comes from the spikes and not
from the estimator. For the middle unit (round 3 H8, round 2 default) and its neighbour round's unit, in both chains:

    step   |Ta - Tb|^2 / |T|^2
    null   what the step would be if the spikes unique to each round were random draws of the same unit:
           s2 (1/na + 1/nb - 2 m/(na nb)), m = shared spikes, s2 = per-spike scatter
    noise  s2 / n / |T|^2: the template's own sampling error; split-half |h0 - h1|^2 / 4 estimates the same thing

Seeds: T of seed 1 and seed 2 at the same round through seed 1's preprocessing, mutual pairs (m from agreement);
the seed-to-seed difference for one cell, without seed 3's drift difference.

    python m1_raw.py [H8] [default]
"""
import sys

import numpy as np

from common import OUT, Link, Sort, ROUNDS, rates, sort_dir


def get(z, u):
    return z['s_all'][u] / max(z['n_all'][u], 1), z['chans'][u], z['n_all'][u], z['sq32'][u] / max(z['n_all'][u], 1)


def diff(a, ca, b, cb):
    """Rows of a and b on their shared channels."""
    shared, ia, ib = np.intersect1d(ca, cb, return_indices=True)
    return a[ia], b[ib], shared.size


def step(za, u, zb, v, shared):
    a, ca, na, ea = get(za, u)
    b, cb, nb, _ = get(zb, v)
    if na == 0 or nb == 0:
        return (np.nan,) * 6
    x, y, k = diff(a, ca, b, cb)
    if k == 0:
        return (np.nan,) * 6
    s2 = ea - (a * a).sum()                       # per-spike scatter on 32 channels (a's set)
    T2 = (a * a).sum() - s2 / na
    d2 = ((x - y) ** 2).sum()
    null = s2 * (k / a.shape[0]) * (1 / na + 1 / nb - 2 * shared / (na * nb))
    c = (x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum())
    h = za['s_h0'][u] / max(za['n_h0'][u], 1) - za['s_h1'][u] / max(za['n_h1'][u], 1)
    sh = (h * h).sum() / 4
    return d2 / T2, null / T2, c, sh / T2, s2 / na / T2, na


def link_shared(l):
    return dict(zip(zip(l.ua.tolist(), l.ub.tolist()), l.n.tolist()))


def run(ex, setting):
    rounds = ROUNDS[setting]
    mid = 3 if setting == 'H8' else 2
    Z = {r: np.load(OUT / f'{ex}_{setting}_s1_r{r}.npz') for r in rounds}
    Zm = {r: {k: Z[r][k] for k in Z[r].files} for r in rounds}
    m = np.load(OUT / f'm1_{ex}_{setting}_s1.npz')
    S = {r: Sort(sort_dir(ex, setting, 1, r), ex, setting) for r in rounds}
    out = {'unit': m['unit'], 'rq': m['rq'], 'aq': m['aq']}
    for o in rounds:
        if o == mid:
            continue
        sh = link_shared(Link(S[mid], S[o]))
        for mode in ('all', 'mut'):
            rows = []
            for u, v in zip(m['unit'], m[f'{mode}_{o}_unit']):
                rows.append(step(Zm[mid], int(u), Zm[o], int(v), sh.get((int(u), int(v)), 0)) if v >= 0 else (np.nan,) * 6)
            out[f'{mode}_{o}'] = np.array(rows)
    last = rounds[-1]
    zs = {r: {k: v for k, v in np.load(OUT / f'{ex}_{setting}_s2_r{r}.npz').items()} for r in rounds if (OUT / f'{ex}_{setting}_s2_r{r}.npz').exists()}
    for r, z2 in zs.items():
        p = np.load(OUT / f'm1_seedpairs_{ex}_{setting}_s1s2_r{r}.npy')
        rows = []
        for u, v, g in p[:, :3]:
            u, v = int(u), int(v)
            n1, n2 = Zm[r]['n_all'][u], z2['n_all'][v]
            rows.append(step(Zm[r], u, z2, v, g * (n1 + n2) / (1 + g)))
        out[f'seed_r{r}'] = np.array(rows)
    np.savez(OUT / f'm1raw_{ex}_{setting}.npz', **out)
    print(ex, setting, 'done', flush=True)


if __name__ == '__main__':
    for setting in sys.argv[1:] or ('H8', 'default'):
        for ex in ('kobe', 'mec'):
            run(ex, setting)
