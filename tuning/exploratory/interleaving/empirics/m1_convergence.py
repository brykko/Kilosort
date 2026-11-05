"""Measurement 1: do templates converge across rounds, and do labels?

From the saved sorts only (Wall.npy, spike times, clusters, depths, cleaning). For each round the middle unit (round 3
for H8, round 2 for default) is followed backwards and forwards:

- 'all': every eligible unit of the middle round, successor/predecessor = the unit of the other round (eligible or not)
  holding most of its cleaned spikes, no threshold. Likely bias: towards more change.
- 'mutual': mutual best at compare.agreement >= 0.5 among eligible units, on both links. Likely bias: towards less.

Same-event link between two rounds: nearest spike within TOL_SAMPLES and TOL_UM in depth, over cleaned spikes. Time alone
can't be used: 12 M spikes in 1200 s is 10 spikes per ms across the probe. Chance is measured by shifting one sort by
1000 samples (33 ms) and rerunning the link.

Template comparison in the time domain (Wall @ wPCA, own sort's wPCA): cosine at lag 0, maximum cosine over lags
-5..5, normalised L2 |A - B| / |A| at lag 0, amplitude ratio |B|/|A|.

Seed noise: same cell in two seeds (mutual >= 0.5 at the same round), pairs s1-s2, s2-s3, s3-s1, every round; and for
H8 the fixed set of cells linked in both seeds through r2<->r4 and across seeds at both r2 and r4, so that the seed
distance at round 2 and round 4 is over the same cells.

    python m1_convergence.py
"""
import gc
import json

import numpy as np

from common import DUR, OUT, ROUNDS, SEEDS, Link, Sort, mutual_pairs, rates, sort_dir, strata

LAGS = range(-5, 6)


def tcache(s):
    o = np.load(s.d / 'ops.npy', allow_pickle=True).item()
    T = np.load(s.d / 'Wall.npy') @ np.asarray(o['wPCA'])              # units x chans x nt
    return T.astype(np.float32)


def compare(a, b):
    """cos0, cos over lags, |a-b|/|a|, |b|/|a| for two chans x nt templates, on the 32 channels of largest union norm."""
    ch = np.argsort(-((a ** 2).sum(1) + (b ** 2).sum(1)))[:32]
    a, b = a[ch], b[ch]
    na, nb = np.sqrt((a ** 2).sum()), np.sqrt((b ** 2).sum())
    if na == 0 or nb == 0:
        return np.nan, np.nan, np.nan, np.nan
    c0 = (a * b).sum() / (na * nb)
    cl = max((a[:, max(0, -k):a.shape[1] - max(0, k)] * b[:, max(0, k):b.shape[1] - max(0, -k)]).sum() for k in LAGS) / (na * nb)
    return c0, cl, np.sqrt(((a - b) ** 2).sum()) / na, nb / na


def assign_stats(link, ctrl, elig, succ):
    e = np.isin(link.ca, list(elig))
    sv = np.full(int(link.ca.max()) + 1, -2)
    for u, (v, _) in succ.items():
        sv[u] = v
    rel = (link.cb != sv[link.ca]) & link.ok
    out = {}
    for k, m in (('all', np.ones(e.size, bool)), ('elig', e)):
        s, c = link.ok[m].mean(), ctrl.ok[m].mean()
        out[k] = dict(n=int(m.sum()), shared=float(s), chance=float(c), not_shared_corr=float(1 - (s - c) / (1 - c)),
                      relabelled_of_shared=float(rel[m].sum() / link.ok[m].sum()))
    return out


def seed_pairs(sa, sb, Ta, Tb):
    p = mutual_pairs(sa, sb)
    rows = [(u, v, g, *compare(Ta[u], Tb[v])) for u, (v, g) in p.items()]
    return np.array(rows), len(sa.eligible()), len(sb.eligible())


def run(ex, setting):
    rounds = ROUNDS[setting]
    sorts = {}
    assign = {}
    Ts = {}
    for seed in SEEDS:
        S = {r: Sort(sort_dir(ex, setting, seed, r), ex, setting) for r in rounds}
        T = {r: tcache(S[r]) for r in rounds}
        mid = rounds[1] if setting == 'H8' else rounds[0]
        m = S[mid]
        elig = list(m.eligible())
        rate, amp = rates(m, elig), m.amp
        st = strata(rate, amp)
        res = dict(unit=np.array(elig), rate=np.array([rate[u] for u in elig]), amp=np.array([amp.get(u, np.nan) for u in elig]),
                   rq=np.array([st[u][0] for u in elig]), aq=np.array([st[u][1] for u in elig]))
        for a, b in zip(rounds[:-1], rounds[1:]):
            print(ex, setting, seed, a, b, flush=True)
            fwd, back = Link(S[a], S[b]), Link(S[b], S[a])
            cf = Link(S[a], S[b], shift=1000)
            cb = Link(S[b], S[a], shift=1000)
            sa, sb = fwd.successors(), back.successors()
            assign[f's{seed}_{a}to{b}'] = {'fwd': assign_stats(fwd, cf, S[a].eligible(), sa),
                                           'back': assign_stats(back, cb, S[b].eligible(), sb),
                                           'dt0': float((np.abs(fwd.ta[fwd.ok] - S[b].cleaned()[0][fwd.m[fwd.ok]]) == 0).mean())}
            mut = mutual_pairs(S[a], S[b], fwd, back)
            # the other round's unit for each middle unit, in both senses
            if a == mid:
                best, pair, other = sa, mut, b
            else:
                best, pair, other = sb, {v: (u, g) for u, (v, g) in mut.items()}, a
            for mode, get in (('all', lambda u: best.get(u, (-1, 0))[0]), ('mut', lambda u: pair.get(u, (-1, 0))[0])):
                rows = []
                for u in elig:
                    v = get(u)
                    rows.append((v, *(compare(T[mid][u], T[other][v]) if v >= 0 else (np.nan,) * 4)))
                rows = np.array(rows)
                res[f'{mode}_{other}_unit'] = rows[:, 0].astype(int)
                res[f'{mode}_{other}_c0'], res[f'{mode}_{other}_cl'] = rows[:, 1], rows[:, 2]
                res[f'{mode}_{other}_l2'], res[f'{mode}_{other}_ar'] = rows[:, 3], rows[:, 4]
            del fwd, back, cf, cb
            gc.collect()
        if setting == 'H8':
            # persistence: cosine between the step T3-T2 and the step T4-T3, and T2 to T4 directly
            for mode in ('all', 'mut'):
                p = np.full(len(elig), np.nan)
                d24 = np.full(len(elig), np.nan)
                for i, u in enumerate(elig):
                    a, b = res[f'{mode}_2_unit'][i], res[f'{mode}_4_unit'][i]
                    if a < 0 or b < 0:
                        continue
                    c = T[3][u]
                    x, y = c - T[2][a], T[4][b] - c
                    ch = np.argsort(-(c ** 2).sum(1))[:32]
                    p[i] = (x[ch] * y[ch]).sum() / np.sqrt((x[ch] ** 2).sum() * (y[ch] ** 2).sum() + 1e-12)
                    d24[i] = np.sqrt(((T[4][b][ch] - T[2][a][ch]) ** 2).sum() / (c[ch] ** 2).sum())
                res[f'{mode}_persist'] = p
                res[f'{mode}_l2_24'] = d24
            # direct r2<->r4 pairs, kept for the fixed-cell seed comparison
            sorts[seed, 'p24'] = mutual_pairs(S[2], S[4])
        res['S_mid'] = np.array([m.eligible()[u] for u in elig])
        for r in rounds:
            if r != mid:
                e = S[r].eligible()
                res[f'S_{r}_all'] = np.array([e.get(int(v), np.nan) for v in res[f'all_{r}_unit']])
                res[f'S_{r}_mut'] = np.array([e.get(int(v), np.nan) for v in res[f'mut_{r}_unit']])
        np.savez(OUT / f'm1_{ex}_{setting}_s{seed}.npz', **res)
        for r in rounds:
            sorts[seed, r] = S[r]
            Ts[seed, r] = T[r]
    seed_out = {}
    fixed = {}
    for a, b in ((1, 2), (2, 3), (3, 1)):
        for r in rounds:
            rows, na, nb = seed_pairs(sorts[a, r], sorts[b, r], Ts[a, r], Ts[b, r])
            np.save(OUT / f'm1_seedpairs_{ex}_{setting}_s{a}s{b}_r{r}.npy', rows)
            seed_out[f's{a}s{b}_r{r}'] = dict(pairs=rows.shape[0], na=na, nb=nb)
            print(ex, setting, 'seed', a, b, r, rows.shape, flush=True)
        if setting == 'H8':
            r2 = {int(x[0]): int(x[1]) for x in np.load(OUT / f'm1_seedpairs_{ex}_{setting}_s{a}s{b}_r2.npy')}
            r4 = {int(x[0]): int(x[1]) for x in np.load(OUT / f'm1_seedpairs_{ex}_{setting}_s{a}s{b}_r4.npy')}
            pa, pb = sorts[a, 'p24'], sorts[b, 'p24']
            rows = []
            for u, (u4, _) in pa.items():
                v = r2.get(u)
                if v is None or v not in pb:
                    continue
                v4 = pb[v][0]
                if r4.get(u4) == v4:
                    rows.append((*compare(Ts[a, 2][u], Ts[b, 2][v]), *compare(Ts[a, 4][u4], Ts[b, 4][v4])))
            fixed[f's{a}s{b}'] = np.array(rows)
    np.savez(OUT / f'm1_seedfixed_{ex}_{setting}.npz', **fixed)
    (OUT / f'm1_assign_{ex}_{setting}.json').write_text(json.dumps({'assign': assign, 'seed': seed_out}, indent=1))


if __name__ == '__main__':
    for setting in ('H8', 'default'):
        for ex in ('kobe', 'mec'):
            run(ex, setting)
            gc.collect()
