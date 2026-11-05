"""Measurements 3 and 4: sample complexity of a template, and non-stationarity. Reads snippets.py's outputs.

All eligible units of the sort's final round (seed 1), labels as saved. Feature space: KS4's, 6 PCs on the 10 nearest
channels of the unit's peak (tF's shape), batch not residual (no overlap subtraction).

Measured: for n in NS, the mean of n random spikes of the unit's reservoir (uniform in time, at most 4000 spikes) against
the mean of all the unit's other spikes, 20 draws (seed 0), median cosine. Requires N >= 2n so that the reference
isn't noisier than the estimate: high-rate bias, so the number of units at each n is given.

Closed form, for all units: per-spike scatter s2 = mean |x|^2 - |mean|^2, signal |T|^2 = |mean|^2 - s2/N (debiased).
For iid spikes around T the cosine with the truth is 1/sqrt(1 + s2/(n |T|^2)), so n*(c) = (s2/|T|^2) / (1/c^2 - 1).

Batches: n*/(2 r), and from the unit's real train the number of consecutive 2-s batches, from 20 random starts,
to accumulate n* spikes.

Non-stationarity: T_first vs T_last (5 min at each end of the 20-min excerpt; 20 min of the 2-hour sorts). Control:
random halves of the union of the two epochs, same spike numbers. Cosines, and debiased shape similarity
<a,b>/sqrt((|a|^2 - s2/na)(|b|^2 - s2/nb)). Requires >= 30 spikes in each epoch; units below are reported, not hidden.
Batch occupancy: fraction of 2-s batches with >= 1, 3, 10 spikes, against Poisson at the unit's mean rate.

    python m34_samples.py [H8] [default] [2h]
"""
import json
import sys

import numpy as np
from scipy.stats import poisson

from common import BATCH, DUR, OUT, SORTS, Sort, rates, scored, sort_dir, strata

NS = (10, 30, 100, 300, 1000)
CS = (0.9, 0.95, 0.99)
RNG = 0
LW = OUT.parents[1] / 'iteration/windows'


def cosine(a, b):
    return (a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12)


def batches_to(cnt, need, rng, n=20):
    """Median and 90th percentile, over random starts, of consecutive batches to accumulate need spikes (circular); nan if never."""
    if cnt.sum() < need:
        return np.nan, np.nan
    c = np.cumsum(np.r_[cnt, cnt])
    starts = rng.integers(0, cnt.size, n)
    base = np.where(starts > 0, c[starts - 1], 0)
    k = np.searchsorted(c, base + need) - starts + 1
    return float(np.median(k)), float(np.quantile(k, .9))


def curves(name, ex, setting, d):
    z = np.load(OUT / f'{name}.npz')
    r = np.load(OUT / f'{name}_res_idx.npz')
    feat = np.load(OUT / f'{name}_res.npy', mmap_mode='r')
    s = Sort(d, ex, setting)
    elig = np.array(sorted(s.eligible()))
    rate = rates(s, list(elig))
    st = strata(rate, s.amp)
    t_, _, c_ = s.cleaned()
    o = np.argsort(r['unit'], kind='stable')
    units, lim = np.unique(r['unit'][o], return_index=True)
    lim = dict(zip(units, zip(lim, np.r_[lim[1:], o.size])))
    rng = np.random.default_rng(RNG)
    nb = int(np.ceil(t_[-1] / BATCH))
    bi = (t_ // BATCH).astype(np.int64)
    oc = np.argsort(c_, kind='stable')
    climit = np.searchsorted(c_[oc], np.arange(s.n_units + 1))
    rows = []
    cur = np.full((elig.size, len(NS)), np.nan)
    for i, u in enumerate(elig):
        N = z['n_all'][u]
        S = z['s_all'][u, :10]
        M = S / N
        sig2 = z['sq10'][u] / N - (M * M).sum()
        T2 = (M * M).sum() - sig2 / N
        a, b = lim[u]
        x = np.asarray(feat[o[a:b]]).astype(np.float64)
        for k, n in enumerate(NS):
            if x.shape[0] >= n and N >= 2 * n:
                v = []
                for _ in range(20):
                    p = rng.choice(x.shape[0], n, replace=False)
                    m = x[p].sum(0)
                    v.append(cosine(m / n, (S - m) / (N - n)))
                cur[i, k] = np.median(v)
        cnt = np.bincount(bi[oc[climit[u]:climit[u + 1]]], minlength=nb)
        row = dict(unit=int(u), rate=rate[u], n=int(N), sig2=sig2, T2=T2, rq=st[u][0], aq=st[u][1], amp=s.amp.get(u, np.nan))
        for c in CS:
            need = sig2 / max(T2, 1e-9) / (1 / c ** 2 - 1)
            row[f'n{c}'] = need
            row[f'b{c}'] = need / (2 * rate[u])
            row[f'bm{c}'], row[f'bp{c}'] = batches_to(cnt, max(1, int(np.ceil(need))), rng)
        row['occ1'], row['occ3'], row['occ10'] = (cnt >= 1).mean(), (cnt >= 3).mean(), (cnt >= 10).mean()
        lam = max(N / nb, 1e-9)
        row['pocc3'], row['pocc10'] = poisson.sf(2, lam), poisson.sf(9, lam)
        row['half_b'] = np.searchsorted(np.cumsum(np.sort(cnt)[::-1]), 0.5 * cnt.sum()) / nb       # batches holding half its spikes
        rows.append(row)
    keys = list(rows[0])
    res = {k: np.array([x[k] for x in rows]) for k in keys}
    res['cur'] = cur
    np.savez(OUT / f'm3_{name}.npz', **res)
    print(name, 'done', flush=True)


def epochs(name, units, tag):
    z = np.load(OUT / f'{name}.npz')
    rows = []
    for u in units:
        nf, nl, N = z['n_first'][u], z['n_last'][u], z['n_all'][u]
        if nf < 30 or nl < 30:
            rows.append((u, nf, nl, *(np.nan,) * 7))
            continue
        F, L = z['s_first'][u, :10] / nf, z['s_last'][u, :10] / nl
        # the two halves of the union of the epochs, same spike numbers
        a0 = z['s_first0'][u, :10] + z['s_last0'][u, :10]
        a1 = z['s_first1'][u, :10] + z['s_last1'][u, :10]
        n0, n1 = z['n_first0'][u] + z['n_last0'][u], z['n_first1'][u] + z['n_last1'][u]
        if n0 < 15 or n1 < 15:
            rows.append((u, nf, nl, *(np.nan,) * 7))
            continue
        A0, A1 = a0 / n0, a1 / n1
        M = z['s_all'][u, :10] / N
        sig2 = z['sq10'][u] / N - (M * M).sum()
        deb = (F * L).sum() / np.sqrt(max((F * F).sum() - sig2 / nf, 1e-9) * max((L * L).sum() - sig2 / nl, 1e-9))
        deb0 = (A0 * A1).sum() / np.sqrt(max((A0 * A0).sum() - sig2 / n0, 1e-9) * max((A1 * A1).sum() - sig2 / n1, 1e-9))
        rows.append((u, nf, nl, cosine(F, L), cosine(A0, A1), deb, deb0, (F * L).sum() / (F * F).sum(),
                     np.sqrt(((F - L) ** 2).sum() / (F * F).sum()), np.sqrt(((A0 - A1) ** 2).sum() / (A0 * A0).sum())))
    np.save(OUT / f'm4_{tag}.npy', np.array(rows, dtype=float))
    print(tag, 'done', flush=True)


def occupancy_2h(ex):
    d = SORTS / f'{ex}_2h_repeat_s1'
    t = np.load(d / 'spike_times.npy', mmap_mode='r').ravel()
    c = np.load(d / 'spike_clusters.npy', mmap_mode='r').ravel()
    bi = (np.asarray(t) // BATCH).astype(np.int64)
    c = np.asarray(c).astype(np.int64)
    nb = int(bi.max()) + 1
    H = np.bincount(c * nb + bi, minlength=(int(c.max()) + 1) * nb).reshape(-1, nb)
    return H


if __name__ == '__main__':
    parts = sys.argv[1:] or ('H8', 'default', '2h')
    for ex in ('kobe', 'mec'):
        if 'H8' in parts:
            curves(f'{ex}_H8_s1_r4', ex, 'H8', sort_dir(ex, 'H8', 1, 4))
            epochs(f'{ex}_H8_s1_r4', sorted(scored(sort_dir(ex, 'H8', 1, 4), ex)), f'{ex}_H8')
        if 'default' in parts:
            curves(f'{ex}_default_s1_r3', ex, 'default', sort_dir(ex, 'default', 1, 3))
            epochs(f'{ex}_default_s1_r3', sorted(scored(sort_dir(ex, 'default', 1, 3), ex, 'default')), f'{ex}_default')
        if '2h' in parts:
            w = LW / f'{ex}_2h_rep4'
            if ex == 'kobe':
                units = sorted(int(k) for k in json.loads((w / 'evaluation.json').read_text())['S_i'])
            else:
                winners = json.loads((OUT.parents[1] / 'iteration/transfer_mec_long.json').read_text())['winner']
                units = sorted(int(k) for x in winners if x['sort'].endswith('mec_2h_rep4') for k in x['S_i'])
            epochs(f'{ex}_2h_r4', units, f'{ex}_2h')
            H = occupancy_2h(ex)[units]
            np.savez(OUT / f'm4_occ_{ex}_2h.npz', rate=H.sum(1) / (H.shape[1] * 2.0), occ1=(H >= 1).mean(1), occ3=(H >= 3).mean(1),
                     occ10=(H >= 10).mean(1), pocc10=poisson.sf(9, np.maximum(H.mean(1), 1e-9)), silent10min=np.array([
                         (np.convolve((h > 0).astype(float), np.ones(300), 'valid') == 0).mean() for h in H]))
