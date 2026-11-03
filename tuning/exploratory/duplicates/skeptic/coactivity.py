"""Are the within-unit 0.3-0.5 ms pairs special in time or place? (competing explanations: population
synchrony, artefacts/common-mode residues, probe ends, high-amplitude units.)

For a sort: 'bump' = same-unit pairs at 9..15 samples, 'ctrl' = same-unit pairs at 60..90 samples,
'rand' = random spikes. For the first member of each pair: number of spikes of OTHER units within +-0.5 ms,
split into near (unit depth within 100 um) and far (> 300 um); whether it lies in a harness multi-unit
event interval (cleaning.npz). Per unit: bump excess (pairs at 9..15 minus 7 x mean per-lag count at
30..39), against amplitude, rate and depth (Spearman), and how concentrated the excess is across units.
Writes coactivity_<name>.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

OUTDIR = Path(__file__).resolve().parent


def pairs(t, c, lo, hi):
    key = c * (1 << 40) + t
    o = np.argsort(key, kind='stable')
    k = key[o]
    I, J = [], []
    d = 1
    while True:
        dk = k[d:] - k[:-d]
        if not (dk <= hi).any():
            break
        m = (dk >= lo) & (dk <= hi)
        I.append(o[:-d][m]); J.append(o[d:][m])
        d += 1
    return np.concatenate(I), np.concatenate(J)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--sort', required=True)
    a = ap.parse_args()
    sort = Path(a.sort)
    rng = np.random.default_rng(0)
    t = np.load(sort / 'spike_times.npy').ravel().astype(np.int64)
    c = np.load(sort / 'spike_clusters.npy').ravel().astype(np.int64)
    amp = np.load(sort / 'amplitudes.npy').ravel()
    y = np.load(sort / 'spike_positions.npy')[:, 1]
    cln = np.load(sort / 'cleaning.npz')
    ev = np.zeros(t.size, bool)
    ev[cln['spike_order']] = cln['removed_event']
    ids = np.unique(c)
    depth = np.zeros(ids.max() + 1); amp_u = np.zeros(ids.max() + 1)
    for u in ids:
        m = c == u
        depth[u] = np.median(y[m]); amp_u[u] = np.median(amp[m])
    n_u = np.bincount(c, minlength=ids.max() + 1)
    yd = depth[c]
    o = np.argsort(t, kind='stable')
    bands = np.arange(np.floor(yd.min() / 50) * 50, yd.max() + 50, 50)
    bt = [np.sort(t[(yd >= b) & (yd < b + 50)]) for b in bands]

    def coact(q):
        tq, dq, cq = t[q], yd[q], c[q]
        near = np.zeros(q.size); far = np.zeros(q.size)
        for b, tb in zip(bands, bt):
            cnt = np.searchsorted(tb, tq + 15, 'right') - np.searchsorted(tb, tq - 15, 'left')
            dist = np.abs(b + 25 - dq)
            near += cnt * (dist <= 100)
            far += cnt * (dist > 300)
        # remove the unit's own spikes from 'near' (same depth band)
        key = np.sort(c * (1 << 40) + t)
        kq = cq * (1 << 40) + tq
        own = np.searchsorted(key, kq + 15, 'right') - np.searchsorted(key, kq - 15, 'left')
        return near - own, far

    bi, bj = pairs(t, c, 9, 15)
    ci, cj = pairs(t, c, 60, 90)
    ri = rng.choice(t.size, 200000, replace=False)
    out = {}
    for name, q in (('bump', bi), ('ctrl', ci), ('rand', ri)):
        q = q if q.size <= 200000 else rng.choice(q, 200000, replace=False)
        nr, fr = coact(q)
        out[name] = {'n': int(q.size), 'near_mean': float(nr.mean()), 'far_mean': float(fr.mean()),
                     'far_ge3': float(np.mean(fr >= 3)), 'in_event': float(ev[q].mean()),
                     'amp_rel': float(np.median(amp[q] / amp_u[c[q]]))}
    # per-unit excess
    lagc = {}
    key = c * (1 << 40) + t
    k = np.sort(key)
    H = {}
    d = 1
    while True:
        dk = k[d:] - k[:-d]
        if not (dk <= 40).any():
            break
        m = dk <= 40
        uu = (k[:-d][m] >> 40)
        np.add.at(lagc.setdefault('dummy', np.zeros(1)), 0, 0)
        H.setdefault('u', []).append(uu); H.setdefault('l', []).append(dk[m])
        d += 1
    uu = np.concatenate(H['u']); ll = np.concatenate(H['l'])
    M = np.zeros((ids.max() + 1, 41))
    np.add.at(M, (uu, ll), 1)
    exc = M[:, 9:16].sum(1) - 7 * M[:, 30:40].mean(1)
    sel = n_u >= 1000
    e = exc[sel]
    es = np.sort(np.clip(e, 0, None))[::-1]
    out['per_unit'] = {
        'n_units': int(sel.sum()), 'total_excess': float(e.sum()),
        'share_top10_units': float(es[:10].sum() / max(es.sum(), 1)),
        'share_top50_units': float(es[:50].sum() / max(es.sum(), 1)),
        'spearman_excess_per_spike_vs_amp': float(spearmanr(e / n_u[sel], amp_u[sel])[0]),
        'spearman_excess_per_spike_vs_rate': float(spearmanr(e / n_u[sel], n_u[sel])[0]),
        'spearman_excess_per_spike_vs_depth': float(spearmanr(e / n_u[sel], depth[sel])[0]),
        'excess_per_1k_by_amp_quartile': [float(e[(amp_u[sel] >= lo) & (amp_u[sel] <= hi)].sum() /
                                                n_u[sel][(amp_u[sel] >= lo) & (amp_u[sel] <= hi)].sum() * 1e3)
                                          for lo, hi in zip(*[np.quantile(amp_u[sel], [0, .25, .5, .75])[None].T.ravel(),
                                                              np.quantile(amp_u[sel], [.25, .5, .75, 1])])],
        'excess_per_1k_by_depth_quintile': [float(e[(depth[sel] >= lo) & (depth[sel] <= hi)].sum() /
                                                  n_u[sel][(depth[sel] >= lo) & (depth[sel] <= hi)].sum() * 1e3)
                                            for lo, hi in zip(np.quantile(depth[sel], [0, .2, .4, .6, .8]),
                                                              np.quantile(depth[sel], [.2, .4, .6, .8, 1]))],
    }
    (OUTDIR / f'coactivity_{a.name}.json').write_text(json.dumps(out, indent=1))
    print(a.name, json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
