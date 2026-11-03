"""Cross-unit coincidences at sample resolution, KS4 vs KS2.5 on the same excerpts: CCG of unit pairs
(>= 1000 spikes each) whose positions (median spike position) are < 60 um apart, 60-150 um, and
300-600 um (random subset), as |lag| histogram 0..300 samples normalised by the 5-10 ms (150-300) level.
Raw sorter output and after the harness's cleaning (removed_event | removed_duplicate).

Answers: is there short-lag excess between neighbouring units in both sorters (real synchrony, or a
shared artefact) or only in KS4 (a sorter mechanism); at which lags; does it depend on distance.
"""
import json
from pathlib import Path

import numpy as np

SORTS = {
    'kobe_ks4_H8': Path(r'C:\data\ks4_tuning\exploratory\autocurate\sorts\kobe_H8_s1'),
    'kobe_ks4_default': Path(r'C:\data\ks4_tuning\exploratory\autocurate\sorts\kobe_default_s1'),
    'kobe_ks25': Path(r'C:\data\ks4_tuning\exploratory\ks25_traced\kobe_s1'),
    'mec_ks4_H8': Path(r'C:\data\ks4_tuning\exploratory\autocurate\sorts\mec_H8_s1'),
    'mec_ks25': Path(r'C:\data\ks4_tuning\exploratory\ks25_traced\mec_s1'),
}
OUT = Path(__file__).parent / 'out'
MAXLAG = 300
RNG = np.random.default_rng(0)


def ccg_abs(ta, tb, w):
    lo = np.searchsorted(tb, ta - w, 'left')
    hi = np.searchsorted(tb, ta + w, 'right')
    n = hi - lo
    i = np.repeat(np.arange(ta.size), n)
    j = lo[i] + np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    return np.bincount(np.abs(tb[j] - ta[i]), minlength=w + 1)[:w + 1]


def run(d, clean):
    t = np.load(d / 'spike_times.npy').ravel().astype(np.int64)
    c = np.load(d / 'spike_clusters.npy').ravel().astype(np.int64)
    pos = np.load(d / 'spike_positions.npy')
    if clean:
        z = np.load(d / 'cleaning.npz')
        assert np.all(z['spike_order'] == np.arange(t.size))
        k = ~(z['removed_event'] | z['removed_duplicate'])
        t, c, pos = t[k], c[k], pos[k]
    o = np.argsort(t, kind='stable')
    t, c, pos = t[o], c[o], pos[o]
    ids, cnt = np.unique(c, return_counts=True)
    ids = ids[cnt >= 1000]
    trains = {u: t[c == u] for u in ids}
    P = np.array([np.median(pos[c == u], 0) for u in ids])
    D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    iu, ju = np.triu_indices(ids.size, 1)
    dd = D[iu, ju]
    res = {}
    for nm, (a, b, cap) in {'lt60': (0, 60, 4000), '60_150': (60, 150, 3000), '300_600': (300, 600, 1500)}.items():
        sel = np.where((dd >= a) & (dd < b))[0]
        if sel.size > cap:
            sel = RNG.choice(sel, cap, replace=False)
        h = np.zeros(MAXLAG + 1)
        for s in sel:
            h += ccg_abs(trains[ids[iu[s]]], trains[ids[ju[s]]], MAXLAG)
        base = h[150:301].mean()
        res[nm] = dict(n_pairs=int(sel.size), norm=(h / base).tolist())
    return res


def bands(n):
    n = np.asarray(n)
    return {k: float(n[a:b + 1].mean()) for k, (a, b) in
            {'0': (0, 0), '1-3': (1, 3), '4-6': (4, 6), '7-9': (7, 9), '10-12': (10, 12), '13-15': (13, 15),
             '16-21': (16, 21), '22-30': (22, 30), '31-45': (31, 45), '60-90': (60, 90)}.items()}


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    allres = {}
    for name, d in SORTS.items():
        for clean in [False, True]:
            key = f"{name}_{'clean' if clean else 'raw'}"
            r = run(d, clean)
            allres[key] = r
            for nm, v in r.items():
                b = bands(v['norm'])
                print(f"{key:26s} {nm:8s} n={v['n_pairs']:5d} " + ' '.join(f"{a}:{x:.2f}" for a, x in b.items()),
                      flush=True)
    json.dump(allres, open(OUT / 'cross_unit_lags.json', 'w'))
