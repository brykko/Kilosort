"""How noisy labels would limit a template however many spikes it sees: the bias floor.

All eligible units of round 4 (seed 1, H8), each with the round-2 unit holding most of its cleaned spikes (no threshold).
The round-2 labels stand in for the labels an online learner would have after its first templates (optimistic: round 2
has had a full clustering and full-data templates). Per unit, with the batch features on the channels shared by the
two units' 32-channel sets:

    cos(T2_full, T4_full)  the plateau: what n -> infinity gives with round-2 labels
    n_equal                spikes at which sampling noise of T4's own estimator equals the bias, s2 / |T2 - T4|^2

    python m3_floor.py
"""
import numpy as np

from common import OUT, Link, Sort, rates, sort_dir, strata


def run(ex):
    a, b = (Sort(sort_dir(ex, 'H8', 1, r), ex, 'H8') for r in (4, 2))
    su = Link(a, b).successors()
    z4 = np.load(OUT / f'{ex}_H8_s1_r4.npz')
    z2 = np.load(OUT / f'{ex}_H8_s1_r2.npz')
    s4, n4, c4, q4 = z4['s_all'], z4['n_all'], z4['chans'], z4['sq32']
    s2, n2, c2 = z2['s_all'], z2['n_all'], z2['chans']
    elig = sorted(a.eligible())
    rate = rates(a, elig)
    st = strata(rate, a.amp)
    rows = []
    for u in elig:
        v = su.get(u, (-1, 0))[0]
        if v < 0 or n2[v] == 0:
            rows.append((u, rate[u], *st[u], np.nan, np.nan, np.nan))
            continue
        _, ia, ib = np.intersect1d(c4[u], c2[v], return_indices=True)
        x = s4[u][ia] / n4[u]
        y = s2[v][ib] / n2[v]
        m = s4[u] / n4[u]
        sig2 = (q4[u] / n4[u] - (m * m).sum()) * ia.size / 32
        d2 = ((x - y) ** 2).sum()
        rows.append((u, rate[u], *st[u], (x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()), sig2 / max(d2, 1e-12), n4[u]))
    r = np.array(rows, float)
    np.save(OUT / f'm3floor_{ex}.npy', r)
    c, ne = r[:, 4], r[:, 5]
    print(ex, 'units', r.shape[0], 'plateau cos .1/.5/.9', np.nanquantile(c, [.1, .5, .9]).round(4),
          'share < 0.99', np.nanmean(c < .99).round(3), 'share < 0.95', np.nanmean(c < .95).round(3), 'n_equal .1/.5/.9', np.nanquantile(ne, [.1, .5, .9]).round(0))
    for i in range(3):
        m = r[:, 2] == i
        print('  rate tercile', i, 'median plateau', np.nanmedian(c[m]).round(4), 'share < .99', np.nanmean(c[m] < .99).round(3), 'median n_equal', np.nanmedian(ne[m]).round(0), 'batches', np.nanmedian(ne[m] / (2 * r[m, 1])).round(1))


if __name__ == '__main__':
    run('kobe')
    run('mec')
