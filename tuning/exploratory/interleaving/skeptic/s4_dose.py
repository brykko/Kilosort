"""Dose-response inside a sort, no new runs. Does the gain sit where the dictionary was redundant?

(1) Depth bins of 200 um. For each bin: templates at round 2 and round 3 (by template peak channel), S at round 2 and
round 3 (units by median spike depth), refractory violations (ISI < 1.5 ms, uncleaned). Spearman, across bins, of
the relative shrink of the dictionary against the relative change in S and in violations per spike.

(2) Stable unit pairs round 2 -> round 4 (s2_pairs.py). Units by how spread their round-2 spikes were over extraction
templates (effective number, 1 / sum p^2): change in violations per spike and in S_i by tertile of spread, within
tertiles of round-2 violations (units with more violations have more to lose: regression to the mean). Same rule
for all seeds, and for default (r2 -> r3) as a comparison.
"""
from common import *  # noqa


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def viol(t, clu, n_units):
    o = np.lexsort((t, clu))
    t_, c_ = t[o], clu[o]
    d = np.diff(t_)
    v = (d < 45) & (c_[1:] == c_[:-1])
    return np.bincount(c_[1:][v], minlength=n_units)


def load(d):
    t = np.load(d / 'spike_times.npy').ravel().astype(np.int64)
    c = np.load(d / 'spike_clusters.npy').ravel().astype(np.int64)
    y = np.load(d / 'spike_positions.npy')[:, 1]
    tpl = np.load(d / 'spike_detection_templates.npy').ravel().astype(np.int64)
    return t, c, y, tpl


def depths(c, y):
    o = np.argsort(c, kind='stable')
    n = np.bincount(c)
    b = np.r_[0, np.cumsum(n)]
    return np.array([np.median(y[o[b[u]:b[u+1]]]) if n[u] else np.nan for u in range(n.size)])


bins = np.arange(0, 4201, 200)
for ex in EXS:
    rows = []
    for s in SEEDS:
        da, db, dc = folder(ex, 'H8', 2, s), folder(ex, 'H8', 3, s), folder(ex, 'H8', 4, s)
        ta, ca, ya, tpa = load(da)
        tb, cb, yb, tpb = load(db)
        oa, ob = ops_of(da), ops_of(db)
        yc = tonp(oa['yc'])
        hA = np.histogram(yc[tonp(oa['iU']).astype(int)], bins)[0].astype(float)
        hB = np.histogram(yc[tonp(ob['iU']).astype(int)], bins)[0].astype(float)
        Sa, Sb = S_i(da, ex), S_i(db, ex)
        dA, dB = depths(ca, ya), depths(cb, yb)
        va, vb = viol(ta, ca, dA.size), viol(tb, cb, dB.size)
        ida, idb = np.array(sorted(Sa)), np.array(sorted(Sb))
        Sbin_a = np.histogram(dA[ida], bins, weights=[Sa[i] for i in ida])[0]
        Sbin_b = np.histogram(dB[idb], bins, weights=[Sb[i] for i in idb])[0]
        spk_a = np.histogram(dA, bins, weights=np.bincount(ca))[0]
        spk_b = np.histogram(dB, bins, weights=np.bincount(cb))[0]
        vbin_a = np.histogram(dA, bins, weights=va)[0] / np.maximum(spk_a, 1)
        vbin_b = np.histogram(dB, bins, weights=vb)[0] / np.maximum(spk_b, 1)
        ok = (hA >= 20) & (Sbin_a > 0)
        shrink = (hA - hB)[ok] / hA[ok]
        rel_dS = ((Sbin_b - Sbin_a)[ok] / Sbin_a[ok])
        rel_dv = ((vbin_b - vbin_a)[ok] / vbin_a[ok])
        print(f'{ex} s{s}: bins {ok.sum()}, shrink median {np.median(shrink):.3f} range {shrink.min():.3f}..{shrink.max():.3f}; '
              f'Spearman shrink~rel dS {spearman(shrink, rel_dS):+.2f}; shrink~rel dviol {spearman(shrink, rel_dv):+.2f}; '
              f'absolute: (hA-hB)~dS {spearman((hA - hB)[ok], (Sbin_b - Sbin_a)[ok]):+.2f}; hA~dS {spearman(hA[ok], (Sbin_b - Sbin_a)[ok]):+.2f}', flush=True)
    for setting, ka, kb in (('H8', 2, 4), ('default', 2, 3)):
        tag = f'{setting}_r2-r{kb}'
        all_rows = []
        for s in SEEDS:
            z = np.load(OUT / f'units_{ex}_{tag}_s{s}.npz')
            u, v = z['u'], z['v']
            da, db = folder(ex, setting, ka, s), folder(ex, setting, kb, s)
            ta, ca, _, tpa = load(da)
            tb, cb, _, _ = load(db)
            na, nb = np.bincount(ca), np.bincount(cb)
            va, vb = viol(ta, ca, na.size) / np.maximum(na, 1), viol(tb, cb, nb.size) / np.maximum(nb, 1)
            _, t_, c_, n_ = owners(tpa, ca)
            sim = np.bincount(c_, (n_ / na[c_])**2, minlength=na.size)
            eff = 1 / np.maximum(sim, 1e-12)
            Sa, Sb = S_i(da, ex), S_i(db, ex)
            big = (na[u] >= 300) & np.array([(i in Sa) for i in u]) & np.array([(j in Sb) for j in v])
            u_, v_ = u[big], v[big]
            all_rows.append(np.c_[eff[u_], va[u_], vb[v_], [Sa[i] for i in u_], [Sb[j] for j in v_], na[u_], nb[v_]])
        R = np.vstack(all_rows)
        eff, va, vb, s0, s1, n0, n1 = R.T
        qe = np.digitize(eff, np.quantile(eff, [1 / 3, 2 / 3]))
        qv = np.digitize(va, np.quantile(va, [1 / 3, 2 / 3]))
        print(f'{ex} {setting} stable pairs (all seeds) {len(R)}: Spearman eff~dviol {spearman(eff, vb - va):+.3f}, eff~dS {spearman(eff, s1 - s0):+.3f}; '
              f'eff tertile edges {np.round(np.quantile(eff, [1 / 3, 2 / 3]), 2)}')
        for e in range(3):
            m = qe == e
            print(f'   eff tertile {e}: n {m.sum()}, mean viol/spike {va[m].mean():.5f} -> {vb[m].mean():.5f} ({100 * (vb[m].mean() / va[m].mean() - 1):+.1f}%), '
                  f'sum dS {(s1 - s0)[m].sum() / 3:+.1f} per seed, dspikes median {np.median((n1 - n0)[m] / n0[m]) * 100:+.1f}%')
        for w in range(3):
            line = []
            for e in range(3):
                m = (qe == e) & (qv == w)
                line.append(f'{100 * (vb[m].mean() / va[m].mean() - 1):+6.1f}% ({m.sum()})')
            print(f'   viol tertile {w}: change in violations by eff tertile: ' + ' '.join(line), flush=True)
