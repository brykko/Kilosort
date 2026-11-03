"""Cross-sorter reading of KS4's within-unit 0.3-0.5 ms pairs: does KS2.5 find both spikes, in one unit or
two? And is the KS4 unit a merge, or does it take the partner's spike only when it overlaps?

Matching is spatially restricted: a KS4 spike of unit u is matched to the nearest-in-time KS2.5 spike
(+-5 samples) among KS2.5 units whose position is within 60 um of u's median position (the probe holds
~10k spikes/s, so matching by time alone is meaningless).
Fate of u's pairs at 9..15 samples ('bump') and at 60..90 samples ('ctrl'): both spikes found in one KS2.5
unit / in two KS2.5 units / one found / none.

For each KS4 unit u (>= 20 pairs at 9..15 samples): v_main = the KS2.5 unit holding most of u's spikes;
for u's 9..15 pairs whose two spikes match two different KS2.5 units, one of which is v_main, v_o = the
other. Per (u, v_main, v_o):
- P(a v_o spike is in u | it lies 7..20 samples from a v_main spike)  vs
  P(a v_o spike is in u | no v_main spike within +-1 ms).
  Ratio >> 1: KS4 takes v_o's spike into u mainly when it overlaps a v_main spike (assignment during
  overlap); ratio ~ 1: u is a merge of v_main and (part of) v_o, and the bump is v_main-v_o synchrony.
- the KS2.5 cross-correlogram v_main x v_o (both directions pooled) at 0..60 samples, relative to its
  60..150 level, and the units' distance.
Writes xsorter2.json.
"""
import json
from pathlib import Path

import numpy as np

KOBE = Path('C:/data/ks4_tuning/26648_kobe_2020-08-21/runs')
MEC = Path('C:/data/ks4_tuning/25843_roger_2019-09-11/runs')
OUT = Path(__file__).resolve().parent / 'xsorter2.json'
TOL = 5
NEAR_UM = 60.0


def load(sort):
    t = np.load(sort / 'spike_times.npy').ravel().astype(np.int64)
    c = np.load(sort / 'spike_clusters.npy').ravel().astype(np.int64)
    y = np.load(sort / 'spike_positions.npy')
    o = np.argsort(t, kind='stable')
    return t[o], c[o], y[o]


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


def match(tq, tb):
    j = np.clip(np.searchsorted(tb, tq), 1, tb.size - 1)
    jl = np.where(np.abs(tb[j - 1] - tq) <= np.abs(tb[j] - tq), j - 1, j)
    return np.where(np.abs(tb[jl] - tq) <= TOL, jl, -1)


def nearest_lag(tq, tr):
    """signed lag from each tq to the nearest tr"""
    j = np.clip(np.searchsorted(tr, tq), 1, tr.size - 1)
    a, b = tr[j - 1] - tq, tr[j] - tq
    return np.where(np.abs(a) <= np.abs(b), a, b)


def ccg(ta, tb, W=150):
    lo = np.searchsorted(tb, ta - W, 'left'); hi = np.searchsorted(tb, ta + W, 'right')
    H = np.zeros(2 * W + 1)
    for d in range(int((hi - lo).max())):
        has = lo + d < hi
        np.add.at(H, tb[lo[has] + d] - ta[has] + W, 1)
    return H


def main():
    res = {}
    for ex, base, a in (('kobe', KOBE, 'search/198_H8_cds10_s1/sort'), ('mec', MEC, 'H8_s1'),
                        ('kobe', KOBE, 'hybrid_default_seed1'), ('mec', MEC, 'default_s1')):
        ta, ca, ya = load(base / a)
        tb, cb, yb = load(base / 'ks25_standard_r1')
        upos_a = {int(k): np.median(ya[ca == k], 0) for k in np.unique(ca)}
        ub = np.unique(cb)
        upos_b = np.array([np.median(yb[cb == k], 0) for k in ub])
        I, J = pairs(ta, ca, 9, 15)
        CI, CJ = pairs(ta, ca, 60, 90)
        ma = np.full(ta.size, -1)
        for u in np.unique(ca):
            su = np.where(ca == u)[0]
            nb = ub[np.linalg.norm(upos_b - upos_a[int(u)], axis=1) <= NEAR_UM]
            ib = np.where(np.isin(cb, nb))[0]
            if ib.size < 2:
                continue
            m = match(ta[su], tb[ib])
            ma[su] = np.where(m >= 0, ib[np.maximum(m, 0)], -1)
        fates = {}
        for nm_, (P, Q) in (('bump', (I, J)), ('ctrl', (CI, CJ))):
            mi, mj = ma[P], ma[Q]
            both = (mi >= 0) & (mj >= 0) & (mi != mj)
            same = both & (cb[np.maximum(mi, 0)] == cb[np.maximum(mj, 0)])
            fates[nm_] = {'n': int(P.size), 'both_found': float(both.mean()),
                          'both_same_ks25_unit': float(same.mean()), 'both_diff_ks25_unit': float((both & ~same).mean()),
                          'one_found': float(np.mean((mi >= 0) ^ (mj >= 0))), 'none': float(np.mean((mi < 0) & (mj < 0))),
                          'same_ks25_spike': float(np.mean((mi >= 0) & (mi == mj)))}
        print(ex, a, 'fates', json.dumps(fates))
        ids, n = np.unique(ca[I], return_counts=True)
        rows = []
        Hsum = np.zeros(301)
        for u in ids[n >= 20]:
            su = np.where(ca == u)[0]
            vb = cb[ma[su][ma[su] >= 0]]
            if vb.size == 0:
                continue
            v_main = np.bincount(vb).argmax()
            sel = (ca[I] == u)
            mi, mj = ma[I[sel]], ma[J[sel]]
            ok = (mi >= 0) & (mj >= 0) & (mi != mj)
            ui, uj = cb[np.maximum(mi, 0)], cb[np.maximum(mj, 0)]
            other = np.where(ok & (ui == v_main) & (uj != v_main), uj, np.where(ok & (uj == v_main) & (ui != v_main), ui, -1))
            other = other[other >= 0]
            if other.size < 5:
                continue
            v_o = np.bincount(other).argmax()
            t_main = tb[cb == v_main]
            io = np.where(cb == v_o)[0]
            lag = nearest_lag(tb[io], t_main)
            in_u = np.zeros(io.size, bool)
            # is v_o's spike in u? (a KS4 spike of u within TOL)
            tu = ta[su]
            j = match(tb[io], tu)
            share_pairs = float(np.mean(other == v_o))
            in_u = j >= 0
            co = (np.abs(lag) >= 7) & (np.abs(lag) <= 20)
            far = np.abs(lag) > 30
            H = ccg(t_main, tb[io])
            Hs = H[150:] + H[150::-1]               # |lag| 0..150, both directions
            Hsum += H
            d = float(np.linalg.norm(np.median(yb[cb == v_main], 0) - np.median(yb[io], 0)))
            rows.append({'u': int(u), 'n_bump': int(sel.sum()), 'v_main': int(v_main), 'v_o': int(v_o),
                         'n_main': int(t_main.size), 'n_o': int(io.size),
                         'share_of_partners_in_v_o': share_pairs, 'n_partner_pairs': int(other.size),
                         'p_in_u_coinc': float(in_u[co].mean()) if co.any() else np.nan, 'n_coinc': int(co.sum()),
                         'p_in_u_alone': float(in_u[far].mean()) if far.any() else np.nan,
                         'share_of_u_spikes_from_o_alone': float(in_u[far].sum() / su.size),
                         'ccg_ratio_9to15': float(Hs[9:16].mean() / max(Hs[60:151].mean(), 1e-9)),
                         'ccg_ratio_0to6': float(Hs[0:7].mean() / max(Hs[60:151].mean(), 1e-9)),
                         'ccg_ratio_30to45': float(Hs[30:46].mean() / max(Hs[60:151].mean(), 1e-9)),
                         'dist_um': d})
        R = rows
        g = lambda k: np.array([r[k] for r in R], float)
        w = g('n_bump')
        summ = {'n_units': len(R), 'bump_pairs_covered': float(w.sum()), 'bump_pairs_total': int(I.size),
                'median_p_in_u_coinc': float(np.nanmedian(g('p_in_u_coinc'))),
                'median_p_in_u_alone': float(np.nanmedian(g('p_in_u_alone'))),
                'weighted_p_in_u_coinc': float(np.nansum(g('p_in_u_coinc') * w) / w.sum()),
                'weighted_p_in_u_alone': float(np.nansum(g('p_in_u_alone') * w) / w.sum()),
                'median_ccg_ratio_9to15': float(np.median(g('ccg_ratio_9to15'))),
                'median_ccg_ratio_0to6': float(np.median(g('ccg_ratio_0to6'))),
                'median_ccg_ratio_30to45': float(np.median(g('ccg_ratio_30to45'))),
                'median_dist_um': float(np.median(g('dist_um'))),
                'median_share_of_partners_in_v_o': float(np.median(g('share_of_partners_in_v_o'))),
                'partner_pairs_over_bump_pairs': float(g('n_partner_pairs').sum() / w.sum()),
                'frac_units_alone_ge_0.3': float(np.mean(g('p_in_u_alone') >= 0.3)),
                'frac_units_coinc_over_alone_gt_3': float(np.mean(g('p_in_u_coinc') > 3 * np.maximum(g('p_in_u_alone'), 1e-3)))}
        lvl = np.r_[Hsum[:91], Hsum[211:]].mean()
        summ['pooled_ccg_ratio_by_lag_-30..30'] = (Hsum[120:181] / lvl).round(3).tolist()
        res[f'{ex}:{a}'] = {'fates': fates, 'summary': summ, 'rows': R}
        print(ex, a, json.dumps({k: v for k, v in summ.items() if not k.startswith('pooled')}, indent=0))
        print(' pooled CCG v_main x v_o, lags -30..30:', ' '.join(f'{v:.2f}' for v in summ['pooled_ccg_ratio_by_lag_-30..30']))
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
