"""How often does one inserted (ground-truth) spike of the kobe hybrid excerpt come out as two sorted
spikes, in the same sorted unit or in another unit, and at which lags (sample resolution)?

The 40 inserted units are time-shifted copies of real units' trains (shift 300-700 s), so their spikes are
independent of every real neuron: a sorted spike locked to an inserted spike above chance is an artefact
of the sorter (a duplicate or a partial detection). GT spikes with a GT spike of a unit inserted within
300 um (including itself and its HD partner) within +-80 samples are dropped.

For each GT unit u: best sorted unit = most hits within +-12 samples of the GT spikes; offset o_u = modal
lag. The PRIMARY detection of a GT spike is the best unit's spike nearest t_gt + o_u (within +-12).
Lags of every other sorted spike are measured from the primary detection.
  best : other spikes of the best unit; chance level per lag sample = the best unit's spikes that hit no
         GT spike of u, per sample of recording.
  other: spikes of other units; chance = the same GT spikes shifted by +-50/100 ms (this includes the
         'shadow' dip: real spikes missed near an inserted one, so excess can be negative).
Per (GT unit, other unit) pair, a locked partner is flagged when its count in a 3-sample window around its
modal lag exceeds chance with Poisson z > 6 and >= 1% of the GT spikes.
"""
import json
from pathlib import Path

import numpy as np

TRUTH = Path(r'C:\data\ks4_tuning\26648_kobe_2020-08-21\search\hybrid\hybrid_truth.npz')
KS4 = Path(r'C:\data\ks4_tuning\exploratory\autocurate\sorts')
KS25T = Path(r'C:\data\ks4_tuning\exploratory\ks25_traced\kobe_s1\stages.npz')
RUNS = Path(r'C:\data\ks4_tuning\26648_kobe_2020-08-21\runs')
OUT = Path(__file__).parent / 'out'
W = 60
SHIFTS = [-3000, -1500, 1500, 3000]
TOL = 12
T_REC = 1200 * 30000


def load_sorts():
    s = {}
    for name in ['kobe_H8_s1', 'kobe_default_s1']:
        d = KS4 / name
        st = np.load(d / 'full_st.npy')
        clu = np.load(d / 'full_clu.npy').astype(np.int64)
        kept = np.load(d / 'kept_spikes.npy')
        t = st[:, 0].astype(np.int64)
        s[name + '_full'] = dict(t=t, clu=clu, tmpl=st[:, 1].astype(np.int64), score=st[:, 2])
        s[name + '_kept'] = dict(t=t[kept], clu=clu[kept], tmpl=st[kept, 1].astype(np.int64),
                                 score=st[kept, 2])
    z = np.load(KS25T)
    for stg in ['extracted', 'merge1', 'split1', 'cut1', 'merge3']:
        s['ks25_' + stg] = dict(t=z[stg + '_t'].astype(np.int64), clu=z[stg + '_clu'].astype(np.int64),
                                tmpl=z['extracted_clu'].astype(np.int64) if stg == 'extracted' else None,
                                score=z[stg + '_score'].astype(float))
    for r in ['ks25_standard_r1', 'ks25_standard_r2', 'ks25_standard_low_r1', 'hybrid_default_seed1',
              'hybrid_default_seed2']:
        d = RUNS / r
        s['final_' + r] = dict(t=np.load(d / 'spike_times.npy').ravel().astype(np.int64),
                               clu=np.load(d / 'spike_clusters.npy').ravel().astype(np.int64),
                               tmpl=None, score=None)
    return s


def pairs(ts, tq, w):
    lo = np.searchsorted(ts, tq - w, 'left')
    hi = np.searchsorted(ts, tq + w, 'right')
    n = hi - lo
    i = np.repeat(np.arange(tq.size), n)
    j = lo[i] + np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    return i, j, ts[j] - tq[i]


def band(L, h, a, b):
    m = (np.abs(L) >= a) & (np.abs(L) <= b)
    return float(np.sum(h[m]))


def analyse(sd, tt, tu, keep):
    o = np.argsort(sd['t'], kind='stable')
    t, clu = sd['t'][o], sd['clu'][o]
    tmpl = None if sd['tmpl'] is None else sd['tmpl'][o]
    score = None if sd['score'] is None else sd['score'][o]
    nu = tu.max() + 1
    i, j, lag = pairs(t, tt, TOL)          # all GT spikes, for best unit / offset / hit flags
    best, off = np.zeros(nu, np.int64), np.zeros(nu, np.int64)
    nonhit = np.zeros(nu)
    for u in range(nu):
        m = tu[i] == u
        cu, cnt = np.unique(clu[j[m]], return_counts=True)
        best[u] = cu[cnt.argmax()]
        mm = m & (clu[j] == best[u])
        lu, lc = np.unique(lag[mm], return_counts=True)
        off[u] = lu[lc.argmax()]
        nonhit[u] = np.sum(clu == best[u]) - np.unique(j[mm]).size
    # primary detections for the isolated GT spikes
    tq, uq = tt[keep], tu[keep]
    i, j, lag = pairs(t, tq + off[uq], TOL)
    m = clu[j] == best[uq[i]]
    i, j, lag = i[m], j[m], lag[m]
    ordr = np.argsort(np.abs(lag), kind='stable')[::-1]
    prim = np.full(tq.size, -1)
    prim[i[ordr]] = j[ordr]
    det = prim >= 0
    tp, up, pp = t[prim[det]], uq[det], prim[det]
    ng = tp.size
    bins = np.arange(-W, W + 2) - 0.5
    L = np.arange(-W, W + 1)
    out = {'lags': L.tolist(), 'n_gt_isolated': int(tq.size), 'n_primary': int(ng),
           'frac_detected': ng / tq.size}
    i, j, lag = pairs(t, tp, W)
    notp = j != pp[i]
    isb = clu[j] == best[up[i]]
    # best unit
    e = notp & isb
    hb = np.histogram(lag[e], bins)[0].astype(float)
    chance_b = np.sum(nonhit[up]) / T_REC          # expected per lag sample, summed over primaries
    out['best_per_gt'] = (hb / ng).tolist()
    out['best_raw_bands'] = {k: band(L, hb, a, b) for k, (a, b) in
                             {'1-6': (1, 6), '7-15': (7, 15), '16-30': (16, 30), '31-60': (31, 60)}.items()}
    out['best_chance_per_gt_per_sample'] = chance_b / ng
    ex = (hb - chance_b) / ng
    out['best_excess_bands'] = {k: band(L, ex, a, b) for k, (a, b) in
                                {'1-6': (1, 6), '7-15': (7, 15), '16-30': (16, 30), '31-60': (31, 60)}.items()}
    if tmpl is not None:
        same = tmpl[j] == tmpl[pp[i]]
        out['best_same_tmpl'] = (np.histogram(lag[e & same], bins)[0] / ng).tolist()
        out['best_diff_tmpl'] = (np.histogram(lag[e & ~same], bins)[0] / ng).tolist()
    if score is not None:
        ratio = score[j] / score[pp[i]]
        q = {}
        for nm, (a, b) in {'1-6': (1, 6), '7-15': (7, 15), '16-30': (16, 30), '31-60': (31, 60)}.items():
            mm = e & (np.abs(lag) >= a) & (np.abs(lag) <= b)
            q[nm] = [int(mm.sum())] + (np.percentile(ratio[mm], [10, 50, 90]).round(3).tolist() if mm.sum() else [])
        out['best_ratio_extra_over_primary'] = q
    # other units: real vs shifted null
    e = notp & ~isb
    ho = np.histogram(lag[e], bins)[0].astype(float)
    key_r = up[i[e]] * 100000 + clu[j[e]]
    lag_r = lag[e]
    hn, keys_n, lags_n = [], [], []
    for s in SHIFTS:
        i2, j2, l2 = pairs(t, tp + s, W)
        e2 = clu[j2] != best[up[i2]]
        hn.append(np.histogram(l2[e2], bins)[0])
        keys_n.append(up[i2[e2]] * 100000 + clu[j2[e2]])
        lags_n.append(l2[e2])
    hn = np.mean(hn, 0)
    out['other_per_gt'] = (ho / ng).tolist()
    out['other_excess_per_gt'] = ((ho - hn) / ng).tolist()
    # locked partners per (GT unit, other unit)
    keys_n, lags_n = np.concatenate(keys_n), np.concatenate(lags_n)
    nprim_u = np.bincount(up, minlength=nu)
    partners = []
    so = np.argsort(key_r, kind='stable')
    ks, ls = key_r[so], lag_r[so]
    uk, start = np.unique(ks, return_index=True)
    groups = np.split(ls, start[1:])
    cnt_null = dict(zip(*np.unique(keys_n, return_counts=True)))
    for k, ll in zip(uk, groups):
        if ll.size < 20:
            continue
        sm = np.convolve(np.bincount(ll + W, minlength=2 * W + 1), [1, 1, 1], 'same')
        c = sm.argmax()
        n3 = sm[c]
        mu = cnt_null.get(k, 0) / len(SHIFTS) * 3 / (2 * W + 1)
        u = k // 100000
        if (n3 - mu) / np.sqrt(mu + 1) > 6 and (n3 - mu) >= 0.01 * nprim_u[u]:
            pk = np.bincount(ll + W, minlength=2 * W + 1)
            # whole locked excess within +-4 of the mode
            lo, hi = max(c - 4, 0), min(c + 5, 2 * W + 1)
            partners.append(dict(gt=int(u), unit=int(k % 100000), mode_lag=int(c - W),
                                 frac=float((pk[lo:hi].sum() - mu * (hi - lo) / 3) / nprim_u[u])))
    out['partners'] = partners
    out['partner_frac_total_per_gt'] = float(sum(p['frac'] * nprim_u[p['gt']] for p in partners) / ng)
    return out


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    z = np.load(TRUTH)
    tt, tu = z['spike_samples'].astype(np.int64), z['spike_unit'].astype(np.int64)
    dep = z['target_depth_um']
    o = np.argsort(tt)
    tt, tu = tt[o], tu[o]
    keep = np.ones(tt.size, bool)
    for u in range(tu.max() + 1):
        near = np.abs(dep - dep[u]) < 300
        tn = tt[near[tu]]
        a = tt[tu == u]
        lo = np.searchsorted(tn, a - 80)
        hi = np.searchsorted(tn, a + 80, 'right')
        keep[np.where(tu == u)[0]] = (hi - lo) == 1
    print('isolated GT spikes', keep.sum(), 'of', tt.size)
    res = {}
    for name, sd in load_sorts().items():
        r = analyse(sd, tt, tu, keep)
        res[name] = r
        L = np.array(r['lags'])
        oe = np.array(r['other_excess_per_gt'])
        print(f"{name:30s} det {r['frac_detected']:.3f} | best excess/GT: " +
              ' '.join(f"{k}:{v:.4f}" for k, v in r['best_excess_bands'].items()) +
              f" | other excess 7-15 {band(L, oe, 7, 15):.4f} 1-6 {band(L, oe, 1, 6):.4f} lag0 {oe[L==0][0]:.4f}"
              f" | locked partners {len(r['partners'])}, dup/GT {r['partner_frac_total_per_gt']:.4f}", flush=True)
        print('    best-unit extra spikes (raw counts):', r['best_raw_bands'],
              'chance/GT/sample %.2e' % r['best_chance_per_gt_per_sample'])
        if 'best_ratio_extra_over_primary' in r:
            print('    ratio extra/primary (n, p10, p50, p90):', r['best_ratio_extra_over_primary'])
        if 'best_same_tmpl' in r:
            s_, d_ = np.array(r['best_same_tmpl']), np.array(r['best_diff_tmpl'])
            print('    best-unit extras per GT, same tmpl / diff tmpl, |lag| 1-6:',
                  f"{band(L, s_, 1, 6):.4f}/{band(L, d_, 1, 6):.4f}  7-15: {band(L, s_, 7, 15):.4f}/{band(L, d_, 7, 15):.4f}"
                  f"  16-30: {band(L, s_, 16, 30):.4f}/{band(L, d_, 16, 30):.4f}; chance per sample {r['best_chance_per_gt_per_sample']:.5f}")
    json.dump(res, open(OUT / 'gt_dup_lags.json', 'w'))
