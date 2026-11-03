"""Which member of a within-unit duplicate pair does KS4 keep, and would a different rule matter?

Question: KS4's remove_duplicates (kilosort/postprocessing.py:9-29) keeps the EARLIEST spike of a
same-cluster run and measures its window from the last KEPT spike. On the scored sorts (spikes
before KS4's dedup, with per-spike scores), (1) reproduce KS4's kept mask; (2) for each removed spike,
compare its score/extraction template with the kept spike it was dropped against; (3) compare KS4's
rule with alternatives at the same window (7 samples):
  - 'best'   : non-maximum suppression by score (keep the higher-score member, cluster-wise),
  - 'chain'  : window measured from the last SEEN spike (a run of close spikes keeps only its first),
and with KS4's rule at larger windows (12, 15 samples), counting the within-unit pairs left in lag
bands; (4) how many KEPT pairs at 7-15 samples are chain products (a removed spike lies between).
"""
import json
from pathlib import Path

import numpy as np
from numba import njit

AUTO = Path(r'C:\data\ks4_tuning\exploratory\autocurate\sorts')
NAMES = ('kobe_default_s1', 'kobe_H8_s1', 'mec_default_s1', 'mec_H8_s1')
OUT = Path(__file__).parent / 'ks4_dedup_choice.json'
DT = 7


@njit(cache=True)
def ks4_rule(t, c, dt, nclu):
    keep = np.zeros(t.size, np.bool_)
    last = np.full(nclu, -(1 << 40), np.int64)
    ref = np.full(t.size, -1, np.int64)          # index of the kept spike it was dropped against
    lastidx = np.full(nclu, -1, np.int64)
    for i in range(t.size):
        k = c[i]
        if t[i] >= last[k] + dt:
            keep[i] = True; last[k] = t[i]; lastidx[k] = i
        else:
            ref[i] = lastidx[k]
    return keep, ref


@njit(cache=True)
def chain_rule(t, c, dt, nclu):
    keep = np.zeros(t.size, np.bool_)
    last = np.full(nclu, -(1 << 40), np.int64)
    for i in range(t.size):
        k = c[i]
        if t[i] >= last[k] + dt:
            keep[i] = True
        last[k] = t[i]
    return keep


@njit(cache=True)
def nms_rule(t, s, order_c, starts, dt):
    """order_c: indices sorted by (cluster, time); starts: cluster block starts (len nclu+1).
    Within each cluster, accept spikes in descending score if no accepted spike within dt."""
    keep = np.zeros(t.size, np.bool_)
    for b in range(starts.size - 1):
        idx = order_c[starts[b]:starts[b + 1]]
        m = idx.size
        if m == 0:
            continue
        acc = np.zeros(m, np.bool_)
        ss = np.empty(m)
        for j in range(m):
            ss[j] = s[idx[j]]
        o = np.argsort(-ss, kind='mergesort')
        for q in range(m):
            j = o[q]
            ok = True
            l = j - 1
            while l >= 0 and t[idx[j]] - t[idx[l]] < dt:
                if acc[l]:
                    ok = False; break
                l -= 1
            if ok:
                r = j + 1
                while r < m and t[idx[r]] - t[idx[j]] < dt:
                    if acc[r]:
                        ok = False; break
                    r += 1
            if ok:
                acc[j] = True
        for j in range(m):
            keep[idx[j]] = acc[j]
    return keep


@njit(cache=True)
def band_pairs(tc, cc, maxlag):
    H = np.zeros(maxlag + 1, np.int64)
    for i in range(tc.size):
        j = i + 1
        while j < tc.size and cc[j] == cc[i] and tc[j] - tc[i] <= maxlag:
            H[tc[j] - tc[i]] += 1
            j += 1
    return H


@njit(cache=True)
def chain_products(tc, cc, kc, lo, hi):
    """Among consecutive KEPT within-unit pairs at lag lo..hi, how many have a removed spike between."""
    n_pairs = 0; n_chain = 0
    i = 0
    n = tc.size
    while i < n:
        if not kc[i]:
            i += 1; continue
        j = i + 1
        between = False
        while j < n and cc[j] == cc[i] and not kc[j]:
            between = True; j += 1
        if j < n and cc[j] == cc[i]:
            d = tc[j] - tc[i]
            if lo <= d <= hi:
                n_pairs += 1
                if between:
                    n_chain += 1
        i = j if j > i else i + 1
    return n_pairs, n_chain


def bands(t, c, keep):
    tt, cc = t[keep], c[keep]
    o = np.lexsort((tt, cc))
    H = band_pairs(tt[o], cc[o], 45)
    return {'removed': int((~keep).sum()), 'lag0-6': int(H[:7].sum()), 'lag7': int(H[7]),
            'lag8-15': int(H[8:16].sum()), 'lag16-30': int(H[16:31].sum()), 'lag31-45': int(H[31:46].sum())}


def main():
    res = {}
    for name in NAMES:
        p = AUTO / name
        fs = np.load(p / 'full_st.npy')
        t = fs[:, 0].astype(np.int64)
        tmpl = fs[:, 1].astype(np.int64)
        s = fs[:, 2].astype(np.float64)
        c = np.load(p / 'full_clu.npy').astype(np.int64)
        kept = np.load(p / 'kept_spikes.npy')
        nclu = int(c.max()) + 1
        keep, ref = ks4_rule(t, c, DT, nclu)
        r = {'n_spikes': int(t.size), 'reproduces_kept_spikes': bool(np.array_equal(keep, kept)),
             'n_removed': int((~kept).sum()), 'frac_removed': float((~kept).mean())}
        rm = np.where(~keep)[0]
        k = ref[rm]
        lag = t[rm] - t[k]
        r['removed_lag_hist_0_6'] = np.bincount(lag, minlength=7).tolist()
        ratio = s[rm] / s[k]
        r['removed_has_higher_score_frac'] = float((s[rm] > s[k]).mean())
        r['score_ratio_removed_over_kept_median'] = float(np.median(ratio))
        r['score_ratio_quartiles'] = np.percentile(ratio, [25, 75]).tolist()
        r['min_over_max_median'] = float(np.median(np.minimum(s[rm], s[k]) / np.maximum(s[rm], s[k])))
        r['same_extraction_template_frac'] = float((tmpl[rm] == tmpl[k]).mean())
        r['removed_score_median'] = float(np.median(s[rm]))
        r['kept_partner_score_median'] = float(np.median(s[k]))
        r['all_spikes_score_median'] = float(np.median(s))
        # alternatives
        o = np.lexsort((t, c))
        starts = np.searchsorted(c[o], np.arange(nclu + 1))
        rules = {'ks4_earliest_dt7': keep, 'chain_dt7': chain_rule(t, c, DT, nclu),
                 'best_score_nms_dt7': nms_rule(t, s, o, starts, DT),
                 'ks4_earliest_dt12': ks4_rule(t, c, 12, nclu)[0], 'ks4_earliest_dt15': ks4_rule(t, c, 15, nclu)[0],
                 'best_score_nms_dt15': nms_rule(t, s, o, starts, 15)}
        r['rules'] = {kk: bands(t, c, v) for kk, v in rules.items()}
        nms = rules['best_score_nms_dt7']
        r['nms_vs_ks4_differ_spikes'] = int((nms != keep).sum())
        r['nms_kept_mean_score_minus_ks4'] = float(s[nms].sum() - s[keep].sum())
        # chain products among kept pairs
        tc, cc, kc = t[o], c[o], keep[o]
        r['kept_consecutive_pairs_7_15_with_removed_between'] = list(chain_products(tc, cc, kc, 7, 15))
        r['kept_consecutive_pairs_7_8_with_removed_between'] = list(chain_products(tc, cc, kc, 7, 8))
        res[name] = r
        print(name, json.dumps({kk: v for kk, v in r.items() if kk != 'rules'}), flush=True)
        for kk, v in r['rules'].items():
            print('   ', kk.ljust(22), v, flush=True)
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
