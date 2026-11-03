"""Are short-lag within-unit pairs one spike split into two fits, a residual re-fit, or two full spikes?

Question (for choosing between deleting one member and merging the pair into one event): for
within-unit spike pairs in KS4's output BEFORE its own dedup (full_st, with per-spike scores), how do
the two members' scores compare with the unit's typical score, by lag band?
  - residual re-fit: one member typical (~1x unit median), the other small;
  - split fit of one spike: both members well below typical (~0.5-0.7x), similar to each other;
  - two genuine full spikes (contamination / real): both ~1x.
Also: which member is earlier (is the higher-score member earlier or later?), same extraction
template share, and for control, pairs at 2-5 ms (mostly genuine spike pairs / contamination).
Only consecutive within-unit pairs are used.
"""
import json
from pathlib import Path

import numpy as np

AUTO = Path(r'C:\data\ks4_tuning\exploratory\autocurate\sorts')
NAMES = ('kobe_default_s1', 'kobe_H8_s1', 'mec_default_s1', 'mec_H8_s1')
OUT = Path(__file__).parent / 'pair_member_scores.json'
BANDS = {'1-6': (1, 6), '7': (7, 7), '8-10': (8, 10), '11-15': (11, 15), '16-30': (16, 30),
         '31-45': (31, 45), '60-150 (control)': (60, 150)}


def main():
    res = {}
    for name in NAMES:
        p = AUTO / name
        fs = np.load(p / 'full_st.npy')
        t = fs[:, 0].astype(np.int64); tm = fs[:, 1].astype(np.int64); s = fs[:, 2]
        c = np.load(p / 'full_clu.npy').astype(np.int64)
        o = np.lexsort((t, c))
        t, tm, s, c = t[o], tm[o], s[o], c[o]
        med = np.zeros(c.max() + 1)
        for k in np.unique(c):
            a, b = np.searchsorted(c, [k, k + 1])
            med[k] = np.median(s[a:b])
        rel = s / med[c]
        same = c[1:] == c[:-1]
        d = np.diff(t)
        i = np.where(same)[0]
        d = d[i]
        r = {}
        for band, (lo, hi) in BANDS.items():
            j = i[(d >= lo) & (d <= hi)]
            e, l = rel[j], rel[j + 1]           # earlier, later member (relative to unit median)
            hi_m, lo_m = np.maximum(e, l), np.minimum(e, l)
            r[band] = {'n_pairs': int(j.size),
                       'higher_member_rel_median': float(np.median(hi_m)),
                       'lower_member_rel_median': float(np.median(lo_m)),
                       'min_over_max_median': float(np.median(lo_m / hi_m)),
                       'sum_rel_median': float(np.median(e + l)),
                       'frac_both_below_0.75': float(np.mean(hi_m < 0.75)),
                       'frac_one_typical_other_below_0.6': float(np.mean((hi_m >= 0.8) & (lo_m < 0.6))),
                       'frac_both_above_0.8': float(np.mean(lo_m >= 0.8)),
                       'frac_earlier_is_higher': float(np.mean(e > l)),
                       'frac_same_extraction_template': float(np.mean(tm[j] == tm[j + 1]))}
        r['all_spikes_rel_quartiles'] = np.percentile(rel, [10, 25, 50, 75, 90]).tolist()
        res[name] = r
        print(name, 'all-spike rel score q10/25/50/75/90', np.round(r['all_spikes_rel_quartiles'], 2))
        for band in BANDS:
            v = r[band]
            print(f"  lag {band:>16}: n={v['n_pairs']:7d} hi={v['higher_member_rel_median']:.2f} "
                  f"lo={v['lower_member_rel_median']:.2f} min/max={v['min_over_max_median']:.2f} "
                  f"sum={v['sum_rel_median']:.2f} bothLow={v['frac_both_below_0.75']:.2f} "
                  f"oneTyp={v['frac_one_typical_other_below_0.6']:.2f} bothTyp={v['frac_both_above_0.8']:.2f} "
                  f"earlierHi={v['frac_earlier_is_higher']:.2f} sameTmpl={v['frac_same_extraction_template']:.2f}",
                  flush=True)
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
