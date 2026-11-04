"""Exploratory: per-unit MSR on scored kobe sorts, against the harness's per-unit heading scores,
firing rate and isolation; and against hybrid ground truth.

Per sort (common.SORTS), eligible units (evaluation.json S_i keys), cleaned trains (§6.1-6.3):
- MSR (counts, 1 ms points), MSR binarised, and dMSR = MSR - mean MSR of homogeneous Poisson trains
  with the same spike count (msr.Baseline; a rate/count-corrected MSR);
- S_i, S_track_i (bits/s), S_i per spike, rate, spike count;
- refractory contamination (Hill et al. 2011; isolation/comod.py 'plain' and 'local' estimates);
- Spearman correlations, and partial Spearman with log rate partialled out.
Hybrid check (all four sorts are of the hybrid excerpt): for each of the 40 inserted units, MSR of the
true train and of its best-matching sorted unit (raw sorter output, hybrid.recovery's match), with
the sorted unit's false-positive and missed fractions.
Outputs: C:/data/ks4_tuning/exploratory/msr/real_units_<sort>.csv, real_units.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'isolation'))
import common  # noqa: E402
import msr  # noqa: E402
from comod import unit_estimates  # noqa: E402
import hybrid  # noqa: E402

t0 = time.time()
common.check_loader(common.SORTS['default_s1'])
print('loader matches load_clean_units on default_s1', round(time.time() - t0, 1), 's')
base = msr.Baseline(common.DUR)
print('Poisson baseline done', round(time.time() - t0, 1), 's')
truth = np.load(common.TRUTH)
res = {'baseline': base.as_dict(), 'sorts': {}}
for name, sort in common.SORTS.items():
    by, iv, T = common.clean_units(sort)
    ev = common.evaluation(sort)
    units = [int(k) for k in ev['S_i']]
    rows = []
    for c in units:
        t = by[c]
        n, rate, con = unit_estimates(t, T)
        m = msr.msr(t, common.DUR)
        rows.append([c, n, rate, m, msr.msr(t, common.DUR, binarize=True), m - base(n), ev['S_i'][str(c)],
                     ev['S_track_i'][str(c)], con['plain'], con['local']])
    a = np.array(rows, float)
    cols = ['unit', 'n', 'rate', 'msr', 'msr_bin', 'dmsr', 'S_i', 'S_track_i', 'contam_plain', 'contam_local']
    np.savetxt(common.OUT / f'real_units_{name}.csv', a, delimiter=',', header=','.join(cols), comments='')
    d = dict(zip(cols, a.T))
    lr = np.log(d['rate'])
    out = {'n_units': len(units), 'S': float(d['S_i'].sum()), 'msr_median': float(np.median(d['msr'])),
           'sum_msr': float(d['msr'].sum()), 'sum_dmsr': float(d['dmsr'].sum())}
    spk = d['S_i'] / d['rate']
    for mname in ('msr', 'dmsr'):
        x = d[mname]
        out[mname] = {
            'vs_S_i': common.spearman(x, d['S_i']),
            'vs_S_track_i': common.spearman(x, d['S_track_i']),
            'vs_S_i_per_spike': common.spearman(x, spk),
            'vs_rate': common.spearman(x, d['rate']),
            'vs_contam_plain': common.spearman(x, d['contam_plain']),
            'vs_contam_local': common.spearman(x, d['contam_local']),
            'partial_lograte': {
                'vs_S_i': common.partial_spearman(x, d['S_i'], lr),
                'vs_S_track_i': common.partial_spearman(x, d['S_track_i'], lr),
                'vs_S_i_per_spike': common.partial_spearman(x, spk, lr),
                'vs_contam_plain': common.partial_spearman(x, d['contam_plain'], lr),
                'vs_contam_local': common.partial_spearman(x, d['contam_local'], lr)},
            'partial_lograte_and_contam': {
                'vs_S_i': common.partial_spearman(x, d['S_i'], np.column_stack([lr, d['contam_plain']]))},
        }
        # within firing-rate quartiles
        q = np.quantile(d['rate'], [0, .25, .5, .75, 1])
        out[mname]['vs_S_i_by_rate_quartile'] = [
            common.spearman(x[(d['rate'] >= lo) & (d['rate'] <= hi)], d['S_i'][(d['rate'] >= lo) & (d['rate'] <= hi)])
            for lo, hi in zip(q[:-1], q[1:])]
    out['S_i_vs_rate'] = common.spearman(d['S_i'], d['rate'])
    out['contam_plain_vs_rate'] = common.spearman(d['contam_plain'], d['rate'])
    out['S_i_vs_contam_plain_partial_lograte'] = common.partial_spearman(d['S_i'], d['contam_plain'], lr)
    # top-decile overlap: do the highest-MSR units carry the most S?
    k = max(1, len(units) // 10)
    top_msr, top_s = set(np.argsort(-d['dmsr'])[:k]), set(np.argsort(-d['S_i'])[:k])
    out['top_decile_overlap_dmsr_S'] = len(top_msr & top_s) / k

    # hybrid: true vs sorted MSR
    rec = hybrid.recovery(sort, truth)
    st = np.load(Path(sort) / 'spike_times.npy').ravel()
    cl = np.load(Path(sort) / 'spike_clusters.npy').ravel()
    hy = []
    for u in rec['per_unit']:
        tt = np.sort(truth['spike_samples'][truth['spike_unit'] == u['inserted']]) / common.FS
        ts = np.sort(st[cl == u['best_sorted']]) / common.FS
        i = np.clip(np.searchsorted(tt, ts), 1, tt.size - 1)
        hit = np.minimum(np.abs(tt[i] - ts), np.abs(tt[i - 1] - ts)) <= 0.4e-3
        j = np.clip(np.searchsorted(ts, tt), 1, ts.size - 1)
        found = np.minimum(np.abs(ts[j] - tt), np.abs(ts[j - 1] - tt)) <= 0.4e-3
        m_t, m_s = msr.msr(tt, common.DUR), msr.msr(ts, common.DUR)
        hy.append([u['inserted'], u['agreement'], tt.size, ts.size, 1 - hit.mean(), 1 - found.mean(),
                   m_t, m_s, m_t - base(tt.size), m_s - base(ts.size)])
    h = np.array(hy, float)
    hcols = ['inserted', 'agreement', 'n_true', 'n_sorted', 'fp_frac', 'miss_frac', 'msr_true', 'msr_sorted',
             'dmsr_true', 'dmsr_sorted']
    np.savetxt(common.OUT / f'hybrid_{name}.csv', h, delimiter=',', header=','.join(hcols), comments='')
    hd = dict(zip(hcols, h.T))
    dm = hd['msr_sorted'] - hd['msr_true']
    ddm = hd['dmsr_sorted'] - hd['dmsr_true']
    out['hybrid'] = {
        'n': int(h.shape[0]),
        'median_msr_sorted_minus_true': float(np.median(dm)),
        'median_dmsr_sorted_minus_true': float(np.median(ddm)),
        'dmsr_change_vs_fp': common.spearman(ddm, hd['fp_frac']),
        'dmsr_change_vs_miss': common.spearman(ddm, hd['miss_frac']),
        'msr_change_vs_fp': common.spearman(dm, hd['fp_frac']),
        'msr_change_vs_miss': common.spearman(dm, hd['miss_frac']),
        'msr_change_vs_agreement': common.spearman(dm, hd['agreement']),
        'frac_units_sorted_msr_below_true': float(np.mean(dm < 0)),
        'msr_true_vs_sorted': common.spearman(hd['msr_true'], hd['msr_sorted']),
    }
    res['sorts'][name] = out
    print(name, json.dumps({k: v for k, v in out.items() if k not in ('msr', 'dmsr')}), round(time.time() - t0, 1), 's')
    for mname in ('msr', 'dmsr'):
        o = out[mname]
        print(f"  {mname}: vs S_i {o['vs_S_i'][0]:+.3f}  vs S_track {o['vs_S_track_i'][0]:+.3f}  vs rate {o['vs_rate'][0]:+.3f}"
              f"  vs contam {o['vs_contam_plain'][0]:+.3f}  partial(logr) vs S_i {o['partial_lograte']['vs_S_i'][0]:+.3f}"
              f"  vs contam {o['partial_lograte']['vs_contam_plain'][0]:+.3f}  quartiles "
              + ' '.join(f"{r[0]:+.2f}" for r in o['vs_S_i_by_rate_quartile']))
res['runtime_s'] = time.time() - t0
(common.OUT / 'real_units.json').write_text(json.dumps(res, indent=1))
