"""Exploratory: sort-level MSR summaries across scored kobe sorts, against the sorts' S.

Sorts: the five default replicates, the six KS2.5 sorts ("standard" and "standard_low", r1-3), and
every search run scored under the current S (log result records with S_default 1019.66) whose folder
still holds its spike files (on 2026-11-04 only round 4 and its ablation, runs 191-230, all seeds; the
earlier runs' spike files have been pruned), all of the hybrid excerpt.
Per sort, over eligible units (evaluation.json S_i keys, cleaned trains):
  n_eligible, S, S_track; sum / mean / median MSR; sum / mean dMSR (MSR minus the homogeneous-Poisson
  MSR at the same spike count); spike-weighted dMSR yield sum_i N_i dMSR_i; sum of positive dMSR.
Then Spearman (and Pearson) of each summary with S across sorts, partial Spearman with n_eligible
partialled out, and the SD of each summary across the five default replicates (its noise scale).
Outputs: C:/data/ks4_tuning/exploratory/msr/sort_level.csv, sort_level.json
"""
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import msr  # noqa: E402

RUNS = common.KOBE / 'runs'
_base = None


def summarise(item):
    global _base
    if _base is None:
        _base = msr.Baseline(common.DUR)
    label, sort = item
    by, _, _ = common.clean_units(sort)
    ev = common.evaluation(sort)
    units = [int(k) for k in ev['S_i']]
    m = np.array([msr.msr(by[c], common.DUR) for c in units])
    n = np.array([by[c].size for c in units], float)
    d = m - _base(n)
    return {'label': label, 'n_eligible': len(units), 'S': ev['S'], 'S_track': ev['S_track'],
            'sum_msr': float(m.sum()), 'mean_msr': float(m.mean()), 'median_msr': float(np.median(m)),
            'sum_dmsr': float(d.sum()), 'mean_dmsr': float(d.mean()), 'yield_dmsr': float((n * d).sum()),
            'sum_pos_dmsr': float(np.clip(d, 0, None).sum()), 'n_spikes': float(n.sum())}


def sorts():
    out = [(f'default_s{k}', RUNS / f'hybrid_default_seed{k}') for k in range(1, 6)]
    out += [(f'ks25_{p}_r{k}', RUNS / f'ks25_{p}_r{k}') for p in ('standard', 'standard_low') for k in (1, 2, 3)]
    recs = [json.loads(l) for l in (RUNS / 'search_log.jsonl').read_text().splitlines()]
    for r in recs:
        if r['type'] != 'result' or abs(r['S_default'] - 1019.665) > 0.01:
            continue
        rid = r['run_id']
        if int(re.match(r'(\d+)', rid).group(1)) < 84 or 'ks25' in rid:
            continue
        f = RUNS / 'search' / rid / 'sort'
        if all((f / x).exists() for x in ('evaluation.json', 'spike_times.npy', 'spike_clusters.npy', 'cleaning.npz')):
            out.append((rid, f))
    return out


if __name__ == '__main__':
    t0 = time.time()
    items = sorts()
    print(len(items), 'sorts', flush=True)
    rows = []
    with ProcessPoolExecutor(4) as ex:
        for r in ex.map(summarise, items):
            rows.append(r)
            print(r['label'], r['n_eligible'], round(r['S'], 1), round(r['sum_dmsr'], 2), round(r['mean_dmsr'], 5),
                  f'({time.time() - t0:.0f} s)', flush=True)
    keys = [k for k in rows[0] if k != 'label']
    with open(common.OUT / 'sort_level.csv', 'w') as fh:
        fh.write('label,' + ','.join(keys) + '\n')
        for r in rows:
            fh.write(r['label'] + ',' + ','.join(f'{r[k]:.6g}' for k in keys) + '\n')
    from scipy.stats import pearsonr
    A = {k: np.array([r[k] for r in rows]) for k in keys}
    kinds = {'all': np.ones(len(rows), bool),
             'ks4_only': np.array([not r['label'].startswith('ks25') for r in rows])}
    stats = {}
    for kind, sel in kinds.items():
        stats[kind] = {}
        for k in ('n_eligible', 'S_track', 'sum_msr', 'mean_msr', 'median_msr', 'sum_dmsr', 'mean_dmsr',
                  'yield_dmsr', 'sum_pos_dmsr', 'n_spikes'):
            x = A[k][sel]
            stats[kind][k] = {'spearman_vs_S': common.spearman(x, A['S'][sel]),
                              'pearson_vs_S': float(pearsonr(x, A['S'][sel])[0]),
                              'partial_spearman_vs_S_given_n_eligible': common.partial_spearman(x, A['S'][sel], A['n_eligible'][sel])}
    dsel = np.array([r['label'].startswith('default') for r in rows])
    noise = {k: float(np.std(A[k][dsel], ddof=1)) for k in keys}
    means = {k: float(np.mean(A[k][dsel])) for k in keys}
    h8 = [r for r in rows if r['label'].startswith('198_H8')]
    out = {'n_sorts': len(rows), 'stats': stats, 'default_mean': means, 'default_sd': noise,
           'H8_s1_minus_default_in_default_sd': {k: (h8[0][k] - means[k]) / noise[k] for k in keys} if h8 else None,
           'runtime_s': time.time() - t0}
    (common.OUT / 'sort_level.json').write_text(json.dumps(out, indent=1))
    for kind in stats:
        print(kind)
        for k, v in stats[kind].items():
            print(f"  {k:14s} rho {v['spearman_vs_S'][0]:+.3f}  r {v['pearson_vs_S']:+.3f}  partial|n {v['partial_spearman_vs_S_given_n_eligible'][0]:+.3f}")
    print('default sd', {k: round(v, 4) for k, v in noise.items()})
    print('H8 vs default (in default SDs)', {k: round(v, 1) for k, v in (out['H8_s1_minus_default_in_default_sd'] or {}).items()})
