"""E1 scoring: per inserted hybrid unit, agreement in a window sort vs a full-length sort cut to the same window.

Agreement as tuning/hybrid.py (hit within +-0.4 ms; hits / (n_true + n_sorted - hits); best sorted unit),
computed on truth spikes and sorted spikes inside [a, b) only.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FS, TOL = 30000, 12
EX = Path('C:/data/ks4_tuning/26648_kobe_2020-08-21')
SA = EX / 'runs' / 'scaling_audit'
truth = np.load(EX / 'search/hybrid/hybrid_truth.npz')
TT, TU = truth['spike_samples'].astype(np.int64), truth['spike_unit'].astype(np.int64)
NINS = int(TU.max()) + 1


def load(sort):
    st = np.load(Path(sort) / 'spike_times.npy').ravel().astype(np.int64)
    cl = np.load(Path(sort) / 'spike_clusters.npy').ravel().astype(np.int64)
    o = np.argsort(st, kind='stable')
    lab = pd.read_csv(Path(sort) / 'cluster_KSLabel.tsv', sep='\t')
    return st[o], cl[o], int((lab.KSLabel == 'good').sum()), lab.shape[0]


def agreement(st, cl, a, b):
    m = (st >= a) & (st < b)
    st, cl = st[m], cl[m]
    ids, ci = np.unique(cl, return_inverse=True)
    ns = np.bincount(ci, minlength=ids.size)
    mt = (TT >= a) & (TT < b)
    tt, tu = TT[mt], TU[mt]
    nt = np.bincount(tu, minlength=NINS)
    lo, hi = np.searchsorted(st, tt - TOL, 'left'), np.searchsorted(st, tt + TOL, 'right')
    pairs = []
    for d in range(int((hi - lo).max())):
        has = lo + d < hi
        pairs.append(np.stack([np.where(has)[0], ci[lo[has] + d]], 1))
    key = np.unique(np.concatenate(pairs), axis=0)
    hits = np.zeros((NINS, ids.size), np.int64)
    np.add.at(hits, (tu[key[:, 0]], key[:, 1]), 1)
    ag = hits / np.maximum(nt[:, None] + ns[None, :] - hits, 1)
    frag = (hits / np.maximum(nt[:, None], 1) >= 0.1).sum(1)    # sorted units holding >= 10% of a unit's spikes
    return ag.max(1), nt, frag


def windows():
    for p in sorted(SA.glob('w*_s1')):
        if (p / 'spike_times.npy').exists():
            a, b = (int(x) for x in p.name.split('_')[1].split('-'))
            yield p, a, b


full = [SA / 'full_0-1200_s1'] + [EX / f'runs/hybrid_default_seed{k}' for k in range(1, 6)]
full = [f for f in full if (f / 'spike_times.npy').exists()]
fl = {f: load(f) for f in full}
rows, summ = [], []
for p, a, b in windows():
    st, cl, ngood, nunits = load(p)
    A, B = a * FS, b * FS
    ag_w, nt, fr_w = agreement(st, cl, A, B)
    res_f = {f.name: agreement(fl[f][0], fl[f][1], A, B) for f in full}
    ag_f = {k: v[0] for k, v in res_f.items()}
    fr_f = {k: v[2] for k, v in res_f.items()}
    for u in range(NINS):
        rows.append({'window': p.name, 'unit': u, 'n_true': int(nt[u]), 'rate_hz': nt[u] / (b - a),
                     'ag_window': ag_w[u], 'frag_window': int(fr_w[u]), **{f'ag_{k}': v[u] for k, v in ag_f.items()},
                     **{f'frag_{k}': int(v[u]) for k, v in fr_f.items()}})
    summ.append({'window': p.name, 'min': (b - a) / 60, 'units': nunits, 'good': ngood,
                 'rec_window': int((ag_w >= .8).sum()), 'med_window': float(np.median(ag_w)), 'split_window': int((fr_w > 1).sum()),
                 **{f'split_{k}': int((v > 1).sum()) for k, v in fr_f.items()},
                 **{f'rec_{k}': int((v >= .8).sum()) for k, v in ag_f.items()},
                 **{f'med_{k}': float(np.median(v)) for k, v in ag_f.items()}})
for f in full:
    st, cl, ng, nu = fl[f]
    summ.append({'window': f.name, 'min': 20, 'units': nu, 'good': ng})
df, sm = pd.DataFrame(rows), pd.DataFrame(summ)
df.to_csv(SA / 'e1_per_unit.csv', index=False)
sm.to_csv(SA / 'e1_summary.csv', index=False)
pd.set_option('display.width', 250, 'display.max_columns', 40)
print(sm.round(3).to_string())
if len(df):
    ref = 'ag_full_0-1200_s1' if 'ag_full_0-1200_s1' in df else 'ag_hybrid_default_seed1'
    df['d'] = df.ag_window - df[ref]
    sd = df[[c for c in df if c.startswith('ag_hybrid')]].std(1)
    print(f'\npaired window - full ({ref}): mean d {df.d.mean():+.3f}, median {df.d.median():+.3f}, '
          f'n units better {int((df.d > .05).sum())}, worse {int((df.d < -.05).sum())} of {len(df)}; '
          f'seed-to-seed SD of full sorts (median over unit-windows) {sd.median():.3f}')
    df['rate_bin'] = pd.qcut(df.rate_hz, 4)
    print(df.groupby(['rate_bin'], observed=True)[['ag_window', ref, 'd']].mean().round(3))
