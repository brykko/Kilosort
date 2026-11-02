"""Template-age test: early- vs late-learned templates, matched over the whole file, compared per half.

If template age matters, each arm does better in the half its templates were learned from:
interaction = (early - late in the first half) - (early - late in the second half) > 0.
"""
import json
import sys
from pathlib import Path

import numpy as np

FS, TOL, CUT = 30000, 12, 600 * 30000
SA = Path('C:/data/ks4_tuning/26648_kobe_2020-08-21/runs/scaling_audit')
truth = np.load('C:/data/ks4_tuning/26648_kobe_2020-08-21/search/hybrid/hybrid_truth.npz')


def load(d):
    st = np.load(d / 'st_learned.npy')
    return st[np.argsort(st[:, 0])]


def recall(st):
    """Per inserted unit and half: fraction of inserted spikes with a learned spike within +-0.4 ms (label-free)."""
    t = st[:, 0].astype(np.int64)
    tt, tu = truth['spike_samples'].astype(np.int64), truth['spike_unit']
    lo, hi = np.searchsorted(t, tt - TOL), np.searchsorted(t, tt + TOL, 'right')
    hit = hi > lo
    half = (tt >= CUT).astype(int)
    n = int(tu.max()) + 1
    r = np.zeros((n, 2))
    for h in (0, 1):
        m = half == h
        r[:, h] = np.bincount(tu[m], weights=hit[m], minlength=n) / np.maximum(np.bincount(tu[m], minlength=n), 1)
    return r


def summary(st):
    h = (st[:, 0] >= CUT).astype(int)
    return {'n': [int((h == k).sum()) for k in (0, 1)],
            'amp_median': [float(np.median(st[h == k, 2])) for k in (0, 1)],
            'amp_p10': [float(np.percentile(st[h == k, 2], 10)) for k in (0, 1)]}


for ds in sys.argv[1:]:
    e, l = SA / f'tage_{ds}_early', SA / f'tage_{ds}_late'
    if not ((e / 'st_learned.npy').exists() and (l / 'st_learned.npy').exists()):
        print(ds, 'missing'); continue
    se, sl = load(e), load(l)
    ie, il = json.loads((e / 'info.json').read_text()), json.loads((l / 'info.json').read_text())
    a, b = summary(se), summary(sl)
    print(f'\n== {ds}: pass-1 templates early {ie["n_templates_pass1"]} / late {il["n_templates_pass1"]}; '
          f'matched {ie["n_templates_matched"]} / {il["n_templates_matched"]}')
    for k, nm in ((0, 'first half'), (1, 'second half')):
        print(f'  {nm}: spikes early {a["n"][k]:,} late {b["n"][k]:,} (early/late {a["n"][k] / b["n"][k]:.4f}); '
              f'median th_amp {a["amp_median"][k]:.2f} / {b["amp_median"][k]:.2f}; p10 {a["amp_p10"][k]:.2f} / {b["amp_p10"][k]:.2f}')
    inter = np.log(a['n'][0] / b['n'][0]) - np.log(a['n'][1] / b['n'][1])
    print(f'  interaction (log spike-count ratio, own half advantage x2): {inter:+.4f}')
    if ds == 'kobe':
        re, rl = recall(se), recall(sl)
        d = (re[:, 0] - rl[:, 0]) - (re[:, 1] - rl[:, 1])
        print(f'  hybrid detection recall, mean over 40 units: early {re.mean(0).round(4)}, late {rl.mean(0).round(4)}')
        print(f'  per-unit interaction: mean {d.mean():+.4f}, median {np.median(d):+.4f}, '
              f'units > +0.02: {(d > .02).sum()}, < -0.02: {(d < -.02).sum()}')
