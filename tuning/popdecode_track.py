"""Variant: decode tracked heading (not theta_ref), ridge strength chosen per sort by nested CV."""
import json
import numpy as np
import popdecode as P

GRID = [3e-3, 1e-2, 3e-2, 1e-1]


def design_track(by, units, mask):
    bounds = P.REF['cycle_bounds_s'] - P.T0
    half = (bounds[:-1] + bounds[1:]) / 2 >= P.DUR / 2
    frac = P.masked_fraction(bounds, mask)
    dur = np.diff(bounds) * (1 - frac)
    hd = P.REF['hd_tracked']
    valid = P.REF['moving'] & (frac <= 0.5) & np.isfinite(hd)
    X = P.cycle_counts(by, units, bounds)[valid] / dur[valid, None]
    return X, hd[valid], half[valid], dur[valid]


def score(sort):
    by, u, m = P.load(sort)
    X, th, half, dur = design_track(by, u, m)
    lam = P.choose_lambda(X, th, half, GRID)
    return P.score_D(X, th, half, dur, lam)[1], lam, len(u)


S = P.BASE / 'runs/search'
sets = {'default': [P.BASE / f'runs/hybrid_default_seed{k}' for k in range(1, 6)],
        'final merge 0.7': [S / r / 'sort' for r in ('010_S6_corr2_070_s1', '021_S6_corr2_070_s2', '022_S6_corr2_070_s3')],
        'standard': [S / r / 'sort' for r in ('001_S1_standard_s1', '087_S1_standard_s2', '088_S1_standard_s3')],
        'R4 (gamma 2)': [S / r / 'sort' for r in ('094_R4_g2_f1_f2095_s1', '108_R4_g2_f1_f2095_s2', '121_R4_g2_f1_f2095_s3')],
        'gamma 8': [S / r / 'sort' for r in ('140_X4_r4_g8_s1', '142_X4_r4_g8_s2')],
        'Th_universal 7': [S / '148_D5_thu7_s1' / 'sort']}
res = {}
for name, sorts in sets.items():
    res[name] = [score(s) for s in sorts]
    v = np.array([r[0] for r in res[name]])
    if name == 'default':
        d0, sd = v.mean(), v.std(ddof=1)
        print(f'default {d0:.2f} +- {sd:.2f} bits/s (mean/SD {d0 / sd:.0f}); lambdas {[r[1] for r in res[name]]}', flush=True)
    else:
        print(f'{name:16s} dD {v.mean() - d0:+6.2f} bits/s ({(v.mean() - d0) / sd:+5.1f} SD) seeds {np.round(v, 2).tolist()} lambdas {[r[1] for r in res[name]]}', flush=True)
(P.BASE / 'runs/popdecode_track.json').write_text(json.dumps({k: [list(r) for r in v] for k, v in res.items()}))
