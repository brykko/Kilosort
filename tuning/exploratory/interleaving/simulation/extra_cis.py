"""Paired CIs the verdicts need beyond analyse.py: poor-match units against R2 (the best R arm on them), wide units across rounds,
sparse units for S, and I's dictionary size."""
import glob
import json
import os

import numpy as np

OUT = r'C:\data\ks4_tuning\exploratory\interleaving\simulation'
rng = np.random.default_rng(2)


def boot(x, n=10000):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    i = rng.integers(0, len(x), (n, len(x)))
    return f'{x.mean():+.3f} [{np.percentile(x[i].mean(1), 2.5):+.3f}, {np.percentile(x[i].mean(1), 97.5):+.3f}] (n={len(x)})'


E, TR = {}, {}
for f in glob.glob(os.path.join(OUT, 'main', '*.json')):
    d = json.load(open(f))
    v, s = d['job']['variant'], d['job']['seed']
    TR[(v, s)] = d['truth']
    for a in d['arms']:
        for r in a['records']:
            if 'C_found' not in r:
                continue
            if a['name'] == 'R4':
                E[(v, s, 'R' + r['tag'][-1])] = r
            elif a['name'] == 'Rd3':
                E[(v, s, 'Rd' + r['tag'][-1])] = r
            elif r['tag'] == 'final':
                E[(v, s, a['name'])] = r


def fr(v, s, arm, lab, mask):
    t = TR[(v, s)]
    m = mask(t)
    if not m.any():
        return np.nan
    return np.array(E[(v, s, arm)][f'{lab}_found'], float)[m].mean()


poor = lambda t: np.array(t['um']) < .75
wide = lambda t: (np.array(t['kinds']) == 'wide') & (np.array(t['um']) < .75)
sparse = lambda t: np.array(t['rates']) < .5
for v in ('stationary', 'nonstat', 'drift'):
    S = range(1, 11)
    d = lambda a, la, b, lb, m: boot([fr(v, s, a, la, m) - fr(v, s, b, lb, m) for s in S])
    print(v)
    print('  poor: I(1) T - R2 C', d('I1_shuffled', 'T', 'R2', 'C', poor))
    print('  poor: I(1) C - R2 C', d('I1_shuffled', 'C', 'R2', 'C', poor))
    print('  poor: Rd3 C - R2 C', d('Rd3', 'C', 'R2', 'C', poor))
    print('  wide: R4 C - R2 C ', d('R4', 'C', 'R2', 'C', wide))
    print('  wide: R3 C - R2 C ', d('R3', 'C', 'R2', 'C', wide))
    print('  sparse: S C - R3 C', d('S0.25_3', 'C', 'R3', 'C', sparse))
    print('  sparse: S C - I(1) C', d('S0.25_3', 'C', 'I1_shuffled', 'C', sparse))
    print('  T vs C: I(1) C - I(1) T acc', boot([E[(v, s, 'I1_shuffled')]['C']['acc'] - E[(v, s, 'I1_shuffled')]['T']['acc'] for s in S]))
    print('  T vs C: R4 C - R4 T acc', boot([E[(v, s, 'R4')]['C']['acc'] - E[(v, s, 'R4')]['T']['acc'] for s in S]))
    print('  S te_mean - R3', boot([E[(v, s, 'S0.25_3')]['te_mean'] - E[(v, s, 'R3')]['te_mean'] for s in S]))
