"""Post hoc: where does I's template-error deficit sit? Per-unit error on units both arms find, by unit type."""
import glob
import json
import os

import numpy as np

OUT = r'C:\data\ks4_tuning\exploratory\interleaving\simulation'
rng = np.random.default_rng(1)


def boot(x, n=10000):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    i = rng.integers(0, len(x), (n, len(x)))
    return x.mean(), *np.percentile(x[i].mean(1), [2.5, 97.5])


E, TR = {}, {}
for f in glob.glob(os.path.join(OUT, 'main', '*.json')):
    d = json.load(open(f))
    v, s = d['job']['variant'], d['job']['seed']
    TR[(v, s)] = d['truth']
    for a in d['arms']:
        for r in a['records']:
            if 'te' in r and 'C_found' in r:
                if a['name'] == 'R4':
                    E[(v, s, 'R' + r['tag'][-1])] = r
                elif r['tag'] == 'final':
                    E[(v, s, a['name'])] = r

print('Template error by unit group, mean over seeds of the group median (I(1) at end of learning vs R3); paired diff CI')
print('variant | group | n | R3 | I(1) | I(1) - R3')
for v in ('stationary', 'nonstat', 'drift'):
    seeds = range(1, 11)
    def group(name, mk):
        a, b, n = [], [], []
        for s in seeds:
            t = TR[(v, s)]
            ra, rb = E[(v, s, 'R3')], E[(v, s, 'I1_shuffled')]
            m = mk(t, ra, rb)
            if m.sum() == 0:
                continue
            a.append(np.median(np.array(ra['te'])[m])); b.append(np.median(np.array(rb['te'])[m])); n.append(m.sum())
        d_ = np.array(b) - np.array(a)
        c = boot(d_)
        print(f'{v} | {name} | {np.mean(n):.1f} | {np.mean(a):.3f} | {np.mean(b):.3f} | {c[0]:+.3f} [{c[1]:+.3f}, {c[2]:+.3f}]')

    def pm(t):
        m = np.zeros(len(t['rates']), bool)
        for a, b in t['pairs']:
            m[a] = m[b] = True
        return m

    group('all', lambda t, a, b: np.ones(len(t['rates']), bool))
    both = lambda t, a, b: (np.array(a['C_found']) > 0) & (np.array(b['C_found']) > 0)
    group('found by both (C)', both)
    group('found by both, not in a pair', lambda t, a, b: both(t, a, b) & ~pm(t))
    group('found by both, in a pair', lambda t, a, b: both(t, a, b) & pm(t))
    group('rate > 3 Hz, not in a pair', lambda t, a, b: (np.array(t['rates']) > 3) & ~pm(t))
    group('rate < 3 Hz, not in a pair', lambda t, a, b: (np.array(t['rates']) < 3) & ~pm(t))
    group('poor', lambda t, a, b: np.array(t['um']) < .75)
