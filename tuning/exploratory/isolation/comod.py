"""Contamination with the co-modulation of firing (theta, bursts, shared fluctuations) allowed for.

The plain estimate solves V = 2 dtau D c (1 - c/2) with D = N^2 / T, the density of forward spike pairs
per second of lag if spikes were independent of each other at the mean rate. Contaminating spikes that
share the unit's rate fluctuations coincide with its own spikes more often than that. Two replacements
for D, each assuming the contaminants share the unit's own modulation:

- binned: D = sum_j k_j (k_j - 1) / bin over 25 ms bins (an estimate of the integral of r(t)^2). It follows
  theta, but also counts the unit's own bursts within a bin, so it is an upper bound on the correction;
- local: D = forward pairs at lags 10-30 ms / 20 ms, the unit's autocorrelogram just past its bursts. It
  skips most burst intervals but sits a quarter of a theta cycle off zero lag, so it corrects less.

Reported per sorter (seeds pooled, all units, hybrid copies out): % misassigned under each estimate, by
firing-rate band, and the split of gaps between sorters into rate mix and within rate.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from isolation import TAU_C, violations  # noqa: E402
from rate_mix import EDGES, SETS, TAU, units  # noqa: E402  (units() is re-implemented below for spike trains)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import hybrid  # noqa: E402
from cleaning import load_clean_units  # noqa: E402

BIN = 0.025
LOCAL = (0.010, 0.030)
K = Path('C:/data/ks4_tuning/26648_kobe_2020-08-21')
M = Path('C:/data/ks4_tuning/25843_roger_2019-09-11')


def solve(v, D):
    a = v / (2 * (TAU - TAU_C) * max(D, 1e-12))
    return 1 - np.sqrt(1 - 2 * a) if a <= 0.5 else 1.0


def unit_estimates(t, T):
    t = np.sort(t)
    n, v = t.size, violations(t, TAU)
    k = np.bincount((t // BIN).astype(int))
    pairs = np.sum(np.searchsorted(t, t + LOCAL[1], 'right') - np.searchsorted(t, t + LOCAL[0], 'right'))
    return n, n / T, {'plain': solve(v, n * n / T), 'binned': solve(v, np.sum(k * (k - 1)) / BIN),
                      'local': solve(v, pairs / (LOCAL[1] - LOCAL[0]))}


def sort_units(sort, ex):
    base = K if ex == 'kobe' else M
    by, _, iv = load_clean_units(sort, base / 'search/raw_artefacts_car.npz')
    if ex == 'kobe':
        copies = set(hybrid.recovery(sort, np.load(base / 'search/hybrid/hybrid_truth.npz'))['hybrid_copies'])
        by = {c: t for c, t in by.items() if c not in copies}
    T = 1200 - float(np.sum(iv[:, 1] - iv[:, 0])) if iv.size else 1200.0
    return [unit_estimates(t, T) for t in by.values() if t.size]


if __name__ == '__main__':
    METHODS = ('plain', 'binned', 'local')
    for ex, groups in SETS.items():
        data = {}
        for g, sorts in groups.items():
            rows = [r for s in sorts for r in sort_units(s, ex)]
            n = np.array([r[0] for r in rows]); rate = np.array([r[1] for r in rows])
            b = np.digitize(rate, EDGES) - 1
            data[g] = {}
            for m in METHODS:
                c = np.array([r[2][m] for r in rows])
                w = np.array([n[b == k].sum() for k in range(len(EDGES) - 1)]) / n.sum()
                cb = np.array([(n * c)[b == k].sum() / max(n[b == k].sum(), 1) for k in range(len(EDGES) - 1)])
                data[g][m] = (w, cb, (n * c).sum() / n.sum(), np.mean(c >= 1))
        print(f'\n{ex}: % misassigned within firing-rate band (Hz), and overall; [share of units saturated at c = 1]')
        print(f"{'':16s}" + ''.join(f'{f"{a}-{bb}":>9s}' for a, bb in zip(EDGES[:-1], EDGES[1:])) + '   overall  saturated')
        for g in groups:
            for m in METHODS:
                w, cb, tot, sat = data[g][m]
                print(f'{g:8s}{m:8s}' + ''.join(f'{x:9.1%}' for x in cb) + f'   {tot:7.1%}  {sat:8.1%}')
        print('gaps: total = rate mix + within rate')
        for g1, g2 in (('default', 'H8'), ('default', 'ks25'), ('H8', 'ks25')):
            for m in METHODS:
                (w1, c1, t1, _), (w2, c2, t2, _) = data[g1][m], data[g2][m]
                mix = np.sum((w2 - w1) * (c1 + c2) / 2)
                within = np.sum((c2 - c1) * (w1 + w2) / 2)
                print(f'  {g2} - {g1} ({m:6s}): {t2 - t1:+6.1%} = rate mix {mix:+6.1%} + within rate {within:+6.1%}')
