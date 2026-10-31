"""Seed-stability check (spec §8.3): agreement between two theta_ref builds from different sorts.

Cycles are detected separately in each sort, so they are matched by time: each moving cycle of
the first build is paired with the cycle of the second build that contains its midpoint, if that
cycle is also moving.
"""
import argparse
import json

import numpy as np

from theta_reference import fisher_lee

p = argparse.ArgumentParser()
p.add_argument('ref_a')
p.add_argument('ref_b')
args = p.parse_args()

a, b = np.load(args.ref_a), np.load(args.ref_b)
mid = (a['cycle_bounds_s'][:-1] + a['cycle_bounds_s'][1:]) / 2
j = np.searchsorted(b['cycle_bounds_s'], mid) - 1
ok = a['moving'] & (j >= 0) & (j < b['moving'].size)
ok[ok] &= b['moving'][j[ok]]
ta, tb = a['theta_ref'][ok], b['theta_ref'][j[ok]]
err = np.angle(np.exp(1j * (ta - tb)))
print(json.dumps({
    'n_cycles_a': int(a['moving'].sum()), 'n_cycles_b': int(b['moving'].sum()), 'n_matched': int(ok.sum()),
    'fisher_lee': fisher_lee(ta, tb),
    'median_abs_difference_deg': float(np.degrees(np.median(np.abs(err)))),
    'fraction_over_45_deg': float(np.mean(np.abs(err) > np.pi / 4)),
}, indent=2))
