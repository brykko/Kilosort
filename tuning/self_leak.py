"""Phase 0 item 8.4 (spec §8.4): self-leak check by depth split.

theta_ref is rebuilt (same procedure as build_reference.py) twice from the reference sort's eligible
units: once from units deeper than the median depth + 75 um, once from units shallower than the
median - 75 um (150 um buffer). Each candidate unit is scored only against the half-reference built
from the other side of the probe, so no unit is scored against a reference containing its own
spikes. Reports S under the full and split references for each sort, and the pairwise differences
between sorts under each; the split reference is noisier (half the units), so levels drop, and the
check is whether comparisons change (< 0.25 delta).
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

import hybrid
import score
from cleaning import load_clean_units, masked_fraction
from theta_reference import cycle_counts, internal_direction

BUFFER_UM = 150.0

p = argparse.ArgumentParser()
p.add_argument('--reference-sort', required=True)
p.add_argument('--reference', required=True, help='frozen theta_ref.npz of the reference sort')
p.add_argument('--sorts', nargs='+', required=True, help='name=folder')
p.add_argument('--raw-artefacts', required=True)
p.add_argument('--t0', type=float, required=True)
p.add_argument('--duration', type=float, required=True)
p.add_argument('--hybrid-truth')
p.add_argument('--delta', type=float, required=True)
p.add_argument('--out', required=True)
args = p.parse_args()

ref = np.load(args.reference)
bounds = ref['cycle_bounds_s'] - args.t0
half = (bounds[:-1] + bounds[1:]) / 2 >= args.duration / 2
usable = np.isfinite(ref['theta_ref'])
hd = ref['hd_tracked']

# half-references from the reference sort
by_ref, depth_ref, mask_ref = load_clean_units(args.reference_sort, args.raw_artefacts)
units_ref = [int(u) for u in ref['units']]
med = float(np.median([depth_ref[u] for u in units_ref]))
frac_ref = masked_fraction(bounds, mask_ref)
dur_ref = np.diff(bounds) * (1 - frac_ref)
theta_half = {}
for side, sel in (('deep', lambda d: d >= med + BUFFER_UM / 2), ('shallow', lambda d: d <= med - BUFFER_UM / 2)):
    us = [u for u in units_ref if sel(depth_ref[u])]
    th = internal_direction(cycle_counts(by_ref, us, bounds), dur_ref, usable, np.where(usable, hd, 0.0),
                            'poisson_mean')[-1]
    theta_half[side] = np.where(usable, th, np.nan)
    print(f'{side} half-reference: {len(us)} units')

S = {}
for name, folder in (a.split('=', 1) for a in args.sorts):
    by_cluster, depth, mask = load_clean_units(folder, args.raw_artefacts)
    if args.hybrid_truth:
        copies = set(hybrid.recovery(folder, np.load(args.hybrid_truth))['hybrid_copies'])
        by_cluster = {c: t for c, t in by_cluster.items() if c not in copies}
    units = score.eligible(folder, by_cluster, args.duration)[0]
    frac = masked_fraction(bounds, mask)
    dur = np.diff(bounds) * (1 - frac)
    valid = usable & (frac <= 0.5)
    counts = cycle_counts(by_cluster, units, bounds)
    full = score.corrected_scores(ref['theta_ref'], counts, dur, valid, half).sum()
    shallow_units = np.array([depth[u] < med for u in units])   # scored against the deep reference
    split = (score.corrected_scores(theta_half['deep'], counts[:, shallow_units], dur, valid, half).sum()
             + score.corrected_scores(theta_half['shallow'], counts[:, ~shallow_units], dur, valid, half).sum())
    S[name] = {'full': float(full), 'split': float(split)}
    print(f'{name:12s} S full {full:7.1f}   split {split:7.1f}')

pairs = {f'{b} - {a}': {'full': S[b]['full'] - S[a]['full'], 'split': S[b]['split'] - S[a]['split']}
         for a, b in itertools.combinations(S, 2)}
worst = max(abs(d['full'] - d['split']) for d in pairs.values())
print(f'largest change in a pairwise difference: {worst:.2f} bits/s (limit 0.25 delta = {0.25 * args.delta:.2f})')
Path(args.out).write_text(json.dumps({'median_depth_um': med, 'S': S, 'pairwise': pairs,
                                      'max_change': worst, 'pass': worst < 0.25 * args.delta}, indent=2))
