"""Compare reference-heading variants by how well they predict held-out units (moving theta cycles).

Eligible units are split into two groups by alternating 100 µm depth blocks, so a neuron detected
twice (same channels) stays within one group. Each variant is built from one group only; every unit
of the other group then gets a tuning curve against that variant on one time half, scored on the
other half (Poisson log-likelihood gain over the unit's mean rate, bits/s); halves and groups are
swapped and summed. Tracked heading is scored the same way as the baseline, and a time-shifted
tracked heading as a null control.

The sort is cleaned first (spec §6.1-6.3): cycles more than 50% masked are dropped, others have
their duration scaled by the unmasked fraction.

Agreement with tracked heading is reported only as a sanity floor; it is not the criterion.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from score import unit_scores
from cleaning import load_clean_units, masked_fraction
from score import eligible as eligible_units_of
from theta_reference import (DECODERS, N_ITER, compare_to_tracking, cycle_counts,
                             internal_direction, theta_cycles)

BLOCK_UM = 100

p = argparse.ArgumentParser()
p.add_argument('--sort', required=True, help='KS4 results folder (with acg_class.npz)')
p.add_argument('--tracking', required=True)
p.add_argument('--t0', type=float, required=True)
p.add_argument('--duration', type=float, required=True)
p.add_argument('--out', required=True, help='output .json')
p.add_argument('--raw-artefacts', required=True)
p.add_argument('--n-iter', type=int, default=N_ITER)
args = p.parse_args()

by_cluster, depth, mask = load_clean_units(args.sort, args.raw_artefacts)
acg = np.load(Path(args.sort) / 'acg_class.npz')
source = [int(c) for c, lab in zip(acg['cluster'], acg['raw_acg_label']) if lab in (1, 2)]
bounds, dur, moving, hd = theta_cycles(by_cluster, source, np.load(args.tracking), args.t0, args.duration)
frac = masked_fraction(bounds, mask)
moving, dur = moving & (frac <= 0.5), dur * (1 - frac)
eligible = np.array(eligible_units_of(args.sort, by_cluster, args.duration))
counts = cycle_counts(by_cluster, eligible, bounds)
group = np.array([int(depth[c] // BLOCK_UM) % 2 for c in eligible])
half = ((bounds[:-1] + bounds[1:]) / 2) >= args.duration / 2


def heldout_scores(theta, cnt):
    return unit_scores(theta, cnt, dur, moving, half)


variants = {}
per_unit = {}
for build in (0, 1):
    b, s = group == build, group != build
    thetas = {'tracked_heading': hd,
              'tracked_heading_shifted_null': np.roll(hd, hd.size // 2)}
    for dec in DECODERS:
        for it, th in enumerate(internal_direction(counts[:, b], dur, moving, hd, dec, args.n_iter), 1):
            thetas[f'{dec}_iter{it}'] = th
    for name, th in thetas.items():
        per_unit.setdefault(name, np.zeros(eligible.size))[s] = heldout_scores(th, counts[:, s])
        variants.setdefault(name, {})[f'built_from_group{build}_vs_tracking'] = compare_to_tracking(th, hd, moving)

base = per_unit['tracked_heading']
rows = {}
for name, v in per_unit.items():
    diff = v - base
    rows[name] = {'heldout_S_bits_per_s': float(v.sum()),
                  'delta_vs_tracked_bits_per_s': float(diff.sum()),
                  'per_unit_delta_mean_se': [float(diff.mean()), float(diff.std(ddof=1) / np.sqrt(diff.size))],
                  'fraction_units_improved': float(np.mean(diff > 0)),
                  **variants[name]}
report = {'n_eligible': int(eligible.size), 'group_sizes': [int(np.sum(group == 0)), int(np.sum(group == 1))],
          'n_moving_cycles': int(moving.sum()),
          'moving_cycles_per_half': [int(np.sum(moving & ~half)), int(np.sum(moving & half))],
          'variants': rows}
Path(args.out).write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k != 'variants'}))
print(f"{'variant':34s} {'held-out S':>11s} {'d vs tracked':>13s} {'per-unit d (SE)':>20s} {'units better':>12s}  Fisher-Lee vs tracking (built from group 0 / 1)")
for name, r in sorted(rows.items(), key=lambda kv: -kv[1]['heldout_S_bits_per_s']):
    fl = [r[f'built_from_group{g}_vs_tracking']['fisher_lee'] for g in (0, 1)]
    print(f"{name:34s} {r['heldout_S_bits_per_s']:11.1f} {r['delta_vs_tracked_bits_per_s']:+13.1f} "
          f"{r['per_unit_delta_mean_se'][0]:+9.3f} ({r['per_unit_delta_mean_se'][1]:.3f}) {r['fraction_units_improved']:12.2f}  {fl[0]:.3f} / {fl[1]:.3f}")
