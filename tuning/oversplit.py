"""Over-splitting response of S (search_round2.md §4): does S peak and fall as a sort is split past
its real cells? Takes one sort, splits every eligible unit into k fragments, re-applies eligibility
(fragments inherit their unit's waveform-screen result) and scores S, without re-sorting.

Three ways to split, each at k = 2, 4, 8, 16:
- random: each spike to a random fragment (a split S should be indifferent to while fragments are
  large; small fragments lose to estimation noise and the 100-spikes-per-half floor);
- amplitude: by quantile of the spike's KS4 amplitude (a common real over-split);
- time: into k consecutive blocks of the excerpt (a drift-like split; fragments active in one half
  can't be predicted from the other).

    python oversplit.py --sort <sort> --reference <theta_ref.npz> --raw-artefacts <npz> --t0 1668
        --duration 1200 --hybrid-truth <npz> --out <json>
"""
import argparse
import json
from pathlib import Path

import numpy as np

import hybrid
import score
import waveform
from cleaning import load_clean_units, masked_fraction
from theta_reference import cycle_counts, eligible_units

KS = (2, 4, 8, 16)

p = argparse.ArgumentParser()
p.add_argument('--sort', required=True)
p.add_argument('--reference', required=True)
p.add_argument('--raw-artefacts', required=True)
p.add_argument('--t0', type=float, required=True)
p.add_argument('--duration', type=float, required=True)
p.add_argument('--hybrid-truth')
p.add_argument('--out', required=True)
args = p.parse_args()
rng = np.random.default_rng(0)

ref = np.load(args.reference)
bounds = ref['cycle_bounds_s'] - args.t0
half = (bounds[:-1] + bounds[1:]) / 2 >= args.duration / 2
amps = np.load(Path(args.sort) / 'amplitudes.npy')
by0, _, mask, amp0 = load_clean_units(args.sort, args.raw_artefacts, values=amps)
if args.hybrid_truth:
    copies = set(hybrid.recovery(args.sort, np.load(args.hybrid_truth))['hybrid_copies'])
    by0 = {c: t for c, t in by0.items() if c not in copies}
screen = waveform.screen(args.sort)
frac = masked_fraction(bounds, mask)
dur = np.diff(bounds) * (1 - frac)
valid = ref['moving'] & (frac <= 0.5) & np.isfinite(ref['theta_ref'])


def S_of(by, passes):
    units = [c for c in eligible_units(by, args.duration, None) if passes[c]]
    s = score.corrected_scores(ref['theta_ref'], cycle_counts(by, units, bounds), dur, valid, half)
    return float(s.sum()), len(units), sum(by[c].size for c in units)


passes0 = {c: screen[c][1] for c in by0}
S0, n0, spikes0 = S_of(by0, passes0)
units0 = [c for c in eligible_units(by0, args.duration, None) if passes0[c]]
print(f'baseline S {S0:.1f} bits/s, {n0} eligible units', flush=True)


def split(mode, k):
    by, passes = {c: t for c, t in by0.items() if c not in units0}, {c: passes0[c] for c in by0}
    new = max(by0) + 1
    for c in units0:
        t, a = by0[c], amp0[c]
        if mode == 'random':
            part = rng.integers(k, size=t.size)
        elif mode == 'amplitude':
            part = np.searchsorted(np.quantile(a, np.linspace(0, 1, k + 1)[1:-1]), a, side='right')
        else:
            part = np.minimum((t / args.duration * k).astype(int), k - 1)
        for j in range(k):
            by[new], passes[new] = t[part == j], passes0[c]
            new += 1
    return by, passes


out = {'sort': args.sort, 'baseline': {'S': S0, 'n_eligible': n0}, 'splits': {}}
for mode in ('random', 'amplitude', 'time'):
    for k in KS:
        S, n, spikes = S_of(*split(mode, k))
        out['splits'][f'{mode}_{k}'] = {'S': S, 'dS': S - S0, 'n_eligible': n, 'spikes_in_eligible': spikes / spikes0}
        print(f'{mode:9s} k={k:2d}: S {S:7.1f} (dS {S - S0:+7.1f}), eligible units {n:5d}, '
              f'spikes in eligible units {spikes / spikes0:.1%}', flush=True)
Path(args.out).write_text(json.dumps(out, indent=2))
