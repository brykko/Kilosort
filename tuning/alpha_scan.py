"""Phase 0 item 8.5 (spec §5.2, §8.5): the shrinkage alpha and the time-shift control.

For each alpha, scores every default replicate against theta_ref, raw (score.unit_scores) and
null-corrected (score.corrected_scores), and reports the mean, SD across replicates and mean/SD.
Then, at the chosen alpha (score.SCORE_PSEUDO_S), the time-shift control: each replicate's
null-corrected S against theta_ref rotated among the scored cycles by CONTROL_FRACTIONS, which lie
halfway between the null rotations (score.NULL_FRACTIONS), so no combination returns to zero shift.
Pass: |S_shifted| within 2 sigma_S of 0.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import hybrid
import score
from cleaning import load_clean_units, masked_fraction
from theta_reference import cycle_counts

ALPHAS = [0.1, 0.25, 0.5, 1, 2, 4]
CONTROL_FRACTIONS = [(k + 0.5) / 7 for k in range(1, 7)]

p = argparse.ArgumentParser()
p.add_argument('--sorts', nargs='+', required=True, help='default replicate sort folders')
p.add_argument('--reference', required=True)
p.add_argument('--raw-artefacts', required=True)
p.add_argument('--t0', type=float, required=True)
p.add_argument('--duration', type=float, required=True)
p.add_argument('--hybrid-truth')
p.add_argument('--out', required=True)
args = p.parse_args()

ref = np.load(args.reference)
theta = ref['theta_ref']
bounds = ref['cycle_bounds_s'] - args.t0
half = (bounds[:-1] + bounds[1:]) / 2 >= args.duration / 2
data = []
for s in args.sorts:
    by_cluster, _, mask = load_clean_units(s, args.raw_artefacts)
    if args.hybrid_truth:
        copies = set(hybrid.recovery(s, np.load(args.hybrid_truth))['hybrid_copies'])
        by_cluster = {c: t for c, t in by_cluster.items() if c not in copies}
    units = score.eligible(s, by_cluster, args.duration)
    frac = masked_fraction(bounds, mask)
    data.append((cycle_counts(by_cluster, units, bounds), np.diff(bounds) * (1 - frac),
                 ref['moving'] & (frac <= 0.5) & np.isfinite(theta)))

chosen = score.SCORE_PSEUDO_S
out = {'alpha': {}, 'chosen_alpha': chosen}
for a in ALPHAS:
    score.SCORE_PSEUDO_S = a
    raw = np.array([score.unit_scores(theta, c, d, v, half).sum() for c, d, v in data])
    cor = np.array([score.corrected_scores(theta, c, d, v, half).sum() for c, d, v in data])
    out['alpha'][a] = {'raw': raw.tolist(), 'corrected': cor.tolist()}
    print(f'alpha {a:5.2f}: raw S {raw.mean():6.1f} +- {raw.std(ddof=1):4.1f} (mean/SD {raw.mean() / raw.std(ddof=1):5.1f})'
          f'   corrected S {cor.mean():6.1f} +- {cor.std(ddof=1):4.1f} (mean/SD {cor.mean() / cor.std(ddof=1):5.1f})')

score.SCORE_PSEUDO_S = chosen
sigma_S = float(np.std(out['alpha'][chosen]['corrected'], ddof=1))
control = []
for c, d, v in data:
    for f in CONTROL_FRACTIONS:
        control.append(float(score.corrected_scores(score.rotate(theta, v, f), c, d, v, half).sum()))
out['control'] = {'fractions': CONTROL_FRACTIONS, 'S_shifted': control, 'sigma_S': sigma_S,
                  'pass': bool(abs(np.mean(control)) <= 2 * sigma_S)}
print(f'\ntime-shift control at alpha {chosen}: S_shifted {np.mean(control):+.2f} (SD {np.std(control, ddof=1):.2f}, '
      f'range {min(control):+.1f} to {max(control):+.1f}) vs 2 sigma_S = {2 * sigma_S:.1f} -> '
      f"{'pass' if out['control']['pass'] else 'FAIL'}")
Path(args.out).write_text(json.dumps(out, indent=2))
