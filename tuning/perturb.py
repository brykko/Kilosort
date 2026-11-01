"""Phase 0 item 8.6 (spec §8.6): perturbation audit of a default sort, without re-sorting.

Each perturbation is applied at a 5% rate to the cleaned sort (hybrid copies removed), eligibility
(spike criteria; waveform screen inherited from the source unit) is recomputed, and S is rescored:
- merge: 5% of eligible units each merged with an eligible unit whose preferred direction (against
  theta_ref) differs by >= 90°;
- drop: 5% of eligible units removed;
- contaminate: every eligible unit gains 5% extra spikes drawn from other units;
- thin: every eligible unit loses 5% of its spikes;
- split: 5% of eligible units split in two at random (S should barely change);
- duplicate: 5% of eligible units copied (80% of spikes, +-2 samples jitter), then the §6.3 duplicate
  test is run on the result (S should change by < sigma_S; the cleaning-burden guard should trip).
Pass (spec): merge, drop, contaminate and thin each lower S by >= 3 sigma_S.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import cleaning
import hybrid
import score
import waveform
from cleaning import load_clean_units, masked_fraction
from theta_reference import CENTRES, cycle_counts, eligible_units, tuning

RATE = 0.05
FS = 30000

p = argparse.ArgumentParser()
p.add_argument('--sort', required=True)
p.add_argument('--reference', required=True)
p.add_argument('--raw-artefacts', required=True)
p.add_argument('--t0', type=float, required=True)
p.add_argument('--duration', type=float, required=True)
p.add_argument('--hybrid-truth')
p.add_argument('--sigma-s', type=float, required=True)
p.add_argument('--draws', type=int, default=5)
p.add_argument('--out', required=True)
p.add_argument('--max-contamination', type=float, default=None,
               help='refractory eligibility limit (0.10 up to spec v0.8; none since v0.9)')
args = p.parse_args()
rng = np.random.default_rng(0)

ref = np.load(args.reference)
bounds = ref['cycle_bounds_s'] - args.t0
half = (bounds[:-1] + bounds[1:]) / 2 >= args.duration / 2
by0, _, mask = load_clean_units(args.sort, args.raw_artefacts)
if args.hybrid_truth:
    copies = set(hybrid.recovery(args.sort, np.load(args.hybrid_truth))['hybrid_copies'])
    by0 = {c: t for c, t in by0.items() if c not in copies}
screen = waveform.screen(args.sort)
frac = masked_fraction(bounds, mask)
dur = np.diff(bounds) * (1 - frac)
valid = ref['moving'] & (frac <= 0.5) & np.isfinite(ref['theta_ref'])


def S_of(by, passes):
    """S of a perturbed sort; passes: unit -> waveform-screen result (inherited)."""
    units = [c for c in eligible_units(by, args.duration, args.max_contamination) if passes[c]]
    return float(score.corrected_scores(ref['theta_ref'], cycle_counts(by, units, bounds), dur, valid, half).sum()), units


passes0 = {c: screen[c][1] for c in by0}
S0, units0 = S_of(by0, passes0)
num, occ = tuning(np.where(valid, ref['theta_ref'], 0.0), cycle_counts(by0, units0, bounds), dur, valid)
pd = dict(zip(units0, np.angle((num / np.maximum(occ, 1e-9)) @ np.exp(1j * CENTRES))))
n_pert = int(round(RATE * len(units0)))
print(f'baseline S {S0:.1f} bits/s, {len(units0)} eligible units; perturbing {n_pert} units')


def merge(by):
    by, passes = dict(by), dict(passes0)
    for a in rng.choice(units0, n_pert, replace=False):
        if a not in by:
            continue
        cand = [b for b in units0 if b != a and b in by
                and abs(np.angle(np.exp(1j * (pd[a] - pd[b])))) >= np.pi / 2]
        b = rng.choice(cand)
        by[a] = np.sort(np.concatenate([by[a], by.pop(b)]))
    return by, passes


def drop(by):
    by = dict(by)
    for a in rng.choice(units0, n_pert, replace=False):
        by.pop(a)
    return by, passes0


def contaminate(by):
    all_t = np.concatenate(list(by.values()))
    return {c: (np.sort(np.concatenate([t, rng.choice(all_t, int(RATE * t.size))])) if c in units0 else t)
            for c, t in by.items()}, passes0


def thin(by):
    return {c: (t[rng.random(t.size) >= RATE] if c in units0 else t) for c, t in by.items()}, passes0


def split(by):
    by, passes = dict(by), dict(passes0)
    new_id = max(by) + 1
    for a in rng.choice(units0, n_pert, replace=False):
        t = by[a]
        side = rng.random(t.size) < 0.5
        by[a], by[new_id] = t[side], t[~side]
        passes[new_id] = passes0[a]
        new_id += 1
    return by, passes


results = {'baseline_S': S0, 'n_eligible': len(units0), 'n_perturbed': n_pert, 'sigma_S': args.sigma_s}
for name, fn in (('merge', merge), ('drop', drop), ('contaminate', contaminate), ('thin', thin)):
    dS = [S_of(*fn(by0))[0] - S0 for _ in range(args.draws)]
    results[name] = {'dS': dS, 'mean_dS': float(np.mean(dS)), 'in_sigma': float(np.mean(dS) / args.sigma_s),
                     'pass': bool(np.mean(dS) <= -3 * args.sigma_s)}
    print(f"{name:12s} dS {np.mean(dS):+7.1f} bits/s ({np.mean(dS) / args.sigma_s:+5.1f} sigma_S) "
          f"{'pass' if results[name]['pass'] else 'FAIL'}   draws: " + ' '.join(f'{x:+.1f}' for x in dS))

by_s, passes_s = split(by0)
dS = S_of(by_s, passes_s)[0] - S0
results['split'] = {'dS': dS, 'of': n_pert}
print(f'split        dS {dS:+7.1f} bits/s')

# duplicates: copy units, then run the duplicate test on the result
t_, u_, ids, xy = cleaning.load_sort(args.sort)
pos = dict(zip(ids.tolist(), xy))
by_d, passes_d, new_id = dict(by0), dict(passes0), max(by0) + 1
for a in rng.choice(units0, n_pert, replace=False):
    t = by0[a][rng.random(by0[a].size) < 0.8]
    by_d[new_id] = np.sort(t + rng.integers(-2, 3, t.size) / FS)
    passes_d[new_id], pos[new_id] = passes0[a], pos[a]
    new_id += 1
keys = sorted(by_d)
tt = np.concatenate([np.round(by_d[k] * FS).astype(np.int64) for k in keys])
uu = np.concatenate([np.full(by_d[k].size, i) for i, k in enumerate(keys)])
o = np.argsort(tt, kind='stable')
tt, uu = tt[o], uu[o]
dup, pairs, remove = cleaning.duplicates(tt, uu, np.array([pos[k] for k in keys]), rng)
by_clean = {k: tt[(uu == i) & ~remove] / FS for i, k in enumerate(keys)}
dS_raw = S_of(by_d, passes_d)[0] - S0
dS_clean = S_of(by_clean, passes_d)[0] - S0
results['duplicate'] = {'dS_before_cleaning': dS_raw, 'dS_after_cleaning': dS_clean,
                        'flagged_pairs': dup['n_flagged'], 'fraction_spikes_removed': float(remove.mean()),
                        'pass_S': bool(abs(dS_clean) < args.sigma_s)}
print(f"duplicate    dS before cleaning {dS_raw:+.1f}, after {dS_clean:+.1f} bits/s "
      f"({'pass' if abs(dS_clean) < args.sigma_s else 'FAIL'}: |dS| < sigma_S); flagged pairs {dup['n_flagged']}, "
      f'spikes removed {100 * remove.mean():.2f}%')
Path(args.out).write_text(json.dumps(results, indent=2))
