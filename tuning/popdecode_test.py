"""Driver for popdecode.py: decoding theta_ref; default replicates, round 1-3 leaders, perturbations, unit subsets."""
import json
import time
import numpy as np

from popdecode import BASE, REF, T0, DUR, load, design, score_D, choose_lambda, tuning, cycle_counts, CENTRES, masked_fraction

S = BASE / 'runs/search'
rng = np.random.default_rng(0)
out = {}

# 1. ridge strength, once, on default seed 1
by0, units0, mask0 = load(BASE / 'runs/hybrid_default_seed1')
X, th, half, dur = design(by0, units0, mask0)
print(f'default s1: {len(units0)} units, {len(th)} moving cycles ({half.sum()} in the second half)', flush=True)
lam = choose_lambda(X, th, half, [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2])
print('lambda', lam, flush=True)
out['lambda'] = lam

# 2. default replicates: noise
D = []
for k in range(1, 6):
    by, u, m = load(BASE / f'runs/hybrid_default_seed{k}')
    D.append(score_D(*design(by, u, m), lam))
    print(f'default s{k}: {D[-1][0]:.3f} bits/cycle, {D[-1][1]:.2f} bits/s ({len(u)} units)', flush=True)
Dc = np.array([d[0] for d in D]); Ds = np.array([d[1] for d in D])
sd = Ds.std(ddof=1)
print(f'default: {Ds.mean():.2f} +- {sd:.2f} bits/s, sensitivity mean/SD {Ds.mean() / sd:.1f} '
      f'({Dc.mean():.3f} bits/cycle)', flush=True)
out['default'] = {'bits_per_cycle': Dc.tolist(), 'bits_per_s': Ds.tolist()}

# 3. ranking
configs = {'final merge 0.7': ['010_S6_corr2_070_s1', '021_S6_corr2_070_s2', '022_S6_corr2_070_s3'],
           'standard': ['001_S1_standard_s1', '087_S1_standard_s2', '088_S1_standard_s3'],
           'R4 (gamma 2)': ['094_R4_g2_f1_f2095_s1', '108_R4_g2_f1_f2095_s2', '121_R4_g2_f1_f2095_s3'],
           'gamma 8': ['140_X4_r4_g8_s1', '142_X4_r4_g8_s2'],
           'Th_universal 7': ['148_D5_thu7_s1']}
out['configs'] = {}
for name, runs in configs.items():
    v = []
    for r in runs:
        by, u, m = load(S / r / 'sort')
        v.append(score_D(*design(by, u, m), lam)[1])
    out['configs'][name] = v
    print(f'{name:16s} dD {np.mean(v) - Ds.mean():+6.2f} bits/s ({(np.mean(v) - Ds.mean()) / sd:+5.1f} SD)  seeds {[round(x, 2) for x in v]}', flush=True)

# 4. perturbations of default seed 1 (5% of units, 3 draws), no cleaning of what is injected
bounds = REF['cycle_bounds_s'] - T0
frac = masked_fraction(bounds, mask0)
dd = np.diff(bounds) * (1 - frac)
valid = REF['moving'] & (frac <= 0.5) & np.isfinite(REF['theta_ref'])
num, occ = tuning(np.where(valid, REF['theta_ref'], 0.0), cycle_counts(by0, units0, bounds), dd, valid)
pd = dict(zip(units0, np.angle((num / np.maximum(occ, 1e-9)) @ np.exp(1j * CENTRES))))
base = score_D(X, th, half, dur, lam)[1]
n = int(round(0.05 * len(units0)))
out['perturbations'] = {}
for kind in ('duplicate', 'merge', 'drop'):
    v = []
    for draw in range(3):
        by, units = dict(by0), list(units0)
        new = max(by0) + 1
        for a in rng.choice(units0, n, replace=False):
            if kind == 'duplicate':
                t = by0[a]
                keep = rng.random(t.size) < 0.8
                by[new] = np.sort(t[keep] + rng.integers(-2, 3, keep.sum()) / 30000)
                units.append(new); new += 1
            elif kind == 'drop':
                units.remove(a)
            else:
                if a not in units:
                    continue
                cand = [b for b in units if b != a and abs(np.angle(np.exp(1j * (pd[a] - pd[b])))) >= np.pi / 2]
                b = rng.choice(cand)
                by[a] = np.sort(np.concatenate([by[a], by[b]])); units.remove(b)
        v.append(score_D(*design(by, units, mask0), lam)[1] - base)
    out['perturbations'][kind] = v
    print(f'{kind:9s} dD {np.mean(v):+6.2f} bits/s ({np.mean(v) / sd:+5.1f} SD)  draws {[round(x, 2) for x in v]}', flush=True)

# 5. saturation: random unit subsets of default seed 1
out['subsets'] = {}
for k in (50, 100, 200, 400, len(units0)):
    v = []
    for draw in range(3 if k < len(units0) else 1):
        sub = list(rng.choice(units0, k, replace=False))
        v.append(score_D(*design(by0, sub, mask0), lam)[1])
    out['subsets'][k] = v
    print(f'{k:4d} units: {np.mean(v):.2f} bits/s', flush=True)
Path = __import__('pathlib').Path
Path(BASE / 'runs/popdecode_test.json').write_text(json.dumps(out, indent=1))
