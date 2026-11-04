"""Exploratory: MSR on synthetic spike trains (validation of msr.py).

1. Homogeneous Poisson trains: MSR against rate (0.2-50 Hz) at 1200 s, and against duration at 5 Hz.
2. Matched-rate inhomogeneous Poisson trains, 1200 s:
   - head-direction cells driven by the real kobe tracked heading (hd_azimuth, 120 Hz), von Mises
     tuning exp(kappa cos(theta - mu)) scaled to the target mean rate, kappa 0-8; the true
     single-unit heading information (bits/s and bits/spike) is computed from the rate and occupancy;
   - 8 Hz theta-modulated trains (depth 0.5, 1.0);
   - slow, behaviour-free rate modulation: a 60 s on/off square wave (depth 0.5), and a
     drift-like dropout (no spikes in one contiguous 10% / 30% of the recording);
   - Poisson with a 2 ms dead time (refractoriness).
3. Each at rates 1, 5 and 20 Hz; REPS replicates; binarised and count versions for one case.
Output: C:/data/ks4_tuning/exploratory/msr/synthetic.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import msr  # noqa: E402

OUT = Path('C:/data/ks4_tuning/exploratory/msr')
OUT.mkdir(parents=True, exist_ok=True)
T = 1200.0
REPS = 10
DT = 1e-3
rng = np.random.default_rng(1)

trk = np.load('C:/data/ks4_tuning/26648_kobe_2020-08-21/search/tracking_1668-2868s.npz')
grid = np.arange(0, T, DT)
hd = np.interp(grid, trk['t'] - 1668.0, np.unwrap(trk['hd_azimuth'])) % (2 * np.pi)


def inhom(rate_fn_vals):
    """Inhomogeneous Poisson on the 1 ms grid from per-point rates (Hz)."""
    n = rng.poisson(rate_fn_vals * DT)
    idx = np.repeat(np.arange(grid.size), n)
    return np.sort(grid[idx] + rng.uniform(0, DT, idx.size))


def hd_rate(r, kappa, mu=1.0):
    f = np.exp(kappa * np.cos(hd - mu))
    return r * f / f.mean()


def hd_info(lam):
    """Single-unit heading information of rate lam(t) along the trajectory: bits/s and bits/spike."""
    b = np.floor(hd / (2 * np.pi) * 60).astype(int)
    occ = np.bincount(b, minlength=60) / b.size
    lb = np.bincount(b, weights=lam, minlength=60) / np.maximum(np.bincount(b, minlength=60), 1)
    rbar = (occ * lb).sum()
    ok = lb > 0
    bps = float(np.sum(occ[ok] * lb[ok] * np.log2(lb[ok] / rbar)))
    return bps, bps / rbar


def dead_time(t, d=2e-3):
    keep, last = [], -1.0
    for x in t:
        if x - last >= d:
            keep.append(x)
            last = x
    return np.asarray(keep)


def stats(gen, reps=REPS):
    v = [msr.msr(gen(), T) for _ in range(reps)]
    return {'mean': float(np.mean(v)), 'sd': float(np.std(v))}


t0 = time.time()
res = {'T_s': T, 'reps': REPS}

# 1. homogeneous Poisson: rate and duration
res['poisson_rate'] = {}
for r in (0.2, 0.5, 1, 2, 5, 10, 20, 50):
    res['poisson_rate'][r] = {**stats(lambda: np.sort(rng.uniform(0, T, rng.poisson(r * T)))), 'M': r * T}
res['poisson_duration_5Hz'] = {}
for d in (150, 300, 600, 1200, 2400):
    v = [msr.msr(np.sort(rng.uniform(0, d, rng.poisson(5 * d))), d) for _ in range(REPS)]
    res['poisson_duration_5Hz'][d] = {'mean': float(np.mean(v)), 'sd': float(np.std(v)), 'M': 5 * d}
print('poisson done', round(time.time() - t0, 1), 's')

# 2. inhomogeneous at matched rate
res['matched'] = {}
for r in (1, 5, 20):
    row = {'poisson': stats(lambda: np.sort(rng.uniform(0, T, rng.poisson(r * T))))}
    for kappa in (0.5, 1, 2, 4, 8):
        lam = hd_rate(r, kappa)
        bps, bpsp = hd_info(lam)
        row[f'hd_kappa{kappa}'] = {**stats(lambda: inhom(lam)), 'hd_bits_per_s': bps, 'hd_bits_per_spike': bpsp}
    for depth in (0.5, 1.0):
        lam = r * (1 + depth * np.cos(2 * np.pi * 8 * grid))
        row[f'theta8Hz_depth{depth}'] = stats(lambda: inhom(lam))
    lam = r * (1 + 0.5 * np.sign(np.sin(2 * np.pi * grid / 60)))
    row['square60s_depth0.5'] = stats(lambda: inhom(lam))
    for frac in (0.1, 0.3):
        lam = np.where(grid < frac * T, 0.0, r / (1 - frac))
        row[f'dropout{int(frac * 100)}pct'] = stats(lambda: inhom(lam))
    row['poisson_dead2ms'] = stats(lambda: dead_time(np.sort(rng.uniform(0, T, rng.poisson(r * T * 1.05)))))
    res['matched'][r] = row
    print('rate', r, 'done', round(time.time() - t0, 1), 's')

# binarised vs counts (5 Hz kappa 2)
lam = hd_rate(5, 2)
tr = [inhom(lam) for _ in range(REPS)]
res['binarize_check_5Hz_kappa2'] = {
    'counts': float(np.mean([msr.msr(t, T) for t in tr])),
    'binarized': float(np.mean([msr.msr(t, T, binarize=True) for t in tr]))}

# 3. contamination and thinning of a synthetic HD cell (5 Hz, kappa 2) by homogeneous Poisson spikes
res['synthetic_hd_errors'] = {}
for f in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5):
    v_c, v_t, v_p = [], [], []
    for t in tr:
        v_c.append(msr.msr(np.sort(np.concatenate([t, rng.uniform(0, T, int(f * t.size))])), T))
        v_t.append(msr.msr(t[rng.random(t.size) >= f], T))
    # a homogeneous Poisson train thinned the same way (rate-only effect)
    for _ in range(REPS):
        p = np.sort(rng.uniform(0, T, rng.poisson(5 * T)))
        v_p.append(msr.msr(p[rng.random(p.size) >= f], T))
    res['synthetic_hd_errors'][f] = {'contaminate_poisson': float(np.mean(v_c)), 'thin': float(np.mean(v_t)),
                                    'thin_of_homogeneous_poisson': float(np.mean(v_p))}
res['runtime_s'] = time.time() - t0
(OUT / 'synthetic.json').write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
