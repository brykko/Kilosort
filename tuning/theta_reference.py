"""Reference heading theta_ref quantised to theta cycles (spec §5.1, prototype).

Follows elv:
- speed: position smoothed with a 0.5 s Gaussian (elv RecordingData.calc_speed), on a 10 ms grid;
  moving = speed >= 0.05 m/s (elv `speed005`);
- theta cycles: population rate of elv's theta-source units (raw-ACG label sharp or medium, from
  acg_class.py), each unit smoothed with a 20 ms Gaussian on the 10 ms grid, spline-upsampled x10;
  every local peak is one cycle and boundaries lie midway between successive peaks
  (elv ThetaPhase, peak_interp);
- internal direction: tuning curves (60 bins) fitted against tracked heading on moving cycles, each
  cycle decoded independently (no temporal prior), tuning refitted against the decoded estimate
  for a fixed number of iterations (elv IntDirEstimate uses 3).

Decoders: Poisson likelihood of each cycle's spike counts (best bin, or posterior circular mean),
and elv's population-vector correlation of each cycle's rates (best bin).

Run as a script to write one sort's cycles and decoded variants; reference_heldout.py imports the
functions to compare variants by how well they predict held-out units.

Preliminary: units pass a simple eligibility check (spike counts, refractory contamination), but
the artefact masks and duplicate removal of spec §6.1-6.3 are not applied yet.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

FS = 30000
GRID_S = 0.01
N_BINS = 60
SPEED_SIGMA_S = 0.5
MOVING_SPEED = 0.05
RATE_SIGMA_S = 0.02
N_ITER = 3
PSEUDO_S = 1.0  # Poisson decoder: pseudo-occupancy pulling tuning towards the unit's mean rate
CENTRES = (np.arange(N_BINS) + 0.5) * 2 * np.pi / N_BINS - np.pi


def load_units(sort):
    """Spike times (s) and median depth (µm) per cluster."""
    sort = Path(sort)
    st = np.load(sort / 'spike_times.npy').ravel() / FS
    clu = np.load(sort / 'spike_clusters.npy').ravel()
    y = np.load(sort / 'spike_positions.npy')[:, 1]
    ids = np.unique(clu)
    return {int(c): st[clu == c] for c in ids}, {int(c): float(np.median(y[clu == c])) for c in ids}


def eligible_units(by_cluster, duration):
    """Spec §6.4, simplified: >= 100 spikes per half, refractory contamination <= 10%."""
    out = []
    for c, t in by_cluster.items():
        t = np.sort(t)
        if min(np.sum(t < duration / 2), np.sum(t >= duration / 2)) < 100:
            continue
        if np.sum(np.diff(t) < 1.5e-3) * duration / (2 * (1.5e-3 - 0.25e-3) * t.size ** 2) <= 0.10:
            out.append(c)
    return out


def theta_cycles(by_cluster, source_units, tracking, t0, duration):
    """Cycle boundaries, durations, moving flags and tracked heading per cycle (excerpt time)."""
    grid = np.arange(0, duration, GRID_S) + GRID_S / 2
    tt = tracking['t'] - t0
    sig = int(np.ceil(SPEED_SIGMA_S / np.median(np.diff(tt))))
    xs, ys, ts = (gaussian_filter1d(v, sig, mode='nearest') for v in (tracking['x'], tracking['y'], tt))
    speed = np.interp(grid, tt, np.r_[0.0, np.hypot(np.diff(xs), np.diff(ys)) / np.diff(ts)])
    tracked = tracking['valid'] & tracking['tracked']
    track_ok = np.interp(grid, tt, tracked.astype(float)) > 0.99

    edges = np.r_[grid - GRID_S / 2, grid[-1] + GRID_S / 2]
    pop = sum(gaussian_filter1d(np.histogram(by_cluster[c], edges)[0].astype(float) / GRID_S,
                                RATE_SIGMA_S / GRID_S, mode='constant') for c in source_units)
    fine = np.linspace(grid[0], grid[-1], grid.size * 10)
    peaks = fine[find_peaks(CubicSpline(grid, pop)(fine))[0]]
    bounds = (peaks[:-1] + peaks[1:]) / 2
    n = bounds.size - 1

    gi = np.searchsorted(grid, bounds[:-1]), np.searchsorted(grid, bounds[1:])
    slow = np.r_[0, np.cumsum((speed < MOVING_SPEED) | ~track_ok)]
    moving = (slow[gi[1]] - slow[gi[0]] == 0) & (gi[1] > gi[0])

    cyc = np.searchsorted(bounds, tt) - 1
    ok = (cyc >= 0) & (cyc < n) & tracked
    z = np.zeros(n, complex)
    np.add.at(z, cyc[ok], np.exp(1j * tracking['hd_azimuth'][ok]))
    return bounds, np.diff(bounds), moving & (np.abs(z) > 0), np.angle(z)


def cycle_counts(by_cluster, units, bounds):
    return np.stack([np.histogram(by_cluster[c], bounds)[0] for c in units], axis=1).astype(float)


def tuning(angle, counts, dur, mask):
    """Spike counts and occupancy (s) per heading bin over the masked cycles: (units x bins, bins)."""
    k = np.clip(np.floor((np.angle(np.exp(1j * angle)) + np.pi) / (2 * np.pi) * N_BINS).astype(int), 0, N_BINS - 1)
    occ = np.bincount(k[mask], dur[mask], N_BINS)
    num = np.stack([np.bincount(k[mask], counts[mask, j], N_BINS) for j in range(counts.shape[1])])
    return num, occ


def decode_poisson(counts, dur, num, occ, mask, posterior_mean=False):
    rate = counts[mask].sum(0) / dur[mask].sum()
    lam = (num + PSEUDO_S * rate[:, None]) / (occ[None, :] + PSEUDO_S)
    ll = counts @ np.log(lam) - dur[:, None] * lam.sum(0)[None, :]
    if not posterior_mean:
        return CENTRES[np.argmax(ll, 1)]
    post = np.exp(ll - ll.max(1, keepdims=True))
    return np.angle(post @ np.exp(1j * CENTRES))


def decode_pv(counts, dur, num, occ, mask):
    lam = num / np.maximum(occ, 1e-9)[None, :]
    rates = counts / dur[:, None]
    a = rates - rates.mean(1, keepdims=True)
    b = lam - lam.mean(0, keepdims=True)
    corr = (a @ b) / (np.linalg.norm(a, axis=1)[:, None] * np.linalg.norm(b, axis=0)[None, :] + 1e-12)
    return CENTRES[np.argmax(corr, 1)]


DECODERS = {
    'poisson_best': lambda *a: decode_poisson(*a),
    'poisson_mean': lambda *a: decode_poisson(*a, posterior_mean=True),
    'pvcorr_best': decode_pv,
}


def internal_direction(counts, dur, mask, hd, decoder, n_iter=N_ITER):
    """Decoded heading per cycle after each iteration (tuning starts from tracked heading)."""
    angle, out = hd, []
    for _ in range(n_iter):
        angle = DECODERS[decoder](counts, dur, *tuning(angle, counts, dur, mask), mask)
        out.append(angle)
    return out


def fisher_lee(a, b):
    n = a.size
    A = np.sum(np.cos(a) * np.cos(b)); B = np.sum(np.sin(a) * np.sin(b))
    C = np.sum(np.cos(a) * np.sin(b)); D = np.sum(np.sin(a) * np.cos(b))
    E = np.sum(np.cos(2 * a)); F = np.sum(np.sin(2 * a)); G = np.sum(np.cos(2 * b)); H = np.sum(np.sin(2 * b))
    return float(4 * (A * B - C * D) / np.sqrt((n * n - E * E - F * F) * (n * n - G * G - H * H)))


def compare_to_tracking(theta, hd, mask):
    err = np.angle(np.exp(1j * (theta[mask] - hd[mask])))
    off = np.angle(np.mean(np.exp(1j * err)))
    return {'fisher_lee': fisher_lee(theta[mask], hd[mask]), 'offset_deg': float(np.degrees(off)),
            'median_abs_error_deg': float(np.degrees(np.median(np.abs(np.angle(np.exp(1j * (err - off)))))))}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--sort', required=True, help='KS4 results folder (with acg_class.npz)')
    p.add_argument('--tracking', required=True)
    p.add_argument('--t0', type=float, required=True, help="synced time of the excerpt's first sample")
    p.add_argument('--duration', type=float, required=True)
    p.add_argument('--out', required=True, help='output .npz (a .json report is written beside it)')
    args = p.parse_args()

    by_cluster, _ = load_units(args.sort)
    acg = np.load(Path(args.sort) / 'acg_class.npz')
    source = [int(c) for c, lab in zip(acg['cluster'], acg['raw_acg_label']) if lab in (1, 2)]
    bounds, dur, moving, hd = theta_cycles(by_cluster, source, np.load(args.tracking), args.t0, args.duration)
    eligible = eligible_units(by_cluster, args.duration)
    counts = cycle_counts(by_cluster, eligible, bounds)
    report = {'n_clusters': len(by_cluster), 'n_eligible': len(eligible), 'n_theta_source_units': len(source),
              'n_cycles': int(dur.size), 'n_moving_cycles': int(moving.sum()),
              'moving_time_fraction': float(dur[moving].sum() / args.duration),
              'cycle_duration_ms_median_iqr': [float(np.median(dur) * 1e3), *(np.percentile(dur, [25, 75]) * 1e3).tolist()],
              'vs_tracking': {}}
    out = {'c_start': bounds[:-1] + args.t0, 'c_stop': bounds[1:] + args.t0, 'moving': moving, 'hd_tracked': hd,
           'eligible_units': np.array(eligible), 'theta_source_units': np.array(source)}
    for name in DECODERS:
        for it, theta in enumerate(internal_direction(counts, dur, moving, hd, name), 1):
            out[f'theta_{name}_iter{it}'] = theta
            report['vs_tracking'][f'{name}_iter{it}'] = compare_to_tracking(theta, hd, moving)
    np.savez(args.out, **out)
    Path(args.out).with_suffix('.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
