"""Tracked-position score for the transfer test (spec §10, Appendix A).

Per unit, lambda_i(z) = r_i * P_i(x) * H_i(theta) * V_i(v) on the tracking frames (120 Hz):
position in 5 cm bins (Gaussian smoothing sigma = 5 cm), head direction in 10° bins (sigma = 15°,
circular), speed in 5 cm/s bins over 5-50 cm/s (sigma = 5 cm/s; faster frames go in the top bin).
The factors are fitted on one time half by iterative proportional fitting, each shrunk towards 1 by
alpha seconds of pseudo-occupancy at the unit's mean rate, and the unit is scored on the other half:
Poisson log-likelihood gain over its training mean rate, bits/s. Both halves are scored and summed.

Frames count if tracking is valid and speed (position smoothed with a 0.5 s Gaussian, as for
theta_ref, §5.1) is >= 5 cm/s. Masks (§6.1-6.2) are missing data, as in score.py: a frame more than
50% masked is dropped, otherwise its duration is scaled by its unmasked fraction.

Each unit's score is null-corrected as in §5.2: its mean score with the behavioural covariates
rotated among the scored frames by NULL_FRACTIONS of their number is subtracted, so an untuned unit
scores about zero instead of paying the cross-validation penalty of a model fitted to noise.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from cleaning import masked_fraction
from score import NULL_FRACTIONS

POS_BIN_M = 0.05
POS_SIGMA_BINS = 1.0
HD_BINS = 36
HD_SIGMA_BINS = 1.5
SPEED_EDGES = np.arange(0.05, 0.501, 0.05)
SPEED_SIGMA_BINS = 1.0
MIN_SPEED = 0.05
SPEED_SIGMA_S = 0.5
IPF_ITER = 10


def covariates(tracking, t0, duration):
    """Frame edges (excerpt s), bin indices of position, heading and speed, and the usable frames."""
    tt = tracking['t'] - t0
    inside = (tt >= 0) & (tt < duration)
    sig = SPEED_SIGMA_S / np.median(np.diff(tt))
    xs, ys = (gaussian_filter1d(v, sig, mode='nearest') for v in (tracking['x'], tracking['y']))
    speed = np.r_[0.0, np.hypot(np.diff(xs), np.diff(ys)) / np.diff(tt)]
    ok = tracking['valid'] & tracking['tracked'] & (speed >= MIN_SPEED) & inside
    ok &= np.isfinite(tracking['x']) & np.isfinite(tracking['y']) & np.isfinite(tracking['hd_azimuth'])
    tt, x, y, hd, speed, ok = (a[inside] for a in (tt, tracking['x'], tracking['y'], tracking['hd_azimuth'], speed, ok))
    mid = (tt[:-1] + tt[1:]) / 2
    edges = np.r_[max(tt[0] - (mid[0] - tt[0]), 0), mid, min(tt[-1] + (tt[-1] - mid[-1]), duration)]
    lo = np.array([x[ok].min(), y[ok].min()])
    shape = tuple(np.floor((np.array([x[ok].max(), y[ok].max()]) - lo) / POS_BIN_M).astype(int) + 1)
    ix = np.clip(np.floor((x - lo[0]) / POS_BIN_M).astype(int), 0, shape[0] - 1)
    iy = np.clip(np.floor((y - lo[1]) / POS_BIN_M).astype(int), 0, shape[1] - 1)
    ih = np.clip(np.floor(np.mod(hd, 2 * np.pi) / (2 * np.pi) * HD_BINS).astype(int), 0, HD_BINS - 1)
    iv = np.clip(np.searchsorted(SPEED_EDGES, speed, 'right') - 1, 0, SPEED_EDGES.size - 2)
    return {'edges': edges, 'pos': np.where(ok, ix * shape[1] + iy, 0), 'shape': shape, 'hd': ih,
            'speed': iv, 'ok': ok, 'second_half': tt >= duration / 2}


def _smooth(kind, a, shape):
    if kind == 'pos':
        return gaussian_filter(a.reshape(shape), POS_SIGMA_BINS, mode='constant').ravel()
    if kind == 'hd':
        return gaussian_filter1d(a, HD_SIGMA_BINS, mode='wrap')
    return gaussian_filter1d(a, SPEED_SIGMA_BINS, mode='nearest')


def fit(cov, n, d, train, alpha):
    """Factors {'pos','hd','speed'} and rate r of the multiplicative model, fitted on frames `train`."""
    sizes = {'pos': int(np.prod(cov['shape'])), 'hd': HD_BINS, 'speed': SPEED_EDGES.size - 1}
    idx = {k: cov[k][train] for k in sizes}
    nt, dt = n[train], d[train]
    rate = nt.sum() / dt.sum()
    F = {k: np.ones(s) for k, s in sizes.items()}
    for _ in range(IPF_ITER):
        for k in sizes:
            others = np.prod([F[j][idx[j]] for j in sizes if j != k], axis=0)
            num = _smooth(k, np.bincount(idx[k], nt, sizes[k]), cov['shape'])
            den = _smooth(k, np.bincount(idx[k], rate * others * dt, sizes[k]), cov['shape'])
            F[k] = (num + alpha * rate) / (den + alpha * rate)
    pred = rate * np.prod([F[k][idx[k]] for k in sizes], axis=0)
    return F, rate * nt.sum() / np.sum(pred * dt), rate


def _gain(F, r, mean_rate, cov, n, d, test, shift=0):
    frames = np.flatnonzero(test)
    src = np.roll(frames, shift) if shift else frames
    lam = r * np.prod([F[k][cov[k][src]] for k in F], axis=0)
    return float(np.sum(n[frames] * np.log2(lam / mean_rate) - (lam - mean_rate) * d[frames] / np.log(2)))


def score_sort(by_cluster, units, tracking, t0, duration, mask, alpha, null=True):
    """S (bits/s) and per-unit S_i; S is the sum over units, both halves scored, test-time weighted."""
    cov = covariates(tracking, t0, duration)
    frac = masked_fraction(cov['edges'], mask)
    d = np.diff(cov['edges']) * (1 - frac)
    usable = cov['ok'] & (frac <= 0.5) & (d > 0)
    halves = [(usable & ~cov['second_half'], usable & cov['second_half']),
              (usable & cov['second_half'], usable & ~cov['second_half'])]
    T = sum(d[test].sum() for _, test in halves)
    S_i = {}
    for c in units:
        n = np.histogram(by_cluster[c], cov['edges'])[0].astype(float)
        total = 0.0
        for train, test in halves:
            F, r, mean_rate = fit(cov, n, d, train, alpha)
            g = _gain(F, r, mean_rate, cov, n, d, test)
            if null:
                m = int(test.sum())
                g -= np.mean([_gain(F, r, mean_rate, cov, n, d, test, int(round(f * m))) for f in NULL_FRACTIONS])
            total += g
        S_i[c] = total / T
    return {'S': float(sum(S_i.values())), 'S_i': S_i, 'scored_time_s': float(T)}
