"""Primary score S of a sort against a frozen reference heading (spec §5.2-5.3, prototype).

Each eligible unit gets a tuning curve against theta_ref (60 bins, 12° circular smoothing, shrunk
towards its mean rate by SCORE_PSEUDO_S of pseudo-occupancy) on one time half's moving cycles and is
scored on the other half: Poisson log-likelihood gain over its mean rate, bits/s. Both halves are
scored and summed.

Spikes come from cleaning.load_clean_units (spec §6.1-6.3). A cycle more than 50% masked is
dropped; otherwise its duration is scaled by its unmasked fraction (§5.3). Cycles where the
reference itself is undefined (non-moving, or masked in the reference sort) are not scored.

Preliminary: alpha is a placeholder (spec §5.2 sets it in Phase 0).
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

from cleaning import masked_fraction
from theta_reference import N_BINS, cycle_counts, eligible_units, tuning

SCORE_SMOOTH_BINS = 2   # 12°
SCORE_PSEUDO_S = 1.0


def unit_scores(theta, counts, dur, moving, second_half):
    """Per-unit cross-validated log-likelihood gain (bits/s) of counts' units against theta."""
    theta = np.where(moving, theta, 0.0)
    gain = np.zeros(counts.shape[1])
    t_test = 0.0
    for h in (False, True):
        train, test = moving & (second_half == h), moving & (second_half != h)
        num, occ = tuning(theta, counts, dur, train)
        num = gaussian_filter1d(num, SCORE_SMOOTH_BINS, axis=1, mode='wrap')
        occ = gaussian_filter1d(occ, SCORE_SMOOTH_BINS, mode='wrap')
        rate = counts[train].sum(0) / dur[train].sum()
        lam = (num + SCORE_PSEUDO_S * rate[:, None]) / (occ[None, :] + SCORE_PSEUDO_S)
        k = np.clip(np.floor((np.angle(np.exp(1j * theta[test])) + np.pi) / (2 * np.pi) * N_BINS).astype(int), 0, N_BINS - 1)
        lam_t = lam[:, k].T
        n, d = counts[test], dur[test][:, None]
        gain += np.sum(n * np.log2(lam_t / rate) - (lam_t - rate) * d / np.log(2), axis=0)
        t_test += dur[test].sum()
    return gain / t_test


def score_sort(by_cluster, reference, t0, duration, mask):
    """S (bits/s) and per-unit scores of a cleaned sort (spike times in excerpt seconds, masked
    intervals in excerpt seconds) against a theta_ref.npz."""
    bounds = reference['cycle_bounds_s'] - t0
    mid = (bounds[:-1] + bounds[1:]) / 2
    frac = masked_fraction(bounds, mask)
    valid = reference['moving'] & np.isfinite(reference['theta_ref']) & (frac <= 0.5)
    units = eligible_units(by_cluster, duration)
    s = unit_scores(reference['theta_ref'], cycle_counts(by_cluster, units, bounds), np.diff(bounds) * (1 - frac),
                    valid, mid >= duration / 2)
    return float(s.sum()), dict(zip(units, s))
