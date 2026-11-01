"""Primary score S, tracked-heading score S_track and calibration C of a sort (spec §5.2-5.3, §7.2,
§7.5).

Each eligible unit gets a tuning curve against the heading (60 bins, 12° circular smoothing, shrunk
towards its mean rate by SCORE_PSEUDO_S of pseudo-occupancy) on one time half's moving cycles and is
scored on the other half: Poisson log-likelihood gain over its mean rate, bits/s. Both halves are
scored and summed. Each unit's score is null-corrected: its mean score against the heading rotated
among the scored cycles by NULL_FRACTIONS of their number (each >= ~100 s of moving time) is
subtracted, so an untuned unit scores about zero for any alpha instead of paying the
cross-validation penalty of a curve fitted to noise (spec §5.2). Rotating within the scored cycles
keeps the null's cycles and time identical to the real score's, so its penalty matches.
S uses theta_ref; S_track uses the tracked heading of each cycle (circular mean).
C decodes theta_ref from the same cross-validated tuning curves (flat prior) and averages
log2(60 x posterior at the theta_ref bin) over test cycles.

Spikes come from cleaning.load_clean_units (spec §6.1-6.3). A cycle more than 50% masked is
dropped; otherwise its duration is scaled by its unmasked fraction (§5.3). Cycles where the
reference itself is undefined (non-moving, or masked in the reference sort) are not scored.

SCORE_PSEUDO_S (alpha) was set in Phase 0 (§8.5): sensitivity across default replicates barely
depends on it, and 0.25 s keeps nearly all of it.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

import waveform
from cleaning import masked_fraction
from theta_reference import N_BINS, cycle_counts, eligible_units, tuning

SCORE_SMOOTH_BINS = 2   # 12°
SCORE_PSEUDO_S = 0.25
NULL_FRACTIONS = [k / 7 for k in range(1, 7)]


def eligible(sort, by_cluster, duration, screen=None, max_contamination=None):
    """Eligible units (§6.4: spike counts, waveform screen; since v0.9 no refractory criterion unless
    max_contamination is given, as for theta_ref) and the subset failing the stricter screen (G5)."""
    screen = waveform.screen(sort) if screen is None else screen
    units = [c for c in eligible_units(by_cluster, duration, max_contamination) if screen[c][1]]
    return units, [c for c in units if not screen[c][2]]


def _bins(theta):
    return np.clip(np.floor((np.angle(np.exp(1j * theta)) + np.pi) / (2 * np.pi) * N_BINS).astype(int), 0, N_BINS - 1)


def _fit(theta, counts, dur, train):
    num, occ = tuning(theta, counts, dur, train)
    num = gaussian_filter1d(num, SCORE_SMOOTH_BINS, axis=1, mode='wrap')
    occ = gaussian_filter1d(occ, SCORE_SMOOTH_BINS, mode='wrap')
    rate = counts[train].sum(0) / dur[train].sum()
    return (num + SCORE_PSEUDO_S * rate[:, None]) / (occ[None, :] + SCORE_PSEUDO_S), rate


def unit_scores(theta, counts, dur, moving, second_half):
    """Per-unit cross-validated log-likelihood gain (bits/s) of counts' units against theta."""
    theta = np.where(moving, theta, 0.0)
    gain = np.zeros(counts.shape[1])
    t_test = 0.0
    for h in (False, True):
        train, test = moving & (second_half == h), moving & (second_half != h)
        lam, rate = _fit(theta, counts, dur, train)
        lam_t = lam[:, _bins(theta[test])].T
        n, d = counts[test], dur[test][:, None]
        gain += np.sum(n * np.log2(lam_t / rate) - (lam_t - rate) * d / np.log(2), axis=0)
        t_test += dur[test].sum()
    return gain / t_test


def rotate(theta, mask, fraction):
    """theta with its values on the masked cycles circularly shifted by fraction of their number."""
    idx = np.where(mask)[0]
    out = theta.copy()
    out[idx] = np.roll(theta[idx], int(round(fraction * idx.size)))
    return out


def corrected_scores(theta, counts, dur, moving, second_half, fractions=NULL_FRACTIONS):
    """unit_scores minus each unit's mean score against theta rotated among the moving cycles."""
    null = sum(unit_scores(rotate(theta, moving, f), counts, dur, moving, second_half) for f in fractions)
    return unit_scores(theta, counts, dur, moving, second_half) - null / len(fractions)


def calibration(theta, counts, dur, moving, second_half):
    """C: mean over test cycles of log2(60 x posterior at theta's bin), flat prior, cross-validated."""
    theta = np.where(moving, theta, 0.0)
    total, n_cyc = 0.0, 0
    for h in (False, True):
        train, test = moving & (second_half == h), moving & (second_half != h)
        lam, _ = _fit(theta, counts, dur, train)
        ll = counts[test] @ np.log(lam) - dur[test][:, None] * lam.sum(0)[None, :]
        ll -= ll.max(1, keepdims=True)
        logpost = ll - np.log(np.exp(ll).sum(1, keepdims=True))
        total += np.sum(logpost[np.arange(test.sum()), _bins(theta[test])] + np.log(N_BINS)) / np.log(2)
        n_cyc += test.sum()
    return total / n_cyc


def score_sort(by_cluster, units, reference, t0, duration, mask, track=False):
    """S (bits/s) and per-unit scores of a cleaned sort (spike times and masked intervals in excerpt
    seconds) against a theta_ref.npz; with track=True also S_track, per-unit S_track and C."""
    bounds = reference['cycle_bounds_s'] - t0
    half = (bounds[:-1] + bounds[1:]) / 2 >= duration / 2
    frac = masked_fraction(bounds, mask)
    dur = np.diff(bounds) * (1 - frac)
    usable = reference['moving'] & (frac <= 0.5)
    valid = usable & np.isfinite(reference['theta_ref'])
    counts = cycle_counts(by_cluster, units, bounds)
    s = corrected_scores(reference['theta_ref'], counts, dur, valid, half)
    out = {'S': float(s.sum()), 'S_i': dict(zip(units, s.tolist()))}
    if track:
        hd = reference['hd_tracked']
        st = corrected_scores(hd, counts, dur, usable & np.isfinite(hd), half)
        out.update(S_track=float(st.sum()), S_track_i=dict(zip(units, st.tolist())),
                   C=float(calibration(reference['theta_ref'], counts, dur, valid, half)))
    return out
