"""Infer the reference heading theta_ref from a sort with a ring hidden Markov model (spec §5.1).

Preliminary milestone version: units pass a simple eligibility check (spike counts and
refractory contamination, spec §6.4) but the artefact masks and duplicate removal of
§6.1-6.3 are not applied yet.

Model: 72 ring states, 125 ms bins, wrapped-Gaussian random-walk transitions (SD fitted),
Poisson emissions with circularly smoothed per-unit tuning curves. Initialised from tracked
head direction, then EM to convergence (relative log-likelihood change < 1e-6; the total
log-likelihood is ~1e6 nats, so the spec's 1e-4 stopped EM after one step). Tracking is used only for that initialisation and
for the validation statistics.
"""
import argparse
import json
from pathlib import Path

import numpy as np

FS = 30000
K = 72
BIN_S = 0.125
SMOOTH_DEG = 15.0
PSEUDO_S = 1.0          # pseudo-occupancy (s) pulling tuning curves towards the unit's mean rate
SD_GRID_DEG = np.array([1, 2, 3, 5, 7, 10, 15, 20, 30, 45, 60, 90], float)

p = argparse.ArgumentParser()
p.add_argument('--sort', required=True, help='KS4 results folder')
p.add_argument('--tracking', required=True, help='tracking .npz from extract_tracking.py')
p.add_argument('--t0', type=float, required=True, help='synced time of the excerpt\'s first sample')
p.add_argument('--duration', type=float, required=True)
p.add_argument('--out', required=True, help='output .npz (a .json report is written beside it)')
p.add_argument('--sd-deg', type=float, default=None, help='fix the transition SD instead of fitting it')
p.add_argument('--min-mvl', type=float, default=0.0, help='keep only units this tuned to tracked heading')
p.add_argument('--max-iter', type=int, default=100)
args = p.parse_args()

# ---- units and eligibility ----
st = np.load(Path(args.sort) / 'spike_times.npy').ravel() / FS
clu = np.load(Path(args.sort) / 'spike_clusters.npy').ravel()
half = args.duration / 2
n_bins = int(round(args.duration / BIN_S))
eligible, contamination = [], {}
for c in np.unique(clu):
    t = np.sort(st[clu == c])
    if min(np.sum(t < half), np.sum(t >= half)) < 100:
        continue
    # Hill et al. (2011) contamination estimate, 1.5 ms refractory, 0.25 ms censored
    n_viol = np.sum(np.diff(t) < 1.5e-3)
    fp = n_viol * args.duration / (2 * (1.5e-3 - 0.25e-3) * t.size ** 2)
    contamination[int(c)] = float(fp)
    if fp <= 0.10:
        eligible.append(int(c))
eligible = np.array(eligible)
if eligible.size == 0:
    raise SystemExit('No eligible units')
counts = np.zeros((n_bins, eligible.size))
for j, c in enumerate(eligible):
    b = np.floor(st[clu == c] / BIN_S).astype(int)
    counts[:, j] = np.bincount(b[b < n_bins], minlength=n_bins)

# ---- tracked heading per bin (circular mean of valid samples) ----
tr = np.load(args.tracking)
ok = tr['valid'] & tr['tracked'] & np.isfinite(tr['hd_azimuth'])
tb = np.floor((tr['t'][ok] - args.t0) / BIN_S).astype(int)
keep = (tb >= 0) & (tb < n_bins)
hd = tr['hd_azimuth'][ok][keep]
z = np.bincount(tb[keep], np.cos(hd), n_bins) + 1j * np.bincount(tb[keep], np.sin(hd), n_bins)
hd_valid = np.bincount(tb[keep], minlength=n_bins) > 0
hd_bin = np.angle(z)
centres = (np.arange(K) + 0.5) * 2 * np.pi / K - np.pi


def circ_smooth(x):
    """Circularly smooth along the last axis with a Gaussian of SMOOTH_DEG."""
    d = np.minimum(np.arange(K), K - np.arange(K)) * 360 / K
    kern = np.exp(-0.5 * (d / SMOOTH_DEG) ** 2)
    return np.real(np.fft.ifft(np.fft.fft(x, axis=-1) * np.fft.fft(kern / kern.sum()), axis=-1))


def transition(sd_deg):
    d = (np.arange(K)[None, :] - np.arange(K)[:, None]) * 360 / K
    P = sum(np.exp(-0.5 * ((d + 360 * m) / sd_deg) ** 2) for m in (-1, 0, 1))
    return P / P.sum(1, keepdims=True)


def m_step(gamma):
    num = circ_smooth(counts.T @ gamma)                    # units x K
    den = circ_smooth(gamma.sum(0) * BIN_S)                # K
    rate = counts.sum(0) / (n_bins * BIN_S)
    return (num + PSEUDO_S * rate[:, None]) / (den[None, :] + PSEUDO_S)


def e_step(lam, P):
    loglik = counts @ np.log(lam * BIN_S) - BIN_S * lam.sum(0)[None, :]   # bins x K
    mx = loglik.max(1, keepdims=True)
    e = np.exp(loglik - mx)
    alpha = np.empty((n_bins, K)); c = np.empty(n_bins)
    a = e[0] / K
    for t in range(n_bins):
        if t:
            a = (alpha[t - 1] @ P) * e[t]
        c[t] = a.sum(); alpha[t] = a / c[t]
    beta = np.ones((n_bins, K))
    for t in range(n_bins - 2, -1, -1):
        b = P @ (e[t + 1] * beta[t + 1])
        beta[t] = b / c[t + 1]
    gamma = alpha * beta
    gamma /= gamma.sum(1, keepdims=True)
    xi = np.zeros((K, K))
    for t in range(n_bins - 1):
        xi += np.outer(alpha[t], e[t + 1] * beta[t + 1]) / c[t + 1]
    xi *= P
    return gamma, xi, float(np.log(c).sum() + mx.sum())


def circ_corr(a, b):
    """Fisher-Lee circular correlation. Unlike Jammalamadaka-SenGupta it needs no mean direction,
    which is undefined for the near-uniform heading of open-field foraging."""
    n = a.size
    A = np.sum(np.cos(a) * np.cos(b)); B = np.sum(np.sin(a) * np.sin(b))
    C = np.sum(np.cos(a) * np.sin(b)); D = np.sum(np.sin(a) * np.cos(b))
    E = np.sum(np.cos(2 * a)); F = np.sum(np.sin(2 * a)); G = np.sum(np.cos(2 * b)); H = np.sum(np.sin(2 * b))
    return float(4 * (A * B - C * D) / np.sqrt((n * n - E * E - F * F) * (n * n - G * G - H * H)))


def compare(theta):
    v = hd_valid
    diff = np.angle(np.exp(1j * (theta[v] - hd_bin[v])))
    return {'circ_corr': circ_corr(theta[v], hd_bin[v]),
            'offset_deg': float(np.degrees(np.angle(np.mean(np.exp(1j * diff))))),
            'median_abs_error_deg': float(np.degrees(np.median(np.abs(np.angle(np.exp(1j * (diff - np.angle(np.mean(np.exp(1j * diff)))))))))),
            'n_bins': int(v.sum())}


# ---- initialise from tracked heading, then EM ----
gamma0 = np.full((n_bins, K), 1 / K)
k_hd = np.clip(np.floor((hd_bin + np.pi) / (2 * np.pi) * K).astype(int), 0, K - 1)
gamma0[hd_valid] = 0
gamma0[np.where(hd_valid)[0], k_hd[hd_valid]] = 1
lam_track = m_step(gamma0)
mvl = np.abs((lam_track * np.exp(1j * centres)).sum(1)) / lam_track.sum(1)
n_tuned_all = int(np.sum(mvl >= 0.2))
if args.min_mvl > 0:
    sel = mvl >= args.min_mvl
    counts, eligible, mvl = counts[:, sel], eligible[sel], mvl[sel]
lam = m_step(gamma0)
sd = args.sd_deg or 10.0
history = []
for it in range(args.max_iter):
    gamma, xi, ll = e_step(lam, transition(sd))
    if it == 0:
        theta_first = np.angle(gamma @ np.exp(1j * centres))
    history.append({'iter': it, 'loglik': ll, 'sd_deg': sd})
    if it and abs(ll - history[-2]['loglik']) < 1e-6 * abs(ll):
        break
    lam = m_step(gamma)
    if args.sd_deg is None:
        sd = float(SD_GRID_DEG[np.argmax([np.sum(xi * np.log(transition(s) + 1e-300)) for s in SD_GRID_DEG])])
theta = np.angle(gamma @ np.exp(1j * centres))
resultant = np.abs(gamma @ np.exp(1j * centres))

report = {
    'sort': str(args.sort),
    'options': {'sd_deg': args.sd_deg, 'min_mvl': args.min_mvl, 'max_iter': args.max_iter},
    'n_clusters': int(np.unique(clu).size),
    'n_eligible': int(np.sum([fp <= 0.10 for fp in contamination.values()])),
    'n_units_mvl_ge_0.2_vs_tracking': n_tuned_all,
    'n_units_in_model': int(eligible.size),
    'em_iterations': len(history),
    'final_transition_sd_deg': sd,
    'median_posterior_resultant': float(np.median(resultant)),
    'vs_tracking_after_first_e_step': compare(theta_first),
    'vs_tracking_final': compare(theta),
    'history': history,
}
np.savez(args.out, theta_ref=theta, resultant=resultant, bin_s=BIN_S, t0=args.t0,
         eligible_units=eligible, tuning=lam, mvl_vs_tracking=mvl, hd_tracked=hd_bin, hd_valid=hd_valid)
Path(args.out).with_suffix('.json').write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k != 'history'}, indent=2))
