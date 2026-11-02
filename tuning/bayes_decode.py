"""Bayesian decoding of tracked head direction and 2D position from sorts (reported, not a criterion;
spec §14).

Units: each sort's eligible units (§6.4) after cleaning (§6.1-6.3), hybrid copies excluded on the hybrid
excerpt, as reproducibility.py takes them. Usable tracking frames are those of Appendix A (valid
tracking, speed >= 5 cm/s with position smoothed 0.5 s); a frame more than 50% masked (§6.1-6.2) is
dropped, otherwise its duration is scaled by its unmasked fraction.

Tuning, fitted on one time half's usable frames: head direction in 60 bins of 6° (circular Gaussian
smoothing 12°, as S, §5.2) and position in 5 cm bins (Gaussian smoothing 5 cm, as Appendix A). Spike
counts and occupancy are smoothed alike, and each map is shrunk towards the unit's mean rate by
ALPHA_S seconds of pseudo-occupancy (§5.2's alpha), which also keeps log rates finite.

Decoding (Zhang et al. 1998): the other half is cut into non-overlapping windows; a window is decoded if
all its frames are usable, with its duration scaled by its unmasked fraction. Poisson likelihood of the
window's spike counts under independent units, flat prior over the bins visited in the training half;
the estimate is the posterior's maximum. Then the halves swap. Error: absolute angle between decoded
and tracked head direction (circular mean over the window's frames), and distance between decoded and
tracked position (mean over the frames).

Reported per sort: median error over the decoded windows of both halves, for each window length in
WINDOWS_S; and, at MAIN_WINDOW_S, the median error with N randomly drawn units (N_UNITS, DRAWS draws
each), which shows how close each sorter is to saturation and compares sorters at equal unit counts.

    python bayes_decode.py --group default <sort> ... --group ks25 <sort> ... --group H8 <sort> ...
        --tracking <npz> --raw-artefacts <npz> --t0 <s> --duration 1200 [--hybrid-truth <npz>] --out <json>
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from cleaning import mask_intervals, masked_fraction
from position_score import POS_BIN_M, covariates
from reproducibility import eligible_units
from score import SCORE_PSEUDO_S

ALPHA_S = SCORE_PSEUDO_S          # 0.25 s
HD_BINS = 60                      # 6°
HD_SIGMA_BINS = 2.0               # 12°
POS_SIGMA_BINS = 1.0              # 5 cm
WINDOWS_S = [0.02, 0.1, 0.5]
MAIN_WINDOW_S = 0.1
N_UNITS = [25, 50, 100, 200, 400, 800, 1600]
DRAWS = 10


def behaviour(tracking, t0, duration, mask):
    """Usable frames with their edges, durations, halves, bins and tracked values; the position grid."""
    cov = covariates(tracking, t0, duration)
    tt = tracking['t'] - t0
    inside = (tt >= 0) & (tt < duration)
    x, y, hd = (tracking[k][inside] for k in ('x', 'y', 'hd_azimuth'))
    edges = cov['edges']
    frac = masked_fraction(edges, mask)
    ok = cov['ok'] & (frac <= 0.5)
    lo = np.array([x[cov['ok']].min(), y[cov['ok']].min()])
    ih = np.clip(np.floor(np.mod(hd, 2 * np.pi) / (2 * np.pi) * HD_BINS).astype(int), 0, HD_BINS - 1)
    return {'edges': edges, 'd': np.diff(edges) * (1 - frac), 'ok': ok, 'second': cov['second_half'],
            'bin': {'hd': ih, 'pos': cov['pos']}, 'n_bins': {'hd': HD_BINS, 'pos': int(np.prod(cov['shape']))},
            'shape': cov['shape'], 'lo': lo, 'x': x, 'y': y, 'hd': hd}


def smooth(var, a, shape):
    """Smooth the last axis of a (units x bins, or bins) as a direction or position map."""
    if var == 'hd':
        return gaussian_filter1d(a, HD_SIGMA_BINS, axis=-1, mode='wrap')
    s = a.reshape(*a.shape[:-1], *shape)
    sig = (0,) * (a.ndim - 1) + (POS_SIGMA_BINS, POS_SIGMA_BINS)
    return gaussian_filter(s, sig, mode='constant').reshape(a.shape)


def fit(var, beh, T, U, n_units, train):
    """Rate maps (units x bins) from the usable frames of `train`, and the bins visited."""
    B = beh['n_bins'][var]
    b, use = beh['bin'][var], beh['ok'] & train
    occ = np.bincount(b[use], beh['d'][use], B)
    f = np.searchsorted(beh['edges'], T, 'right') - 1
    keep = (f >= 0) & (f < b.size)
    keep[keep] = use[f[keep]]
    C = np.bincount(U[keep] * B + b[f[keep]], minlength=n_units * B).reshape(n_units, B).astype(float)
    mean_rate = C.sum(1) / occ.sum()
    rate = (smooth(var, C, beh['shape']) + ALPHA_S * mean_rate[:, None]) / (smooth(var, occ, beh['shape']) + ALPHA_S)
    return np.maximum(rate, 1e-9), occ > 0


def windows(beh, T, U, n_units, w, duration, mask, test):
    """Spike counts (windows x units), durations and tracked values of the usable windows of `test`."""
    K = int(round(duration / w))
    centre = (beh['edges'][:-1] + beh['edges'][1:]) / 2
    k = np.minimum((centre // w).astype(int), K - 1)
    n_frames = np.bincount(k, minlength=K)
    usable = (n_frames > 0) & (np.bincount(k, ~beh['ok'], K) == 0) & np.bincount(k, test, K).astype(bool)
    idx = np.full(K, -1)
    idx[usable] = np.arange(usable.sum())
    kw = np.minimum((T // w).astype(int), K - 1)
    sel = idx[kw] >= 0
    n = np.bincount(idx[kw[sel]] * n_units + U[sel], minlength=usable.sum() * n_units).reshape(-1, n_units)
    bounds = np.arange(K + 1) * w
    tau = w * (1 - masked_fraction(bounds, mask))[usable]
    hd = np.angle(np.bincount(k, np.cos(beh['hd']), K) + 1j * np.bincount(k, np.sin(beh['hd']), K))[usable]
    xy = np.stack([np.bincount(k, beh[c], K) / np.maximum(n_frames, 1) for c in ('x', 'y')], 1)[usable]
    return n.astype(float), tau, hd, xy


def decode(var, beh, n, tau, rate, visited, true):
    """Errors (degrees or cm) of the posterior maximum for each window."""
    R = rate[:, visited]
    L = n @ np.log(R) - tau[:, None] * R.sum(0)[None, :]
    b = np.flatnonzero(visited)[np.argmax(L, 1)]
    if var == 'hd':
        est = (b + 0.5) * 2 * np.pi / HD_BINS
        return np.degrees(np.abs(np.angle(np.exp(1j * (est - true)))))
    ny = beh['shape'][1]
    est = beh['lo'] + (np.stack([b // ny, b % ny], 1) + 0.5) * POS_BIN_M
    return 100 * np.hypot(*(est - true).T)


def decode_sort(sort, tracking, raw_artefacts, t0, duration, truth, rng):
    units, _ = eligible_units(sort, raw_artefacts, duration, truth)
    ids = sorted(units)
    T = np.concatenate([units[c] for c in ids])
    U = np.repeat(np.arange(len(ids)), [units[c].size for c in ids])
    mask = mask_intervals(sort, raw_artefacts)
    beh = behaviour(tracking, t0, duration, mask)
    nu = len(ids)
    halves = [(~beh['second'], beh['second']), (beh['second'], ~beh['second'])]
    fits = {v: [fit(v, beh, T, U, nu, tr) for tr, _ in halves] for v in ('hd', 'pos')}
    out = {'sort': str(sort), 'n_units': nu, 'median_error': {'hd': {}, 'pos': {}}, 'n_windows': {},
           'curve': {'hd': {}, 'pos': {}}}
    for w in WINDOWS_S:
        data = [windows(beh, T, U, nu, w, duration, mask, te) for _, te in halves]
        out['n_windows'][str(w)] = int(sum(d[0].shape[0] for d in data))
        for v in ('hd', 'pos'):
            err = np.concatenate([decode(v, beh, n, tau, *fits[v][h], hd if v == 'hd' else xy)
                                  for h, (n, tau, hd, xy) in enumerate(data)])
            out['median_error'][v][str(w)] = float(np.median(err))
        if w == MAIN_WINDOW_S:
            for N in [m for m in N_UNITS if m < nu]:
                draws = [rng.choice(nu, N, replace=False) for _ in range(DRAWS)]
                for v in ('hd', 'pos'):
                    meds = []
                    for s in draws:
                        err = np.concatenate([decode(v, beh, n[:, s], tau, fits[v][h][0][s], fits[v][h][1],
                                                     hd if v == 'hd' else xy)
                                              for h, (n, tau, hd, xy) in enumerate(data)])
                        meds.append(float(np.median(err)))
                    out['curve'][v][str(N)] = meds
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--group', nargs='+', action='append', required=True, metavar=('LABEL', 'SORT'),
                   help='a label followed by its sort folders')
    p.add_argument('--tracking', required=True)
    p.add_argument('--raw-artefacts', required=True)
    p.add_argument('--t0', type=float, required=True)
    p.add_argument('--duration', type=float, required=True)
    p.add_argument('--hybrid-truth')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    tracking = dict(np.load(args.tracking))
    truth = np.load(args.hybrid_truth) if args.hybrid_truth else None
    rng = np.random.default_rng(0)
    res = {'windows_s': WINDOWS_S, 'main_window_s': MAIN_WINDOW_S, 'alpha_s': ALPHA_S, 'groups': {}}
    for label, *sorts in args.group:
        res['groups'][label] = []
        for s in sorts:
            r = decode_sort(s, tracking, args.raw_artefacts, args.t0, args.duration, truth, rng)
            res['groups'][label].append(r)
            me = r['median_error']
            print(f"{label:8s} {Path(s).name if Path(s).name != 'sort' else Path(s).parent.name:28s} "
                  f"units {r['n_units']:5d}  direction " + ' '.join(f"{me['hd'][str(w)]:5.1f}" for w in WINDOWS_S)
                  + ' deg   position ' + ' '.join(f"{me['pos'][str(w)]:5.1f}" for w in WINDOWS_S) + ' cm', flush=True)
        Path(args.out).write_text(json.dumps(res, indent=1))
    print('mean median error (direction deg / position cm) at windows', WINDOWS_S)
    for label, rs in res['groups'].items():
        m = {v: [np.mean([r['median_error'][v][str(w)] for r in rs]) for w in WINDOWS_S] for v in ('hd', 'pos')}
        print(f"{label:8s} " + ' '.join(f'{x:5.1f}' for x in m['hd']) + '   ' + ' '.join(f'{x:5.1f}' for x in m['pos']))
