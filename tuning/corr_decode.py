"""Correlation-based (population-vector) decoding of tracked head direction and 2D position from sorts
(reported, not a criterion; spec §14).

Same units, tuning maps, time halves and windows as bayes_decode.py. Each decoded window's population
vector of spike counts is compared with every visited bin's expected population vector (the training
half's maps) by Pearson correlation across units; the estimate is the bin with the highest correlation.
Unlike the Poisson decoder, evidence is not multiplied across units as if they were independent, so a
unit that repeats another's spikes adds a vote rather than a factor of likelihood. Two variants:

- 'raw': counts and maps as they are, so units weigh in by their rates;
- 'z': each unit z-scored first (its counts by their mean and SD over the half's decoded windows, its
  map by its mean and SD over the visited bins), so every unit weighs the same.

A window whose population vector has no variance across units (no spikes) is decoded to a random bin.
Reported per sort and variant: median error at each window length of bayes_decode.WINDOWS_S, and at
MAIN_WINDOW_S the median error with N randomly drawn units, as in bayes_decode.py.

    python corr_decode.py --group default <sort> ... --group ks25 <sort> ... --group H8 <sort> ...
        --tracking <npz> --raw-artefacts <npz> --t0 <s> --duration 1200 [--hybrid-truth <npz>] --out <json>
"""
import argparse
import json
from pathlib import Path

import numpy as np

import bayes_decode as bd
from cleaning import mask_intervals
from position_score import POS_BIN_M
from reproducibility import eligible_units

VARIANTS = ['raw', 'z']


def _standardise(a, axis):
    """Centre and scale along `axis`; constant rows or columns become 0."""
    c = a - a.mean(axis, keepdims=True)
    sd = c.std(axis, keepdims=True)
    return np.divide(c, sd, out=np.zeros_like(c), where=sd > 0)


def errors(var, beh, n, rate, visited, true, variant, rng):
    """Errors (degrees or cm) of the best-correlated bin for each window."""
    X, F = n, rate[:, visited]
    if variant == 'z':
        X, F = _standardise(X, 0), _standardise(F, 1)
    Xc = X - X.mean(1, keepdims=True)
    Fc = F - F.mean(0, keepdims=True)
    C = Xc @ (Fc / np.maximum(np.linalg.norm(Fc, axis=0, keepdims=True), 1e-12))   # each window's norm is the same for every bin
    k = np.argmax(C, 1)
    flat = ~(np.abs(Xc).sum(1) > 0)
    k[flat] = rng.integers(0, F.shape[1], flat.sum())
    b = np.flatnonzero(visited)[k]
    if var == 'hd':
        est = (b + 0.5) * 2 * np.pi / bd.HD_BINS
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
    beh = bd.behaviour(tracking, t0, duration, mask)
    nu = len(ids)
    halves = [(~beh['second'], beh['second']), (beh['second'], ~beh['second'])]
    fits = {v: [bd.fit(v, beh, T, U, nu, tr) for tr, _ in halves] for v in ('hd', 'pos')}
    out = {'sort': str(sort), 'n_units': nu, 'median_error': {m: {'hd': {}, 'pos': {}} for m in VARIANTS},
           'curve': {m: {'hd': {}, 'pos': {}} for m in VARIANTS}}
    run = lambda v, m, data, s: float(np.median(np.concatenate(
        [errors(v, beh, n[:, s], fits[v][h][0][s], fits[v][h][1], hd if v == 'hd' else xy, m, rng)
         for h, (n, tau, hd, xy) in enumerate(data)])))
    for w in bd.WINDOWS_S:
        data = [bd.windows(beh, T, U, nu, w, duration, mask, te) for _, te in halves]
        for m in VARIANTS:
            for v in ('hd', 'pos'):
                out['median_error'][m][v][str(w)] = run(v, m, data, np.arange(nu))
        if w == bd.MAIN_WINDOW_S:
            for N in [k for k in bd.N_UNITS if k < nu]:
                draws = [rng.choice(nu, N, replace=False) for _ in range(bd.DRAWS)]
                for m in VARIANTS:
                    for v in ('hd', 'pos'):
                        out['curve'][m][v][str(N)] = [run(v, m, data, s) for s in draws]
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
    res = {'windows_s': bd.WINDOWS_S, 'main_window_s': bd.MAIN_WINDOW_S, 'variants': VARIANTS, 'groups': {}}
    for label, *sorts in args.group:
        res['groups'][label] = []
        for s in sorts:
            r = decode_sort(s, tracking, args.raw_artefacts, args.t0, args.duration, truth, rng)
            res['groups'][label].append(r)
            name = Path(s).name if Path(s).name != 'sort' else Path(s).parent.name
            for m in VARIANTS:
                me = r['median_error'][m]
                print(f"{label:8s} {name:28s} {m:3s} units {r['n_units']:5d}  direction "
                      + ' '.join(f"{me['hd'][str(w)]:5.1f}" for w in bd.WINDOWS_S) + ' deg   position '
                      + ' '.join(f"{me['pos'][str(w)]:5.1f}" for w in bd.WINDOWS_S) + ' cm', flush=True)
        Path(args.out).write_text(json.dumps(res, indent=1))
    print('mean median error (direction deg / position cm) at windows', bd.WINDOWS_S)
    for m in VARIANTS:
        for label, rs in res['groups'].items():
            e = {v: [np.mean([r['median_error'][m][v][str(w)] for r in rs]) for w in bd.WINDOWS_S] for v in ('hd', 'pos')}
            print(f"{m:3s} {label:8s} " + ' '.join(f'{x:5.1f}' for x in e['hd']) + '   ' + ' '.join(f'{x:5.1f}' for x in e['pos']))
