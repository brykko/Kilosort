"""Population decoding as an alternative objective (tested in round 3, not adopted; spec §12).

A discriminative population score: cross-validated von Mises regression from the population's
per-cycle rates to theta_ref, scored by the held-out log-likelihood of theta_ref (bits relative to a
uniform guess). Same cycles, halves, masks and eligibility as S. Drivers: popdecode_test.py (theta_ref)
and popdecode_track.py (tracked heading).
"""
import json
from pathlib import Path

import numpy as np
import torch

import hybrid
import score
import waveform
from cleaning import load_clean_units, masked_fraction
from theta_reference import CENTRES, cycle_counts, tuning

BASE = Path(r'C:\data\ks4_tuning\26648_kobe_2020-08-21')
REF = np.load(BASE / 'runs/reference_seed1000/theta_ref.npz')
RAW = BASE / 'search/raw_artefacts_car.npz'
TRUTH = np.load(BASE / 'search/hybrid/hybrid_truth.npz')
T0, DUR = 1668.0, 1200.0
torch.set_num_threads(8)
LN2 = np.log(2)


def load(sort):
    by, _, mask = load_clean_units(sort, RAW)
    copies = set(hybrid.recovery(sort, TRUTH)['hybrid_copies'])
    by = {c: t for c, t in by.items() if c not in copies}
    units = score.eligible(sort, by, DUR, waveform.screen(sort))
    return by, units, mask


def design(by, units, mask):
    bounds = REF['cycle_bounds_s'] - T0
    half = (bounds[:-1] + bounds[1:]) / 2 >= DUR / 2
    frac = masked_fraction(bounds, mask)
    dur = np.diff(bounds) * (1 - frac)
    valid = REF['moving'] & (frac <= 0.5) & np.isfinite(REF['theta_ref'])
    X = cycle_counts(by, units, bounds)[valid] / dur[valid, None]
    return X, REF['theta_ref'][valid], half[valid], dur[valid]


def fit(X, th, lam, iters=200):
    Xt, tht = torch.from_numpy(X).float(), torch.from_numpy(th).float()
    W = torch.zeros(X.shape[1], 2, requires_grad=True)
    b = torch.zeros(2, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=iters, line_search_fn='strong_wolfe')
    c, s = torch.cos(tht), torch.sin(tht)

    def closure():
        opt.zero_grad()
        eta = Xt @ W + b
        k = torch.sqrt(eta[:, 0] ** 2 + eta[:, 1] ** 2 + 1e-12)
        ll = eta[:, 0] * c + eta[:, 1] * s - (torch.log(torch.special.i0e(k)) + k)
        loss = -ll.mean() + lam * (W ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return W.detach(), b.detach()


def loglik(X, th, W, b):
    eta = torch.from_numpy(X).float() @ W + b
    k = torch.sqrt(eta[:, 0] ** 2 + eta[:, 1] ** 2 + 1e-12)
    tht = torch.from_numpy(th).float()
    ll = eta[:, 0] * torch.cos(tht) + eta[:, 1] * torch.sin(tht) - (torch.log(torch.special.i0e(k)) + k)
    return (ll.numpy() + np.log(2 * np.pi)) / LN2          # bits relative to uniform, per cycle


def standardise(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd


def score_D(X, th, half, dur, lam):
    """Cross-validated over the two halves: bits per cycle and bits per second."""
    bits, t = 0.0, 0.0
    for h in (False, True):
        tr, te = half == h, half != h
        Xtr, Xte = standardise(X[tr], X[te])
        W, b = fit(Xtr, th[tr], lam)
        bits += loglik(Xte, th[te], W, b).sum()
        t += dur[te].sum()
    return bits / len(th), bits / t


def choose_lambda(X, th, half, grid):
    """Nested CV inside each training half: early vs late part of the half."""
    out = {}
    for lam in grid:
        tot = 0.0
        for h in (False, True):
            idx = np.where(half == h)[0]
            a, b2 = idx[:len(idx) // 2], idx[len(idx) // 2:]
            for tr, te in ((a, b2), (b2, a)):
                Xtr, Xte = standardise(X[tr], X[te])
                W, b = fit(Xtr, th[tr], lam)
                tot += loglik(Xte, th[te], W, b).sum()
        out[lam] = tot
        print(f'  lambda {lam:g}: inner held-out bits {tot:.0f}', flush=True)
    return max(out, key=out.get)
