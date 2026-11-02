"""How much of each score difference between sorters is firing rate (more spikes) and how much is per spike.

Every score here is a product: S = (spike rate of the scored units) x (bits per spike, spike-weighted);
the isolation score = (spikes) x (fraction correctly assigned). For two sorts, the difference splits
exactly into a rate term and a per-spike term (Shapley: dX = dA (B1 + B2)/2 + dB (A1 + A2)/2).
Then the same on matched cells (one-to-one, agreement >= 0.5, as compare.py), seed 1 of each sorter.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cleaning import load_clean_units  # noqa: E402
from compare import MATCH_MIN, MATCH_UM, agreement  # noqa: E402

K = Path('C:/data/ks4_tuning/26648_kobe_2020-08-21')
M = Path('C:/data/ks4_tuning/25843_roger_2019-09-11')
ISO = Path('C:/data/ks4_tuning/exploratory/isolation')
kobe = {'default': [K / f'runs/hybrid_default_seed{k}' for k in (1, 2, 3, 4, 5)],
        'ks25': [K / f'runs/ks25_standard_r{k}' for k in (1, 2, 3)],
        'H8': [K / f'runs/search/{r}/sort' for r in ('198_H8_cds10_s1', '208_H8_cds10_s2', '218_H8_cds10_s3')]}
mec = {'default': [M / f'runs/default_s{k}' for k in (1, 2, 3)], 'ks25': [M / f'runs/ks25_standard_r{k}' for k in (1, 2, 3)],
       'H8': [M / f'runs/H8_s{k}' for k in (1, 2, 3)]}
tr = json.loads((M / 'runs/transfer_ks25.json').read_text())
mec_Si = {Path(e['sort']).as_posix(): e['S_i'] for e in tr['default'] + tr['winner']}
CACHE = {}


def units(sort, raw):
    if sort not in CACHE:
        by, depth, _ = load_clean_units(sort, raw)
        CACHE[sort] = (by, depth)
    return CACHE[sort]


def per_unit(sort, ex, key):
    """{unit: (rate Hz, S_i bits/s)} for the scored (eligible) units."""
    raw = (K if ex == 'kobe' else M) / 'search/raw_artefacts_car.npz'
    by, _ = units(sort, raw)
    Si = json.loads((sort / 'evaluation.json').read_text())[key] if ex == 'kobe' else mec_Si[sort.as_posix()]
    return {int(c): (by[int(c)].size / 1200, v) for c, v in Si.items()}


def product(groups, ex, key):
    """Per sorter: mean over seeds of total rate (Hz) and S; bits/spike = S / rate."""
    out = {}
    for g, sorts in groups.items():
        R, S = [], []
        for s in sorts:
            u = per_unit(s, ex, key)
            R.append(sum(r for r, _ in u.values()))
            S.append(sum(v for _, v in u.values()))
        out[g] = (np.mean(R), np.mean(S))
    return out


def shapley(a1, b1, a2, b2):
    dx = a2 * b2 - a1 * b1
    rate = (a2 - a1) * (b1 + b2) / 2
    return dx, rate, dx - rate


def report(title, vals, unit_a, unit_b, unit_x):
    print(f'\n{title}')
    for g, (a, x) in vals.items():
        print(f'  {g:8s} {unit_a} {a:10.1f}   {unit_b} {x / a:7.4f}   {unit_x} {x:10.1f}')
    for g1, g2 in (('default', 'H8'), ('ks25', 'H8'), ('default', 'ks25')):
        (a1, x1), (a2, x2) = vals[g1], vals[g2]
        dx, rate, per = shapley(a1, x1 / a1, a2, x2 / a2)
        print(f'  {g2} - {g1}: {dx:+10.1f} = rate {rate:+10.1f} ({rate / dx:6.0%}) + per spike {per:+10.1f} ({per / dx:6.0%})')


report('Kobe S (theta_ref), eligible units', product(kobe, 'kobe', 'S_i'), 'rate (spikes/s)', 'bits/spike', 'S (bits/s)')
report('Kobe S_track (tracked heading)', product(kobe, 'kobe', 'S_track_i'), 'rate (spikes/s)', 'bits/spike', 'S_track')
report('MEC position S', product(mec, 'mec', None), 'rate (spikes/s)', 'bits/spike', 'S (bits/s)')
for ex in ('kobe', 'mec'):
    r = json.loads((ISO / f'{ex}.json').read_text())
    vals = {}
    for g in ('default', 'ks25', 'H8'):
        ys = [v['yield']['1.5ms'] for k, v in r.items() if k.rsplit('_', 1)[0] == g]
        vals[g] = (np.mean([y['spikes'] for y in ys]) / 1e6, np.mean([y['score'] for y in ys]) / 1e6)
    report(f'{ex} isolation score (all units)', vals, 'spikes (M)', 'fraction clean', 'score (M)')


# ---------- matched cells ----------
def best_pairs(ua, da, ub, db):
    order = sorted(ub, key=lambda c: db[c])
    dep = np.array([db[c] for c in order])
    cand = []
    for c, t in ua.items():
        lo, hi = np.searchsorted(dep, [da[c] - MATCH_UM, da[c] + MATCH_UM + 1e-9])
        for k in range(lo, hi):
            g = agreement(t, ub[order[k]])
            if g >= MATCH_MIN:
                cand.append((g, c, order[k]))
    ua_used, ub_used, pairs = set(), set(), []
    for g, c, d in sorted(cand, reverse=True):
        if c not in ua_used and d not in ub_used:
            pairs.append((c, d)); ua_used.add(c); ub_used.add(d)
    return pairs


print('\nMatched cells (seed 1 of each; one-to-one, agreement >= 0.5): totals over the shared cells')
for ex, groups, key in (('kobe', kobe, 'S_i'), ('mec', mec, None)):
    raw = (K if ex == 'kobe' else M) / 'search/raw_artefacts_car.npz'
    for g1, g2 in (('default', 'H8'), ('ks25', 'H8'), ('default', 'ks25')):
        s1, s2 = groups[g1][0], groups[g2][0]
        u1, u2 = per_unit(s1, ex, key), per_unit(s2, ex, key)
        (b1, d1), (b2, d2) = units(s1, raw), units(s2, raw)
        pr = best_pairs({c: np.sort(b1[c]) for c in u1}, d1, {c: np.sort(b2[c]) for c in u2}, d2)
        R1 = sum(u1[a][0] for a, _ in pr); X1 = sum(u1[a][1] for a, _ in pr)
        R2 = sum(u2[b][0] for _, b in pr); X2 = sum(u2[b][1] for _, b in pr)
        dx, rate, per = shapley(R1, X1 / R1, R2, X2 / R2)
        r_ratio = np.median([u2[b][0] / u1[a][0] for a, b in pr])
        print(f'  {ex:4s} {g2} vs {g1}: {len(pr)} cells; rate {R1:.0f} -> {R2:.0f} Hz (median per cell x{r_ratio:.2f}); '
              f'S {X1:.1f} -> {X2:.1f}: {dx:+.1f} = rate {rate:+.1f} + per spike {per:+.1f}', flush=True)
