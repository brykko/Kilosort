"""Exploratory: are KS2.5's sparsely firing eligible units what helps its MEC position decoding?

Per sort: eligible units (as bayes_decode.py), their rates after cleaning, their position S_i (from
transfer_ks25.json, alpha 0.0625 s), and position decoding (20 and 100 ms) with all units, with units
>= 1 Hz only, and with units < 1 Hz only. Then: how many of KS2.5's sparse units have a match among
H8's / default's eligible units.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import bayes_decode as bd  # noqa: E402
from cleaning import mask_intervals  # noqa: E402
from compare import MATCH_MIN, MATCH_UM, agreement  # noqa: E402
from reproducibility import eligible_units  # noqa: E402

M = Path('C:/data/ks4_tuning/25843_roger_2019-09-11')
R = M / 'runs'
RAW = str(M / 'search/raw_artefacts_car.npz')
T0, DUR = 13575.0, 1200.0
tracking = dict(np.load(M / 'search/tracking_13575-14775s.npz'))
groups = {'default': [R / f'default_s{k}' for k in (1, 2, 3)], 'ks25': [R / f'ks25_standard_r{k}' for k in (1, 2, 3)],
          'H8': [R / f'H8_s{k}' for k in (1, 2, 3)]}
tr = json.loads((R / 'transfer_ks25.json').read_text())
S_i = {Path(e['sort']).name: e['S_i'] for e in tr['default'] + tr['winner']}
BANDS = [0, 0.5, 1, 2, 5, np.inf]
SPLIT_HZ = 1.0


def decode_subsets(sort, units):
    ids = sorted(units)
    rate = np.array([units[c].size / DUR for c in ids])
    T = np.concatenate([units[c] for c in ids])
    U = np.repeat(np.arange(len(ids)), [units[c].size for c in ids])
    mask = mask_intervals(sort, RAW)
    beh = bd.behaviour(tracking, T0, DUR, mask)
    nu = len(ids)
    halves = [(~beh['second'], beh['second']), (beh['second'], ~beh['second'])]
    fits = [bd.fit('pos', beh, T, U, nu, trn) for trn, _ in halves]
    subsets = {'all': np.arange(nu), '>=1Hz': np.flatnonzero(rate >= SPLIT_HZ), '<1Hz': np.flatnonzero(rate < SPLIT_HZ)}
    out = {}
    for w in (0.02, 0.1):
        data = [bd.windows(beh, T, U, nu, w, DUR, mask, te) for _, te in halves]
        for name, s in subsets.items():
            err = np.concatenate([bd.decode('pos', beh, n[:, s], tau, fits[h][0][s], fits[h][1], xy)
                                  for h, (n, tau, hd, xy) in enumerate(data)])
            out[(name, w)] = float(np.median(err))
    return out, {k: v.size for k, v in subsets.items()}


res = {}
for g, sorts in groups.items():
    for s in sorts:
        units, depth = eligible_units(s, RAW, DUR, None)
        ids = sorted(units)
        rate = np.array([units[c].size / DUR for c in ids])
        si = np.array([S_i[s.name][str(c)] for c in ids])
        band = np.digitize(rate, BANDS) - 1
        dec, sizes = decode_subsets(s, units)
        res[s.name] = {'group': g, 'rate': rate, 'S_i': si, 'band': band, 'dec': dec, 'sizes': sizes,
                       'units': units, 'depth': depth, 'ids': ids}
        print(s.name, sizes, {f'{k[0]}@{int(k[1]*1000)}ms': round(v, 1) for k, v in dec.items()}, flush=True)

print('\nEligible units and summed position S_i by rate band (mean over replicates)')
print(f"{'':8s}" + ''.join(f'{f"{a}-{b} Hz":>18s}' for a, b in zip(BANDS[:-1], BANDS[1:])))
for g in groups:
    rs = [r for r in res.values() if r['group'] == g]
    cells = []
    for k in range(len(BANDS) - 1):
        n = np.mean([np.sum(r['band'] == k) for r in rs])
        sb = np.mean([r['S_i'][r['band'] == k].sum() for r in rs])
        cells.append(f'{n:7.0f} / {sb:7.1f}')
    print(f'{g:8s}' + ''.join(f'{c:>18s}' for c in cells))
print('\nBits per spike (median S_i / rate) by band')
for g in groups:
    rs = [r for r in res.values() if r['group'] == g]
    print(f'{g:8s}' + ''.join(f'{np.median(np.concatenate([r["S_i"][r["band"] == k] / r["rate"][r["band"] == k] for r in rs])):18.3f}'
                              for k in range(len(BANDS) - 1)))
print('\nPosition decoding, median error (cm), mean over replicates; units in each subset')
for g in groups:
    rs = [r for r in res.values() if r['group'] == g]
    print(f'{g:8s}', '  '.join(f"{name} {np.mean([r['sizes'][name] for r in rs]):5.0f} u: "
                               + ' / '.join(f"{np.mean([r['dec'][(name, w)] for r in rs]):5.1f}" for w in (0.02, 0.1))
                               for name in ('all', '>=1Hz', '<1Hz')))


def matched(a, b):
    """For each unit of a (dict c -> times, depth), whether b has a unit with agreement >= MATCH_MIN."""
    ua, da = a
    ub, db = b
    order = sorted(ub, key=lambda c: db[c])
    dep = np.array([db[c] for c in order])
    hit = {}
    for c, t in ua.items():
        lo, hi = np.searchsorted(dep, [da[c] - MATCH_UM, da[c] + MATCH_UM + 1e-9])
        hit[c] = max((agreement(t, ub[order[k]]) for k in range(lo, hi)), default=0.0) >= MATCH_MIN
    return hit


print('\nKS2.5 r1 eligible units with a match (agreement >= 0.5) in H8 s1 / default s1, by rate band')
k = res['ks25_standard_r1']
ka = ({c: k['units'][c] for c in k['ids']}, k['depth'])
for other in ('H8_s1', 'default_s1'):
    o = res[other]
    hit = matched(ka, ({c: o['units'][c] for c in o['ids']}, o['depth']))
    h = np.array([hit[c] for c in k['ids']])
    print(f'  in {other:10s}', '  '.join(f'{a}-{b} Hz: {h[k["band"] == i].sum()}/{np.sum(k["band"] == i)} '
                                          f'(S {k["S_i"][(k["band"] == i) & h].sum():.0f}/{k["S_i"][k["band"] == i].sum():.0f})'
                                          for i, (a, b) in enumerate(zip(BANDS[:-1], BANDS[1:]))))
print('\nH8 s1 eligible units with a match in KS2.5 r1, by rate band')
o = res['H8_s1']
hit = matched(({c: o['units'][c] for c in o['ids']}, o['depth']), ka)
h = np.array([hit[c] for c in o['ids']])
print('  ', '  '.join(f'{a}-{b} Hz: {h[o["band"] == i].sum()}/{np.sum(o["band"] == i)} '
                       f'(S {o["S_i"][(o["band"] == i) & h].sum():.0f}/{o["S_i"][o["band"] == i].sum():.0f})'
                       for i, (a, b) in enumerate(zip(BANDS[:-1], BANDS[1:]))))
