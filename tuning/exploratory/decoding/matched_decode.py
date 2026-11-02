"""Exploratory: decode MEC position from the cells KS2.5 and H8 share (each sorter's own version), and
from each sorter's unshared units. KS2.5 replicate k paired with H8 seed k."""
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
tr = json.loads((R / 'transfer_ks25.json').read_text())
S_i = {Path(e['sort']).name: e['S_i'] for e in tr['default'] + tr['winner']}


def best_pairs(a, b):
    """One-to-one pairs (unit of a, unit of b) with agreement >= MATCH_MIN, best first."""
    (ua, da), (ub, db) = a, b
    order = sorted(ub, key=lambda c: db[c])
    dep = np.array([db[c] for c in order])
    cand = []
    for c, t in ua.items():
        lo, hi = np.searchsorted(dep, [da[c] - MATCH_UM, da[c] + MATCH_UM + 1e-9])
        for k in range(lo, hi):
            g = agreement(t, ub[order[k]])
            if g >= MATCH_MIN:
                cand.append((g, c, order[k]))
    used_a, used_b, pairs = set(), set(), []
    for g, c, d in sorted(cand, reverse=True):
        if c not in used_a and d not in used_b:
            pairs.append((c, d, g)); used_a.add(c); used_b.add(d)
    return pairs


def prepare(sort, units):
    ids = sorted(units)
    T = np.concatenate([units[c] for c in ids])
    U = np.repeat(np.arange(len(ids)), [units[c].size for c in ids])
    mask = mask_intervals(sort, RAW)
    beh = bd.behaviour(tracking, T0, DUR, mask)
    halves = [(~beh['second'], beh['second']), (beh['second'], ~beh['second'])]
    fits = [bd.fit('pos', beh, T, U, len(ids), trn) for trn, _ in halves]
    data = {w: [bd.windows(beh, T, U, len(ids), w, DUR, mask, te) for _, te in halves] for w in (0.02, 0.1)}
    return {'ids': ids, 'beh': beh, 'fits': fits, 'data': data}


def err(P, idx, w):
    pos = {c: i for i, c in enumerate(P['ids'])}
    s = np.array([pos[c] for c in idx])
    e = np.concatenate([bd.decode('pos', P['beh'], n[:, s], tau, P['fits'][h][0][s], P['fits'][h][1], xy)
                        for h, (n, tau, hd, xy) in enumerate(P['data'][w])])
    return float(np.median(e))


for k in (1, 2, 3):
    sk, sh = R / f'ks25_standard_r{k}', R / f'H8_s{k}'
    uk, dk = eligible_units(sk, RAW, DUR, None)
    uh, dh = eligible_units(sh, RAW, DUR, None)
    pairs = best_pairs((uk, dk), (uh, dh))
    mk, mh = [p[0] for p in pairs], [p[1] for p in pairs]
    rest_k, rest_h = sorted(set(uk) - set(mk)), sorted(set(uh) - set(mh))
    Pk, Ph = prepare(sk, uk), prepare(sh, uh)
    rk = np.array([uk[c].size for c in mk]) / DUR
    rh = np.array([uh[c].size for c in mh]) / DUR
    Sk = np.array([S_i[sk.name][str(c)] for c in mk]); Sh = np.array([S_i[sh.name][str(c)] for c in mh])
    print(f'pair r{k}/s{k}: {len(pairs)} shared cells (agreement median {np.median([p[2] for p in pairs]):.2f}); '
          f'KS2.5 unshared {len(rest_k)}, H8 unshared {len(rest_h)}')
    print(f'  shared cells: rate KS2.5 {np.median(rk):.2f} Hz vs H8 {np.median(rh):.2f} Hz (H8/KS2.5 median {np.median(rh / rk):.2f}); '
          f'S {Sk.sum():.0f} vs {Sh.sum():.0f}; bits/spike {np.median(Sk / rk):.3f} vs {np.median(Sh / rh):.3f}')
    print(f'  unshared: S KS2.5 {sum(S_i[sk.name][str(c)] for c in rest_k):.0f}, H8 {sum(S_i[sh.name][str(c)] for c in rest_h):.0f}')
    for w in (0.02, 0.1):
        print(f'  {int(w * 1000):3d} ms: shared KS2.5 {err(Pk, mk, w):5.1f} | shared H8 {err(Ph, mh, w):5.1f} | '
              f'unshared KS2.5 {err(Pk, rest_k, w):5.1f} | unshared H8 {err(Ph, rest_h, w):5.1f} | '
              f'all KS2.5 {err(Pk, sorted(uk), w):5.1f} | all H8 {err(Ph, sorted(uh), w):5.1f}', flush=True)
