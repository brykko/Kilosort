"""Contamination of the same cells in two sorts: one-to-one matched eligible units (agreement >= 0.5,
within 100 um, as compare.py), seed 1 of each sorter. Per pair: rates and contamination (plain and local
estimates of comod.py). If KS2.5's lower contamination came from lower firing rates, it would vanish on
cells whose rates match."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cleaning import load_clean_units  # noqa: E402
from comod import unit_estimates  # noqa: E402
from compare import MATCH_MIN, MATCH_UM, agreement  # noqa: E402

K = Path('C:/data/ks4_tuning/26648_kobe_2020-08-21')
M = Path('C:/data/ks4_tuning/25843_roger_2019-09-11')
SORTS = {'kobe': {'default': K / 'runs/hybrid_default_seed1', 'ks25': K / 'runs/ks25_standard_r1',
                  'H8': K / 'runs/search/198_H8_cds10_s1/sort'},
         'mec': {'default': M / 'runs/default_s1', 'ks25': M / 'runs/ks25_standard_r1', 'H8': M / 'runs/H8_s1'}}
tr = json.loads((M / 'runs/transfer_ks25.json').read_text())
MEC_ELIG = {Path(e['sort']).as_posix(): [int(c) for c in e['S_i']] for e in tr['default'] + tr['winner']}


def load(sort, ex):
    base = K if ex == 'kobe' else M
    by, depth, iv = load_clean_units(sort, base / 'search/raw_artefacts_car.npz')
    T = 1200 - float(np.sum(iv[:, 1] - iv[:, 0])) if iv.size else 1200.0
    elig = [int(c) for c in json.loads((sort / 'evaluation.json').read_text())['S_i']] if ex == 'kobe' else MEC_ELIG[sort.as_posix()]
    return {c: np.sort(by[c]) for c in elig}, depth, T


def pairs(ua, da, ub, db):
    order = sorted(ub, key=lambda c: db[c])
    dep = np.array([db[c] for c in order])
    cand = []
    for c, t in ua.items():
        lo, hi = np.searchsorted(dep, [da[c] - MATCH_UM, da[c] + MATCH_UM + 1e-9])
        cand += [(g, c, order[k]) for k in range(lo, hi) if (g := agreement(t, ub[order[k]])) >= MATCH_MIN]
    used_a, used_b, out = set(), set(), []
    for g, c, d in sorted(cand, reverse=True):
        if c not in used_a and d not in used_b:
            out.append((c, d)); used_a.add(c); used_b.add(d)
    return out


if __name__ == '__main__':
    for ex, s in SORTS.items():
        L = {g: load(p, ex) for g, p in s.items()}
        for g1, g2 in (('default', 'ks25'), ('H8', 'ks25'), ('default', 'H8')):
            (u1, d1, T1), (u2, d2, T2) = L[g1], L[g2]
            pr = pairs(u1, d1, u2, d2)
            e1 = [unit_estimates(u1[a], T1) for a, _ in pr]
            e2 = [unit_estimates(u2[b], T2) for _, b in pr]
            r1, r2 = np.array([e[1] for e in e1]), np.array([e[1] for e in e2])
            print(f'\n{ex}: {g2} vs {g1}, {len(pr)} shared cells; median rate {np.median(r1):.2f} vs {np.median(r2):.2f} Hz '
                  f'(median ratio {np.median(r2 / r1):.2f})')
            for m in ('plain', 'local'):
                c1, c2 = np.array([e[2][m] for e in e1]), np.array([e[2][m] for e in e2])
                w1 = np.array([e[0] for e in e1]); w2 = np.array([e[0] for e in e2])
                close = np.abs(np.log(r2 / r1)) < np.log(1.1)      # rates within 10%
                print(f'  {m:6s}: misassigned {np.sum(w1 * c1) / w1.sum():6.1%} ({g1}) vs {np.sum(w2 * c2) / w2.sum():6.1%} ({g2}); '
                      f'{g2} cleaner in {np.mean(c2 < c1):.0%} of pairs, dirtier in {np.mean(c2 > c1):.0%}; '
                      f'pairs with rates within 10% ({close.sum()}): {np.sum((w1 * c1)[close]) / w1[close].sum():6.1%} vs '
                      f'{np.sum((w2 * c2)[close]) / w2[close].sum():6.1%}')
