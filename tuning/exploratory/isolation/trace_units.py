"""Where the KS4 sorts' contamination comes from, relative to KS2.5, and where KS2.5's own units went.

Spike labels: each spike of one sort is labelled with the unit of the other sort it coincides with
(within 0.4 ms), among the other sort's units within 100 um in depth whose coincidences with it clearly
exceed chance (>= 10, >= 5 x chance, >= 1% of the smaller unit); unlabelled if none. The other sort's
units are all units of its cleaned sort.

1. KS4 (default, H8; seed 1) eligible units, split into cells shared with KS2.5 (one-to-one match, as
   matched_contam.py) and own units. Each refractory violation (tau_c < interval < 1.5 ms) of a KS4 unit
   is classed by the KS2.5 labels of its two spikes: both in the same KS2.5 unit (KS2.5 has it too), in
   two different KS2.5 units (KS4 joins spikes KS2.5 separates), or at least one spike KS2.5 doesn't
   output. Each unit's misassigned spikes (n c, plain and local) are shared out in proportion.
2. KS2.5's own eligible units (seed 1, unmatched to the KS4 sort): fraction of their spikes in the KS4
   output, and the KS4 unit holding most of them. Classes: absent (< 50% of spikes in KS4's output),
   split (no KS4 unit holds 50%), inside a larger KS4 unit (one holds >= 50%, but they are < 50% of it),
   own counterpart (otherwise: unmatched only because agreement < 0.5 or the match went elsewhere).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cleaning import load_clean_units  # noqa: E402
from compare import MATCH_MIN, MATCH_UM, TOL_S, agreement  # noqa: E402
from isolation import TAU_C  # noqa: E402

TAU = 1.5e-3
LOCAL = (0.010, 0.030)
DUR = 1200.0
K = Path('C:/data/ks4_tuning/26648_kobe_2020-08-21')
M = Path('C:/data/ks4_tuning/25843_roger_2019-09-11')
SORTS = {'kobe': {'default': K / 'runs/hybrid_default_seed1', 'ks25': K / 'runs/ks25_standard_r1',
                  'H8': K / 'runs/search/198_H8_cds10_s1/sort'},
         'mec': {'default': M / 'runs/default_s1', 'ks25': M / 'runs/ks25_standard_r1', 'H8': M / 'runs/H8_s1'}}
tr = json.loads((M / 'runs/transfer_ks25.json').read_text())
MEC_ELIG = {Path(e['sort']).as_posix(): [int(c) for c in e['S_i']] for e in tr['default'] + tr['winner']}
CATS = ('same KS2.5 unit', 'two KS2.5 units', 'not in KS2.5')


def load(sort, ex):
    base = K if ex == 'kobe' else M
    by, depth, iv = load_clean_units(sort, base / 'search/raw_artefacts_car.npz')
    T = DUR - float(np.sum(iv[:, 1] - iv[:, 0])) if iv.size else DUR
    elig = [int(c) for c in json.loads((sort / 'evaluation.json').read_text())['S_i']] if ex == 'kobe' else MEC_ELIG[sort.as_posix()]
    by = {c: np.sort(t) for c, t in by.items() if t.size}
    return by, depth, T, [c for c in elig if c in by]


def solve(v, D):
    a = v / (2 * (TAU - TAU_C) * max(D, 1e-12))
    return 1 - np.sqrt(1 - 2 * a) if a <= 0.5 else 1.0


def violation_pairs(t):
    lo, hi = np.searchsorted(t, t + TAU_C, 'right'), np.searchsorted(t, t + TAU, 'left')
    cnt = np.maximum(hi - lo, 0)
    i = np.repeat(np.arange(t.size), cnt)
    return i, lo[i] + np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)


def contamination(t, T):
    v = violation_pairs(t)[0].size
    pairs = np.sum(np.searchsorted(t, t + LOCAL[1], 'right') - np.searchsorted(t, t + LOCAL[0], 'right'))
    return {'plain': solve(v, t.size ** 2 / T), 'local': solve(v, pairs / (LOCAL[1] - LOCAL[0]))}


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


def label(ua, da, ub, db, T):
    """Per unit of ua, the ub unit each spike coincides with (-1 if none)."""
    order = [c for c in sorted(ub, key=lambda c: db[c]) if ub[c].size >= 2]
    dep = np.array([db[c] for c in order])
    out = {}
    for a, t in ua.items():
        lo, hi = np.searchsorted(dep, [da[a] - MATCH_UM, da[a] + MATCH_UM + 1e-9])
        lab, best = np.full(t.size, -1), np.full(t.size, np.inf)
        for b in order[lo:hi]:
            tb = ub[b]
            i = np.clip(np.searchsorted(tb, t), 1, tb.size - 1)
            d = np.minimum(np.abs(tb[i] - t), np.abs(tb[i - 1] - t))
            hit = d <= TOL_S
            if hit.sum() < max(10, 5 * t.size * tb.size * 2 * TOL_S / T, 0.01 * min(t.size, tb.size)):
                continue
            better = hit & (d < best)
            lab[better], best[better] = b, d[better]
        out[a] = lab
    return out


def contributors(lab, n, frac=0.1):
    v = lab[lab >= 0]
    if not v.size:
        return 0, 0.0
    _, cnt = np.unique(v, return_counts=True)
    return int(np.sum(cnt >= frac * n)), cnt.max() / n


def main():
    for ex, s in SORTS.items():
        by25, d25, T25, el25 = load(s['ks25'], ex)
        for g in ('default', 'H8'):
            by4, d4, T4, el4 = load(s[g], ex)
            pr = pairs({c: by4[c] for c in el4}, d4, {c: by25[c] for c in el25}, d25)
            sh4, sh25 = {a for a, _ in pr}, {b for _, b in pr}
            T = (T4 + T25) / 2
            lab4 = label(by4, d4, by25, d25, T)
            est4 = {c: contamination(by4[c], T4) for c in by4}
            print(f'\n=== {ex}: {g} against KS2.5 ({len(pr)} shared cells) ===', flush=True)

            # 1. KS4's misassigned spikes by the KS2.5 labels of their violations
            for grp, units in (('shared', [c for c in el4 if c in sh4]), ('own', [c for c in el4 if c not in sh4])):
                n_tot, absent, dom, mis = 0, 0, 0.0, {m: np.zeros(3) for m in ('plain', 'local')}
                ncontrib = np.zeros(3)                               # spikes in units with 0, 1, >= 2 KS2.5 contributors
                for c in units:
                    t, lab = by4[c], lab4[c]
                    i, j = violation_pairs(t)
                    li, lj = lab[i], lab[j]
                    cat = np.where((li < 0) | (lj < 0), 2, np.where(li == lj, 0, 1))
                    frac = np.bincount(cat, minlength=3) / max(cat.size, 1)
                    for m in mis:
                        mis[m] += t.size * est4[c][m] * frac
                    k, share = contributors(lab, t.size)
                    n_tot += t.size; absent += np.sum(lab < 0); dom += share * t.size
                    ncontrib[min(k, 2)] += t.size
                print(f'  KS4 {grp} units ({len(units)}, {n_tot / 1e6:.2f} M spikes): {absent / n_tot:.0%} of spikes not in KS2.5\'s output; '
                      f'{dom / n_tot:.0%} in the unit\'s main KS2.5 unit; spikes in units drawing >= 10% from 0 / 1 / >= 2 KS2.5 units: '
                      + ' / '.join(f'{x / n_tot:.0%}' for x in ncontrib))
                for m, v in mis.items():
                    print(f'     misassigned ({m}) {v.sum() / 1e6:.2f} M: ' + ', '.join(f'{cn} {x / 1e6:.2f} M ({x / max(v.sum(), 1):.0%})' for cn, x in zip(CATS, v)))

            # 2. Where KS2.5's own units went in the KS4 sort
            own25 = [c for c in el25 if c not in sh25]
            lab25 = label({c: by25[c] for c in own25}, d25, by4, d4, T)
            cls = {k: [] for k in ('absent', 'split', 'inside a larger KS4 unit', 'own counterpart')}
            for c in own25:
                t, lab = by25[c], lab25[c]
                found = np.mean(lab >= 0)
                if found < 0.5:
                    cls['absent'].append((c, None)); continue
                hosts, cnt = np.unique(lab[lab >= 0], return_counts=True)
                h = int(hosts[cnt.argmax()])
                if cnt.max() < 0.5 * t.size:
                    cls['split'].append((c, None))
                elif cnt.max() < 0.5 * by4[h].size:
                    cls['inside a larger KS4 unit'].append((c, h))
                else:
                    cls['own counterpart'].append((c, h))
            est25 = {c: contamination(by25[c], T25) for c in own25}
            n_own = sum(by25[c].size for c in own25)
            print(f'  KS2.5 own units ({len(own25)}, {n_own / 1e6:.2f} M spikes), by where they are in {g}:')
            for k, rows in cls.items():
                if not rows:
                    print(f'     {k:26s}: 0'); continue
                n = np.array([by25[c].size for c, _ in rows], float)
                cp = np.array([est25[c]['plain'] for c, _ in rows]); cl = np.array([est25[c]['local'] for c, _ in rows])
                found = np.array([np.mean(lab25[c] >= 0) for c, _ in rows])
                line = (f'     {k:26s}: {len(rows):4d} units, {n.sum() / 1e6:.2f} M spikes ({n.sum() / n_own:.0%}), '
                        f'contamination {np.sum(n * cp) / n.sum():.1%} / {np.sum(n * cl) / n.sum():.1%}, '
                        f'{np.sum(n * found) / n.sum():.0%} of spikes in {g}\'s output')
                hs = [h for _, h in rows if h is not None]
                if hs:
                    uh = sorted(set(hs))
                    nh = np.array([by4[h].size for h in uh], float)
                    hp = np.array([est4[h]['plain'] for h in uh]); hl = np.array([est4[h]['local'] for h in uh])
                    grp = [('shared' if h in sh4 else 'own' if h in el4 else 'ineligible') for h in uh]
                    k25 = [contributors(lab4[h], by4[h].size)[0] for h in uh]
                    line += (f'; {len(uh)} host units ({", ".join(f"{grp.count(x)} {x}" for x in ("shared", "own", "ineligible"))}), '
                             f'{nh.sum() / 1e6:.2f} M spikes, contamination {np.sum(nh * hp) / nh.sum():.1%} / {np.sum(nh * hl) / nh.sum():.1%}, '
                             f'hosts drawing >= 10% from >= 2 KS2.5 units: {np.mean(np.array(k25) >= 2):.0%}')
                print(line, flush=True)


if __name__ == '__main__':
    main()
