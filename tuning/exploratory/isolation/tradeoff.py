"""Trade-off curves: correctly assigned against misassigned spikes as units are admitted in order of
their contamination, cross-validated over the two halves of the recording.

Per unit (all units of the cleaned sort, hybrid copies out), contamination is estimated separately in
each half (plain and local estimates, as comod.py). Units are admitted by their contamination in one
half and counted with the point estimate in the other half; both directions are summed, so the selection
can't flatter itself. Admission uses the 1-SD upper bound of the half's violation count (Gehrels 1986),
so units with too few spikes to judge come late; 'point' ranks by the point estimate instead, as a check.
Sweeping the admission threshold traces a curve from the cleanest units to the whole output: a sorter
whose curve lies above another's gives more correctly assigned spikes at every level of misassigned ones.

KS2.5 is also shown before set_cutoff: the units of the traced rerun (ks25_setup/ks25_sort_traced.py,
seed 1) at stage split1, and, as its own reference, at stage merge3 (the exported sort). Stage units are
taken as the rerun's templates with the rerun's masked intervals dropped but no duplicate cleaning (both
stages alike); hybrid copies are found as hybrid.py does, on the stage's spikes.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import hybrid  # noqa: E402
from cleaning import FS, load_clean_units, mask_intervals  # noqa: E402
from isolation import TAU_C, violations  # noqa: E402

TAU = 1.5e-3
LOCAL = (0.010, 0.030)
DUR = 1200.0
K = Path('C:/data/ks4_tuning/26648_kobe_2020-08-21')
M = Path('C:/data/ks4_tuning/25843_roger_2019-09-11')
SETS = {
    'kobe': {'default': [K / f'runs/hybrid_default_seed{k}' for k in (1, 2, 3, 4, 5)],
             'ks25': [K / f'runs/ks25_standard_r{k}' for k in (1, 2, 3)],
             'ks25 low': [K / f'runs/ks25_standard_low_r{k}' for k in (1, 2, 3)],
             'H8': [K / f'runs/search/{r}/sort' for r in ('198_H8_cds10_s1', '208_H8_cds10_s2', '218_H8_cds10_s3')]},
    'mec': {'default': [M / f'runs/default_s{k}' for k in (1, 2, 3)], 'ks25': [M / f'runs/ks25_standard_r{k}' for k in (1, 2, 3)],
            'ks25 low': [M / f'runs/ks25_standard_low_r{k}' for k in (1, 2, 3)], 'H8': [M / f'runs/H8_s{k}' for k in (1, 2, 3)]},
}
RERUN = Path('C:/data/ks4_tuning/exploratory/ks25_traced')
STAGES = {'ks25 pre-cut': 'split1', 'ks25 post-cut': 'merge3'}
OUT = Path('C:/data/ks4_tuning/exploratory/isolation/tradeoff')
RETAINED = (0.01, 0.02, 0.05, 0.10)


def solve(v, D):
    a = v / (2 * (TAU - TAU_C) * max(D, 1e-12))
    return 1 - np.sqrt(1 - 2 * a) if a <= 0.5 else 1.0


def half_estimates(t, T):
    """Spikes, and per estimate (point, upper-bound) contamination, in one half."""
    n, v = t.size, violations(t, TAU)
    pairs = int(np.sum(np.searchsorted(t, t + LOCAL[1], 'right') - np.searchsorted(t, t + LOCAL[0], 'right')))
    D = {'plain': n * n / T, 'local': pairs / (LOCAL[1] - LOCAL[0])}
    return n, {m: (solve(v, d), solve(v + 1 + np.sqrt(v + 0.75), d)) for m, d in D.items()}


def masked(iv, lo, hi):
    return float(np.sum(np.clip(np.minimum(iv[:, 1], hi) - np.maximum(iv[:, 0], lo), 0, None))) if iv.size else 0.0


def sort_rows(sort, ex):
    base = K if ex == 'kobe' else M
    by, _, iv = load_clean_units(sort, base / 'search/raw_artefacts_car.npz')
    if ex == 'kobe':
        copies = set(hybrid.recovery(sort, np.load(base / 'search/hybrid/hybrid_truth.npz'))['hybrid_copies'])
        by = {c: t for c, t in by.items() if c not in copies}
    return unit_rows(by, iv)


def stage_rows(ex, stage):
    """Units of the traced KS2.5 rerun at one stage of autocurate (rez.st3, 1-based samples)."""
    base = K if ex == 'kobe' else M
    R = RERUN / f'{ex}_s1'
    st = np.load(R / 'stages.npz')
    s, clu = st[f'{stage}_t'].astype(np.int64) - 1, st[f'{stage}_clu'].astype(np.int64)
    iv = mask_intervals(R, base / 'search/raw_artefacts_car.npz')
    copies = set()
    if ex == 'kobe':
        tt = np.sort(np.load(base / 'search/hybrid/hybrid_truth.npz')['spike_samples'])
        k = np.searchsorted(tt, s - hybrid.TOL, 'left')
        hit = (k < tt.size) & (tt[np.minimum(k, tt.size - 1)] <= s + hybrid.TOL)
        ids, idx = np.unique(clu, return_inverse=True)
        copies = set(ids[np.bincount(idx, weights=hit) / np.bincount(idx) >= hybrid.COPY].tolist())
    t = s / FS
    i = np.searchsorted(iv[:, 0], t, 'right') - 1
    keep = ~((i >= 0) & (t < iv[np.maximum(i, 0), 1])) if iv.size else np.ones(t.size, bool)
    t, clu = t[keep], clu[keep]
    o = np.argsort(clu, kind='stable')
    ids, start = np.unique(clu[o], return_index=True)
    by = {int(c): x for c, x in zip(ids, np.split(t[o], start[1:])) if int(c) not in copies}
    return unit_rows(by, iv)


def unit_rows(by, iv):
    T = (DUR / 2 - masked(iv, 0, DUR / 2), DUR / 2 - masked(iv, DUR / 2, DUR))
    rows = []
    for t in by.values():
        if t.size:
            t = np.sort(t)
            k = np.searchsorted(t, DUR / 2)
            rows.append((half_estimates(t[:k], T[0]), half_estimates(t[k:], T[1])))
    return rows


def curve(rows, m, rank):
    """Cumulative misassigned and correctly assigned spikes as (unit, half) entries are admitted."""
    j = 1 if rank == 'upper' else 0
    key = np.array([r[h][1][m][j] for h in (0, 1) for r in rows])
    n = np.array([r[1 - h][0] for h in (0, 1) for r in rows], float)
    c = np.array([r[1 - h][1][m][0] for h in (0, 1) for r in rows])
    o = np.argsort(key, kind='stable')
    return np.cumsum((n * c)[o]), np.cumsum((n * (1 - c))[o])


def good_at_mis(mis, good, level):
    i = np.searchsorted(mis, level, 'right') - 1
    return good[i] if i >= 0 else 0.0


def good_at_retained(mis, good, p):
    ok = mis <= p * (mis + good)
    return good[ok].max() if ok.any() else 0.0


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {ex: {g: [sort_rows(s, ex) for s in sorts] for g, sorts in groups.items()} for ex, groups in SETS.items()}
    for ex in rows:
        rows[ex].update({g: [stage_rows(ex, stage)] for g, stage in STAGES.items()})
    saved = {}
    for ex, groups in rows.items():
        for m in ('plain', 'local'):
            print(f'\n{ex}, {m} estimate (millions of spikes; mean +- SD over seeds)')
            curves = {rank: {g: [curve(r, m, rank) for r in rs] for g, rs in groups.items()} for rank in ('upper', 'point')}
            ks_mis = np.mean([c[0][-1] for c in curves['upper']['ks25']])
            for g, cs in curves['upper'].items():
                full_m = np.array([c[0][-1] for c in cs]); full_g = np.array([c[1][-1] for c in cs])
                at_ks = np.array([good_at_mis(*c, ks_mis) for c in cs])
                at_ks_point = np.array([good_at_mis(*c, ks_mis) for c in curves['point'][g]])
                at_half = np.array([good_at_mis(*c, ks_mis / 2) for c in cs])
                ret = [np.array([good_at_retained(*c, p) for c in cs]) for p in RETAINED]
                f = lambda a: f'{a.mean() / 1e6:5.2f}+-{a.std(ddof=1) / 1e6:4.2f}' if a.size > 1 else f'{a[0] / 1e6:5.2f}      '  # noqa: E731
                print(f'  {g:13s} whole output: correct {f(full_g)}, misassigned {f(full_m)} ({np.mean(full_m / (full_m + full_g)):5.1%}); '
                      f'correct at KS2.5\'s misassigned ({ks_mis / 1e6:.2f}) {f(at_ks)} [point ranking {f(at_ks_point)}], '
                      f'at half of it {f(at_half)}; at retained contamination '
                      + ', '.join(f'{p:.0%}: {f(r)}' for p, r in zip(RETAINED, ret)))
                for k, (mis, good) in enumerate(cs):
                    idx = np.unique(np.r_[np.linspace(0, mis.size - 1, 600).astype(int), mis.size - 1])
                    saved[f'{ex}|{m}|{g}|{k}'] = np.c_[mis[idx], good[idx]]
    np.savez(OUT / 'curves.npz', **saved)
