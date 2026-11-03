"""How much does the 0.3-0.5 ms within-unit pile-up move the refractory-based comparisons?

Per sort, cleaned units (tuning/cleaning.load_clean_units: §6.1-6.3; hybrid copies excluded on kobe),
units with >= 600 spikes. Hill et al. contamination (as exploratory/isolation/isolation.py, censored
window) with three windows: (tau_c, tau_r) = (0.25, 1.5) ms [the harness's], (0.5, 1.5) ms [bump
excluded] and (1.0, 2.0) ms [bump and its tail excluded]. Reported: misassigned spikes sum N c, median c,
share of units with c <= 0.1, and the KS4 / KS2.5 ratios of the misassigned totals per excerpt.
Also the set_cutoff view: share of within-unit pairs in set_cutoff's first non-zero bin (0.5-1.5 ms; its
central bin |lag| < 0.5 ms is zeroed, set_cutoff.py:52) that is the bump's tail (excess over the
1.0-1.3 ms level at 0.5-1.0 ms).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, 'C:/code/github/Kilosort/tuning')
import hybrid  # noqa: E402
from cleaning import load_clean_units  # noqa: E402

KOBE = Path('C:/data/ks4_tuning/26648_kobe_2020-08-21')
MEC = Path('C:/data/ks4_tuning/25843_roger_2019-09-11')
OUT = Path(__file__).resolve().parent / 'consequences.json'
DUR = 1200.0
WINDOWS = ((0.25e-3, 1.5e-3), (0.5e-3, 1.5e-3), (1.0e-3, 2.0e-3))


def viol(t, tc, tr):
    return int(np.sum(np.searchsorted(t, t + tr, 'left') - np.searchsorted(t, t + tc, 'right')))


def contam(n, v, T, tc, tr):
    a = v * T / (2 * (tr - tc) * max(n, 1) ** 2)
    return 1 - np.sqrt(1 - 2 * a) if a <= 0.5 else 1.0


def lagsamples(units, maxl=45):
    H = np.zeros(maxl + 1)
    for t in units.values():
        s = np.round(np.sort(t) * 30000).astype(np.int64)
        d = 1
        while True:
            dk = s[d:] - s[:-d]
            m = dk <= maxl
            if not m.any():
                break
            H += np.bincount(dk[m], minlength=maxl + 1)[:maxl + 1]
            d += 1
    return H


def main():
    sorts = {'kobe': {'ks4_default': [f'hybrid_default_seed{i}' for i in (1, 2, 3)],
                      'ks4_H8': [f'search/{n}_H8_cds10_s{i}/sort' for i, n in ((1, 198), (2, 208), (3, 218))],
                      'ks25_standard': [f'ks25_standard_r{i}' for i in (1, 2, 3)],
                      'ks25_standard_low': [f'ks25_standard_low_r{i}' for i in (1, 2, 3)]},
             'mec': {'ks4_default': [f'default_s{i}' for i in (1, 2, 3)], 'ks4_H8': [f'H8_s{i}' for i in (1, 2, 3)],
                     'ks25_standard': [f'ks25_standard_r{i}' for i in (1, 2, 3)],
                     'ks25_standard_low': [f'ks25_standard_low_r{i}' for i in (1, 2, 3)]}}
    res = {}
    for ex, groups in sorts.items():
        base = KOBE if ex == 'kobe' else MEC
        truth = np.load(base / 'search/hybrid/hybrid_truth.npz') if ex == 'kobe' else None
        for g, names in groups.items():
            for nm in names:
                sort = base / 'runs' / nm
                by, _, iv = load_clean_units(sort, base / 'search/raw_artefacts_car.npz')
                if truth is not None:
                    cp = set(hybrid.recovery(sort, truth)['hybrid_copies'])
                    by = {k: v for k, v in by.items() if k not in cp}
                by = {k: np.sort(v) for k, v in by.items() if v.size >= 600}
                T = DUR - float(np.sum(iv[:, 1] - iv[:, 0])) if iv.size else DUR
                r = {'n_units': len(by), 'n_spikes': int(sum(v.size for v in by.values()))}
                for tc, tr in WINDOWS:
                    c = np.array([contam(v.size, viol(v, tc, tr), T, tc, tr) for v in by.values()])
                    n = np.array([v.size for v in by.values()])
                    r[f'{tc * 1e3:g}-{tr * 1e3:g}'] = {'misassigned': float((n * c).sum()), 'median_c': float(np.median(c)),
                                                     'share_c_le_0.1': float(np.mean(c <= 0.1))}
                H = lagsamples(by)
                floor = H[30:40].mean()
                r['setcut_bin1_tail_share'] = float(np.clip(H[15:30] - floor, 0, None).sum() / H[15:45].sum())
                r['hill_bump_share'] = float(np.clip(H[8:30] - floor, 0, None).sum() / H[8:45].sum())
                res[f'{ex}/{g}/{nm}'] = r
                print(ex, g, nm, json.dumps({k: (v if not isinstance(v, dict) else {kk: round(vv, 3) for kk, vv in v.items()}) for k, v in r.items()}), flush=True)
    # ratios
    summ = {}
    for ex in sorts:
        for w in ('0.25-1.5', '0.5-1.5', '1-2'):
            m = {g: np.mean([res[f'{ex}/{g}/{nm}'][w]['misassigned'] for nm in sorts[ex][g]]) for g in sorts[ex]}
            med = {g: np.mean([res[f'{ex}/{g}/{nm}'][w]['median_c'] for nm in sorts[ex][g]]) for g in sorts[ex]}
            summ[f'{ex} {w}'] = {'misassigned': {g: round(v) for g, v in m.items()},
                                 'H8/ks25std': m['ks4_H8'] / m['ks25_standard'], 'default/ks25std': m['ks4_default'] / m['ks25_standard'],
                                 'median_c': {g: round(v, 4) for g, v in med.items()}}
            print(ex, w, summ[f'{ex} {w}'])
    res['summary'] = summ
    OUT.write_text(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()
