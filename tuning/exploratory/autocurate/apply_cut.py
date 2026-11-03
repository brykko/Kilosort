"""KS2.5's set_cutoff (set_cutoff.py, cycle 1: strict) applied to KS4 sorts with per-spike scores
(ks4_sort_scored.py), and the result on isolation/tradeoff.py's curves next to KS2.5.

The cut runs on the sort's own spikes, before our cleaning, as in KS2.5. Each unit's cut-off starts at
Th_learned + w and is lowered towards Th_learned (the extraction threshold: every spike's score is above it),
for widths w = 1-4 (KS2.5 standard sweeps 10 -> 6, i.e. w = 4; standard_low 9 -> 4, w = 5). The cleaned sort's
spikes (cleaning.py, run on the uncut sort; hybrid copies out, also found on the uncut sort) are then kept if
the cut kept them (the same rows of the sort).

Curves as tradeoff.py (units admitted cleanest first by the upper bound in one half, counted in the other;
plain and local estimates). Reported: whole output, correctly assigned spikes at KS2.5 standard's (seed 1)
whole-output level of misassigned spikes and at half of it, and at 1-10% retained contamination. References:
KS2.5 standard and standard_low (seed 1) and the original seed-1 KS4 sorts.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / 'isolation'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import hybrid  # noqa: E402
from cleaning import FS, load_clean_units  # noqa: E402
from set_cutoff import set_cutoff  # noqa: E402
from tradeoff import RETAINED, curve, good_at_mis, good_at_retained, sort_rows, unit_rows  # noqa: E402

K = Path('C:/data/ks4_tuning/26648_kobe_2020-08-21')
M = Path('C:/data/ks4_tuning/25843_roger_2019-09-11')
D = Path('C:/data/ks4_tuning/exploratory/autocurate')
WIDTHS = (1, 2, 3, 4)
REF = {'kobe': {'KS2.5 standard': K / 'runs/ks25_standard_r1', 'KS2.5 standard_low': K / 'runs/ks25_standard_low_r1',
                'default (original)': K / 'runs/hybrid_default_seed1', 'H8 (original)': K / 'runs/search/198_H8_cds10_s1/sort'},
       'mec': {'KS2.5 standard': M / 'runs/ks25_standard_r1', 'KS2.5 standard_low': M / 'runs/ks25_standard_low_r1',
               'default (original)': M / 'runs/default_s1', 'H8 (original)': M / 'runs/H8_s1'}}


def cut_variants(sort, ex):
    """{label: rows} for the uncut sort and each width; and the spikes deleted per width."""
    base = K if ex == 'kobe' else M
    th_floor = json.loads((sort / 'provenance.json').read_text())['settings'].get('Th_learned', 8)
    t = np.load(sort / 'spike_times.npy').ravel().astype(np.int64)
    clu = np.load(sort / 'spike_clusters.npy').ravel().astype(np.int64)
    score = np.load(sort / 'score.npy').astype(np.float64)
    by, _, iv, row = load_clean_units(sort, base / 'search/raw_artefacts_car.npz', values=np.arange(t.size))
    if ex == 'kobe':
        copies = set(hybrid.recovery(sort, np.load(base / 'search/hybrid/hybrid_truth.npz'))['hybrid_copies'])
        by = {c: x for c, x in by.items() if c not in copies}
    out, info = {'uncut': unit_rows(by, iv)}, {}
    for w in WIDTHS:
        keep, units = set_cutoff(t.astype(np.float64), clu, score, FS, (th_floor + w, th_floor), strict=True)
        cut = {c: x[keep[row[c]]] for c, x in by.items()}
        out[f'cut w={w}'] = unit_rows(cut, iv)
        info[w] = (1 - keep.mean(), np.mean([g for _, g in units.values()]))
    return out, info, th_floor


if __name__ == '__main__':
    for ex in ('kobe', 'mec'):
        rows = {g: sort_rows(s, ex) for g, s in REF[ex].items()}
        for g in ('default', 'H8'):
            var, info, th = cut_variants(D / f'sorts/{ex}_{g}_s1', ex)
            print(f'\n{ex} {g} rerun: Th_learned {th}; set_cutoff (strict) deletes ' + ', '.join(
                f'w={w}: {d:.1%} of spikes ({gd:.0%} of units "good")' for w, (d, gd) in info.items()), flush=True)
            rows.update({f'{g} {k}': r for k, r in var.items()})
        for m in ('plain', 'local'):
            curves = {g: curve(r, m, 'upper') for g, r in rows.items()}
            ks_mis = curves['KS2.5 standard'][0][-1]
            print(f'\n{ex}, {m} estimate (millions of spikes; seed 1). Correctly assigned at KS2.5 standard\'s '
                  f'misassigned ({ks_mis / 1e6:.2f} M) and half of it, and at retained contamination:')
            print(f'  {"":24s} {"correct":>8s} {"misas.":>7s} {"share":>6s}  {"@KS2.5":>7s} {"@half":>7s}  '
                  + ' '.join(f'{p:>6.0%}' for p in RETAINED))
            for g, (mis, good) in curves.items():
                print(f'  {g:24s} {good[-1] / 1e6:8.2f} {mis[-1] / 1e6:7.2f} {mis[-1] / (mis[-1] + good[-1]):6.1%}  '
                      f'{good_at_mis(mis, good, ks_mis) / 1e6:7.2f} {good_at_mis(mis, good, ks_mis / 2) / 1e6:7.2f}  '
                      + ' '.join(f'{good_at_retained(mis, good, p) / 1e6:6.2f}' for p in RETAINED), flush=True)
            np.savez(D / f'curves_{ex}_{m}.npz', **{g: np.c_[c[0], c[1]] for g, c in curves.items()})
