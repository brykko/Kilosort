"""Rates per second inside the 1000 um band from m6_{ex}.json; null-corrected counts over the batches done.

    python m6_summary.py
"""
import json

import numpy as np

from common import OUT

for ex in ('kobe', 'mec'):
    f = OUT / f'm6_{ex}.json'
    if not f.exists():
        continue
    J = json.loads(f.read_text())
    for r, rows in J.items():
        if not rows:
            continue
        S = 2.0 * len(rows)
        g = lambda n, k: sum(x[n][k] for x in rows) / S  # noqa: E731
        fnd = sum(x['found'] for x in rows) / S
        print(f"{ex} {r}: batches {[x['batch'] for x in rows]}, band {rows[0]['band']}; found {fnd:.0f}/s; X events {g('X', 'n'):.0f}/s (st0 {sum(x['n_st0'] for x in rows) / S:.0f}/s); "
              f"var X {np.mean([x['var_X'] for x in rows]):.3f}, Xres {np.mean([x['var_Xres'] for x in rows]):.3f}")
        print(f"   Xres events {g('Xres', 'n'):.1f}/s, spike-signed {g('Xres', 'pos'):.1f}/s, near a found spike {g('Xres', 'near'):.1f}/s (chance {g('Xres', 'near_chance'):.1f}), "
              f"isolated spike-signed {g('Xres', 'iso_pos'):.1f}/s; null {g('null', 'n'):.1f}/s, spike-signed {g('null', 'pos'):.1f}/s, isolated {g('null', 'iso_pos'):.1f}/s")
        xs = g('Xres', 'iso_pos') - g('null', 'iso_pos')
        print(f"   isolated excess over null {xs:.1f}/s = {xs / fnd:.3f} of found spikes; isolated amp .5/.9/.99 (per batch medians) {np.median([x['Xres']['iso_amp'] for x in rows], 0).round(2).tolist()}; "
              f"Xres amp .1/.5/.9/.99 {np.median([x['Xres']['amp'] for x in rows], 0).round(2).tolist()}; X amp {np.median([x['X']['amp'] for x in rows], 0).round(2).tolist()}")
        print(f"   isolated by depth (10 bins, summed) {np.sum([x['Xres']['depth'] for x in rows], 0).tolist()}, per-batch counts {[x['Xres']['iso_pos'] for x in rows]}")
