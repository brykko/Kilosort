"""Pick R and I settings from the tuning runs by the preregistered objective; write tuned.json and a table."""
import glob
import json
import os

import numpy as np

from run_queue import OUT, R_GRID, I_GRID

score = {}
for f in glob.glob(os.path.join(OUT, 'tune', '*.json')):
    d = json.load(open(f))
    for a in d['arms']:
        rec = [r for r in a['records'] if r['tag'] in ('round3', 'final')][-1]
        s = rec['C'] if a['name'].startswith('R') else rec['T']
        nu = len(d['truth']['rates'])
        obj = s['acc'] - 0.5 * s['n_false'] / nu
        score.setdefault(a['name'], []).append((d['job']['variant'], obj, s['acc'], s['n_false'], s['found'], rec['te_med']))


def table(prefix, grid):
    rows = []
    for i, g in enumerate(grid):
        r = score[f'{prefix}{i}']
        rows.append((np.mean([x[1] for x in r]), i, g, r))
    return rows


out = {}
lines = []
for prefix, grid, key in (('Rg', R_GRID, 'R'), ('Ig', I_GRID, 'I')):
    rows = table(prefix, grid)
    best = max(rows, key=lambda x: (round(x[0], 6), -x[1]))
    out[key] = dict(best[2])
    lines.append(f'{key} best: index {best[1]} {best[2]} objective {best[0]:.3f}')
    for o, i, g, r in rows:
        lines.append(f'  {i:2d} {o:.3f} ' + ' '.join(f'{x[0][:4]}:{x[1]:.3f}' for x in sorted(r)) + f'  {g}')
print('\n'.join(lines))
json.dump(out, open(os.path.join(OUT, 'tuned.json'), 'w'), indent=1)
open(os.path.join(OUT, 'tuning_table.txt'), 'w').write('\n'.join(lines))
