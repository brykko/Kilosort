import json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

OUT = Path(__file__).parent
fn = sys.argv[1] if len(sys.argv) > 1 else 'results.jsonl'
R = [json.loads(l) for l in open(OUT / fn)]
CONDS = ['v', 'i', 'ii', 'iii', 'iv', 'L5', 'L10', 'L10s10']
by = defaultdict(list)
for r in R:
    by[(r['pctl'], r['group'], r['cond'])].append(r)

def fmt(v):
    v = np.array(v)
    return f'{v.mean():6.1f} [{v.min()}-{v.max()}]'

METS = [m for m in ['leaves', 'final', 'final_nometa', 'niter'] if m in R[0]]
for metric in METS:
    print(f'\n== {metric} (mean [min-max] over seeds) ==')
    print(f'{"pctl/grp":>10} ' + ' '.join(f'{c:>16}' for c in CONDS))
    for key in sorted({(p, g) for p, g, _ in by}):
        row = [fmt([r[metric] for r in by[(*key, c)]]) if by[(*key, c)] else '-' for c in CONDS]
        print(f'{key[0]:>4}/{key[1]:<5} ' + ' '.join(f'{x:>16}' for x in row))

print('\n== n spikes / n landmarks ==')
for key in sorted({(p, g) for p, g, _ in by}):
    print(key, {c: (by[(*key, c)][0]['n_spikes'], by[(*key, c)][0]['n_landmarks']) for c in CONDS if by[(*key, c)]})

# paired ratios vs (i), per seed, pooled over groups
print('\n== paired ratio to (i), geometric mean over group x seed [min-max] ==')
idx = {(r['group'], r['seed'], r['cond']): r for r in R}
for metric in [m for m in METS if m != 'niter']:
    for c in CONDS:
        rs = [idx[(g, s, c)][metric] / idx[(g, s, 'i')][metric] for (g, s, cc) in idx if cc == c and (g, s, 'i') in idx]
        if rs:
            rs = np.array(rs)
            print(f'{metric:>6} {c:>7}: {np.exp(np.log(rs).mean()):.3f} [{rs.min():.2f}-{rs.max():.2f}]  n={len(rs)}')
# totals summed over the 6 groups (mean over seeds)
print('\n== sum over groups, mean over seeds ==')
for metric in [m for m in METS if m != 'niter']:
    G = sorted({k[0] for k in idx}); SS = sorted({k[1] for k in idx})
    print(metric, {c: round(float(np.mean([sum(idx[(g, s, c)][metric] for g in G) for s in SS
                                           if all((g, s, c) in idx for g in G)])), 1) for c in CONDS})
print('\ncap hits:', sum(r['cap_hit'] for r in R), 'of', len(R), '; max niter', max(r['niter'] for r in R))
print('time per run: median', np.median([r['t_total'] for r in R]), 'max', max(r['t_total'] for r in R))
dec = defaultdict(lambda: defaultdict(list))
for r in R:
    for k in ['refractory:merge', 'bimodality:merge', 'connectivity:split', 'bimodality:split']:
        dec[r['cond']][k].append(r['decisions'].get(k, 0))
print('\n== splitter decisions, mean per run ==')
for c in CONDS:
    print(c, {k: round(float(np.mean(v)), 1) for k, v in dec[c].items()})
