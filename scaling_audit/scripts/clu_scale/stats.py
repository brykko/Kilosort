"""Paired log-ratio contrasts with a group-cluster bootstrap (resample groups, then seeds)."""
import json, sys
from pathlib import Path
import numpy as np

OUT = Path(__file__).parent
fn = sys.argv[1] if len(sys.argv) > 1 else 'results.jsonl'
R = [json.loads(l) for l in open(OUT / fn)]
idx = {(r['group'], r['seed'], r['cond']): r for r in R}
G = sorted({r['group'] for r in R}); SS = sorted({r['seed'] for r in R})
rng = np.random.default_rng(0)
mets = [m for m in ['leaves', 'final', 'final_nometa'] if m in R[0]]
pairs = [('ii', 'i', 'duration 20 vs 10 min @nskip20'), ('i', 'v', 'duration 10 vs 5 min @nskip20'),
         ('iii', 'i', 'nskip 10 vs 20 @10 min'), ('iii', 'ii', 'equivalence: (iii) vs (ii)'),
         ('iv', 'i', '20 min capped vs 10 min'), ('ii', 'iv', '20 min uncapped vs capped'),
         ('L10', 'i', 'last 10 vs first 10'), ('ii', 'L10', '20 min vs last 10'),
         ('L10s10', 'L10', 'nskip 10 vs 20 @last 10')]
for m in mets:
    print(f'\n== {m}: geometric-mean ratio A/B, 95% group-bootstrap CI, per-group means ==')
    for a, b, lab in pairs:
        L = np.full((len(G), len(SS)), np.nan)
        for gi, g in enumerate(G):
            for si, s in enumerate(SS):
                if (g, s, a) in idx and (g, s, b) in idx:
                    L[gi, si] = np.log(idx[(g, s, a)][m] / idx[(g, s, b)][m])
        if np.isnan(L).all():
            continue
        est = np.nanmean(L)
        boots = []
        for _ in range(4000):
            gs = rng.integers(0, len(G), len(G))
            boots.append(np.nanmean(L[gs][:, rng.integers(0, len(SS), len(SS))]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        pg = ' '.join(f'{np.exp(np.nanmean(L[gi])):.2f}' for gi in range(len(G)))
        print(f'{a:>6}/{b:<6} {np.exp(est):.3f} [{np.exp(lo):.3f}, {np.exp(hi):.3f}]  per-group {pg}   ({lab})')
print('groups (order):', G)
