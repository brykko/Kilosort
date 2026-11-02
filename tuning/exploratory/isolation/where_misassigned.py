"""Where each sorter's spikes and misassigned spikes sit: units shared with the other sorter (one-to-one
match, as matched_contam.py) or units only it has. Eligible units, seed 1, plain and local estimates."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from comod import unit_estimates  # noqa: E402
from matched_contam import SORTS, load, pairs  # noqa: E402

for ex, s in SORTS.items():
    L = {g: load(p, ex) for g, p in s.items()}
    for g1, g2 in (('default', 'ks25'), ('H8', 'ks25')):
        (u1, d1, T1), (u2, d2, T2) = L[g1], L[g2]
        pr = pairs(u1, d1, u2, d2)
        print(f'\n{ex}: {g1} vs {g2} ({len(pr)} shared cells)')
        for g, u, T, shared in ((g1, u1, T1, {a for a, _ in pr}), (g2, u2, T2, {b for _, b in pr})):
            est = {c: unit_estimates(t, T) for c, t in u.items()}
            for m in ('plain', 'local'):
                n = {c: e[0] for c, e in est.items()}
                bad = {c: e[0] * e[2][m] for c, e in est.items()}
                sh = [c for c in u if c in shared]
                un = [c for c in u if c not in shared]
                tot_n, tot_b = sum(n.values()), sum(bad.values())
                print(f'  {g:8s} {m:6s}: units {len(sh)} shared + {len(un)} own; spikes (M) {sum(n[c] for c in sh) / 1e6:5.2f} + '
                      f'{sum(n[c] for c in un) / 1e6:5.2f}; misassigned (M) {sum(bad[c] for c in sh) / 1e6:5.2f} + '
                      f'{sum(bad[c] for c in un) / 1e6:5.2f} ({sum(bad[c] for c in un) / tot_b:.0%} in own units); '
                      f'contamination shared {sum(bad[c] for c in sh) / sum(n[c] for c in sh):5.1%}, own '
                      f'{sum(bad[c] for c in un) / max(sum(n[c] for c in un), 1):5.1%}; correctly assigned (M) '
                      f'{(sum(n[c] for c in sh) - sum(bad[c] for c in sh)) / 1e6:5.2f} + {(sum(n[c] for c in un) - sum(bad[c] for c in un)) / 1e6:5.2f}')
