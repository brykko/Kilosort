"""Floor for the relabelling between rounds: the same statistic (m1_convergence.assign_stats) between two seeds' sorts at the
same round. Seeds differ in clustering's random start and everything downstream of it, so this is an upper bound on what
re-clustering alone does to labels. Pairs s1-s2, H8 rounds 2 and 4, default rounds 2 and 3.

    python m1_seedlabels.py
"""
import json

import numpy as np

from common import OUT, ROUNDS, Link, Sort, sort_dir
from m1_convergence import assign_stats

out = {}
for setting in ('H8', 'default'):
    for ex in ('kobe', 'mec'):
        for r in (ROUNDS[setting][0], ROUNDS[setting][-1]):
            a, b = (Sort(sort_dir(ex, setting, s, r), ex, setting) for s in (1, 2))
            l, c = Link(a, b), Link(a, b, shift=1000)
            k = f'{ex}_{setting}_r{r}'
            out[k] = assign_stats(l, c, a.eligible(), l.successors())
            print(k, out[k]['elig'], flush=True)
(OUT / 'm1_seedlabels.json').write_text(json.dumps(out, indent=1))
