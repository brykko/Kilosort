"""Distance profile of tuning twins, by pair type, per sorter group (reads finished npz files only)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyse import DBINS, load, twin_stat  # noqa: E402
from unique_tuning import FLOOR, MVL_MIN, SI_MIN  # noqa: E402

OUT = Path('C:/data/ks4_tuning/exploratory/uniqueness')
ex, groups = sys.argv[1], sys.argv[2:]
for x in (0.9, 0.95):
    print(f'\n{ex}: fraction of pairs with twin statistic >= {x}, by distance (um)')
    print(f"{'':26s}" + ' '.join(f'{f"{a}-{b}":>9s}' for a, b in zip(DBINS[:-1], DBINS[1:])) + '   excess<=100  per-unit false-twin prob (far rate, mean neighbours<=100)')
    for g in groups:
        files = [f for f in sorted((OUT / ex).glob(f'{g}_[sr][0-9].npz'))]
        for kind in ('any strong', 'both strong', 'both reliable'):
            hit, tot = np.zeros(len(DBINS) - 1), np.zeros(len(DBINS) - 1)
            nb = []
            for f in files:
                tab, pairs = load(f)
                strong = (tab['mvl_c'] >= MVL_MIN) | (tab['si_c'] >= SI_MIN)
                t = twin_stat(tab, pairs)
                i, j = pairs['i'], pairs['j']
                sel = ~np.isnan(t)
                if kind == 'both strong':
                    sel &= strong[i] & strong[j]
                if kind == 'both reliable':
                    rel = np.minimum(tab['rel_hd'], tab['rel_pos'])
                    rel = np.where((tab['mvl_c'] >= MVL_MIN) & (tab['si_c'] < SI_MIN), tab['rel_hd'],
                                   np.where((tab['si_c'] >= SI_MIN) & (tab['mvl_c'] < MVL_MIN), tab['rel_pos'], rel))
                    sel &= strong[i] & strong[j] & (rel[i] > FLOOR) & (rel[j] > FLOOR)
                b = np.digitize(pairs['d'], DBINS) - 1
                for k in range(len(DBINS) - 1):
                    s = sel & (b == k)
                    tot[k] += s.sum()
                    hit[k] += np.sum(t[s] >= x)
                near = sel & (pairs['d'] <= 100)
                cnt = np.bincount(np.r_[i[near], j[near]], minlength=strong.size)
                nb.append(cnt[strong].mean())
            rate = hit / np.maximum(tot, 1)
            near_rate = hit[:3].sum() / max(tot[:3].sum(), 1)
            far_rate = hit[-1] / max(tot[-1], 1)
            p_false = 1 - (1 - far_rate) ** np.mean(nb)
            print(f'{g:8s} {kind:16s} ' + ' '.join(f'{r:9.2%}' for r in rate)
                  + f'   {near_rate - far_rate:+8.2%}   {p_false:6.1%} ({far_rate:.2%}, {np.mean(nb):.0f})')
