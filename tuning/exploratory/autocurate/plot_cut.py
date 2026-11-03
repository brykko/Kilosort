"""Figure of apply_cut.py's curves (seed 1): KS2.5 standard and standard_low against H8 before and after the
set_cutoff-like cut (widths 2 and 4), zoomed to the strict end (up to 1.6x KS2.5 standard's misassigned)."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

D = Path('C:/data/ks4_tuning/exploratory/autocurate')
LINES = {'KS2.5 standard': ('#eb6834', '-'), 'KS2.5 standard_low': ('#eda100', '-'), 'H8 uncut': ('#1baf7a', '-'),
         'H8 cut w=2': ('#1baf7a', (0, (4, 2))), 'H8 cut w=4': ('#1baf7a', (0, (1, 1.5)))}
NAME = {'H8 uncut': 'H8', 'H8 cut w=2': 'H8 + cut, width 2', 'H8 cut w=4': 'H8 + cut, width 4'}
INK, INK2, GRID, BG = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'

plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'text.color': INK, 'figure.facecolor': BG, 'axes.facecolor': BG})
fig, axs = plt.subplots(2, 2, figsize=(9, 7.2))
for r, ex in enumerate(('kobe', 'mec')):
    for k, m in enumerate(('plain', 'local')):
        ax = axs[r, k]
        C = np.load(D / f'curves_{ex}_{m}.npz')
        xmax = 1.6 * C['KS2.5 standard'][-1, 0] / 1e6
        top = 0
        for g, (col, ls) in LINES.items():
            mis, good = C[g].T / 1e6
            ax.plot(mis, good, color=col, ls=ls, lw=1.6)
            if mis[-1] <= xmax:
                ax.plot(mis[-1], good[-1], 'o', ms=6, color=col, mec=BG, mew=1.5, zorder=3)
            top = max(top, np.interp(xmax, mis, good))
        ax.axvline(C['KS2.5 standard'][-1, 0] / 1e6, color=INK2, lw=0.8, ls=(0, (3, 3)), zorder=0)
        ax.set_xlim(0, xmax); ax.set_ylim(0, top * 1.05)
        ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.set_title(f'{"Kobe" if ex == "kobe" else "MEC"}, {m} estimate', fontsize=10, loc='left')
        if r == 1:
            ax.set_xlabel('Misassigned spikes (millions)')
        if k == 0:
            ax.set_ylabel('Correctly assigned spikes (millions)')
handles = [plt.Line2D([], [], color=c, ls=ls, lw=2, label=NAME.get(g, g)) for g, (c, ls) in LINES.items()]
handles.append(plt.Line2D([], [], color=INK2, lw=0.8, ls=(0, (3, 3)), label="KS2.5 standard's whole output"))
fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.995))
fig.suptitle('Units admitted cleanest first (cross-validated across halves); dot = whole output; seed 1',
             y=0.925, fontsize=9, color=INK2)
fig.tight_layout(rect=(0, 0, 1, 0.91))
fig.savefig(D / 'cut_tradeoff.png', dpi=150)
