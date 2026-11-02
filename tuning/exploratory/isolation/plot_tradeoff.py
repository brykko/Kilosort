"""Figure of tradeoff.py's curves: correctly assigned against misassigned spikes, one line per seed."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

D = Path('C:/data/ks4_tuning/exploratory/isolation/tradeoff')
C = np.load(D / 'curves.npz')
COL = {'default': '#2a78d6', 'ks25': '#eb6834', 'H8': '#1baf7a'}
NAME = {'default': 'KS4 default', 'ks25': 'KS2.5 standard', 'H8': 'H8'}
INK, INK2, GRID, BG = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
ZOOM = len(sys.argv) > 1 and sys.argv[1] == 'zoom'

plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'text.color': INK, 'figure.facecolor': BG, 'axes.facecolor': BG})
fig, axs = plt.subplots(2, 2, figsize=(9, 7.2))
for r, ex in enumerate(('kobe', 'mec')):
    for k, m in enumerate(('plain', 'local')):
        ax = axs[r, k]
        ends = {}
        for g in ('default', 'ks25', 'H8'):
            keys = sorted(x for x in C.files if x.startswith(f'{ex}|{m}|{g}|'))
            for key in keys:
                mis, good = C[key].T / 1e6
                ax.plot(mis, good, color=COL[g], lw=1.5, solid_capstyle='round')
                ax.plot(mis[-1], good[-1], 'o', ms=6, color=COL[g], mec=BG, mew=1.5, zorder=3)
            ends[g] = np.mean([C[key][-1] / 1e6 for key in keys], axis=0)
        ks_end = ends['ks25'][0]
        ax.axvline(ks_end, color=INK2, lw=0.8, ls=(0, (3, 3)), zorder=0)
        if ZOOM:
            ax.set_xlim(0, 1.6 * ks_end)
            top = max(np.interp(1.6 * ks_end, *C[f'{ex}|{m}|{g}|0'].T / 1e6) for g in ('default', 'ks25', 'H8'))
            ax.set_ylim(0, top * 1.05)
        else:
            ax.set_xlim(left=0); ax.set_ylim(bottom=0)
        for g, (x, y) in ends.items():
            if not ZOOM or x <= 1.6 * ks_end:
                ax.annotate(NAME[g], (x, y), xytext=(6, -10 if g == 'ks25' else 4), textcoords='offset points',
                            fontsize=8, color=INK2)
        ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.set_title(f'{"Kobe" if ex == "kobe" else "MEC"}, {m} estimate', fontsize=10, loc='left')
        if r == 1:
            ax.set_xlabel('Misassigned spikes (millions)')
        if k == 0:
            ax.set_ylabel('Correctly assigned spikes (millions)')
handles = [plt.Line2D([], [], color=COL[g], lw=2, label=NAME[g]) for g in COL]
handles.append(plt.Line2D([], [], color=INK2, lw=0.8, ls=(0, (3, 3)), label="KS2.5's whole output"))
fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.995))
fig.suptitle('Units admitted cleanest first (cross-validated across halves); dot = whole output; one line per seed',
             y=0.945, fontsize=9, color=INK2)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(D / f'tradeoff{"_zoom" if ZOOM else ""}.png', dpi=150)
