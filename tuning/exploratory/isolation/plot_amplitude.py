"""Figures of amplitude.py's outputs: spike-assignment correctness against unit amplitude.

amp_scatter_<est>.png  contamination of each unit against its amplitude, one panel per sorter and excerpt,
                       with the spike-weighted mean per amplitude bin;
amp_bins_<est>.png     correctly assigned and misassigned spikes per amplitude bin, and % misassigned;
amp_cumulative_<est>.png  correctly assigned and misassigned spikes in units at or above each amplitude.
<est> is the contamination estimate, plain or local (comod.py).
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

D = Path('C:/data/ks4_tuning/exploratory/isolation/amplitude')
GROUPS = ('default', 'ks25', 'H8')
COL = {'default': '#2a78d6', 'ks25': '#eb6834', 'H8': '#1baf7a'}
NAME = {'default': 'KS4 default', 'ks25': 'KS2.5 standard', 'H8': 'H8'}
EXNAME = {'kobe': 'Kobe', 'mec': 'MEC'}
INK, INK2, GRID, BG = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
EDGES = 10 * 2 ** (np.arange(0, 15) / 2)          # 10 to 1280 uV, half-octave bins
MID = np.sqrt(EDGES[:-1] * EDGES[1:])

plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'text.color': INK, 'figure.facecolor': BG, 'axes.facecolor': BG})
DATA = {ex: np.load(D / f'{ex}.npz') for ex in EXNAME}


def unit(ex, g, est):
    z = DATA[ex]
    return z[f'{g}_amp'], z[f'{g}_n'].astype(float), z[f'{g}_c_{est}']


def style(ax):
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.set_xscale('log')
    ax.set_xlim(EDGES[0], EDGES[-1])
    ax.set_xticks([10, 20, 50, 100, 200, 500, 1000])
    ax.set_xticklabels(['10', '20', '50', '100', '200', '500', '1000'])


def legend(fig, y=0.995):
    h = [plt.Line2D([], [], color=COL[g], lw=2, marker='o', ms=5, label=NAME[g]) for g in GROUPS]
    fig.legend(handles=h, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, y))


def binned(amp, n, c):
    b = np.digitize(amp, EDGES) - 1
    ok = (b >= 0) & (b < MID.size)
    good = np.bincount(b[ok], (n * (1 - c))[ok], MID.size)
    bad = np.bincount(b[ok], (n * c)[ok], MID.size)
    return good, bad


for est in ('plain', 'local'):
    # 1. scatter
    fig, axs = plt.subplots(2, 3, figsize=(10, 6.4), sharex=True, sharey=True)
    for r, ex in enumerate(EXNAME):
        for k, g in enumerate(GROUPS):
            ax = axs[r, k]
            amp, n, c = unit(ex, g, est)
            ax.scatter(amp, 100 * c, s=4 + 40 * n / n.max(), color=COL[g], alpha=0.35, lw=0)
            good, bad = binned(amp, n, c)
            tot = good + bad
            keep = tot > 0
            ax.plot(MID[keep], 100 * bad[keep] / tot[keep], color=INK, lw=1.8, marker='o', ms=4, mfc=BG, mew=1.2)
            style(ax)
            ax.set_ylim(-2, 102)
            ax.set_title(f'{EXNAME[ex]}, {NAME[g]} ({amp.size} units)', fontsize=9.5, loc='left')
            if r == 1:
                ax.set_xlabel('Unit amplitude (µV, peak-to-peak)')
            if k == 0:
                ax.set_ylabel(f'Misassigned spikes (%, {est} estimate)')
    h = [plt.Line2D([], [], color=INK2, lw=0, marker='o', ms=5, alpha=0.6, label='Unit (area ∝ spikes)'),
         plt.Line2D([], [], color=INK, lw=1.8, marker='o', ms=4, mfc=BG, label='Spike-weighted mean per half-octave bin')]
    fig.legend(handles=h, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(D / f'amp_scatter_{est}.png', dpi=150)
    plt.close(fig)

    # 2. per bin
    fig, axs = plt.subplots(2, 3, figsize=(10, 6.4), sharex=True)
    for r, ex in enumerate(EXNAME):
        top = 0
        for g in GROUPS:
            good, bad = binned(*unit(ex, g, est))
            tot = good + bad
            keep = tot > 0
            kw = dict(color=COL[g], lw=1.8, marker='o', ms=4, mec=BG, mew=1)
            axs[r, 0].plot(MID[keep], good[keep] / 1e6, **kw)
            axs[r, 1].plot(MID[keep], bad[keep] / 1e6, **kw)
            axs[r, 2].plot(MID[keep], 100 * bad[keep] / tot[keep], **kw)
            top = max(top, good.max(), bad.max())
        for k, lab in enumerate(('Correctly assigned spikes (M)', 'Misassigned spikes (M)', 'Misassigned (%)')):
            ax = axs[r, k]
            style(ax)
            ax.set_ylim(0, 1.05 * top / 1e6 if k < 2 else 100)
            ax.set_title(f'{EXNAME[ex]}: {lab[0].lower() + lab[1:]}', fontsize=9.5, loc='left')
            if r == 1:
                ax.set_xlabel('Unit amplitude (µV, peak-to-peak)')
    legend(fig)
    fig.suptitle(f'Per half-octave amplitude bin; {est} contamination estimate', y=0.955, fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(D / f'amp_bins_{est}.png', dpi=150)
    plt.close(fig)

    # 3. cumulative from the largest units down
    fig, axs = plt.subplots(2, 2, figsize=(9, 6.4), sharex=True)
    for r, ex in enumerate(EXNAME):
        for g in GROUPS:
            amp, n, c = unit(ex, g, est)
            o = np.argsort(-amp)
            a = amp[o]
            for k, v in enumerate((np.cumsum((n * (1 - c))[o]), np.cumsum((n * c)[o]))):
                axs[r, k].plot(a, v / 1e6, color=COL[g], lw=1.8)
        for k, lab in enumerate(('Correctly assigned spikes (M)', 'Misassigned spikes (M)')):
            ax = axs[r, k]
            style(ax)
            ax.invert_xaxis()
            ax.set_ylim(bottom=0)
            ax.set_title(f'{EXNAME[ex]}: {lab[0].lower() + lab[1:]}', fontsize=9.5, loc='left')
            if r == 1:
                ax.set_xlabel('Units at or above this amplitude (µV)')
    legend(fig)
    fig.suptitle(f'Cumulative from the largest units down; {est} contamination estimate', y=0.955, fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(D / f'amp_cumulative_{est}.png', dpi=150)
    plt.close(fig)
