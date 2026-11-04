"""Tables and figures of score_long.py: each sorter on the 20-min excerpt against the 2-hour segment, scored in
the same 20-min window (seed 1, no hybrid data).

Per window: units with spikes and spikes after cleaning; eligible units and S (kobe: also S_track; MEC:
position S from transfer.py, with its guards); correctly assigned and misassigned spikes of the whole output
and correctly assigned spikes at 2% and 5% retained contamination (isolation/tradeoff.py's curves, units
admitted cleanest first, cross-validated over the halves, plain and local estimates); Bayesian and
population-vector decoding errors (uncut sorts). Writes summary_long.txt and three figures (S; curves of the
uncut sorts; curves of KS4 after the cut at w=4) to
C:\\data\\ks4_tuning\\exploratory\\long_data\\windows\\.

    python summarise_long.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'isolation'))
sys.path.insert(0, str(HERE.parents[1]))
from cleaning import load_clean_units  # noqa: E402
from tradeoff import curve, good_at_retained, unit_rows  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

W = Path('C:/data/ks4_tuning/exploratory/long_data/windows')
BASE = {'kobe': Path('C:/data/ks4_tuning/26648_kobe_2020-08-21'), 'mec': Path('C:/data/ks4_tuning/25843_roger_2019-09-11')}
VARIANTS = ('default', 'default_cut4', 'H8', 'H8_cut2', 'H8_cut4', 'ks25_standard', 'ks25_standard_low')
LABEL = {'default': 'KS4 default', 'default_cut4': 'KS4 default + cut w=4', 'H8': 'H8', 'H8_cut2': 'H8 + cut w=2',
         'H8_cut4': 'H8 + cut w=4', 'ks25_standard': 'KS2.5 standard', 'ks25_standard_low': 'KS2.5 standard_low'}
COLOR = {'default': '#3a6fc4', 'H8': '#1baf7a', 'ks25_standard': '#eb6834', 'ks25_standard_low': '#eda100'}
LENGTHS = ('20min', '2h')
LEN_LABEL = {'20min': '20-min sort', '2h': '2-hour sort'}
INK, INK2, GRID, BG = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'text.color': INK, 'figure.facecolor': BG, 'axes.facecolor': BG})


def color(v):
    return COLOR[v.split('_cut')[0]]


def scores(ex):
    """{window name: {'S', 'S_track'?, 'n_eligible', 'guards'?}}"""
    if ex == 'kobe':
        out = {}
        for d in W.glob('kobe_*'):
            if (d / 'evaluation.json').exists():
                e = json.loads((d / 'evaluation.json').read_text())
                out[d.name] = {'S': e['S'], 'S_track': e['S_track'], 'n_eligible': e['n_eligible']}
        return out
    t = json.loads((W / 'transfer_mec.json').read_text()) if (W / 'transfer_mec.json').exists() else {'winner': []}
    return {Path(w['sort']).name: {'S': float(w['S']), 'n_eligible': int(w['n_eligible']),
                                   'guards': 'pass' if all(v[2] for v in w['guards'].values()) else
                                   'FAIL: ' + ', '.join(k for k, v in w['guards'].items() if not v[2])}
            for w in t['winner']}


def isolation(ex, name):
    by, _, iv = load_clean_units(W / name, BASE[ex] / 'search/raw_artefacts_car.npz')
    rows = unit_rows(by, iv)
    out = {'units': sum(t.size > 0 for t in by.values()), 'spikes': sum(t.size for t in by.values())}
    for m in ('plain', 'local'):
        mis, good = curve(rows, m, 'upper')
        out[m] = {'curve': np.c_[mis, good], 'good': good[-1], 'mis': mis[-1],
                  '2%': good_at_retained(mis, good, 0.02), '5%': good_at_retained(mis, good, 0.05)}
    return out


def decoding(ex):
    """{window name: {'bayes': median_error, 'raw': ..., 'z': ...}}; median_error[var][window_s]."""
    out = {}
    for dec in ('bayes', 'corr'):
        f = W / f'{dec}_decode_{ex}.json'
        if not f.exists():
            continue
        for name, runs in json.loads(f.read_text())['groups'].items():
            me = runs[0]['median_error']
            out.setdefault(name, {}).update({'bayes': me} if dec == 'bayes' else me)
    return out


def table(ex, sc, iso, dec, f):
    p = lambda *a: print(*a, file=f)  # noqa: E731
    p(f'\n===== {ex} (window {"1668-2868" if ex == "kobe" else "13575-14775"} s; seed 1)')
    p('S in bits/s' + (' (S_track in brackets)' if ex == 'kobe' else ' (position S from transfer.py; guards against the default MEC sorts)')
      + '; spikes in millions after cleaning; correct = correctly assigned, mis = misassigned (whole output),')
    p('@2%/@5% = correctly assigned at that retained contamination; plain estimate, then local.')
    p(f'  {"":24s} {"length":>6s} {"units":>6s} {"elig.":>6s} {"S":>16s} {"spikes":>7s}   '
      f'{"plain: correct  mis  @2%  @5%":>30s}   {"local: correct  mis  @2%  @5%":>30s}' + ('  guards' if ex == 'mec' else ''))
    for v in VARIANTS:
        for length in LENGTHS:
            n = f'{ex}_{length}_{v}'
            if n not in iso:
                continue
            s, i = sc.get(n, {}), iso[n]
            S = f'{s["S"]:7.1f}' + (f' ({s["S_track"]:6.1f})' if ex == 'kobe' else '') if s else 'n/a'
            q = lambda m: f'{i[m]["good"] / 1e6:6.2f} {i[m]["mis"] / 1e6:5.2f} {i[m]["2%"] / 1e6:5.2f} {i[m]["5%"] / 1e6:5.2f}'  # noqa: E731
            p(f'  {LABEL[v]:24s} {length:>6s} {i["units"]:6d} {s.get("n_eligible", 0):6d} {S:>16s} {i["spikes"] / 1e6:7.2f}   '
              f'{"":7s}{q("plain")}   {"":7s}{q("local")}' + (f'  {s.get("guards", "")}' if ex == 'mec' else ''))
    p('\n  2-hour sort against 20-min sort, same window:')
    for v in VARIANTS:
        a, b = f'{ex}_20min_{v}', f'{ex}_2h_{v}'
        if a in sc and b in sc and a in iso and b in iso:
            p(f'  {LABEL[v]:24s} S {sc[b]["S"] - sc[a]["S"]:+7.1f} ({(sc[b]["S"] / sc[a]["S"] - 1):+6.1%}), eligible '
              f'{sc[b]["n_eligible"] - sc[a]["n_eligible"]:+5d}, spikes {iso[b]["spikes"] / iso[a]["spikes"] - 1:+6.1%}, '
              f'correct (plain) {iso[b]["plain"]["good"] / iso[a]["plain"]["good"] - 1:+6.1%}, misassigned (plain) '
              f'{iso[b]["plain"]["mis"] / iso[a]["plain"]["mis"] - 1:+6.1%}')
    if dec:
        p('\n  Decoding, median error on the held-out half (uncut sorts): head direction (deg) and position (cm),')
        p('  windows 20 / 100 / 500 ms; Bayesian, then population-vector correlation raw and z-scored.')
        for v in VARIANTS:
            for length in LENGTHS:
                n = f'{ex}_{length}_{v}'
                if n in dec:
                    d = dec[n]
                    cell = lambda k, var: '/'.join(f'{d[k][var][w]:5.1f}' for w in ('0.02', '0.1', '0.5')) if k in d else 'n/a'  # noqa: E731
                    p(f'  {LABEL[v]:24s} {length:>6s}  HD ' + '  '.join(f'{k} {cell(k, "hd")}' for k in ('bayes', 'raw', 'z'))
                      + '   pos ' + '  '.join(f'{k} {cell(k, "pos")}' for k in ('bayes', 'raw', 'z')))


def fig_S(sc_all):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, ex in zip(axs, ('kobe', 'mec')):
        sc = sc_all[ex]
        ys = []
        for k, v in enumerate(VARIANTS):
            y = len(VARIANTS) - 1 - k
            pts = [(sc[f'{ex}_{l}_{v}']['S'] if f'{ex}_{l}_{v}' in sc else np.nan) for l in LENGTHS]
            ax.plot(pts, [y, y], color=GRID, lw=2, zorder=1)
            ax.plot(pts[0], y, 'o', ms=7, mfc=BG, mec=color(v), mew=1.6, zorder=2)
            ax.plot(pts[1], y, 'o', ms=7, color=color(v), mec=BG, mew=1.2, zorder=3)
            ys.append(y)
        ax.set_yticks(ys)
        ax.set_yticklabels([LABEL[v] for v in VARIANTS] if ex == 'kobe' else [])
        ax.grid(axis='x', color=GRID, lw=0.6); ax.set_axisbelow(True)
        for sp in ('top', 'right', 'left'):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis='y', length=0)
        ax.set_title('Kobe: S (heading)' if ex == 'kobe' else 'MEC: position S', loc='left', fontsize=10)
        ax.set_xlabel('bits/s, in the 20-min window')
    h = [plt.Line2D([], [], ls='', marker='o', ms=7, mfc=BG, mec=INK2, mew=1.6, label='20-min sort'),
         plt.Line2D([], [], ls='', marker='o', ms=7, color=INK2, label='2-hour sort')]
    fig.legend(handles=h, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(W / 'long_S.png', dpi=150)


def fig_curves(iso_all, variants, fname, title):
    fig, axs = plt.subplots(2, 2, figsize=(9, 7.2))
    styles = {'20min': (0, (4, 2)), '2h': '-'}
    for r, ex in enumerate(('kobe', 'mec')):
        iso = iso_all[ex]
        for k, m in enumerate(('plain', 'local')):
            ax = axs[r, k]
            ref = iso.get(f'{ex}_20min_ks25_standard')
            xmax = 1.6 * ref[m]['mis'] / 1e6 if ref else None
            top = 0
            for v, lw in variants:
                for length in LENGTHS:
                    n = f'{ex}_{length}_{v}'
                    if n not in iso:
                        continue
                    mis, good = iso[n][m]['curve'].T / 1e6
                    ax.plot(mis, good, color=color(v), ls=styles[length], lw=lw)
                    if xmax is None or mis[-1] <= xmax:
                        ax.plot(mis[-1], good[-1], 'o', ms=5, color=color(v), mec=BG, mew=1.2, zorder=3)
                    top = max(top, np.interp(xmax, mis, good) if xmax else good[-1])
            if xmax:
                ax.set_xlim(0, xmax)
            ax.set_ylim(0, top * 1.05)
            ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
            ax.set_title(f'{"Kobe" if ex == "kobe" else "MEC"}, {m} estimate', fontsize=10, loc='left')
            if r == 1:
                ax.set_xlabel('Misassigned spikes (millions)')
            if k == 0:
                ax.set_ylabel('Correctly assigned spikes (millions)')
    h = [plt.Line2D([], [], color=color(v), lw=2.5 if lw > 1.6 else 1.6, label=LABEL[v]) for v, lw in variants]
    h += [plt.Line2D([], [], color=INK2, ls=styles[l], lw=1.6, label=LEN_LABEL[l]) for l in LENGTHS]
    fig.legend(handles=h, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(title, y=0.905, fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(W / fname, dpi=150)


if __name__ == '__main__':
    sc_all, iso_all = {}, {}
    with open(W / 'summary_long.txt', 'w') as f:
        for ex in ('kobe', 'mec'):
            sc = scores(ex)
            iso = {d.name: isolation(ex, d.name) for d in sorted(W.glob(f'{ex}_*'))
                   if d.is_dir() and (d / 'cleaning.npz').exists()}
            sc_all[ex], iso_all[ex] = sc, iso
            table(ex, sc, iso, decoding(ex), f)
    print((W / 'summary_long.txt').read_text())
    fig_S(sc_all)
    fig_curves(iso_all, [(v, 1.6) for v in ('default', 'H8', 'ks25_standard', 'ks25_standard_low')], 'long_tradeoff.png',
               'Units admitted cleanest first (cross-validated across halves), in the 20-min window; dot = whole output; seed 1')
    fig_curves(iso_all, [(v, 1.6) for v in ('default_cut4', 'H8_cut4', 'ks25_standard', 'ks25_standard_low')],
               'long_tradeoff_cut.png', 'KS4 after the set_cutoff-like cut (w=4; on the 2-hour sort, cut on all 2 hours) '
               'against KS2.5; same window; dot = whole output')
    print('figures written', flush=True)
