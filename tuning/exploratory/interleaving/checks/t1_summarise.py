"""Summary of T1: S (kobe), position S (MEC), S_track, eligible units and trade-off curves of the template-labelled variants
against the sorts' own graph labels. Curves as tradeoff_round3.py: upper-bound ranking, plain and local estimates, correct
spikes at 2% retained and at a fixed misassigned count (kobe: 0.42 M, KS2.5 standard's, as round3_results.md).

G(k) = round k's own sort. T(k), TM(k) = its template labels, without and with the final merge.
No-recluster share = (TM(3) - G(2)) / (G(3) - G(2)), per setting; both excerpts; seeds paired.

Control: the G variants must reproduce the sorts' known scores. Writes t1_summary.txt.

    python t1_summarise.py
"""
import json
import os
import sys
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
sys.dont_write_bytecode = True
WT = Path('C:/code/github/Kilosort/.claude/worktrees/ks4-round3/tuning')
sys.path[:0] = [str(WT), str(WT / 'exploratory/isolation'), str(WT / 'exploratory/iteration')]

import numpy as np  # noqa: E402
from cleaning import load_clean_units  # noqa: E402
from tradeoff import curve, good_at_mis, good_at_retained, unit_rows  # noqa: E402

C = Path('C:/data/ks4_tuning/exploratory/interleaving/checks')
D = C / 'variants'
O = Path('C:/data/ks4_tuning/exploratory/iteration')
RAW = {'kobe': Path('C:/data/ks4_tuning/26648_kobe_2020-08-21/search/raw_artefacts_car.npz'),
       'mec': Path('C:/data/ks4_tuning/25843_roger_2019-09-11/search/raw_artefacts_car.npz')}
ROUNDS = {'H8': (2, 3, 4), 'default': (2, 3)}
SEEDS = (1, 2, 3)
MIS = 0.42e6


def source(ex, g, k, s):
    pre = '' if g == 'H8' else 'default_'
    run = O / 'sorts' / f'{ex}_{pre}repeat_s{s}'
    return run if k == ROUNDS[g][-1] else run / f'round{k}'


def curves(sort, ex):
    by, _, iv = load_clean_units(sort, RAW[ex])
    rows = unit_rows(by, iv)
    out = {}
    for m in ('plain', 'local'):
        mis, good = curve(rows, m, 'upper')
        out[m] = (good_at_retained(mis, good, 0.02) / 1e6, good_at_mis(mis, good, MIS) / 1e6, good[-1] / 1e6, mis[-1] / 1e6)
    return out


def mec_scores():
    out = {}
    for f in (O / 'transfer_mec.json', O / 'transfer_mec_default.json', C / 'transfer_mec.json'):
        for w in json.loads(f.read_text())['winner']:
            out[str(Path(w['sort']))] = w
    return out


def score(ex, d, mec):
    if ex == 'kobe':
        e = json.loads((d / 'evaluation.json').read_text())
        return e['S'], e['S_track'], e['n_eligible']
    w = mec[str(d)]
    return w['S'], None, w['n_eligible']


if __name__ == '__main__':
    mec = mec_scores()
    L = []
    P = lambda s='': (L.append(s), print(s, flush=True))  # noqa: E731
    P('Controls (G variant through the same writer and scoring, against the saved sort):')
    for ex in RAW:
        a = score(ex, source(ex, 'H8', 3, 1), mec)
        b = score(ex, D / f'{ex}_H8_r3_s1_G', mec)
        P(f'  {ex}: saved {a}, variant {b}')
    R = {}
    for ex in RAW:
        for g, rr in ROUNDS.items():
            for k in rr:
                for s in SEEDS:
                    for m in ('G', 'T', 'TM'):
                        d = source(ex, g, k, s) if m == 'G' else D / f'{ex}_{g}_r{k}_s{s}_{m}'
                        S, St, el = score(ex, d, mec)
                        R[ex, g, k, s, m] = dict(S=S, St=St, el=el, c=curves(d, ex))
                    print(ex, g, k, s, 'done', flush=True)
    for ex in RAW:
        for g, rr in ROUNDS.items():
            P()
            P(f'=== {ex} {g}: S per seed (mean); eligible; correct at 2% retained, plain (M); local; correct at {MIS / 1e6:.2f} M misassigned, plain')
            for k in rr:
                for m in ('G', 'T', 'TM'):
                    x = [R[ex, g, k, s, m] for s in SEEDS]
                    S_ = [v['S'] for v in x]
                    el_ = ', '.join(str(v['el']) for v in x)
                    S_txt = ', '.join(f'{v:.1f}' for v in S_)
                    P(f"r{k} {m:2s}: S {S_txt} ({np.mean(S_):.1f}); eligible {el_}; "
                      f"@2% {np.mean([v['c']['plain'][0] for v in x]):.3f}; local {np.mean([v['c']['local'][0] for v in x]):.3f}; @mis {np.mean([v['c']['plain'][1] for v in x]):.3f}")
            P('Paired differences (per seed; mean):')
            def diff(lab, a, b, f, u=''):  # noqa: E301
                d = [f(R[(ex, g, a[0], s, a[1])]) - f(R[(ex, g, b[0], s, b[1])]) for s in SEEDS]
                fmt = '{:+.3f}' if u else '{:+.1f}'
                P(f"  {lab:38s}: {', '.join(fmt.format(v) for v in d)}; mean {fmt.format(np.mean(d))}{u}")
                return np.array(d)
            for name, f, u in (('S', lambda v: v['S'], ''), ('correct @2% plain', lambda v: v['c']['plain'][0], ' M'),
                               ('correct @2% local', lambda v: v['c']['local'][0], ' M'), ('correct @mis plain', lambda v: v['c']['plain'][1], ' M')):
                full = diff(f'{name}: G3 - G2 (repeat)', (3, 'G'), (2, 'G'), f, u)
                nr = diff(f'{name}: TM3 - G2 (no recluster)', (3, 'TM'), (2, 'G'), f, u)
                diff(f'{name}: T3 - G2', (3, 'T'), (2, 'G'), f, u)
                diff(f'{name}: TM3 - TM2 (template labels)', (3, 'TM'), (2, 'TM'), f, u)
                if 4 in rr:
                    diff(f'{name}: TM4 - G3 (no recluster at r4)', (4, 'TM'), (3, 'G'), f, u)
                P(f"  {name}: no-recluster share {nr.mean() / full.mean():.0%} (per seed {', '.join(f'{a / b:.0%}' for a, b in zip(nr, full))})")
    if 'kobe' in RAW:
        for g in ROUNDS:
            P()
            P(f'kobe {g}, S_track: ' + '; '.join(f"{m}{k} {np.mean([R['kobe', g, k, s, m]['St'] for s in SEEDS]):.1f}" for k in ROUNDS[g] for m in ('G', 'TM')))
    (C / 't1_summary.txt').write_text('\n'.join(L) + '\n')
