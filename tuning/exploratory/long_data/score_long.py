"""Scores of the long-data round, all in each excerpt's 20-min window, seed 1, no hybrid data.

Sorts: KS4 default and H8, KS2.5 standard and standard_low, each of the 20-min excerpt and of the 2-hour
segment that contains it; KS4 also after the set_cutoff-like cut (default w=4, H8 w=2 and 4), applied to the
whole sort before windowing (window.py). Every sort is first made a window folder (window.py; for a 20-min
sort the window is the whole sort), so all are cleaned and scored the same way:

- kobe: evaluate.py (S, S_track; no hybrid truth);
- MEC: transfer.py at alpha 0.0625 s against the three default MEC sorts (position S, guards);
- both: bayes_decode.py and corr_decode.py on the uncut sorts.

Steps whose output exists are skipped, so the script can be rerun after a failure.

    python score_long.py [--ex kobe|mec]
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TUNING = HERE.parents[1]
PY = sys.executable
K = Path('C:/data/ks4_tuning/26648_kobe_2020-08-21')
M = Path('C:/data/ks4_tuning/25843_roger_2019-09-11')
O = Path('C:/data/ks4_tuning/exploratory/long_data')
S, W = O / 'sorts', O / 'windows'
AC = Path('C:/data/ks4_tuning/exploratory/autocurate/sorts')
SORTERS = ('default', 'H8', 'ks25_standard', 'ks25_standard_low')
CUTS = {'default': (4,), 'H8': (2, 4)}
EX = {
    'kobe': {'t_exc': 1668.0, 'base': K, 'tracking': K / 'search/tracking_1668-2868s.npz',
             '20min': ({'default': S / 'kobe_default_20min_s1', 'H8': S / 'kobe_H8_20min_s1',
                        'ks25_standard': S / 'kobe_ks25_standard_20min_s1',
                        'ks25_standard_low': S / 'kobe_ks25_standard_low_20min_s1'}, 1668.0),
             '2h': ({g: S / f'kobe_{g}_s1' for g in SORTERS}, 354.0)},
    'mec': {'t_exc': 13575.0, 'base': M, 'tracking': M / 'search/tracking_13575-14775s.npz',
            '20min': ({'default': AC / 'mec_default_s1', 'H8': AC / 'mec_H8_s1',
                       'ks25_standard': M / 'runs/ks25_standard_r1', 'ks25_standard_low': M / 'runs/ks25_standard_low_r1'},
                      13575.0),
            '2h': ({g: S / f'mec_{g}_s1' for g in SORTERS}, 7575.0)},
}
DUR = 1200.0


def run(args, log):
    print(f'  {Path(log).name} ... ', end='', flush=True)
    with open(log, 'w') as f:
        r = subprocess.run([str(a) for a in args], cwd=TUNING, stdout=f, stderr=subprocess.STDOUT)
    print('ok' if r.returncode == 0 else f'FAILED (see {log})', flush=True)
    return r.returncode == 0


def windows(ex):
    """{name: window folder}, made if missing: '{ex}_{length}_{sorter}[_cutW]'."""
    out = {}
    for length in ('20min', '2h'):
        sorts, t_seg = EX[ex][length]
        for g, src in sorts.items():
            for w in (None,) + CUTS.get(g, ()):
                name = f'{ex}_{length}_{g}' + (f'_cut{w}' if w else '')
                dst = W / name
                if not (dst / 'window.json').exists():
                    if not src.exists():
                        print(f'  {name}: sort {src} missing', flush=True)
                        continue
                    if dst.exists():
                        raise SystemExit(f'{dst} exists without window.json: remove it and rerun')
                    args = [PY, HERE / 'window.py', '--sort', src, '--out', dst, '--t-seg', t_seg,
                            '--t-exc', EX[ex]['t_exc'], '--duration', DUR] + (['--cut', w] if w else [])
                    if not run(args, W / f'{name}.window.log'):
                        continue
                out[name] = dst
    return out


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ex', choices=tuple(EX), help='one excerpt only')
    W.mkdir(parents=True, exist_ok=True)
    for ex in [p.parse_args().ex] if p.parse_args().ex else EX:
        print(f'=== {ex}: windows', flush=True)
        win = windows(ex)
        b = EX[ex]['base']
        art = b / 'search/raw_artefacts_car.npz'
        if ex == 'kobe':
            print('=== kobe: evaluate', flush=True)
            for name, d in win.items():
                if not (d / 'evaluation.json').exists():
                    run([PY, 'evaluate.py', '--sort', d, '--reference', K / 'runs/reference_seed1000/theta_ref.npz',
                         '--t0', EX[ex]['t_exc'], '--duration', DUR, '--raw-artefacts', art], d / 'evaluate.log')
        else:
            out = W / 'transfer_mec.json'
            done = {Path(x['sort']).name for x in json.loads(out.read_text())['winner']} if out.exists() else set()
            if set(win) - done:
                print('=== mec: transfer', flush=True)
                run([PY, 'transfer.py', '--default', *[M / f'runs/default_s{k}' for k in (1, 2, 3)],
                     '--winner', *win.values(), '--alpha', 0.0625, '--tracking', EX[ex]['tracking'],
                     '--raw-artefacts', art, '--t0', EX[ex]['t_exc'], '--duration', DUR, '--out', out],
                    W / 'transfer_mec.log')
        uncut = {n: d for n, d in win.items() if '_cut' not in n}
        for dec in ('bayes', 'corr'):
            out = W / f'{dec}_decode_{ex}.json'
            if out.exists() and set(json.loads(out.read_text())['groups']) >= set(uncut):
                continue
            print(f'=== {ex}: {dec}_decode', flush=True)
            groups = [a for n, d in uncut.items() for a in ('--group', n, d)]
            run([PY, f'{dec}_decode.py', *groups, '--tracking', EX[ex]['tracking'], '--raw-artefacts', art,
                 '--t0', EX[ex]['t_exc'], '--duration', DUR, '--out', out], W / f'{dec}_decode_{ex}.log')
    print('=== done', flush=True)
