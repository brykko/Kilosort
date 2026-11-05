"""Scores t1_variants.py's folders with the harness's tools, as score_round3.py scores the rounds: kobe evaluate.py (S,
S_track, no hybrid truth); MEC cleaning, then transfer.py at alpha 0.0625 against the three default sorts.
Steps whose output exists are skipped. Run from the worktree's tuning folder.

    python t1_score.py [n_workers]
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.dont_write_bytecode = True
WT = Path('C:/code/github/Kilosort/.claude/worktrees/ks4-round3/tuning')
PY = sys.executable
D = Path('C:/data/ks4_tuning/exploratory/interleaving/checks/variants')
K = Path('C:/data/ks4_tuning/26648_kobe_2020-08-21')
M = Path('C:/data/ks4_tuning/25843_roger_2019-09-11')
ENV = {**os.environ, 'OMP_NUM_THREADS': '3', 'MKL_NUM_THREADS': '3', 'OPENBLAS_NUM_THREADS': '3', 'CUDA_VISIBLE_DEVICES': '-1',
       'PYTHONDONTWRITEBYTECODE': '1'}


def run(args, log):
    with open(log, 'w') as f:
        r = subprocess.run([str(a) for a in args], cwd=WT, stdout=f, stderr=subprocess.STDOUT, env=ENV)
    print(f"{Path(log).parent.name}: {'ok' if r.returncode == 0 else 'FAILED'}", flush=True)
    return r.returncode == 0


def kobe(d):
    if (d / 'evaluation.json').exists():
        return True
    return run([PY, 'evaluate.py', '--sort', d, '--reference', K / 'runs/reference_seed1000/theta_ref.npz', '--t0', 1668.0,
                '--duration', 1200.0, '--raw-artefacts', K / 'search/raw_artefacts_car.npz'], d / 'evaluate.log')


def clean(d):
    if (d / 'cleaning.npz').exists():
        return True
    return run([PY, 'cleaning.py', '--sort', d, '--duration', 1200.0], d / 'cleaning.log')


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    names = sorted(json.loads((D / 'variants.json').read_text()))
    kk = [D / x for x in names if x.startswith('kobe')]
    mm = [D / x for x in names if x.startswith('mec')]
    with ThreadPoolExecutor(n) as ex:
        ok = list(ex.map(kobe, kk)) + list(ex.map(clean, mm))
    print(f'kobe and MEC cleaning: {sum(ok)} of {len(ok)} ok', flush=True)
    out = D.parent / 'transfer_mec.json'
    if not out.exists():
        run([PY, 'transfer.py', '--default', *[M / f'runs/default_s{k}' for k in (1, 2, 3)], '--winner', *mm, '--alpha', 0.0625,
             '--tracking', M / 'search/tracking_13575-14775s.npz', '--raw-artefacts', M / 'search/raw_artefacts_car.npz',
             '--t0', 13575.0, '--duration', 1200.0, '--out', out], D.parent / 'transfer_mec.log')
    print('done', flush=True)
