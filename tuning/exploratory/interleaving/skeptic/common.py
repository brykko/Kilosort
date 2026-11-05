"""Shared loading for the skeptic's read-only checks. CPU only, no bytecode written into the worktree."""
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np  # noqa: E402

WT = Path('C:/code/github/Kilosort/.claude/worktrees/ks4-round3')
O = Path('C:/data/ks4_tuning/exploratory/iteration')
OUT = Path(r'C:/Users/richa/AppData/Local/Temp/claude/8c1cec062ef79d80/45eb8e85-37d5-451f-bed4-1560d425db7e/scratchpad/skeptic_out')
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(WT))
EXS = ('kobe', 'mec')
SEEDS = (1, 2, 3)
FS = 30000.0
DUR = 1200.0


def folder(ex, setting, rnd, seed):
    """setting 'H8', 'default' or 'aug'; rnd 2, 3 or 4 (aug: 3 only; default: 2, 3)."""
    if setting == 'aug':
        return O / 'sorts' / f'{ex}_augment_s{seed}'
    pre = '' if setting == 'H8' else 'default_'
    run = O / 'sorts' / f'{ex}_{pre}repeat_s{seed}'
    last = 4 if setting == 'H8' else 3
    return run if rnd == last else run / f'round{rnd}'


def ops_of(d):
    return np.load(Path(d) / 'ops.npy', allow_pickle=True).item()


def tonp(x):
    return x.cpu().numpy() if hasattr(x, 'cpu') else np.asarray(x)


def spikes(d):
    d = Path(d)
    return dict(t=np.load(d / 'spike_times.npy').ravel().astype(np.int64),
                clu=np.load(d / 'spike_clusters.npy').ravel().astype(np.int64),
                tpl=np.load(d / 'spike_detection_templates.npy').ravel().astype(np.int64),
                score=np.load(d / 'score.npy').ravel().astype(np.float32),
                amp=np.load(d / 'amplitudes.npy').ravel().astype(np.float32),
                pos=np.load(d / 'spike_positions.npy').astype(np.float32))


def S_i(d, ex):
    d = Path(d)
    if ex == 'kobe':
        return {int(k): float(v) for k, v in json.loads((d / 'evaluation.json').read_text())['S_i'].items()}
    for f in ('transfer_mec.json', 'transfer_mec_default.json', 'transfer_mec_long.json'):
        p = O / f
        if p.exists():
            for w in json.loads(p.read_text())['winner']:
                if Path(w['sort']) == d:
                    return {int(k): float(v) for k, v in w['S_i'].items()}
    return None


def owners(tpl, clu):
    """owner[template] = cluster that got most of its spikes (-1 if the template caught none), plus the
    (template, cluster, count) table."""
    n_t, n_c = int(tpl.max()) + 1, int(clu.max()) + 1
    key = tpl * n_c + clu
    u, n = np.unique(key, return_counts=True)
    t, c = u // n_c, u % n_c
    o = np.lexsort((n, t))
    t, c, n = t[o], c[o], n[o]
    last = np.r_[t[1:] != t[:-1], True]
    owner = -np.ones(n_t, np.int64)
    owner[t[last]] = c[last]
    return owner, t, c, n
