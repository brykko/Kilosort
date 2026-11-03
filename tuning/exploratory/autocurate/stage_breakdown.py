"""What each step of KS2.5's autocurate does: the traced reruns (ks25_setup/ks25_sort_traced.py, preset
"standard", seed 1), stage by stage (extracted, merge1, split1, cut1, merge2, split2, cut2, merge3).

Per stage: units with spikes, spikes, and misassigned / correctly assigned spikes (plain and local, as
cutoff_trace.py: masked intervals dropped, no duplicate cleaning, hybrid copies kept). Per step: merges
(templates whose spikes all moved to another), splits (new templates), spikes relabelled, spikes deleted.
Merge and split steps keep the row order of rez.st3 and cuts only delete rows, so stages compare row by row.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / 'isolation'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cleaning import mask_intervals  # noqa: E402
from trace_units import contamination  # noqa: E402

FS = 30000
DUR = 1200.0
K = Path('C:/data/ks4_tuning/26648_kobe_2020-08-21')
M = Path('C:/data/ks4_tuning/25843_roger_2019-09-11')
RERUN = Path('C:/data/ks4_tuning/exploratory/ks25_traced')
STAGES = ('extracted', 'merge1', 'split1', 'cut1', 'merge2', 'split2', 'cut2', 'merge3')


def unmasked(t, iv):
    i = np.searchsorted(iv[:, 0], t, 'right') - 1
    return ~((i >= 0) & (t < iv[np.maximum(i, 0), 1])) if iv.size else np.ones(t.size, bool)


def stage_stats(t, clu, T):
    o = np.argsort(clu, kind='stable')
    ids, start = np.unique(clu[o], return_index=True)
    units = [np.sort(x) for x in np.split(t[o], start[1:])]
    n = np.array([u.size for u in units], float)
    c = [contamination(u, T) for u in units]
    mis = {m: float(np.sum(n * np.array([x[m] for x in c]))) for m in ('plain', 'local')}
    return len(units), n.sum(), mis


def key(st, s):
    return st[f'{s}_t'].astype(np.int64) * 65536 + st[f'{s}_clu'].astype(np.int64)


if __name__ == '__main__':
    for ex in ('kobe', 'mec'):
        base = K if ex == 'kobe' else M
        R = RERUN / f'{ex}_s1'
        st = np.load(R / 'stages.npz')
        iv = mask_intervals(R, base / 'search/raw_artefacts_car.npz')
        T = DUR - float(np.sum(iv[:, 1] - iv[:, 0])) if iv.size else DUR
        print(f'\n===== {ex} (KS2.5 standard, traced rerun, seed 1)')
        print(f'{"stage":9s} {"units":>6s} {"spikes M":>9s} {"mis. plain":>11s} {"mis. local":>11s} '
              f'{"correct plain M":>16s} {"correct local M":>16s}   step')
        prev = None
        for s in STAGES:
            t = (st[f'{s}_t'] - 1) / FS
            clu = st[f'{s}_clu'].astype(np.int64)
            keep = unmasked(t, iv)
            nu, n, mis = stage_stats(t[keep], clu[keep], T)
            step = ''
            if prev is not None:
                pclu = st[f'{prev}_clu'].astype(np.int64)
                if pclu.size == clu.size:
                    moved = pclu != clu
                    gone = set(np.unique(pclu)) - set(np.unique(clu))
                    new = set(np.unique(clu)) - set(np.unique(pclu))
                    step = (f'{len(gone)} templates merged away, {len(new)} new from splits, '
                            f'{moved.sum() / 1e3:.0f} k spikes relabelled')
                else:
                    gone = ~np.isin(key(st, prev), key(st, s))
                    step = f'{gone.sum() / 1e3:.0f} k spikes deleted ({gone.mean():.1%})'
            print(f'{s:9s} {nu:6d} {n / 1e6:9.3f} {mis["plain"] / n:11.1%} {mis["local"] / n:11.1%} '
                  f'{(n - mis["plain"]) / 1e6:16.3f} {(n - mis["local"]) / 1e6:16.3f}   {step}', flush=True)
            prev = s
