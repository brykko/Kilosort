"""Check set_cutoff.py against MATLAB: on the traced KS2.5 reruns (preset "standard", Th [10 6], seed 1), apply
the port to stage split1 (strict) and split2 (not strict) and compare with MATLAB's cut1 and cut2: the spikes
kept, and each template's cut-off (set_cutoff.mat). Scores were saved as single precision, so a spike whose
score lies within float32 rounding of a cut-off could differ."""
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from set_cutoff import set_cutoff

FS = 30000
RERUN = Path('C:/data/ks4_tuning/exploratory/ks25_traced')
TH = (10.0, 6.0)

if __name__ == '__main__':
    for ex in ('kobe', 'mec'):
        R = RERUN / f'{ex}_s1'
        st = np.load(R / 'stages.npz')
        Ths = loadmat(R / 'set_cutoff.mat', squeeze_me=True)['Ths']
        for cyc, (a, b, strict) in enumerate((('split1', 'cut1', True), ('split2', 'cut2', False))):
            t0 = time.time()
            t, clu, score = st[f'{a}_t'].astype(np.float64), st[f'{a}_clu'].astype(np.int64), st[f'{a}_score'].astype(np.float64)
            keep, units = set_cutoff(t, clu, score, FS, TH, strict)
            ka = t.astype(np.int64) * 65536 + clu
            kb = st[f'{b}_t'].astype(np.int64) * 65536 + st[f'{b}_clu'].astype(np.int64)
            matlab_keep = np.isin(ka, kb)         # a duplicated (time, template) row counts as kept twice here
            same = np.array_equal(np.sort(ka[keep]), np.sort(kb))   # so also: the kept rows as a multiset
            mth = np.atleast_1d(Ths[cyc]).astype(float)
            diff_units = [c for c, (th, _) in units.items() if not np.isclose(th, mth[c - 1])]
            print(f'{ex} {a} -> {b} (strict {strict}): {time.time() - t0:.0f} s; spikes kept: port {keep.sum()}, '
                  f'MATLAB {kb.size}, identical {same} (by key: {np.sum(keep != matlab_keep)} differ); templates {len(units)}, '
                  f'cut-off differs in {len(diff_units)}', flush=True)
            for c in diff_units[:10]:
                ix = clu == c
                print(f'    template {c}: port {units[c][0]}, MATLAB {mth[c - 1]}, {ix.sum()} spikes, '
                      f'disagreeing {np.sum((keep != matlab_keep) & ix)}')
