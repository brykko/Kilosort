"""Check that two KS4 output folders hold the same sort (search_round1.md §2.2): every array the
harness or Phy reads must be exactly equal.

    python same_sort.py <sort A> <sort B>
"""
import sys
from pathlib import Path

import numpy as np

FILES = ['spike_times.npy', 'spike_clusters.npy', 'spike_templates.npy', 'amplitudes.npy', 'templates.npy',
         'templates_ind.npy', 'spike_positions.npy', 'similar_templates.npy', 'whitening_mat.npy',
         'whitening_mat_inv.npy', 'spike_detection_templates.npy']

a, b = Path(sys.argv[1]), Path(sys.argv[2])
bad = []
for f in FILES:
    x, y = np.load(a / f), np.load(b / f)
    same = x.shape == y.shape and x.dtype == y.dtype and np.array_equal(x, y)
    print(f'{f:32s} {str(x.shape):>18s} {"identical" if same else "DIFFERENT"}')
    if not same:
        bad.append(f)
for f in ['cluster_KSLabel.tsv']:
    same = (a / f).read_text() == (b / f).read_text()
    print(f'{f:32s} {"":>18s} {"identical" if same else "DIFFERENT"}')
    if not same:
        bad.append(f)
print('SAME SORT' if not bad else f'DIFFERENT: {bad}')
sys.exit(1 if bad else 0)
