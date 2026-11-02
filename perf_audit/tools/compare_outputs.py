"""Compare two KS4 result folders: bit-identical, or how far apart.

Usage: python compare_outputs.py RUN_A RUN_B
"""
import sys
from pathlib import Path

import numpy as np

a, b = Path(sys.argv[1]), Path(sys.argv[2])
FILES = ['spike_times.npy', 'spike_clusters.npy', 'spike_templates.npy', 'amplitudes.npy',
         'templates.npy', 'spike_positions.npy', 'whitening_mat.npy', 'kept_spikes.npy',
         'similar_templates.npy', 'spike_detection_templates.npy', 'templates_ind.npy',
         'pc_features.npy', 'pc_feature_ind.npy']

identical = True
for f in FILES:
    if not (a / f).exists() or not (b / f).exists():
        print(f'{f:30s} missing in {"A" if not (a / f).exists() else "B"}'); continue
    x, y = np.load(a / f), np.load(b / f)
    if x.shape == y.shape and np.array_equal(x, y):
        print(f'{f:30s} identical {x.shape}')
        continue
    identical = False
    if x.shape != y.shape:
        print(f'{f:24s} DIFFER shape {x.shape} vs {y.shape}')
    else:
        d = np.abs(x.astype(np.float64) - y.astype(np.float64))
        print(f'{f:24s} DIFFER {np.mean(d > 0):.2%} of elements, max abs diff {d.max():.3g}')

if not identical:
    # spike-time agreement: fraction of A's spikes with a B spike within 0.5 ms
    sa, sb = np.load(a / 'spike_times.npy').ravel(), np.load(b / 'spike_times.npy').ravel()
    sb_sorted = np.sort(sb)
    i = np.clip(np.searchsorted(sb_sorted, sa), 1, len(sb_sorted) - 1)
    nearest = np.minimum(np.abs(sb_sorted[i] - sa), np.abs(sb_sorted[i - 1] - sa))
    ca, cb = np.load(a / 'spike_clusters.npy'), np.load(b / 'spike_clusters.npy')
    print(f'spikes {len(sa)} vs {len(sb)}; units {len(np.unique(ca))} vs {len(np.unique(cb))}; '
          f'A spikes with a B spike within 15 samples: {np.mean(nearest <= 15):.4%}')
print('BIT-IDENTICAL' if identical else 'NOT IDENTICAL')
