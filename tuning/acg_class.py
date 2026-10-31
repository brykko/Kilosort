"""Classify a KS4 sort's units with elv's frozen short-lag ACG (burst-shape) classifier.

Run in the elv environment (`conda run -n elv python tuning/acg_class.py ...`). Writes the raw-ACG
label per cluster (1 sharp, 2 medium, 3 broad, 0 not classified); elv's theta-phase source
population is the units labelled sharp or medium. No anatomical ROI filter is applied.
"""
import argparse
from pathlib import Path

import numpy as np
from elv.tasks.burst_shape_classification import classify_counts
from elv.tasks.burst_shape_model import FrozenBurstShapeClassifiers, positive_lag_pair_counts

p = argparse.ArgumentParser()
p.add_argument('--sort', required=True)
p.add_argument('--out', required=True)
args = p.parse_args()

st = np.load(Path(args.sort) / 'spike_times.npy').ravel() / 30000
clu = np.load(Path(args.sort) / 'spike_clusters.npy').ravel()
ids = np.unique(clu)
counts = np.stack([positive_lag_pair_counts(np.sort(st[clu == c])) for c in ids])
out = classify_counts(ids.astype(str), counts, FrozenBurstShapeClassifiers.load())
labels = out['methods']['rawAcg']['label']
np.savez(args.out, cluster=ids, raw_acg_label=labels, eligible=out['quality']['eligible'])
print(f'{ids.size} clusters: sharp {np.sum(labels == 1)}, medium {np.sum(labels == 2)}, '
      f'broad {np.sum(labels == 3)}, unclassified {np.sum(labels == 0)}')
