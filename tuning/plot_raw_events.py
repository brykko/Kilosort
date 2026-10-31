"""Plot random raw-data transient events (spec §8.9 inspection): high-passed traces across depth,
before and after common-median referencing, ±5 ms around each event."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt

FS = 30000
HALF = 150  # ±5 ms

p = argparse.ArgumentParser()
p.add_argument('--data', required=True)
p.add_argument('--probe', required=True)
p.add_argument('--n-chan-bin', type=int, required=True)
p.add_argument('--candidates', required=True, help='raw_artefacts.npz')
p.add_argument('--min-channels', type=int, required=True)
p.add_argument('--max-channels', type=int, default=10**6)
p.add_argument('--n', type=int, default=6)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--out', required=True)
args = p.parse_args()

probe = json.loads(Path(args.probe).read_text())
order = np.argsort(probe['yc'], kind='stable')
chans = np.asarray(probe['chanMap'])[order]
ys = np.asarray(probe['yc'], float)[order]
raw = np.memmap(args.data, np.int16, 'r').reshape(-1, args.n_chan_bin)
ev = np.load(args.candidates)['events']
sel = ev[(ev[:, 2] >= args.min_channels) & (ev[:, 2] < args.max_channels)]
pick = sel[np.random.default_rng(args.seed).choice(sel.shape[0], min(args.n, sel.shape[0]), replace=False)]
sos = butter(3, 300, 'highpass', fs=FS, output='sos')

fig, axes = plt.subplots(2, len(pick), figsize=(3 * len(pick), 9), sharey=True, squeeze=False)
for j, e in enumerate(pick):
    c = int(e[0])
    x = sosfiltfilt(sos, raw[c - 3000:c + 3000][:, chans].astype(np.float32), axis=0)[3000 - HALF:3000 + HALF]
    for row, (y, title) in enumerate(((x, 'raw'), (x - np.median(x, axis=1, keepdims=True), 'after common median'))):
        lim = np.percentile(np.abs(x), 99.5)
        axes[row, j].imshow(y.T, aspect='auto', origin='lower', cmap='RdBu_r', vmin=-lim, vmax=lim,
                            extent=[-5, 5, ys[0], ys[-1]])
        axes[row, j].set_title(f't={c / FS:.3f} s, {int(e[2])} ch\n{title}', fontsize=8)
        axes[row, j].set_xlabel('ms')
axes[0, 0].set_ylabel('depth (µm)'); axes[1, 0].set_ylabel('depth (µm)')
fig.suptitle(f'Raw transient candidates with {args.min_channels}-{args.max_channels} channels > 6 MAD spanning >= 1 mm '
             f'({sel.shape[0]} such events in the excerpt)', fontsize=10)
fig.tight_layout()
fig.savefig(args.out, dpi=110)
print(f'{sel.shape[0]} events in range; plotted {len(pick)} -> {args.out}')
