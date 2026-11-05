"""Stage cache to small files: the universal pass's st0 (columns time s, y, amp, template index, batch, centre) and the
first clustering's labels and cluster templates, per H8 seed and excerpt. Writes st0_{ex}_s{seed}.npy, clu1_*.npy,
wall1_*.npy, and the matching ops iU/iCC of the round-2 extraction to the output folder. Loads about 1.5 GB at a time.

    python extract_st0.py
"""
import gc
import json

import numpy as np
import torch

from common import IT, OUT, SEEDS, sort_dir

for ex in ('kobe', 'mec'):
    for seed in SEEDS:
        if (OUT / f'wall1_{ex}_s{seed}.npy').exists():
            continue
        p = json.loads((sort_dir(ex, 'H8', seed, 4) / 'provenance.json').read_text())['cache']
        d = torch.load(IT / f"stage_cache/A_{p['A']}/detect1.pt", map_location='cpu', weights_only=False)
        np.save(OUT / f'st0_{ex}_s{seed}.npy', d['out'][0].astype(np.float64))
        del d
        gc.collect()
        c = torch.load(IT / f"stage_cache/B_{p['B']}/cluster1.pt", map_location='cpu', weights_only=False)
        np.save(OUT / f'clu1_{ex}_s{seed}.npy', np.asarray(c['out'][0]).astype(np.int32))
        np.save(OUT / f'wall1_{ex}_s{seed}.npy', c['out'][1].numpy())
        del c
        gc.collect()
        print(ex, seed, flush=True)
