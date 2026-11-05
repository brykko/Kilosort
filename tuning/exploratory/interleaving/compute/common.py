"""Shared setup for the interleaving compute benchmarks: loads KS4 from the ks4-round3 worktree (read-only),
the preprocessing state of a finished H8 sort and its extraction templates. No results are written to the sort."""
import json
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
WT = Path('C:/code/github/Kilosort/.claude/worktrees/ks4-round3')
sys.path.insert(0, str(WT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import kilosort  # noqa: E402
from kilosort import cuda_kernels, io as ksio, template_matching as tm  # noqa: E402

assert Path(kilosort.__file__).resolve().parent == (WT / 'kilosort').resolve(), kilosort.__file__

SORTS = Path('C:/data/ks4_tuning/exploratory/iteration/sorts')
OUT = Path('C:/data/ks4_tuning/exploratory/interleaving/compute')
CACHE = Path('C:/data/ks4_tuning/exploratory/iteration/stage_cache')
DEVICE = torch.device('cuda')
assert cuda_kernels.available(DEVICE)   # also loads the library


def now():
    torch.cuda.synchronize()
    return time.perf_counter()


def load_sort(name='kobe_repeat_s1'):
    """ops (with tensors on the GPU) and a BinaryFiltered with the sort's whitening and drift correction."""
    folder = SORTS / name
    ops = ksio.load_ops(folder / 'ops.npy', device=DEVICE)
    data = json.loads((folder / 'provenance.json').read_text())['data']
    bfile = ksio.bfile_from_ops(ops, filename=data, device=DEVICE)
    return ops, bfile, folder


def templates_from_wall(ops, wall):
    """Extraction templates (n, n_pcs, n_chan) from cluster templates saved as Wall.npy, aligned."""
    Wall = torch.from_numpy(np.load(wall)).to(DEVICE)
    Wall = Wall[(Wall**2).sum((1, 2)) > 0]
    U, _ = tm.align_U(Wall, ops, device=DEVICE)
    return U.contiguous()


def save_json(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1, default=float))
