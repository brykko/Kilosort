"""Partial KS4 sorts (default settings) for the scaling audit; each stops early.

  sweep: after universal detection, run pass-1 clustering on time subsets of the same spikes; record counts.
  early/late: pass-1 templates learned from spikes with t < tcut (early) or t >= tcut (late) only; template
              matching then runs on the whole file; saves the learned spikes (st: sample, template, amp) and stops.

    python gpu_tests.py MODE DATA PROBE OUT [--tcut 600]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(r'C:\code\github\Kilosort')
sys.path.insert(0, str(REPO))
import kilosort  # noqa: E402
from kilosort import clustering_qr, run_kilosort, template_matching  # noqa: E402
from kilosort.io import load_probe  # noqa: E402

assert Path(kilosort.__file__).resolve().parent == REPO / 'kilosort', kilosort.__file__

p = argparse.ArgumentParser()
p.add_argument('mode', choices=['sweep', 'early', 'late'])
p.add_argument('data')
p.add_argument('probe')
p.add_argument('out')
p.add_argument('--tcut', type=float, default=600.0)
a = p.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=False)


class Stop(Exception):
    pass


orig_run, orig_pp, orig_extract = clustering_qr.run, template_matching.postprocess_templates, template_matching.extract
state = {}


def cluster_patch(ops, st, tF, mode='template', **kw):
    if mode != 'spikes':
        return orig_run(ops, st, tF, mode=mode, **kw)
    t = st[:, 0]                                            # seconds in the universal pass
    if a.mode == 'sweep':
        T = float(t.max())
        conds = [('first5', t < 300, None), ('first10', t < 600, None), ('last10', t >= T - 600, None),
                 ('all20', np.ones_like(t, bool), None), ('first10_cds10', t < 600, 10)]
        res = []
        cds0 = ops['settings']['cluster_downsampling']
        for name, sel, cds in conds:
            ops['settings']['cluster_downsampling'] = cds or cds0
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            clu, Wall = orig_run(ops, st[sel], tF[torch.from_numpy(sel)], mode='spikes', **kw)
            r = {'cond': name, 'cds': cds or cds0, 'n_spikes': int(sel.sum()), 'n_clusters': int(Wall.shape[0]),
                 'seconds': time.time() - t0, 'max_alloc_gb': torch.cuda.max_memory_allocated() / 1e9}
            print(json.dumps(r), flush=True)
            res.append(r)
        ops['settings']['cluster_downsampling'] = cds0
        (out / 'sweep.json').write_text(json.dumps(res, indent=1))
        raise Stop
    sel = t < a.tcut if a.mode == 'early' else t >= a.tcut
    state['sel'] = sel
    clu, Wall = orig_run(ops, st[sel], tF[torch.from_numpy(sel)], mode='spikes', **kw)
    state['n_templates_pass1'] = int(Wall.shape[0])
    return clu, Wall


def pp_patch(Wall, ops, clu, st, tF, device=torch.device('cuda')):
    sel = state['sel']
    return orig_pp(Wall, ops, clu, st[sel], tF[torch.from_numpy(sel)], device=device)


def extract_patch(ops, bfile, U, **kw):
    t0 = time.time()
    st, tF, ops = orig_extract(ops, bfile, U, **kw)
    np.save(out / 'st_learned.npy', st)
    (out / 'info.json').write_text(json.dumps({
        'mode': a.mode, 'tcut': a.tcut, 'n_templates_pass1': state['n_templates_pass1'],
        'n_templates_matched': int(U.shape[0]), 'n_learned': int(len(st)), 'extract_s': time.time() - t0,
        'imin': int(bfile.imin), 'n_univ_used': int(state['sel'].sum())}, indent=1))
    raise Stop


clustering_qr.run, template_matching.postprocess_templates, template_matching.extract = \
    cluster_patch, pp_patch, extract_patch
settings = {'n_chan_bin': 385, 'fs': 30000, 'random_seed': 1, 'cluster_init_seed': 5}
t0 = time.time()
try:
    run_kilosort(settings=settings, probe=load_probe(a.probe), filename=a.data, results_dir=out / 'ks',
                 data_dtype='int16', verbose_console=False, make_plots=False, save_pc_features=False)
except Stop:
    pass
print(f'{a.mode} done in {(time.time() - t0) / 60:.1f} min -> {out}', flush=True)
