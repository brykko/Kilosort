"""Stage cache for KS4 sorts of the search: reuse the stages that a run's settings don't affect.

KS4 itself is not changed. `install` wraps four of its stage functions for the current process; a
sort without `install` (any normal KS4 run) is untouched. Two cached stages:

- A: drift correction (`datashift.run`) and the first detection pass with universal templates
  (`spikedetect.run`). Depend on the data, the KS4 code and every setting except LATE ones.
- B: the first clustering pass (`clustering_qr.run`, mode 'spikes') and template matching
  (`template_matching.extract`). Depend on A's key and every LATE setting except PASS2 ones. The
  first-merge step between them (`postprocess_templates`) is cheap and always recomputed.

The second clustering pass, the final merge and saving always run.

A stage's outputs are its return values plus the `ops` keys it sets (all stage functions add keys,
none mutate existing values in place), and the numpy/torch random-number states after it, restored
on a hit so the rest of the sort sees what it would have seen. Keys hash the KS4 sources and kernel
library, the data file (path, size, modification time), the probe and the relevant settings, so a
change to any of them misses the cache. Cached sorts are checked to be bit-identical to full sorts
(search_round2.md §2.2).
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from kilosort import clustering_qr, datashift, spikedetect, template_matching

REPO = Path(__file__).resolve().parents[1]

# Settings read only by the clustering passes, the merges or saving (stage B onwards).
LATE = {'cluster_downsampling', 'cluster_init_clusters', 'cluster_init_seed', 'cluster_neighbors',
        'cluster_resolution', 'correlogram_shape_criterion', 'max_cluster_subset', 'x_centers',
        'merge_bimodality_threshold', 'tree_connectivity_threshold', 'merge_correlation_threshold_1',
        'merge_correlation_threshold_2', 'acg_threshold', 'ccg_threshold', 'duplicate_spike_ms',
        'tree_refractory_significance', 'tree_walk', 'cluster_min_group_spikes'}
# Of those, the ones read only by the second clustering pass, the final merge or saving.
PASS2 = {'correlogram_shape_criterion', 'merge_correlation_threshold_2', 'acg_threshold',
         'ccg_threshold', 'duplicate_spike_ms', 'tree_refractory_significance'}


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def code_hash():
    ks = REPO / 'kilosort'
    files = sorted([*ks.glob('*.py'), *(ks / 'cuda').glob('*.dll'), *(ks / 'cuda').glob('*.so')])
    return _sha(b''.join(f.name.encode() + _sha(f.read_bytes()).encode() for f in files))


def keys(settings, data, probe, extra):
    """Cache keys of stages A and B for a sort's full settings dict."""
    d = Path(data).stat()
    base = {'code': code_hash(), 'data': [str(Path(data).resolve()), d.st_size, d.st_mtime],
            'probe': _sha(Path(probe).read_bytes()), 'extra': extra}
    a = {k: v for k, v in settings.items() if k not in LATE}
    b = {k: v for k, v in settings.items() if k in LATE and k not in PASS2}
    ka = _sha(json.dumps({**base, 'settings': a}, sort_keys=True, default=str).encode())[:16]
    kb = _sha(json.dumps({'A': ka, 'settings': b}, sort_keys=True, default=str).encode())[:16]
    return ka, kb


def _rng():
    return {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'numpy': np.random.get_state()}


def _set_rng(s):
    torch.set_rng_state(s['torch'])
    if s['cuda'] is not None:
        torch.cuda.set_rng_state_all(s['cuda'])
    np.random.set_state(s['numpy'])


def _new_keys(ops, before):
    return {k: v for k, v in ops.items() if k not in before or before[k] is not v}


def install(cache_dir, key_a, key_b):
    """Wrap the stage functions; returns a dict recording 'hit'/'miss' per stage."""
    folder = {'A': Path(cache_dir) / f'A_{key_a}', 'B': Path(cache_dir) / f'B_{key_b}'}
    status = {}
    inner = {'drift': False}

    def cached(stage, name, compute, ops):
        path = folder[stage] / f'{name}.pt'
        if path.exists():
            c = torch.load(path, weights_only=False)
            ops.update(c['ops'])
            _set_rng(c['rng'])
            status[name] = 'hit'
            return c['out']
        before = dict(ops)
        out = compute()
        # ops is returned first by drift correction, last by detection and extraction; it is
        # stored as its new keys only, not as part of the outputs
        if name == 'drift':
            new_ops, kept = out[0], (None, *out[1:])
        elif name in ('detect1', 'extract'):
            new_ops, kept = out[-1], (*out[:-1], None)
        else:
            new_ops, kept = ops, out
        folder[stage].mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.tmp')
        torch.save({'out': kept, 'ops': _new_keys(new_ops, before), 'rng': _rng()}, tmp, pickle_protocol=4)
        tmp.replace(path)
        status[name] = 'miss'
        return out

    orig = {'drift': datashift.run, 'detect': spikedetect.run, 'cluster': clustering_qr.run,
            'extract': template_matching.extract}

    def drift(ops, bfile, **kw):
        def compute():
            inner['drift'] = True
            try:
                return orig['drift'](ops, bfile, **kw)
            finally:
                inner['drift'] = False
        ops_out, st = cached('A', 'drift', compute, ops)
        return ops, st

    def detect(ops, bfile, **kw):
        if inner['drift']:                      # the call inside drift correction
            return orig['detect'](ops, bfile, **kw)
        st0, tF, _ = cached('A', 'detect1', lambda: orig['detect'](ops, bfile, **kw), ops)
        return st0, tF, ops

    def cluster(ops, st, tF, mode='template', **kw):
        if mode != 'spikes':
            return orig['cluster'](ops, st, tF, mode=mode, **kw)
        return cached('B', 'cluster1', lambda: orig['cluster'](ops, st, tF, mode=mode, **kw), ops)

    def extract(ops, bfile, U, **kw):
        st, tF, _ = cached('B', 'extract', lambda: orig['extract'](ops, bfile, U, **kw), ops)
        return st, tF, ops

    datashift.run, spikedetect.run = drift, detect
    clustering_qr.run, template_matching.extract = cluster, extract
    return status
