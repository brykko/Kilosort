"""Does template matching's peel limit (`max_peels`) bind? (search_round3.md §2.) Behaviour-blind.

Runs one sort up to the end of template matching (drift correction and the first detection pass from
the stage cache when available) and counts, per batch, the peels that found spikes. A batch whose
count reaches `max_peels` stopped because of the limit, not because no spikes were left.

    python peel_diagnostic.py --settings '<json>' --seed 1 [--out <json>]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kilosort import clustering_qr, cuda_kernels, run_kilosort  # noqa: E402
from kilosort.io import load_probe  # noqa: E402
from kilosort.parameters import DEFAULT_SETTINGS  # noqa: E402
import stage_cache  # noqa: E402

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / 'config.json').read_text())
BASE = Path(CFG['excerpt_dir'])

p = argparse.ArgumentParser()
p.add_argument('--settings', default='{}')
p.add_argument('--seed', type=int, default=1)
p.add_argument('--out')
args = p.parse_args()
settings = {'n_chan_bin': CFG['n_chan_bin'], 'fs': 30000, 'random_seed': args.seed,
            'cluster_init_seed': args.seed + 4, **json.loads(args.settings)}
data, probe = BASE / CFG['data'], BASE / CFG['probe']
key_a, _ = stage_cache.keys({**DEFAULT_SETTINGS, **settings}, data, probe,
                            extra={'data_dtype': 'int16', 'do_CAR': True, 'invert_sign': False})
stage_cache.install(BASE / CFG['stage_cache'], key_a, 'peel_diagnostic')   # stage B always recomputed

peels, current = [], [0]
stock_record, stock_run = cuda_kernels.matching_record, clustering_qr.run


def record(*a, **kw):                     # called once per peel that found spikes (fused path)
    current[0] += 1
    return stock_record(*a, **kw)


class Done(Exception):
    pass


def run(ops, st, tF, mode='template', **kw):
    if mode == 'template':                # second clustering pass: template matching is over
        raise Done
    return stock_run(ops, st, tF, mode=mode, **kw)


import kilosort.template_matching as tm  # noqa: E402
stock_matching = tm.run_matching


def run_matching(*a, **kw):
    current[0] = 0
    out = stock_matching(*a, **kw)
    peels.append(current[0])
    return out


cuda_kernels.matching_record, clustering_qr.run, tm.run_matching = record, run, run_matching
try:
    run_kilosort(settings=settings, probe=load_probe(str(probe)), filename=str(data),
                 results_dir=BASE / 'runs' / 'peel_diagnostic_tmp', data_dtype='int16', verbose_console=False,
                 make_plots=False, save_pc_features=False)
except Done:
    pass
limit = {**DEFAULT_SETTINGS, **settings}['max_peels']
peels = np.array(peels)
res = {'settings': settings, 'max_peels': limit, 'batches': int(peels.size),
       'batches_at_limit': int((peels >= limit).sum()), 'peels_median': float(np.median(peels)),
       'peels_max': int(peels.max())}
print(json.dumps(res))
if args.out:
    Path(args.out).write_text(json.dumps(res))
