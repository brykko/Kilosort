"""Instrumented KS4 sort: per-function GPU-synchronised timers, GPU/CPU/disk traces,
and snapshots of intermediate state for later microbenchmarks.

Imports KS4 from --repo (default: the main checkout). Does not modify any KS4 file:
all instrumentation is monkeypatched at import time.
"""
import argparse
import json
import subprocess
import sys
import threading
import time
import types
from collections import defaultdict
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--repo', default=r'C:\code\github\Kilosort')
p.add_argument('--data', default=r'C:\data\ks4_tuning\26648_kobe_2020-08-21\search\ap_1668-2868s.bin')
p.add_argument('--probe', default=r'C:\data\ks4_tuning\26648_kobe_2020-08-21\search\probe_3a_opt3.json')
p.add_argument('--n-chan-bin', type=int, default=385)
p.add_argument('--out', required=True)
p.add_argument('--seed', type=int, default=1)
p.add_argument('--settings', default='{}')
p.add_argument('--tmax', type=float, default=None, help='seconds; sort a prefix only')
p.add_argument('--no-sync', action='store_true', help='time without cuda synchronize')
p.add_argument('--no-pc-features', action='store_true', help='pass save_pc_features=False')
p.add_argument('--no-plots', action='store_true', help='pass make_plots=False (needs the perf-optimizations branch)')
p.add_argument('--snapshot', action='store_true', help='save intermediate state for microbenchmarks')
p.add_argument('--patches', default='', help='comma-separated ks_fast patches, e.g. det,h2d,tm,rm,noplots')
args = p.parse_args()

sys.path.insert(0, args.repo)
import numpy as np  # noqa: E402
import psutil  # noqa: E402
import torch  # noqa: E402
import kilosort  # noqa: E402
from kilosort import run_kilosort  # noqa: E402
from kilosort.io import load_probe  # noqa: E402

assert Path(kilosort.__file__).resolve().parent == Path(args.repo, 'kilosort').resolve(), kilosort.__file__

out = Path(args.out)
if out.exists():
    raise SystemExit(f'{out} already exists')
out.mkdir(parents=True)

if args.patches:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import ks_fast
    ks_fast.apply(args.patches.split(','))

# ---------------------------------------------------------------- timers
T = defaultdict(lambda: [0.0, 0, 0.0])   # name -> [total_s, calls, max_s]
SYNC = not args.no_sync and torch.cuda.is_available()


def wrap(name, fn):
    def w(*a, **k):
        if SYNC:
            torch.cuda.synchronize()
        t = time.perf_counter()
        r = fn(*a, **k)
        if SYNC:
            torch.cuda.synchronize()
        dt = time.perf_counter() - t
        rec = T[name]
        rec[0] += dt; rec[1] += 1; rec[2] = max(rec[2], dt)
        return r
    w.__wrapped__ = fn
    return w


import kilosort.io as kio  # noqa: E402
from kilosort import (spikedetect, template_matching, clustering_qr, datashift,  # noqa: E402
                      swarmsplitter, hierarchical, postprocessing, CCG, preprocessing)
import kilosort.plots as kplots  # noqa: E402

MODS = [spikedetect, template_matching, clustering_qr, datashift, swarmsplitter,
        hierarchical, postprocessing, CCG, preprocessing, kplots, kio,
        sys.modules['kilosort.run_kilosort']]

TARGETS = {
    spikedetect: ['run', 'template_match', 'yweighted', 'extract_wPCA_wTEMP'],
    template_matching: ['extract', 'run_matching', 'prepare_extract', 'prepare_matching',
                        'merging_function', 'postprocess_templates', 'align_U', 'roll_features'],
    clustering_qr: ['run', 'cluster', 'kmeans_plusplus', 'neigh_mat', 'get_data_cpu',
                    'assign_iclust', 'assign_isub', 'Mstats', 'assign_clust', 'assign_iclust0'],
    datashift: ['run', 'bin_spikes', 'align_block2'],
    swarmsplitter: ['split', 'new_clusters', 'check_split', 'bimod_score', 'check_CCG', 'refractoriness'],
    hierarchical: None,  # all plain functions
    postprocessing: None,
    CCG: None,
    kplots: None,
    kio: ['save_to_phy', 'save_ops'],
    sys.modules['kilosort.run_kilosort']: ['compute_preprocessing', 'compute_drift_correction',
                                           'detect_spikes', 'cluster_spikes', 'save_sorting'],
}

for mod, names in TARGETS.items():
    if names is None:
        names = [n for n, v in vars(mod).items()
                 if isinstance(v, types.FunctionType) and v.__module__ == mod.__name__]
    for n in names:
        orig = getattr(mod, n)
        if not isinstance(orig, types.FunctionType):
            continue  # skip numba dispatchers etc.
        w = wrap(f'{mod.__name__.split(".")[-1]}.{n}', orig)
        for m in MODS:  # replace every module-global reference to the same object
            for k, v in list(vars(m).items()):
                if v is orig:
                    setattr(m, k, w)

# per-batch I/O split: raw read+H2D (BinaryRWFile) vs filter (BinaryFiltered)
kio.BinaryRWFile.padded_batch_to_torch = wrap('io.read_to_gpu', kio.BinaryRWFile.padded_batch_to_torch)
kio.BinaryFiltered.filter = wrap('io.filter', kio.BinaryFiltered.filter)

# ---------------------------------------------------------------- snapshots
if args.snapshot:
    snap = out / 'snapshot'
    snap.mkdir()
    # run_kilosort calls both through the module attribute, so replacing that is enough
    _extract = template_matching.extract.__wrapped__
    _crun = clustering_qr.run.__wrapped__
    calls = {'n': 0}

    def extract_snap(ops, bfile, U, *a, **k):
        torch.save(U.cpu(), snap / 'U_first.pt')
        return _extract(ops, bfile, U, *a, **k)

    def crun_snap(ops, st, tF, *a, **k):
        tag = 'first' if calls['n'] == 0 else 'final'
        calls['n'] += 1
        np.save(snap / f'st_{tag}.npy', st)
        torch.save(tF.cpu() if torch.is_tensor(tF) else torch.from_numpy(tF), snap / f'tF_{tag}.pt')
        return _crun(ops, st, tF, *a, **k)

    template_matching.extract = wrap('template_matching.extract', extract_snap)
    clustering_qr.run = wrap('clustering_qr.run', crun_snap)

# ---------------------------------------------------------------- traces
smi = subprocess.Popen(
    ['nvidia-smi', '--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm',
     '--format=csv,noheader,nounits', '-lms', '500'],
    stdout=open(out / 'gpu_trace.csv', 'w'), stderr=subprocess.DEVNULL)
stop = threading.Event()
proc = psutil.Process()


def sample_host():
    with open(out / 'host_trace.csv', 'w') as f:
        f.write('t,cpu_pct_total,proc_cpu_pct,proc_rss_gb,disk_read_mb_s\n')
        d0 = psutil.disk_io_counters().read_bytes
        t0 = time.time()
        proc.cpu_percent(None); psutil.cpu_percent(None)
        while not stop.wait(0.5):
            d1 = psutil.disk_io_counters().read_bytes
            t1 = time.time()
            f.write(f'{t1:.2f},{psutil.cpu_percent(None):.1f},{proc.cpu_percent(None):.1f},'
                    f'{proc.memory_info().rss / 1e9:.2f},{(d1 - d0) / 1e6 / (t1 - t0):.1f}\n')
            f.flush()
            d0, t0 = d1, t1


th = threading.Thread(target=sample_host, daemon=True)
th.start()

# ---------------------------------------------------------------- sort
settings = {'n_chan_bin': args.n_chan_bin, 'fs': 30000, 'random_seed': args.seed,
            'cluster_init_seed': args.seed + 4, **json.loads(args.settings)}
if args.tmax is not None:
    settings['tmax'] = args.tmax
t_start = time.time()
try:
    run_kilosort(settings=settings, probe=load_probe(args.probe), filename=args.data,
                 results_dir=out, data_dtype='int16', verbose_console=False,
                 **({'make_plots': False} if args.no_plots else {}),
                 **({'save_pc_features': False} if args.no_pc_features else {}))
finally:
    wall = time.time() - t_start
    stop.set(); th.join(2)
    smi.terminate()
    rows = sorted(({'name': k, 'total_s': v[0], 'calls': v[1], 'max_s': v[2]} for k, v in T.items()),
                  key=lambda r: -r['total_s'])
    (out / 'timings.json').write_text(json.dumps(
        {'wall_s': wall, 't_start': t_start, 'sync': SYNC, 'patches': args.patches, 'settings': settings,
         'torch': torch.__version__, 'functions': rows}, indent=1))
    print(f'wall {wall:.1f}s')
    for r in rows[:40]:
        print(f"{r['total_s']:9.2f}s {r['calls']:7d}  {r['name']}")
