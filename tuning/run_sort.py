"""Run one KS4 sort of an extracted excerpt with default settings (plus any overrides).

Imports KS4 from this checkout and refuses to run if kilosort resolves anywhere else, and writes
provenance.json with the
commit, the working-tree diff hash, the hash of the fused-kernel library (built from
kilosort/cuda, not tracked by git) and the settings used. Skips KS4's PNG plots and Phy's PC
features (pc_features.npy, pc_feature_ind.npy), which nothing in the harness reads.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import kilosort  # noqa: E402
from kilosort import cuda_kernels, run_kilosort  # noqa: E402
from kilosort.io import load_probe  # noqa: E402

if Path(kilosort.__file__).resolve().parent != REPO / 'kilosort':
    raise SystemExit(f'Imported kilosort from {kilosort.__file__}, not from {REPO}')

p = argparse.ArgumentParser()
p.add_argument('--data', required=True, help='excerpt .bin')
p.add_argument('--probe', required=True, help='KS4 probe .json')
p.add_argument('--n-chan-bin', type=int, required=True)
p.add_argument('--out', required=True, help='results folder (must not exist)')
p.add_argument('--settings', default='{}', help='JSON dict of setting overrides')
p.add_argument('--seed', type=int, default=1,
               help='replicate seed: random_seed = seed, cluster_init_seed = seed + 4 (1 gives KS4 defaults)')
args = p.parse_args()

out = Path(args.out)
if out.exists():
    raise SystemExit(f'{out} already exists')
settings = {'n_chan_bin': args.n_chan_bin, 'fs': 30000, 'random_seed': args.seed,
            'cluster_init_seed': args.seed + 4, **json.loads(args.settings)}
git = lambda *a: subprocess.run(['git', '-C', str(REPO), *a], capture_output=True, text=True, check=True).stdout
diff = git('diff', 'HEAD', '--', 'kilosort')
untracked = git('ls-files', '--others', '--exclude-standard', '--', 'kilosort').split()
kernels = cuda_kernels.LIBRARY_PATH
t0 = time.time()
run_kilosort(settings=settings, probe=load_probe(args.probe), filename=args.data,
             results_dir=out, data_dtype='int16', verbose_console=True, make_plots=False,
             save_pc_features=False)
(out / 'provenance.json').write_text(json.dumps({
    'kilosort_path': kilosort.__file__,
    'commit': git('rev-parse', 'HEAD').strip(),
    'kilosort_diff_sha256': hashlib.sha256(diff.encode()).hexdigest(),
    'kilosort_diff': diff,
    'kilosort_untracked_files': untracked,
    'kernels_used': cuda_kernels._lib is not None,
    'kernels_sha256': hashlib.sha256(kernels.read_bytes()).hexdigest() if kernels.exists() else None,
    'settings': settings,
    'data': args.data,
    'probe': args.probe,
    'runtime_min': (time.time() - t0) / 60,
}, indent=2))
print(f'Sort finished in {(time.time() - t0) / 60:.1f} min -> {out}')
