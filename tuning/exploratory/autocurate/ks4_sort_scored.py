"""A KS4 sort as tuning/run_sort.py runs it, plus each spike's template-matching score: the value KS4 compares
with Th_learned when it extracts the spike (template_matching.run_matching's th_amps, st[:, 2] of extract),
which a set_cutoff-like step thresholds. Settings, data and probe are copied from an existing sort's
provenance.json (--like), so the rerun can be compared with that sort. --data sorts another recording with those
settings and probe (the long-data sorts); --no-clean skips cleaning (a long sort is cleaned in its window).

KS4 writes the scores only with save_extra_vars=True (full_st.npy, every spike before duplicate removal).
Writes score.npy (one per spike of spike_times.npy: full_st[kept_spikes, 2], checked against the spike
times), deletes the large tF.npy that save_extra_vars also writes, and cleans the sort (cleaning.py).

    python ks4_sort_scored.py --like <sort folder> --out <new folder> [--data <.bin>] [--no-clean]
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'tuning'))

import kilosort  # noqa: E402
from kilosort import cuda_kernels, run_kilosort  # noqa: E402
from kilosort.io import load_probe  # noqa: E402
from cleaning import run_cleaning  # noqa: E402

if Path(kilosort.__file__).resolve().parent != REPO / 'kilosort':
    raise SystemExit(f'Imported kilosort from {kilosort.__file__}, not from {REPO}')

p = argparse.ArgumentParser()
p.add_argument('--like', required=True, help='existing sort whose provenance.json gives settings, data and probe')
p.add_argument('--out', required=True, help='results folder (must not exist)')
p.add_argument('--data', help="recording to sort instead of --like's")
p.add_argument('--no-clean', action='store_true', help='skip cleaning')
args = p.parse_args()

out = Path(args.out)
if out.exists():
    raise SystemExit(f'{out} already exists')
like = json.loads((Path(args.like) / 'provenance.json').read_text())
settings = like['settings']
data = args.data or like['data']
git = lambda *a: subprocess.run(['git', '-C', str(REPO), *a], capture_output=True, text=True, check=True).stdout  # noqa: E731
diff = git('diff', 'HEAD', '--', 'kilosort')
untracked = git('ls-files', '--others', '--exclude-standard', '--', 'kilosort').split()
kernels = cuda_kernels.LIBRARY_PATH
t0 = time.time()
run_kilosort(settings=settings, probe=load_probe(like['probe']), filename=data,
             results_dir=out, data_dtype='int16', verbose_console=True, make_plots=False,
             save_pc_features=False, save_extra_vars=True)
runtime = (time.time() - t0) / 60

full_st = np.load(out / 'full_st.npy')
kept = np.load(out / 'kept_spikes.npy')
times = np.load(out / 'spike_times.npy').ravel()
assert kept.dtype == bool and kept.sum() == times.size, 'kept_spikes does not map full_st to spike_times'
assert np.array_equal(full_st[kept, 0].astype(np.int64), times.astype(np.int64)), 'full_st times do not match spike_times'
np.save(out / 'score.npy', full_st[kept, 2].astype(np.float32))
(out / 'tF.npy').unlink(missing_ok=True)

(out / 'provenance.json').write_text(json.dumps({
    'script': 'tuning/exploratory/autocurate/ks4_sort_scored.py',
    'like': str(args.like),
    'kilosort_path': kilosort.__file__,
    'commit': git('rev-parse', 'HEAD').strip(),
    'kilosort_diff_sha256': hashlib.sha256(diff.encode()).hexdigest(),
    'kilosort_diff': diff,
    'kilosort_untracked_files': untracked,
    'kernels_used': cuda_kernels._lib is not None,
    'kernels_sha256': hashlib.sha256(kernels.read_bytes()).hexdigest() if kernels.exists() else None,
    'settings': settings,
    'data': data,
    'probe': like['probe'],
    'runtime_min': runtime,
}, indent=2))
print(f'Sort finished in {runtime:.1f} min -> {out}', flush=True)
if not args.no_clean:
    print(f"cleaning: {run_cleaning(out, 1200.0)['fraction_spikes_removed']:.2%} of spikes removed", flush=True)
