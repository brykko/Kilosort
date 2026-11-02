"""Run ks25_sort_traced.m (KS2.5 with autocurate's stages saved), then adapt its output as ks25_sort.py
does, clean it (cleaning.py) and convert stages.mat to stages.npz (one array per stage and field)."""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat

HERE = Path(__file__).parent
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cleaning import run_cleaning  # noqa: E402
from ks25_sort import MATLAB, adapt  # noqa: E402

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--meta', required=True)
    p.add_argument('--probe', required=True)
    p.add_argument('--out', required=True, help='sort folder (must not exist)')
    p.add_argument('--seed', type=int, default=1)
    args = p.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f'{out} already exists')
    out.mkdir(parents=True)
    q = lambda s: str(s).replace("'", "''")  # noqa: E731
    cmd = f"addpath('{q(HERE)}'); ks25_sort_traced('{q(args.data)}', '{q(args.meta)}', '{q(out)}', {args.seed})"
    t0 = time.time()
    with open(out / 'ks25_sort.log', 'w') as f:
        subprocess.run([str(MATLAB), '-batch', cmd], check=True, stdout=f, stderr=subprocess.STDOUT)
    print(f'sorted in {(time.time() - t0) / 60:.1f} min; adapt: {adapt(out, args.probe)}', flush=True)
    S = loadmat(out / 'stages.mat', squeeze_me=True, struct_as_record=False)
    np.savez(out / 'stages.npz', **{f'{st}_{f}': np.asarray(getattr(S[st], f)).ravel()
                                    for st in S if not st.startswith('__') for f in ('t', 'clu', 'amp', 'score')})
    (out / 'stages.mat').unlink()
    print(f"cleaning: {run_cleaning(out, 1200.0)['fraction_spikes_removed']:.2%} of spikes removed", flush=True)
