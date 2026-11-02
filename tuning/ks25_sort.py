"""Kilosort 2.5 sort of an excerpt with a neuropixels pipeline preset ("standard", Gardner et al. 2022,
or "standard_low"), adapted to the harness's conventions (spec §13, the KS2.5 comparison).

    python ks25_sort.py --data <excerpt .bin> --meta <source.ap.meta> --probe <KS4 probe .json>
        --out <sort folder> --seed <replicate> [--preset standard|standard_low]

Sorts with ks25_sort.m in MATLAB R2025a (the pipeline's own code from C:\\code\\github\\neuropixels;
see that file for the differences from the pipeline), then adapts rezToPhy's output so the harness
reads it as it reads a KS4 sort:

- clusters relabelled 0..n-1 over the clusters that have spikes, so that cluster c is row c of
  templates.npy, as in KS4 (autocurate's merges leave empty templates, its splits append new ones);
- templates.npy: each cluster's template without the 21 leading zero samples rezToPhy pads for Phy,
  i.e. 61 samples with the trough at sample 20, KS4's layout (whitened, as in KS4; un-whitened by
  waveform.py with whitening_mat_inv.npy);
- spike_positions.npy, which rezToPhy does not write: KS4's spike-position rule
  (postprocessing.compute_spike_positions: among the 10 channels nearest the unit's main channel,
  those within 100 um, weighted by the norm of the features there) applied to the unit's whitened
  template: main channel = the channel with the largest L2 norm, weights = each channel's L2 norm over
  time. Every spike of a unit gets its unit's position. The harness uses spike positions only
  through each unit's median (§6.2 span, §6.3 footprint classes, depth matching), so a position per
  unit is enough.

rezToPhy's own files are kept with the suffix _rezToPhy; pc_features.npy and template_features.npy
(several GB, read by nothing in the harness) are deleted, as run_sort.py skips them for KS4.
Checks that the channel positions equal the KS4 probe file's, so units of both sorters share one
frame. Writes provenance.json (neuropixels commit and state, KS2.5 code and mex hashes, seed,
runtime).
"""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NPX = Path(r'C:\code\github\neuropixels')
KS25 = 'ExternalToolboxes/KilosortToolbox/Versions/kilosort_v2.5'
MEX = 'ExternalToolboxes/KilosortToolbox/mexcuda/kilosort_v2.5/v12.9'
MATLAB = Path(r'C:\Program Files\MATLAB\R2025a\bin\matlab.exe')
PAD = 21                 # zero samples rezToPhy prepends (1 + nt0min)
NEAREST = 10             # KS4 nearest_chans
POSITION_LIMIT = 100.0   # KS4 position_limit (um)


def unit_positions(templates, xy):
    """KS4's spike-position rule applied to each (whitened) template: units x 2 (x, y in um)."""
    norm = np.linalg.norm(templates, axis=1)                        # units x channels
    main = norm.argmax(1)
    d2 = ((xy[None, :, :] - xy[:, None, :]) ** 2).sum(-1)          # channels x channels
    near = np.argsort(d2, axis=1, kind='stable')[:, :NEAREST]
    chs = near[main]                                                # units x NEAREST
    w = np.take_along_axis(norm, chs, 1) * (np.take_along_axis(d2[main], chs, 1) < POSITION_LIMIT ** 2)
    w /= w.sum(1, keepdims=True)
    return np.stack([(xy[chs, 0] * w).sum(1), (xy[chs, 1] * w).sum(1)], 1)


def adapt(sort, probe):
    sort = Path(sort)
    pr = json.loads(Path(probe).read_text())
    cmap, xy = np.load(sort / 'channel_map.npy').ravel(), np.load(sort / 'channel_positions.npy')
    if not (np.array_equal(cmap, pr['chanMap']) and np.allclose(xy, np.c_[pr['xc'], pr['yc']])):
        raise SystemExit('KS2.5 channel map or positions differ from the KS4 probe file')
    for f in ('spike_clusters', 'templates'):
        (sort / f'{f}.npy').rename(sort / f'{f}_rezToPhy.npy')
    clu0 = np.load(sort / 'spike_clusters_rezToPhy.npy').ravel()
    W0 = np.load(sort / 'templates_rezToPhy.npy')                   # templates x 82 x channels
    if np.any(W0[:, :PAD]):
        raise SystemExit('rezToPhy templates are not zero-padded as expected')
    ids, clu = np.unique(clu0, return_inverse=True)
    W = np.ascontiguousarray(W0[ids, PAD:], np.float32)
    pos = unit_positions(W, xy)
    np.save(sort / 'spike_clusters.npy', clu.astype(np.int32))
    np.save(sort / 'templates.npy', W)
    np.save(sort / 'cluster_template.npy', ids.astype(np.int32))  # cluster -> rezToPhy template
    np.save(sort / 'spike_positions.npy', pos[clu].astype(np.float32))
    for f in ('pc_features.npy', 'template_features.npy'):
        (sort / f).unlink(missing_ok=True)
    return {'n_clusters': int(ids.size), 'n_templates': int(W0.shape[0]), 'n_spikes': int(clu.size)}


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='excerpt .bin (385 channels)')
    p.add_argument('--meta', required=True, help="source recording's SpikeGLX AP metadata")
    p.add_argument('--probe', required=True, help='KS4 probe .json of the excerpt (frame check)')
    p.add_argument('--out', required=True, help='sort folder (must not exist)')
    p.add_argument('--seed', type=int, required=True, help='replicate: learnAndSolve8b seed (pipeline: 1)')
    p.add_argument('--preset', choices=('standard', 'standard_low'), default='standard', help='pipeline preset')
    args = p.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f'{out} already exists')
    out.mkdir(parents=True)
    git = lambda *a: subprocess.run(['git', '-C', str(NPX), *a], capture_output=True, text=True, check=True).stdout
    q = lambda s: str(s).replace("'", "''")
    cmd = f"addpath('{q(HERE)}'); ks25_sort('{q(args.data)}', '{q(args.meta)}', '{q(out)}', {args.seed}, '{args.preset}')"
    t0 = time.time()
    with open(out / 'ks25_sort.log', 'w') as f:
        subprocess.run([str(MATLAB), '-batch', cmd], check=True, stdout=f, stderr=subprocess.STDOUT)
    minutes = (time.time() - t0) / 60
    stats = adapt(out, args.probe)
    (out / 'provenance.json').write_text(json.dumps({
        'sorter': f'Kilosort 2.5, neuropixels pipeline preset "{args.preset}"', 'preset': args.preset,
        'neuropixels_commit': git('rev-parse', 'HEAD').strip(),
        'neuropixels_status': git('status', '--porcelain'),
        'ks25_tree': git('rev-parse', f'HEAD:{KS25}').strip(),
        'mex_sha256': {m.name: sha256(m) for m in sorted((NPX / MEX).glob('*.mexw64'))},
        'driver_sha256': sha256(HERE / 'ks25_sort.m'),
        'seed': args.seed, 'data': args.data, 'meta': args.meta, 'probe': args.probe,
        'runtime_min': minutes, **stats,
    }, indent=2))
    print(f'KS2.5 sort finished in {minutes:.1f} min -> {out}: {stats}')
