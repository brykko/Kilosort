"""A long sort restricted to its 20-min excerpt's window, as a sort folder the harness scores like a 20-min
sort: spikes in [t_exc - t_seg, + duration) of the segment, times shifted to the excerpt's origin. Per-spike
files (spike_*.npy, amplitudes.npy, score.npy; KS4 or KS2.5) are filtered; unit ids, templates and the other
files are hard-linked (as in make_variants.py). Cleaning and evaluation outputs are left out, so the scoring
tools clean the window afresh, as for a 20-min sort.

--cut W first applies the set_cutoff-like cut (strict, one pass, cut-off lowered from Th_learned + W to
Th_learned) to the whole long sort, so each unit's refractory statistics come from the whole segment.

    python window.py --sort <long sort> --out <folder> --t-seg 354 --t-exc 1668 [--duration 1200] [--cut W]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'autocurate'))
from make_variants import SKIP  # noqa: E402
from set_cutoff import set_cutoff  # noqa: E402

FS = 30000

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--sort', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--t-seg', type=float, required=True, help='recording time of the segment start (s)')
    p.add_argument('--t-exc', type=float, required=True, help='recording time of the excerpt start (s)')
    p.add_argument('--duration', type=float, default=1200.0)
    p.add_argument('--cut', type=float, help='set_cutoff-like cut of width W first')
    args = p.parse_args()
    src, dst = Path(args.sort), Path(args.out)
    s0 = round((args.t_exc - args.t_seg) * FS)
    n = round(args.duration * FS)
    assert abs(s0 - (args.t_exc - args.t_seg) * FS) < 1e-6 and s0 >= 0

    t = np.load(src / 'spike_times.npy').ravel()
    keep = (t >= s0) & (t < s0 + n)
    info = {'source': str(src), 'window_samples_half_open': [s0, s0 + n], 'n_long': int(t.size),
            'n_window_uncut': int(keep.sum())}
    if args.cut is not None:
        th = json.loads((src / 'provenance.json').read_text())['settings'].get('Th_learned', 8)
        clu = np.load(src / 'spike_clusters.npy').ravel().astype(np.int64)
        score = np.load(src / 'score.npy').astype(np.float64)
        kc, _ = set_cutoff(t.astype(np.float64), clu, score, FS, (th + args.cut, th), strict=True)
        keep &= kc
        info.update(cut_width=args.cut, Th_learned=th, long_kept_by_cut=float(kc.mean()))
    info['n_window'] = int(keep.sum())

    dst.mkdir(parents=True, exist_ok=False)
    for f in src.iterdir():
        if not f.is_file() or f.name in SKIP + ('window.json',):
            continue
        if f.suffix == '.npy' and (f.name.startswith('spike_') or f.name in ('amplitudes.npy', 'score.npy')):
            a = np.load(f)
            assert a.shape[0] == t.size, f'{f.name} is not per spike'
            a = a[keep]
            if f.name == 'spike_times.npy':
                a = a - a.dtype.type(s0)
            np.save(dst / f.name, a)
        else:
            os.link(f, dst / f.name)
    (dst / 'window.json').write_text(json.dumps(info, indent=2))
    print(f"{dst.name}: {info['n_window']} of {info['n_long']} spikes"
          + (f", cut kept {info['long_kept_by_cut']:.1%} of the long sort" if args.cut is not None else ''), flush=True)
