"""Per-batch breakdown of the two batch loops (universal-template detection and
learned-template matching), using ops.npy and the U snapshot from a profile_run.

Usage: python bench_detect.py RUN_DIR [--n 40]
"""
import argparse
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('run')
p.add_argument('--repo', default=r'C:\code\github\Kilosort')
p.add_argument('--n', type=int, default=40, help='number of batches, evenly spaced')
args = p.parse_args()
sys.path.insert(0, args.repo)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from kilosort import io, spikedetect, template_matching  # noqa: E402

dev = torch.device('cuda')
run = Path(args.run)
ops = io.load_ops(run / 'ops.npy', device=dev)
bfile = io.bfile_from_ops(ops, device=dev)
raw = io.BinaryRWFile  # parent class: read + H2D only

batches = np.linspace(1, bfile.n_batches - 2, args.n).astype(int)
sync = torch.cuda.synchronize


def timed(fn, *a, **k):
    sync(); t = time.perf_counter(); r = fn(*a, **k); sync()
    return r, time.perf_counter() - t


# warm up (cuDNN autotune, allocator, page cache for these batches)
for ib in batches[:3]:
    X = bfile.padded_batch_to_torch(int(ib), ops)
    spikedetect.template_match(X, ops, ops['iC'], ops['iC2'], ops['weigh'], device=dev)

U = torch.load(run / 'snapshot' / 'U_first.pt').to(dev)
template_matching.prepare_extract(ops['xc'], ops['yc'], U, ops['settings']['nearest_chans'],
                                  ops['settings']['position_limit'], device=dev)
ctc = template_matching.prepare_matching(ops, U)

rows = []
for ib in batches:
    ib = int(ib)
    # 1. raw read from memmap (CPU)
    bstart, bend = bfile._get_batch_edges(ib)
    sync(); t0 = time.perf_counter()
    data = np.asarray(bfile.file[bstart:bend])
    t_read = time.perf_counter() - t0
    # 2. KS4's transpose + H2D path (strided CPU tensor -> GPU float)
    t0 = time.perf_counter()
    X = torch.zeros((bfile.n_chan_bin, bfile.NT + 2 * bfile.nt), device=dev)
    X[:] = torch.from_numpy(data.T).to(dev).float()
    sync(); t_h2d = time.perf_counter() - t0
    # 2b. alternative: contiguous int16 H2D, transpose + cast on GPU
    t0 = time.perf_counter()
    X2 = torch.from_numpy(data).to(dev).T.float()
    sync(); t_h2d_alt = time.perf_counter() - t0
    assert torch.equal(X, X2)
    # 3. filter (CAR, FFT high-pass, whitening + drift matrix)
    Xf, t_filt = timed(bfile.filter, X, ops, ib)
    # 4. universal-template matching
    (xy, imax, amp, adist), t_tm = timed(spikedetect.template_match, Xf, ops, ops['iC'], ops['iC2'],
                                          ops['weigh'], device=dev)
    # 5. learned-template matching (peeling)
    (stt, amps, th_amps, Xres), t_rm = timed(template_matching.run_matching, ops, Xf, U, ctc, device=dev)
    rows.append((t_read, t_h2d, t_h2d_alt, t_filt, t_tm, t_rm, len(xy), len(stt)))

r = np.array(rows)
names = ['read (memmap)', 'H2D (KS4 path)', 'H2D (int16, GPU transpose)', 'filter',
         'template_match (univ)', 'run_matching (learned)']
print(f'{len(batches)} batches; ms per batch, median [min-max]')
for i, n in enumerate(names):
    print(f'  {n:28s} {1e3 * np.median(r[:, i]):8.1f}  [{1e3 * r[:, i].min():.1f}-{1e3 * r[:, i].max():.1f}]')
print(f'  spikes/batch univ {np.median(r[:, 6]):.0f}, learned {np.median(r[:, 7]):.0f}')
nb = bfile.n_batches
print('Projected full-pass seconds (x{} batches):'.format(nb))
print(f'  univ pass  ~ {nb * np.median(r[:, :2].sum(1) + r[:, 3] + r[:, 4]):.0f} s (read+H2D+filter+match)')
print(f'  learned    ~ {nb * np.median(r[:, :2].sum(1) + r[:, 3] + r[:, 5]):.0f} s')
