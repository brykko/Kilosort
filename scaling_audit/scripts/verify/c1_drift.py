import os, sys, time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, r'C:\code\github\Kilosort')
import numpy as np
import torch
torch.set_num_threads(8)
from kilosort.datashift import align_block2, bin_spikes

rng = np.random.default_rng(3)
Nb = 1200          # 40 min of 2 s batches
yc = np.arange(0, 3840, 10.0)   # NP1-like extent (y only matters)
nunits = 150
udep = rng.uniform(100, 3740, nunits)
uamp = 10 ** rng.uniform(np.log10(10), np.log10(60), nunits)
urate = rng.uniform(1, 10, nunits) * 2     # spikes per 2 s batch

def make_st(drift):
    rows = []
    for t in range(Nb):
        n = rng.poisson(urate)
        u = np.repeat(np.arange(nunits), n)
        dep = udep[u] + drift[t] + rng.normal(0, 5, len(u))
        amp = np.maximum(9.5, uamp[u] * np.exp(rng.normal(0, .1, len(u))))
        ok = (dep > yc.min()) & (dep < yc.max())
        st = np.zeros((ok.sum(), 6))
        st[:, 1] = dep[ok]; st[:, 2] = amp[ok]; st[:, 4] = t
        rows.append(st)
    return np.concatenate(rows)

ops = {'yc': yc, 'binning_depth': 5, 'Nbatches': Nb, 'Th_universal': 9,
       'nblocks': 1, 'drift_smoothing': [0.5, 0.5, 0.5]}
tt = np.arange(Nb)
scen = {
    'linear 100um': 100 * tt / Nb,
    'linear 300um': 300 * tt / Nb,
    'linear 600um': 600 * tt / Nb,
    'step 100um @b800': 100.0 * (tt >= 800),
    'step 200um @b800': 200.0 * (tt >= 800),
    'step 400um @b800': 200.0 * (tt >= 800) + 200.0 * (tt >= 1000),
}
for name, d in scen.items():
    t0 = time.time()
    st = make_st(d)
    F, ysamp = bin_spikes(ops, st)
    imin, yblk, _, _ = align_block2(F, ysamp, ops, device=torch.device('cpu'))
    dshift = imin[:, 0] * ops['binning_depth']
    # correction should be -(drift) up to a constant offset
    err = dshift + d
    err = err - np.median(err)
    print(f'{name:18s} max|err|={np.abs(err).max():6.1f} um  frac batches |err|>10um={np.mean(np.abs(err)>10):.3f}  '
          f'dshift range [{dshift.min():.0f},{dshift.max():.0f}] median offset {np.median(dshift + d):.0f}  ({time.time()-t0:.0f}s)')
