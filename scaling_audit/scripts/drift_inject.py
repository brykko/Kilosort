import torch, glob, numpy as np, time
from kilosort import datashift
torch.set_num_threads(8)
f = sorted(glob.glob(r'C:/data/ks4_tuning/26648_kobe_2020-08-21/search/stage_cache/A_*/drift.pt'))[0]
c = torch.load(f, weights_only=False, map_location='cpu'); st0 = c['out'][1]; base = c['ops']['dshift'][:,0]
NB = 600; ymin, ymax = 580., 4400.
ops = {'yc': np.array([ymin, ymax]), 'binning_depth': 5, 'Nbatches': NB, 'Th_universal': 7,
       'nblocks': 1, 'drift_smoothing': [0.5,0.5,0.5]}
b = st0[:,4].astype(int); t = np.arange(NB)
scen = {}
for D in [50, 100, 150, 250]:
    scen[f'ramp +-{D}'] = D*(t-300)/300          # offset from the batch-300 anchor reaches D at both ends
for D in [60, 100, 150]:
    scen[f'step {D} at b450'] = np.where(t>=450, D, 0.)
for name, d in scen.items():
    st = st0.copy(); st[:,1] += d[b]
    keep = (st[:,1] >= ymin) & (st[:,1] <= ymax); st = st[keep]
    F, ys = datashift.bin_spikes(ops, st)
    imin, yblk, F0, F0m = datashift.align_block2(F, ys, ops, device=torch.device('cpu'))
    est = imin[:,0]*5 - base            # remove the excerpt's own drift estimate
    r = est + d; r = r - np.median(r)   # correction should be -d up to a constant
    print(f'{name:16s} maxerr {np.abs(r).max():6.1f} um  frac|err|>10um {np.mean(np.abs(r)>10):.3f}  p99 {np.percentile(np.abs(r),99):5.1f}')
