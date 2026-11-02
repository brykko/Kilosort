"""Load the stage cache (A_0abcf04804c6ea85 / B_73b7f44f64601191, the pair used by sort
198_H8_cds10_s1) and the run's ops, compute pass-2 spatial groups exactly as clustering_qr.run
(mode='template') does, and save the Xd + spike times of selected groups to disk."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys, json, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, r'C:\code\github\Kilosort')
from kilosort import clustering_qr

OUT = Path(__file__).parent
ROOT = Path(r'C:\data\ks4_tuning\26648_kobe_2020-08-21')
CACHE = ROOT / 'search' / 'stage_cache'
RUN = ROOT / 'runs' / 'search' / '198_H8_cds10_s1' / 'sort'

t0 = time.time()
ops = np.load(RUN / 'ops.npy', allow_pickle=True).item()
for f in [CACHE / 'A_0abcf04804c6ea85' / 'drift.pt', CACHE / 'A_0abcf04804c6ea85' / 'detect1.pt']:
    c = torch.load(f, weights_only=False, map_location='cpu')
    print(f.name, 'ops keys:', sorted(c['ops'].keys()))
    ops.update(c['ops'])
    del c
c = torch.load(CACHE / 'B_73b7f44f64601191' / 'extract.pt', weights_only=False, map_location='cpu')
print('extract ops keys:', sorted(c['ops'].keys()))
ops.update(c['ops'])
st, tF = c['out'][0], c['out'][1]
del c
print('loaded', time.time() - t0, 's; st', st.shape, type(st), 'tF', tF.shape, tF.dtype)
for k in ['iU', 'iCC', 'xc', 'yc', 'xcup', 'ycup', 'dmin', 'dminx', 'fs', 'Nchan']:
    v = ops[k]
    print(k, type(v), getattr(v, 'shape', v) if not np.isscalar(v) else v)
if not torch.is_tensor(ops['iU']):
    ops['iU'] = torch.from_numpy(np.asarray(ops['iU']))
if not torch.is_tensor(ops['iCC']):
    ops['iCC'] = torch.from_numpy(np.asarray(ops['iCC']))
print('settings:', json.dumps({k: v for k, v in ops['settings'].items()
                              if 'cluster' in k or 'tree' in k or 'merge' in k or 'correlogram' in k}, default=str))
st = np.asarray(st)
fs = ops['fs']
print('t range (s):', st[:, 0].min() / fs, st[:, 0].max() / fs, 'monotonic:', bool(np.all(np.diff(st[:, 0]) >= 0)))

# --- groups as in clustering_qr.run(mode='template')
xy, iC = clustering_qr.xy_templates(ops)
iclust_template = st[:, 1].astype('int32')
ycent = clustering_qr.y_centers(ops)
xcent = clustering_qr.x_centers(ops)
nearest_center, _, _ = clustering_qr.get_nearest_centers(xy, xcent, ycent)
nearest_center = nearest_center.numpy()
ntemp = xy.shape[1]
cnt_t = np.bincount(iclust_template, minlength=ntemp)
groups = []
for jj in range(len(xcent)):
    for kk in range(len(ycent)):
        ii = kk + jj * ycent.size
        ix = nearest_center == ii
        if not ix.any():
            continue
        groups.append({'ii': int(ii), 'jj': jj, 'kk': kk, 'ntemp': int(ix.sum()), 'n': int(cnt_t[ix].sum())})
sizes = np.array([g['n'] for g in groups])
print('n groups', len(groups), 'total spikes', sizes.sum(), 'pctl 25/50/75/90/max:',
      np.percentile(sizes, [25, 50, 75, 90, 100]).astype(int))
json.dump(groups, open(OUT / 'groups.json', 'w'))

# pick groups nearest the 25/50/75/90th percentiles plus 10th and 97th
picks = []
order = np.argsort(sizes)
for p in [10, 25, 50, 75, 90, 97]:
    target = np.percentile(sizes, p)
    cand = [i for i in np.argsort(np.abs(sizes - target)) if i not in picks]
    picks.append(int(cand[0]))
PID = torch.from_numpy(iclust_template).long()
tF = torch.as_tensor(tF)
for p, gi in zip([10, 25, 50, 75, 90, 97], picks):
    g = groups[gi]
    ix = torch.from_numpy(nearest_center == g['ii'])
    Xd, igood, ichan = clustering_qr.get_data_cpu(ops, xy, iC, PID, tF, ycent[g['kk']], xcent[g['jj']],
                                                  dmin=ops['dmin'], dminx=ops['dminx'], ix=ix)
    st0 = st[igood.numpy(), 0] / fs
    print(f'p{p}: group {g["ii"]} n={Xd.shape[0]} dim={Xd.shape[1]} ntemp={g["ntemp"]}')
    np.savez(OUT / f'group_p{p}_{g["ii"]}.npz', Xd=Xd.numpy().astype(np.float32), st0=st0, ii=g['ii'], pctl=p)
json.dump({'settings': {k: v for k, v in ops['settings'].items()}, 'fs': fs,
           'tmax_s': float(st[:, 0].max() / fs)}, open(OUT / 'settings.json', 'w'), default=str, indent=1)
print('done', time.time() - t0, 's')
