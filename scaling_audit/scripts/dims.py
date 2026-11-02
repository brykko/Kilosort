import numpy as np, sys, torch
sys.path.insert(0, r'C:\code\github\Kilosort')
from kilosort import clustering_qr as cq
ops = np.load(r'C:\data\ks4_tuning\26648_kobe_2020-08-21\runs\hybrid_default_seed1\ops.npy', allow_pickle=True).item()
print({k: ops[k] for k in ('dmin','dminx')}, ops['settings'].get('nearest_chans'))
for mode in ('spikes','template'):
    xy, iC = (cq.xy_up(ops) if mode=='spikes' else cq.xy_templates(ops))
    iC = torch.as_tensor(np.asarray(iC)) if not torch.is_tensor(iC) else iC
    yc, xc = cq.y_centers(ops), cq.x_centers(ops)
    near,_,_ = cq.get_nearest_centers(xy, xc, yc)
    D = [torch.unique(iC[:, near==ii]).numel()*6 for ii in np.unique(near)]
    print(mode, 'groups', len(D), 'x_centers', xc, 'D pct', np.percentile(D,[0,50,100]))
