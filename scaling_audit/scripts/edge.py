import numpy as np, torch
from kilosort.preprocessing import get_drift_matrix
ops = np.load(r'C:/data/ks4_tuning/26648_kobe_2020-08-21/runs/hybrid_default_seed1/ops.npy', allow_pickle=True).item()
ops['iKxx'] = torch.as_tensor(np.asarray(ops['iKxx'])).float()
for s in [0, 10, 25, 50, 100, 150]:
    M = get_drift_matrix(ops, np.array([float(s)]), device=torch.device('cpu')).numpy()
    g = M.sum(1)
    print(s, 'um: channels with gain<0.8:', int((g<0.8).sum()), ' <0.5:', int((g<0.5).sum()), ' of', len(g))
