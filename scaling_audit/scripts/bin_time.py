import numpy as np, time
from kilosort import datashift
rng = np.random.default_rng(0)
for nb in [300, 600, 1200]:
    n = nb * 6800
    st = np.zeros((n, 6)); st[:,4] = np.sort(rng.integers(0, nb, n)); st[:,1] = rng.uniform(580, 4400, n); st[:,2] = rng.uniform(9, 60, n)
    ops = {'yc': np.array([580., 4400.]), 'binning_depth': 5, 'Nbatches': nb, 'Th_universal': 9}
    t = time.time(); F, ys = datashift.bin_spikes(ops, st); print(nb, n, F.shape, f'{time.time()-t:.1f}s')
