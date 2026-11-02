import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, r'C:\code\github\Kilosort')
import numpy as np
from kilosort.swarmsplitter import check_split

# Synthetic single Gaussian unit, D features, split into two "clusters" (leaf labels)
D = 30
rng = np.random.default_rng(0)
xtree = np.array([[0, 1, 2]])
my_clus = [np.array([0]), np.array([1]), np.array([0, 1])]
reps = 30
print('labels  n      median  p90   frac>=0.6')
for mode in ['halfspace']:
    for n in [400, 500, 600, 800, 1000]:
        sc = []
        for r in range(reps):
            X = rng.standard_normal((n, D)).astype('float32')
            # mimic feature scale: add a common mean, anisotropic variance
            X = X * np.linspace(3, 0.3, D) + 5
            if mode == 'halfspace':
                v = rng.standard_normal(D)
                p = (X - X.mean(0)) @ v
                iclust = (p > np.median(p)).astype(int)
            else:
                iclust = rng.integers(0, 2, n)
            _, s = check_split(X.copy(), 0, xtree, iclust, my_clus)
            sc.append(s)
        sc = np.array(sc)
        print(f'{mode:9s} {n:6d}  {np.median(sc):.3f}  {np.percentile(sc,90):.3f}  {np.mean(sc>=0.6):.2f}')
