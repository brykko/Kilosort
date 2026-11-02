import os, sys
os.environ['NUMBA_CACHE_DIR'] = os.path.dirname(os.path.abspath(__file__)) + '/numba_cache'
import numpy as np
from kilosort import CCG
for d in sys.argv[1:]:
    st = np.load(os.path.join(d,'spike_times.npy')).astype(np.float64)/30000
    clu = np.load(os.path.join(d,'spike_clusters.npy')).astype(np.int64)
    N = clu.max()+1
    g20, R20 = CCG.refract(clu, st); g20 = g20.astype(bool)
    saved = sum(l.endswith('good') for l in open(os.path.join(d,'cluster_KSLabel.tsv')).read().split('\n')[1:])
    m = st < 600
    g10, R10 = CCG.refract(clu[m], st[m]); g10 = np.pad(g10, (0, N-len(g10))).astype(bool)
    n10 = np.bincount(clu[m], minlength=N); sel = n10 >= 50
    gm = int((g20 & ~g10 & sel).sum()); mg = int((~g20 & g10 & sel).sum())
    f = (gm+mg)/sel.sum()
    name = d.split('runs/')[-1]
    print(f'{name:40s} units {N:5d} good20 {int(g20.sum()):5d} (saved {saved}) good10 {int(g10.sum()):5d} | eligible {sel.sum():5d}: good->mua {gm:3d} mua->good {mg:3d} flip {100*f:.1f}% '
          f'| predicted 20min-vs-1h {100*f*np.sqrt(2/3):.1f}%  vs-3h {100*f*np.sqrt(8/9):.1f}%  vs-inf {100*f:.1f}%')
