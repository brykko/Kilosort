import os, sys
os.environ['NUMBA_CACHE_DIR'] = os.path.dirname(os.path.abspath(__file__)) + '/numba_cache'
import numpy as np
from kilosort import CCG
rng=np.random.default_rng(0)
d = sys.argv[1]
st = np.load(os.path.join(d,'spike_times.npy')).astype(np.float64)/30000
clu = np.load(os.path.join(d,'spike_clusters.npy')).astype(np.int64)
N=clu.max()+1
def halves(maskA):
    ra,Ra = CCG.refract(np.append(clu[maskA],N-1), np.append(st[maskA],0.0))
    rb,Rb = CCG.refract(np.append(clu[~maskA],N-1), np.append(st[~maskA],0.0))
    return ra[:N].astype(bool), Ra[:N], rb[:N].astype(bool), Rb[:N]
contig = st < 600
chunk = np.repeat(rng.random(120) < 0.5, 10)  # 10-s chunks
rand = chunk[np.minimum(st.astype(int),1199)]
n = np.bincount(clu, minlength=N); sel = n>=200
for name,m in [('contiguous halves (0-600 / 600-1200 s)',contig),('random 10-s chunks',rand)]:
    ra,Ra,rb,Rb = halves(m)
    print(f'{name}: label disagreements {int((ra!=rb)[sel].sum())}/{sel.sum()}, median|R12a-R12b| {np.median(np.abs(Ra-Rb)[sel]):.4f}, rate CV between halves {np.median(np.abs(np.log((np.bincount(clu[m],minlength=N)+1)/(np.bincount(clu[~m],minlength=N)+1)))[sel]):.3f} (median |log ratio|)')
