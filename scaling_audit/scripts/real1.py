import os, sys
os.environ['NUMBA_CACHE_DIR'] = os.path.dirname(os.path.abspath(__file__)) + '/numba_cache'
import numpy as np
from kilosort import CCG
d = sys.argv[1]
st = np.load(os.path.join(d,'spike_times.npy')).astype(np.float64)/30000
clu = np.load(os.path.join(d,'spike_clusters.npy')).astype(np.int64)
lab = open(os.path.join(d,'cluster_KSLabel.tsv')).read().split('\n')[1:]
print('units', clu.max()+1, 'span s', st.max()-st.min())
full_ref, fullR = CCG.refract(clu, st); full_ref = full_ref.astype(bool)
print('full: good', int(full_ref.sum()), 'saved good', sum('good' in l for l in lab))
for Tw in [300, 600, 1200]:
    m = st < Tw
    ref, R = CCG.refract(clu[m], st[m])
    ref = np.pad(ref, (0, len(full_ref)-len(ref)))
    # compare against labels of the 20-min sort: agreement among units with >=50 spikes in window
    n = np.bincount(clu[m], minlength=len(full_ref))
    sel = n >= 50
    print(f'first {Tw:5d}s: good {int(ref.sum()):4d}; of units with >=50 spikes ({sel.sum()}): good->mua {int((full_ref & ~ref.astype(bool) & sel).sum())}, mua->good {int((~full_ref.astype(bool) & ref.astype(bool) & sel).sum())}')
# rate floor: good fraction by rate bin at full
rate = np.bincount(clu, minlength=len(full_ref))/1200
for lo,hi in [(0,0.1),(0.1,0.3),(0.3,1),(1,3),(3,10),(10,1e9)]:
    s=(rate>=lo)&(rate<hi); print(f'rate [{lo},{hi}) Hz: n={s.sum():4d} good frac={full_ref[s].mean() if s.sum() else np.nan:.2f}  median R12={np.median(fullR[s]) if s.sum() else np.nan:.3f}')
