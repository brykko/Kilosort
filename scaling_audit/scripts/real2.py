import os, sys
os.environ['NUMBA_CACHE_DIR'] = os.path.dirname(os.path.abspath(__file__)) + '/numba_cache'
import numpy as np
from kilosort import CCG, swarmsplitter as ss
d = sys.argv[1]
st = np.load(os.path.join(d,'spike_times.npy')).astype(np.float64)/30000
clu = np.load(os.path.join(d,'spike_clusters.npy')).astype(np.int64)
S = np.load(os.path.join(d,'similar_templates.npy'))
N = clu.max()+1
np.fill_diagonal(S, 0)
iu = np.argwhere(np.triu(S >= 0.5))
print('pairs with similarity >= 0.5:', len(iu))
order = np.argsort(clu, kind='stable'); b = np.concatenate(([0], np.cumsum(np.bincount(clu, minlength=N))))
trains = [st[order[b[k]:b[k+1]]] for k in range(N)]
for Tw in [300, 600, 1200]:
    nm=0; nv=0; nvs=0; npair=0
    for a,c in iu:
        t1 = trains[a][trains[a]<Tw]; t2 = trains[c][trains[c]<Tw]
        if len(t1)<2 or len(t2)<2: continue
        npair+=1
        nm += CCG.check_CCG(t1,t2)[1]
        v = ss.check_CCG(t1,t2); nv += v[1]
        nvs += ss.check_CCG(t1,t2,require_significance=True)[1]
    print(f'first {Tw}s: pairs {npair}, final-merge cross_refractory {nm}, splitter veto {nv} (of which significance-backed {nvs})')
