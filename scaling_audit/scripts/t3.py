import os, sys
os.environ['NUMBA_CACHE_DIR'] = os.path.dirname(os.path.abspath(__file__)) + '/numba_cache'
import numpy as np
from kilosort import CCG, swarmsplitter as ss
rng = np.random.default_rng(0)
d = sys.argv[1]; T = 1200.0; mid = 600.0
st = np.load(os.path.join(d,'spike_times.npy')).astype(np.float64)/30000
clu = np.load(os.path.join(d,'spike_clusters.npy')).astype(np.int64)
S = np.load(os.path.join(d,'similar_templates.npy')); np.fill_diagonal(S, 0)
N = clu.max()+1
good = np.array([l.split('\t')[1]=='good' for l in open(os.path.join(d,'cluster_KSLabel.tsv')).read().strip().split('\n')[1:]])
n = np.bincount(clu, minlength=N); rate = n/T
order = np.argsort(clu, kind='stable'); b = np.concatenate(([0], np.cumsum(n)))
tr = [st[order[b[k]:b[k+1]]] for k in range(N)]
def tests(a, c):
    return (CCG.check_CCG(a, c)[1], ss.check_CCG(a, c)[1], ss.check_CCG(a, c, require_significance=True)[1])
def split(t, ov):
    inA = t < mid+ov/2; inB = t >= mid-ov/2; both = inA & inB; pick = rng.random(len(t)) < 0.5
    return t[inA & (~both | pick)], t[inB & (~both | ~pick)]
units = np.nonzero(good & (rate >= 5))[0]
print(f'same-unit split: {len(units)} good units with rate >= 5 Hz (median {np.median(rate[units]):.1f} Hz)')
print('  overlap   final-merge   veto(sig=False)   veto(sig=True)')
for ov in [0, 60, 300]:
    r = np.array([tests(*split(tr[k], ov)) for k in units])
    print(f'  {ov:5d} s     {r[:,0].mean():.2f}          {r[:,1].mean():.2f}             {r[:,2].mean():.2f}')
# different neurons, template-similar, both good
for thr in [0.8, 0.5]:
    iu = np.argwhere(np.triu(S >= thr))
    iu = [(i,j) for i,j in iu if good[i] and good[j] and rate[i] >= 1 and rate[j] >= 1]
    print(f'different neurons: {len(iu)} good pairs, similarity >= {thr}, both >= 1 Hz')
    print('  config                      final-merge   veto(sig=False)   veto(sig=True)')
    for lab, f in [('full overlap (as sorted)', lambda a,c: (a,c)),
                   ('A<600 s, B>=600 s (disjoint)', lambda a,c: (a[a<mid], c[c>=mid])),
                   ('A<630, B>=570 (60 s overlap)', lambda a,c: (a[a<mid+30], c[c>=mid-30])),
                   ('A<750, B>=450 (300 s overlap)', lambda a,c: (a[a<mid+150], c[c>=mid-150]))]:
        r = np.array([tests(*f(tr[i], tr[j])) for i,j in iu])
        print(f'  {lab:28s} {r[:,0].mean():.2f}          {r[:,1].mean():.2f}             {r[:,2].mean():.2f}')
