import os, sys
os.environ['NUMBA_CACHE_DIR'] = os.path.dirname(os.path.abspath(__file__)) + '/numba_cache'
import numpy as np
from kilosort import CCG, swarmsplitter as ss
d = sys.argv[1]; T = 1200.0
o = np.load(os.path.join(d,'ops.npy'), allow_pickle=True).item()
st = np.load(os.path.join(d,'spike_times.npy')).astype(np.float64)/30000
clu = np.load(os.path.join(d,'spike_clusters.npy')).astype(np.int64)
tpl = np.load(os.path.join(d,'spike_detection_templates.npy')).astype(np.int64)
S = np.load(os.path.join(d,'similar_templates.npy')); np.fill_diagonal(S, 0)
N = clu.max()+1
dmin = float(o['dmin']); ycup = o['ycup']
ycent = np.arange(ycup.min()+dmin-1, ycup.max()+dmin+1, 2*dmin)
ty = o['yc'][o['iU']]                       # template y = its peak channel (xy_templates)
tg = np.argmin(np.abs(ty[:,None]-ycent[None,:]), 1)
sg = tg[tpl]
grp = np.zeros(N, int); purity = np.zeros(N)
for k in range(N):
    c = np.bincount(sg[clu==k]); grp[k] = c.argmax(); purity[k] = c.max()/c.sum()
n = np.bincount(clu, minlength=N).astype(float)
print(f'units {N}, groups used {len(np.unique(grp))}, slab {2*dmin:.0f} um, median unit purity (spikes in modal group) {np.median(purity):.2f}, units with purity<0.9: {(purity<0.9).sum()}')
order = np.argsort(clu, kind='stable'); b = np.concatenate(([0], np.cumsum(np.bincount(clu, minlength=N))))
tr = [st[order[b[k]:b[k+1]]] for k in range(N)]
pairs = [(i,j) for g in np.unique(grp) for idx in [np.nonzero(grp==g)[0]] for a,i in enumerate(idx) for j in idx[a+1:]]
pairs = np.array(pairs); q = n[pairs[:,0]]*n[pairs[:,1]]*1e-3/T
sim = S[pairs[:,0], pairs[:,1]]
def regimes(qq):
    return [np.mean(qq<0.25), np.mean((qq>=0.25)&(qq<3)), np.mean(qq>=3)]
print(f'same-group pairs: {len(pairs)}  (sim>=0.5: {(sim>=0.5).sum()}, sim>=0.8: {(sim>=0.8).sum()})')
print('fractions [sparse Q00<0.25, dead zone 0.25-3, powered >=3] (expected Q00 = n1 n2 1ms / T, stationary extrapolation)')
for lab, m in [('all same-group pairs', np.ones(len(q),bool)), ('sim >= 0.5', sim>=0.5), ('sim >= 0.8', sim>=0.8)]:
    print(f'  {lab:22s} 20 min {np.round(regimes(q[m]),3)}   1 h {np.round(regimes(3*q[m]),3)}   3 h {np.round(regimes(9*q[m]),3)}')
# per unit: most template-similar same-group partner
best = {}
for (i,j),s,qq in zip(pairs, sim, q):
    for u in (i,j):
        if s > best.get(u,(-1,0))[0]: best[u] = (s,qq)
qb = np.array([v[1] for v in best.values()])
print(f'  per unit, most similar same-group partner ({len(qb)} units): 20 min {np.round(regimes(qb),3)}  1 h {np.round(regimes(3*qb),3)}  3 h {np.round(regimes(9*qb),3)}')
rate = n/T
print(f'  unit rates: median {np.median(rate):.2f} Hz; <0.5 Hz {np.mean(rate<0.5):.2f}; <1 Hz {np.mean(rate<1):.2f}')
# actual tests on 20-min trains for template-similar pairs
m = sim >= 0.5
vs=vg=mg=0; Qe=[]
for i,j in pairs[m]:
    a,c = tr[i], tr[j]
    K,TT = CCG.compute_CCG(a,c); R12,Q12,Q00 = CCG.CCG_metrics(a,c,K,TT,nbins=500,tbin=1/1000); Qe.append(Q00)
    v = ss.check_CCG(a,c)[1]; vsig = ss.check_CCG(a,c,require_significance=True)[1]
    vs += v and not vsig; vg += vsig; mg += CCG.check_CCG(a,c)[1]
Qe=np.array(Qe)
print(f'  sim>=0.5 pairs, actual 20-min tests: veto via sparse shortcut only {vs}, veto significance-backed {vg}, final-merge test passes {mg}; empirical Q00 regimes {np.round(regimes(Qe),3)}')
