import numpy as np, sys
d = sys.argv[1]
t = np.load(d+'/spike_times.npy').astype(np.int64)/30000.; c = np.load(d+'/spike_clusters.npy')
pf = np.load(d+'/pc_features.npy', mmap_mode='r')   # (n, n_pcs, nC)
T = t.max(); nb = 10; b = np.minimum((t/T*nb).astype(int), nb-1)   # 2-min bins
order = np.argsort(c, kind='stable'); bd = np.r_[0, np.cumsum(np.bincount(c))]
res=[]; amp=[]
for u in range(c.max()+1):
    i = order[bd[u]:bd[u+1]]
    if len(i) < 3000: continue
    bb = b[i]
    if np.bincount(bb, minlength=nb).min() < 100: continue
    F = np.asarray(pf[i]).reshape(len(i), -1)
    M = np.stack([F[bb==k].mean(0) for k in range(nb)])
    Mn = M/np.linalg.norm(M,axis=1,keepdims=True)
    C = Mn @ Mn.T
    # mean 1-cos as function of lag (bins)
    res.append([np.mean([1-C[k,k+l] for k in range(nb-l)]) for l in range(1,nb)])
    amp.append(np.log(np.linalg.norm(M[-1])/np.linalg.norm(M[0])))
r = np.array(res); print('units', len(r))
for l in [1,3,5,9]:
    print('lag %2d min: 1-cos(bin means) pct 50/90/99 ='%(2*l), np.round(np.percentile(r[:,l-1],[50,90,99]),4))
# noise floor estimate: lag-1 value includes mean-estimation noise; growth = lag9 - lag1
g = r[:,8]-r[:,0]; print('growth (lag18 - lag2) pct 50/90/99', np.round(np.percentile(g,[50,90,99]),4))
amp=np.array(amp); print('log |mean feature| last/first bin pct 5/50/95', np.round(np.percentile(amp,[5,50,95]),3))
