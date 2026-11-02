exec(open('t3.py').read().split("units = np.nonzero")[0])
units = np.nonzero(good & (rate >= 5))[0]
res = np.array([tests(*split(tr[k], 0)) for k in units])
for lo,hi in [(5,10),(10,20),(20,40),(40,1e9)]:
    s=(rate[units]>=lo)&(rate[units]<hi)
    print(f'same unit, 0 s overlap, rate [{lo},{hi}) n={s.sum()}: final-merge {res[s,0].mean():.2f}')
# different neurons disjoint, pairs by geometric-mean rate (any good pair with sim>=0.5)
iu = [(i,j) for i,j in np.argwhere(np.triu(S>=0.5)) if good[i] and good[j]]
gm = np.array([np.sqrt(rate[i]*rate[j]) for i,j in iu])
r = np.array([tests(tr[i][tr[i]<mid], tr[j][tr[j]>=mid]) for i,j in iu])
for lo,hi in [(0,5),(5,10),(10,20),(20,1e9)]:
    s=(gm>=lo)&(gm<hi)
    print(f'different neurons disjoint, sqrt(r1 r2) [{lo},{hi}) n={s.sum()}: final-merge {r[s,0].mean() if s.sum() else np.nan:.2f}, veto(sig=True) {r[s,2].mean() if s.sum() else np.nan:.2f}')
