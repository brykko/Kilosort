import sys, numpy as np, torch
c = torch.load(sys.argv[1], weights_only=False, map_location='cpu')
st = c['out'][0]; print('keys ops', list(c['ops'].keys())[:10])
th = st[:,2]; tpl = st[:,1].astype(int); t = st[:,0]
Th = th.min(); print('n', len(th), 'min th_amp', th.min(), 'templates', tpl.max()+1, 'dur s', t.max()/30000)
for x in [0.05, 0.1, 0.2, 0.3]:
    print('decline %d%%: fraction of all spikes that fall below threshold: %.3f'%(100*x, (th < Th/(1-x)).mean()))
# per template fraction within 10% / 20% of threshold
n = np.bincount(tpl); near10 = np.bincount(tpl, weights=(th < Th/0.9)); near20 = np.bincount(tpl, weights=(th<Th/0.8))
ok = n >= 2000
f10 = near10[ok]/n[ok]; f20 = near20[ok]/n[ok]
print('templates>=2000 spikes', ok.sum(), 'per-template frac lost at 10%% decline pct 25/50/75/90', np.round(np.percentile(f10,[25,50,75,90]),3))
print('per-template frac lost at 20%% decline pct 25/50/75/90', np.round(np.percentile(f20,[25,50,75,90]),3))
# per-template log th_amp slope over the excerpt (median per quarter)
T = t.max(); q = np.minimum((t/T*4).astype(int),3)
order = np.argsort(tpl, kind='stable'); b = np.r_[0, np.cumsum(n)]
sl=[]
for k in np.nonzero(ok)[0]:
    i = order[b[k]:b[k+1]]
    m = [np.median(th[i][q[i]==j]) for j in range(4) if (q[i]==j).sum()>100]
    if len(m)==4: sl.append(np.log(m[3]/m[0]))
sl=np.array(sl); print('per-template log(median th_amp Q4/Q1) pct 5/25/50/75/95', np.round(np.percentile(sl,[5,25,50,75,95]),3))
