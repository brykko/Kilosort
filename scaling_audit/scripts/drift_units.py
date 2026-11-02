import numpy as np, sys
d = sys.argv[1]
t = np.load(d+'/spike_times.npy').astype(np.int64); c = np.load(d+'/spike_clusters.npy')
a = np.load(d+'/amplitudes.npy'); dt = np.load(d+'/spike_detection_templates.npy')
fs=30000; T = t.max()/fs; nq=4
q = np.minimum((t/fs/T*nq).astype(int), nq-1)
units, n = np.unique(c, return_counts=True)
print('spikes', len(t), 'units', len(units), 'dur s', round(T))
rows=[]
order = np.argsort(c, kind='stable'); bounds = np.r_[0, np.cumsum(np.bincount(c))]
for u in units:
    idx = order[bounds[u]:bounds[u+1]]
    if len(idx) < 2000: continue
    qq = q[idx]; aa = a[idx]; tt = dt[idx]
    if min(np.bincount(qq, minlength=nq)) < 200: continue
    med = np.array([np.median(aa[qq==k]) for k in range(nq)])
    vals, cnt = np.unique(tt, return_counts=True); dom = vals[cnt.argmax()]
    fdom = np.array([(tt[qq==k]==dom).mean() for k in range(nq)])
    ntemp = (cnt >= 0.05*len(idx)).sum()
    # slope of log amp vs time (per 20 min)
    s = np.polyfit(t[idx]/fs/T, np.log(aa), 1)[0]
    # temporal centroid of each detection template's spikes in this unit
    rows.append((u, len(idx), med[0], med[-1], s, fdom[0], fdom[-1], ntemp, fdom.min(), fdom.max()))
r = np.array(rows)
print('units analysed', len(r))
s = r[:,4]
print('log-amp slope per %.0f min: pct 5/25/50/75/95:'%(T/60), np.round(np.percentile(s,[5,25,50,75,95]),3))
print('|slope|>0.1 (>~10%% per excerpt): %.1f%%'%(100*(np.abs(s)>0.1).mean()), ' >0.2: %.1f%%'%(100*(np.abs(s)>0.2).mean()))
print('dominant-template fraction Q1 median %.3f, Q4 median %.3f'%(np.median(r[:,5]), np.median(r[:,6])))
sw = r[:,9]-r[:,8]
print('range of dominant fraction across quarters: pct 50/75/90/95', np.round(np.percentile(sw,[50,75,90,95]),3))
print('units with >=2 detection templates each >=5%% of spikes: %.1f%%'%(100*(r[:,7]>=2).mean()))
