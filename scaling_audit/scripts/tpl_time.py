import numpy as np, sys
d = sys.argv[1]
t = np.load(d+'/spike_times.npy').astype(np.int64); c = np.load(d+'/spike_clusters.npy'); dt = np.load(d+'/spike_detection_templates.npy')
q = np.minimum((t/t.max()*8).astype(int), 7)
tv=[]; share=[]
order = np.argsort(c, kind='stable'); b = np.r_[0, np.cumsum(np.bincount(c))]
for u in range(c.max()+1):
    i = order[b[u]:b[u+1]]
    if len(i) < 2000: continue
    pu = np.bincount(q[i], minlength=8)/len(i)
    v, n = np.unique(dt[i], return_counts=True)
    for vv, nn in zip(v, n):
        if nn < 500 or nn < 0.05*len(i): continue
        pt = np.bincount(q[i][dt[i]==vv], minlength=8)/nn
        tv.append(0.5*np.abs(pt-pu).sum()); share.append(nn/len(i))
tv=np.array(tv); share=np.array(share)
print('detection-template/unit pairs', len(tv))
print('TV distance of template time-profile vs its unit (8 bins): pct 50/75/90/95/99', np.round(np.percentile(tv,[50,75,90,95,99]),3))
print('minor templates (share<0.5): TV pct 50/90', np.round(np.percentile(tv[share<0.5],[50,90]),3), 'n', (share<0.5).sum())
