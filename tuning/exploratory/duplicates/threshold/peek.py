import numpy as np, glob, os
for s in glob.glob(r'C:/data/ks4_tuning/exploratory/autocurate/sorts/*_s1'):
    st=np.load(s+'/full_st.npy'); clu=np.load(s+'/full_clu.npy'); k=np.load(s+'/kept_spikes.npy')
    t=st[:,0]
    print(os.path.basename(s), st.shape, 'sorted', np.all(np.diff(t)>=0), 'frac int', np.mean(t==np.round(t)), 'nclu', len(np.unique(clu)), 'kept', k.sum(), 'Tmax s', t.max()/3e4)
    print(open(s+'/provenance.json').read()[400:1200])
z=np.load(r'C:/data/ks4_tuning/exploratory/ks25_traced/kobe_s1/stages.npz', allow_pickle=True)
for k in z.files: print(k, z[k].shape, z[k].dtype, z[k][:3])
