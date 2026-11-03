"""Peek at array formats of the scored KS4 sorts and the KS2.5 traced stages."""
import numpy as np
from pathlib import Path

d = Path(r'C:\data\ks4_tuning\exploratory\autocurate\sorts\kobe_H8_s1')
for f in ['full_st', 'full_clu', 'kept_spikes', 'score', 'spike_times', 'spike_clusters',
          'spike_detection_templates', 'full_amp']:
    a = np.load(d / f'{f}.npy')
    print(f, a.shape, a.dtype, a[:3].tolist())
fs = np.load(d / 'full_st.npy')
print('full_st sorted:', np.all(np.diff(fs[:, 0]) >= 0))
c = np.load(d / 'cleaning.npz')
print({k: c[k].shape for k in c})
s = np.load(r'C:\data\ks4_tuning\exploratory\ks25_traced\kobe_s1\stages.npz')
print({k: s[k].shape for k in s})
k = Path(r'C:\data\ks4_tuning\26648_kobe_2020-08-21\runs\ks25_standard_r1')
for f in ['spike_times', 'spike_clusters', 'spike_clusters_rezToPhy', 'spike_templates', 'amplitudes']:
    a = np.load(k / f'{f}.npy')
    print(f, a.shape, a.dtype, a.ravel()[:3].tolist())
st = np.load(k / 'spike_times.npy').ravel()
print('ks25 sorted', np.all(np.diff(st.astype(np.int64)) >= 0))
