import numpy as np
from common import TRUTH
z=np.load(TRUTH); t=z['spike_samples']; u=z['spike_unit']
for k in range(40):
    tt=np.sort(t[u==k]); d=np.diff(tt)
    if k<40: print(k, tt.size, 'isi<=30', (d<=30).sum(), 'isi<=60', (d<=60).sum(), 'min', d.min())
