import os
os.environ['NUMBA_CACHE_DIR'] = os.path.dirname(os.path.abspath(__file__)) + '/numba_cache'
import numpy as np
from kilosort import CCG, swarmsplitter as ss
rng = np.random.default_rng(3)
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'common.py')).read())
T=1200
print('Expected Q00 = r1*r2*T*1ms. Same-neuron halves (2% contam) vs independent pair (2 ms dead time each); 300 reps')
print(' Q00   same:veto  same:merge   indep:veto  indep:merge')
for q in [0.05,0.1,0.2,0.3,0.5,0.75,1.0,1.5,2.0,3.0,5.0]:
    r = np.sqrt(q/(T*1e-3))   # each fragment rate
    sv=sm=iv=im=0; R=300
    for _ in range(R):
        st = contaminated(2*r, 0.02, T); m = rng.random(len(st))<0.5
        a,b = st[m], st[~m]
        sv += ss.check_CCG(a,b)[1]; sm += CCG.check_CCG(a,b)[1]
        a,b = refr_train(r,T), refr_train(r,T)
        iv += ss.check_CCG(a,b)[1]; im += CCG.check_CCG(a,b)[1]
    print(f'{q:5.2f}   {sv/R:.2f}       {sm/R:.2f}         {iv/R:.2f}        {im/R:.2f}')
