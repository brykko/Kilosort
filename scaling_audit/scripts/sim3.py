import os
os.environ['NUMBA_CACHE_DIR'] = os.path.dirname(os.path.abspath(__file__)) + '/numba_cache'
import numpy as np
from kilosort import CCG
rng = np.random.default_rng(2)
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'common.py')).read())

def modulated(rate, T, g, block, dead=0.002):
    # piecewise-constant gain g[k] per block of `block` s (time-rescaled renewal)
    out=[]
    for k in range(int(np.ceil(T/block))):
        r = rate*g[k]
        if r<=0: continue
        t = refr_train(r, block, dead) + k*block
        out.append(t[t<T])
    return np.concatenate(out)

def poisson_mod(rate, T, g, block):
    out=[]
    for k in range(int(np.ceil(T/block))):
        n = rng.poisson(rate*g[k]*block); out.append(k*block + rng.uniform(0,block,n))
    return np.sort(np.concatenate(out))

T=10800; block=60; nb=T//block; c=0.08; rate=5
print('unit 5 Hz, contamination 8 pct of spikes (Poisson); slow gain per 60 s block; R12 mean over reps (stationary expectation 1-(1-c)^2 = %.3f)'%(1-(1-c)**2))
for name, mk in [
    ('stationary', lambda: (np.ones(nb), np.ones(nb))),
    ('common gain, lognormal sd .7', lambda: (lambda g:(g,g))(rng.lognormal(-.245,.7,nb))),
    ('independent gains sd .7', lambda: (rng.lognormal(-.245,.7,nb), rng.lognormal(-.245,.7,nb))),
    ('anti-correlated states (unit on/contam off alternating 10 min)', lambda: ((np.arange(nb)//10%2)*1.6+0.2, (1-np.arange(nb)//10%2)*1.6+0.2)),
    ]:
    R=[]; G=[]
    for _ in range(20):
        gu, gc = mk()
        u = modulated(rate*(1-c), T, gu/gu.mean(), block)
        p = poisson_mod(rate*c, T, gc/gc.mean(), block)
        st = np.sort(np.concatenate((u,p)))
        ir, _, r12 = CCG.check_CCG(st); R.append(r12); G.append(ir)
    print(f'  {name:60s} R12={np.mean(R):.3f} (sd {np.std(R):.3f})  P(good)={np.mean(G):.2f}')
