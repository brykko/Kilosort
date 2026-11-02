import os, sys, time
os.environ['NUMBA_CACHE_DIR'] = os.path.dirname(os.path.abspath(__file__)) + '/numba_cache'
import numpy as np
from kilosort import CCG, swarmsplitter as ss
rng = np.random.default_rng(0)

def refr_train(rate, T, dead=0.002, rng=rng):
    # renewal process: dead time + exponential
    n = int(rate*T*1.5)+100
    mean_isi = 1/rate
    isi = dead + rng.exponential(mean_isi-dead, n)
    t = np.cumsum(isi); t = t[t<T]
    return t

def contaminated(rate, c, T):
    # total rate `rate`, fraction c Poisson contamination
    u = refr_train(rate*(1-c), T)
    p = rng.uniform(0, T, rng.poisson(rate*c*T))
    return np.sort(np.concatenate((u,p)))

def good(st):
    return CCG.check_CCG(st)  # (is_ref, cross_ref, R12)

print('== Good label: P(good), mean R12 (sd) ==')
for rate in [5.0]:
    for c in [0.005, 0.05, 0.09, 0.105, 0.12, 0.15]:
        line=[]
        for T in [300, 1200, 10800]:
            reps = 200 if T<10800 else 60
            res = [good(contaminated(rate,c,T)) for _ in range(reps)]
            g = np.mean([r[0] for r in res]); R = np.array([r[2] for r in res])
            line.append(f'T={T}: P={g:.2f} R12={R.mean():.3f}({R.std():.3f})')
        print(f'rate {rate} c {c}: ' + ' | '.join(line))
print('== Sparse clean units (c=0): P(good) ==')
for rate in [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
    line=[]
    for T in [300,1200,10800]:
        res=[good(contaminated(rate,0.0,T)) for _ in range(200)]
        line.append(f'T={T}: P={np.mean([r[0] for r in res]):.2f}')
    print(f'rate {rate}: '+' | '.join(line))
