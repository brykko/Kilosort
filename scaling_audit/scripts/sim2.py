import os
os.environ['NUMBA_CACHE_DIR'] = os.path.dirname(os.path.abspath(__file__)) + '/numba_cache'
import numpy as np
from kilosort import CCG, swarmsplitter as ss
rng = np.random.default_rng(1)
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'common.py')).read())

def tests(a, b):
    sp = ss.check_CCG(a, b)[1]                       # splitter veto (sparse clause on)
    sp_sig = ss.check_CCG(a, b, require_significance=True)[1]
    fm = CCG.check_CCG(a, b)[1]                       # final merge cross test
    fm_sh = CCG.check_CCG(a, b, use_shape_criterion=True)[1]
    return sp, sp_sig, fm, fm_sh

def run(label, gen, Ts=(1200, 10800), reps=150):
    out=[]
    for T in Ts:
        r = np.array([tests(*gen(T)) for _ in range(reps if T<5000 else reps//2)])
        out.append(f'T={T}: veto={r[:,0].mean():.2f} veto_sig={r[:,1].mean():.2f} merge={r[:,2].mean():.2f} merge_shape={r[:,3].mean():.2f}')
    print(label + '\n   ' + '\n   '.join(out))

print('== Same neuron, random split into fragments (frac f), 2% contamination ==')
for rate, f in [(5,0.5),(5,0.1),(1,0.5),(0.5,0.5),(0.3,0.3),(0.2,0.5)]:
    def gen(T, rate=rate, f=f):
        st = contaminated(rate, 0.02, T); m = rng.random(len(st))<f
        return st[m], st[~m]
    run(f'rate {rate} Hz, f {f}', gen)

print('== Independent neurons (Poisson + 2ms dead time) ==')
for r1, r2 in [(5,5),(1,0.5),(0.5,0.3),(0.3,0.2),(0.2,0.1)]:
    def gen(T, r1=r1, r2=r2):
        return refr_train(r1,T), refr_train(r2,T)
    run(f'{r1} Hz x {r2} Hz', gen)

print('== Same neuron split in time (drift): A before T/2-ov/2, B after T/2+... overlap ov seconds ==')
for rate, ov in [(5,0),(5,60),(5,300),(1,300),(5,1200)]:
    def gen(T, rate=rate, ov=ov):
        st = contaminated(rate, 0.02, T); mid=T/2
        inA = st < mid+ov/2; inB = st >= mid-ov/2; both = inA & inB
        pick = rng.random(len(st))<0.5
        a = st[inA & (~both | pick)]; b = st[inB & (~both | ~pick)]
        return a, b
    run(f'rate {rate}, overlap {ov}s', gen, reps=60)

print('== Two DIFFERENT neurons, temporally disjoint (one silent in each half) ==')
for rate in [5, 1]:
    def gen(T, rate=rate):
        a = refr_train(rate,T); b = refr_train(rate,T)
        return a[a<T/2], b[b>=T/2]
    run(f'rate {rate}', gen, reps=60)
