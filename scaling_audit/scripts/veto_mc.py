"""CPU Monte Carlo: swarmsplitter.check_CCG cross-refractory veto vs recording length.
Independent Poisson pairs (should NOT be vetoed) and same-neuron random halves with 2 ms
refractory period (should be vetoed)."""
import numpy as np, sys
sys.path.insert(0, r'C:\code\github\Kilosort')
from kilosort.swarmsplitter import check_CCG

rng = np.random.default_rng(0)

def poisson(rate, T, refr=0.0):
    n = rng.poisson(rate * T)
    if refr == 0:
        return np.sort(rng.uniform(0, T, n))
    isi = rng.exponential(1 / rate, int(rate * T * 1.3) + 50) + refr
    t = np.cumsum(isi); return t[t < T]

def frac(fn, reps):
    v = np.array([fn() for _ in range(reps)]); return v.mean(0)

for T in (1200, 10800, 18000):
    for r1, r2 in ((0.2, 0.2), (0.5, 0.5), (1, 0.2), (1, 1), (5, 0.05), (5, 0.5)):
        def indep():
            a, b = poisson(r1, T, 0.002), poisson(r2, T, 0.002)
            return [check_CCG(a, b)[1], check_CCG(a, b, require_significance=True)[1]]
        def same():
            t = poisson(r1 + r2, T, 0.002); m = rng.random(len(t)) < r1 / (r1 + r2)
            return [check_CCG(t[m], t[~m])[1], check_CCG(t[m], t[~m], require_significance=True)[1]]
        q00 = r1 * r2 * T * 1e-3
        fi, fs = frac(indep, 40), frac(same, 40)
        print(f'T={T:6d} r=({r1},{r2}) Q00~{q00:6.2f}  distinct: veto {fi[0]:.2f} (sig-only {fi[1]:.2f})   '
              f'same-neuron: veto {fs[0]:.2f} (sig-only {fs[1]:.2f})', flush=True)
