import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, r'C:\code\github\Kilosort')
import numpy as np
from kilosort.CCG import check_CCG

rng = np.random.default_rng(1)

def unit(rate, T, refr=0.002):
    # Poisson with dead time
    n = int(rate * T * 1.5) + 100
    isi = refr + rng.exponential(1 / rate - refr, n)
    t = np.cumsum(isi)
    return t[t < T]

def contaminated(rate, T, c):
    # total rate = rate; fraction c from an independent Poisson source
    t1 = unit(rate * (1 - c), T)
    nc = rng.poisson(rate * c * T)
    t2 = rng.uniform(0, T, nc)
    return np.sort(np.concatenate((t1, t2)))

cs = [0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.2]
reps = 40
for rate in [2, 5, 20]:
    for T, lab in [(1200, '20min'), (10800, '3h')]:
        row = []
        for c in cs:
            g = 0
            for r in range(reps):
                st = contaminated(rate, T, c)
                is_ref, _, R12 = check_CCG(st, acg_threshold=0.2, ccg_threshold=0.25)
                g += bool(is_ref)
            row.append(g / reps)
        print(f'rate {rate:3d} Hz {lab:6s} P(good) at c={cs}: ' + ' '.join(f'{x:.2f}' for x in row))
