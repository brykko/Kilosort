import time, numpy as np
from msr import msr, poisson, curve

rng = np.random.default_rng(1)
t0 = time.time()
print("E1a homogeneous Poisson, 20 min, MSR vs M (mean +- sd over 20 reps)")
dur = 1200.0
for rate in [0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 40]:
    v = [msr(poisson(rate, dur, rng), dur) for _ in range(20)]
    M = rate * dur
    print(f"rate {rate:6.2f} Hz  M~{M:7.0f}  logM {np.log(M):5.2f}  MSR {np.mean(v):.4f} +- {np.std(v):.4f}")
print("elapsed", time.time() - t0)

print("\nE1b same M=6000, different durations (rescaling invariance test)")
for dur in [300.0, 1200.0, 7200.0]:
    rate = 6000 / dur
    v = [msr(poisson(rate, dur, rng), dur) for _ in range(20)]
    print(f"dur {dur:6.0f}s rate {rate:6.2f}  MSR {np.mean(v):.4f} +- {np.std(v):.4f}")

print("\nE1c curve shape for Poisson M=6000 (H[s], H[K]) at selected bins")
t = poisson(5, 1200, rng)
c = curve(np.floor(t * 1000).astype(int), 1200000)
for i in range(0, len(c), 8):
    print(f"Hs {c[i,0]:.3f}  HK {c[i,1]:.3f}  approx (1-Hs)/2+0.5log(2pi e)/logM = {(1-c[i,0])/2+0.5*np.log(2*np.pi*np.e)/np.log(t.size):.3f}")
