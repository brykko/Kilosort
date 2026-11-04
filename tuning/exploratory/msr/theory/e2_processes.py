import numpy as np
from msr import msr, poisson, inhom, refractory, multiscale_rate, gamma_renewal, ou

rng = np.random.default_rng(2)
dur = 1200.0
R = 10


def rep(gen, label):
    v, ms, sh = [], [], []
    for _ in range(R):
        t = gen()
        v.append(msr(t, dur))
        ms.append(t.size)
        # ISI-shuffled surrogate: keeps ISI distribution, destroys slow modulation / serial correlation
        isi = np.diff(np.concatenate([[0.0], t]))
        rng.shuffle(isi)
        ts = np.cumsum(isi)
        sh.append(msr(ts[ts < dur], dur))
        # Poisson with same M
    pm = [msr(poisson(np.mean(ms) / dur, dur, rng), dur) for _ in range(R)]
    print(f"{label:55s} M {np.mean(ms):7.0f}  MSR {np.mean(v):.4f}+-{np.std(v):.4f}  "
          f"ISIshuf {np.mean(sh):.4f}  PoissonSameM {np.mean(pm):.4f}  excess {np.mean(v)-np.mean(pm):+.4f}")


print("E2a renewal processes, rate 5 Hz")
for k in [0.3, 0.5, 1.0, 2.0, 4.0, 10.0]:
    rep(lambda: gamma_renewal(5, k, dur, rng), f"gamma ISI shape {k}")
for tr in [0.002, 0.005, 0.02]:
    rep(lambda: refractory(poisson(5 / (1 - 5 * tr) if 5 * tr < 0.9 else 5, dur, rng), tr), f"Poisson+dead time {tr*1e3:.0f} ms (5 Hz)")
for tr in [0.002, 0.005]:
    rep(lambda: refractory(poisson(30 / (1 - 30 * tr), dur, rng), tr), f"Poisson+dead time {tr*1e3:.0f} ms (30 Hz)")


def bursty(rate_ev=2.0, nmean=3.0, isi_b=0.004):
    ev = poisson(rate_ev, dur, rng)
    n = 1 + rng.poisson(nmean - 1, ev.size)
    t = np.concatenate([e + isi_b * np.arange(m) for e, m in zip(ev, n)])
    return np.sort(t[t < dur])


print("\nE2b bursts (Poisson burst onsets, 4 ms intra-burst)")
rep(lambda: bursty(2.0, 2.0), "bursts mean 2 spikes, 2 bursts/s")
rep(lambda: bursty(1.25, 4.0), "bursts mean 4 spikes, 1.25 bursts/s")
rep(lambda: bursty(0.5, 10.0), "bursts mean 10 spikes, 0.5 bursts/s")

print("\nE2c inhomogeneous Poisson, single-timescale OU log-rate (mean 5 Hz)")
for tau in [0.01, 0.1, 1.0, 10.0, 100.0]:
    for sig in [0.5, 1.0]:
        def g(tau=tau, sig=sig):
            n = int(dur * 1000)
            r = np.exp(sig * ou(n, tau * 1000, rng)); r *= 5 / r.mean()
            return inhom(r, dur, rng)
        rep(g, f"OU tau {tau:6.2f}s sigma {sig}")

print("\nE2d multi-timescale (0.3,3,30 s) log-rate, sigma 0.6, + 2 ms refractory")
for rate in [1, 5, 20]:
    rep(lambda: refractory(inhom(multiscale_rate(dur, rate, rng), dur, rng), 0.002), f"multiscale rate {rate} Hz")

print("\nE2e periodic (theta 8 Hz) modulation, 5 Hz mean")
tt = np.arange(int(dur * 1000)) * 1e-3
for d in [0.5, 1.0]:
    rep(lambda: inhom(5 * (1 + d * np.cos(2 * np.pi * 8 * tt)), dur, rng), f"theta depth {d}")
rep(lambda: inhom(5 * (1 + 0.9 * np.sign(np.cos(2 * np.pi * tt / 600))), dur, rng), "square-wave 10-min on/off (drift-like), depth .9")
rep(lambda: inhom(5 * (1 + 0.5 * np.cos(2 * np.pi * tt / 1200)), dur, rng), "slow sinus 20-min depth .5")
