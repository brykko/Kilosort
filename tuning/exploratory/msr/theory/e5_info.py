"""Does MSR track covariate information?  1D linear-track place cells on a random-walk trajectory
(own simulation).  Skaggs information (bits/spike) computed from the true tuning and occupancy."""
import numpy as np
from msr import msr, inhom, refractory, poisson, ou

dur = 1200.0
n = int(dur * 1000)

def trajectory(rng):
    # smooth random walk on [0,1] track with reflecting walls, speed ~ 0.2 track/s
    v = ou(n, 2000, rng) * 0.25e-3
    x = np.cumsum(v) % 2.0
    return np.where(x > 1, 2 - x, x)

def skaggs(rate_x, x):
    lam = rate_x.mean()
    r = rate_x / lam
    return np.mean(r * np.log2(np.where(r > 0, r, 1)))

rng = np.random.default_rng(7)
x = trajectory(rng)
print("place cell: Gaussian field width w, peak rate A, baseline b")
for w in [0.03, 0.08, 0.2, 1e9]:
    for A, b in [(5, 0.1), (20, 0.4), (40, 2.0)]:
        rate = b + A * np.exp(-0.5 * ((x - 0.5) / w) ** 2) if w < 1e8 else np.full(n, b + A * 0.3)
        I = skaggs(rate, x)
        v = []
        for s in range(5):
            t = refractory(inhom(rate, dur, np.random.default_rng(s)), 0.002)
            v.append((msr(t, dur), t.size))
        v = np.array(v)
        pm = np.mean([msr(poisson(v[:, 1].mean() / dur, dur, np.random.default_rng(99 + s)), dur) for s in range(5)])
        print(f"  w {w if w<1e8 else 'flat':>5}  A {A:3d} b {b:4.1f}  M {v[:,1].mean():7.0f}  info {I:5.2f} bits/spk "
              f"({I*v[:,1].mean()/dur:6.2f} bits/s)  MSR {v[:,0].mean():.4f}  PoissonSameM {pm:.4f}  excess {v[:,0].mean()-pm:+.4f}")

print("\nuninformative but structured units")
tt = np.arange(n) * 1e-3
def show(label, t):
    pm = np.mean([msr(poisson(t.size / dur, dur, np.random.default_rng(50 + s)), dur) for s in range(5)])
    print(f"  {label:60s} M {t.size:6d}  MSR {msr(t, dur):.4f}  excess {msr(t, dur)-pm:+.4f}")
r = np.random.default_rng(3)
# drift-induced amplitude modulation: constant 5 Hz neuron, detection prob follows slow drift
t = poisson(10, dur, r); t = t[r.uniform(size=t.size) < 0.5 * (1 + np.sin(2 * np.pi * t / 900))]
show("Poisson 10 Hz with drift detection prob (15-min cycle, 0..1)", t)
# lick artefact: licking bouts (Poisson onsets 1/30 s, 10 s long), 7 Hz licks, each lick 1-3 artefact 'spikes'
bouts = poisson(1 / 30, dur, r)
licks = np.concatenate([b0 + np.arange(0, 10, 1 / 7) + r.normal(0, 0.01, 70) for b0 in bouts])
art = np.concatenate([l + 0.001 * np.arange(r.integers(1, 4)) for l in licks])
show("lick-artefact unit only", np.sort(art[(art > 0) & (art < dur)]))
base = poisson(5, dur, r)
mix = np.sort(np.concatenate([base, art[(art > 0) & (art < dur)]]))
show("Poisson 5 Hz unit + lick artefacts", mix)
show("Poisson 5 Hz unit alone", base)
# movement/state modulation: up/down (sleep-like) states, 1 s up 1 s down, uninformative about task
st = (np.floor(tt / np.repeat(1.0, n)) % 2 == 0).astype(float)
show("up/down states 1 s period (rate 10/0.5 Hz)", inhom(0.5 + 9.5 * st, dur, r))
# 'informative but low-MSR': fast covariate with symmetric rate-preserving code
# covariate flips every 50 ms (random), rate 5->7 / 5->3 Hz (weak modulation at fast scale)
c = np.repeat(r.integers(0, 2, n // 50 + 1), 50)[:n]
t = inhom(np.where(c == 1, 7.0, 3.0), dur, r)
show("fast binary covariate (50 ms), rate 3 vs 7 Hz", t)
# same covariate but high-rate neuron (more total info, rate 30 vs 70 Hz)
t = inhom(np.where(c == 1, 70.0, 30.0), dur, r)
show("fast binary covariate (50 ms), rate 30 vs 70 Hz", t)
# pure timing code: constant-rate periodic-ish neuron whose phase encodes a slow covariate
ph = np.cumsum(np.full(n, 1e-3 * 8)) + 0.3 * np.sin(2 * np.pi * tt / 20)  # 8 Hz, phase modulated by covariate
t = inhom(5 * (1 + np.cos(2 * np.pi * ph)), dur, r)
show("phase-coded (8 Hz theta, phase precesses with covariate), const rate", t)
