"""Paired deltas (error - clean) for key error types, 20 seeds, two ground-truth types:
U5  : multiscale-modulated 5 Hz unit (as in e4)
P5r : near-Poisson 5 Hz unit with 2 ms refractory (structureless)."""
import numpy as np
from msr import msr, poisson, inhom, refractory, multiscale_rate

dur = 1200.0
NS = 20

def gen(kind, rng):
    if kind == "U5":
        return refractory(inhom(multiscale_rate(dur, 5, rng), dur, rng), 0.002)
    if kind == "U20":
        return refractory(inhom(multiscale_rate(dur, 20, rng), dur, rng), 0.002)
    return refractory(poisson(5 / (1 - 5 * 0.002), dur, rng), 0.002)

errs = {
    "Poisson contam 10% (f=.11)": lambda t, r: np.sort(np.concatenate([t, r.uniform(0, dur, int(0.111 * t.size))])),
    "Poisson contam 33% (f=.5)": lambda t, r: np.sort(np.concatenate([t, r.uniform(0, dur, int(0.5 * t.size))])),
    "random thinning keep .5": lambda t, r: t[r.uniform(size=t.size) < 0.5],
    "drift loss keep in [.4,1]": lambda t, r: t[r.uniform(size=t.size) < 0.4 + 0.6 * 0.5 * (1 + np.sin(2 * np.pi * t / 1200 + r.uniform(0, 6.3)))],
    "duplicates 10% at 0.1-0.5 ms": lambda t, r: np.sort(np.concatenate([t, (lambda s: s + r.uniform(1e-4, 5e-4, s.size))(t[r.uniform(size=t.size) < 0.1])])),
    "drift split hard, fragment A": lambda t, r: t[r.uniform(size=t.size) < 1 / (1 + np.exp(-5 * np.sin(2 * np.pi * t / 1200 + 1)))],
}
for kind in ["U5", "U20", "P5r"]:
    print(f"\nGround truth {kind}")
    clean, dl = [], {k: [] for k in errs}
    for s in range(NS):
        rng = np.random.default_rng(500 + s)
        t = gen(kind, rng)
        m0 = msr(t, dur); clean.append(m0)
        for k, f in errs.items():
            dl[k].append(msr(f(t, np.random.default_rng(9000 + s)), dur) - m0)
    print(f"  clean MSR {np.mean(clean):.4f} (across-seed sd {np.std(clean):.4f})")
    for k in errs:
        d = np.array(dl[k])
        print(f"  {k:32s} dMSR {d.mean():+.4f}  sd {d.std():.4f}  t {d.mean()/(d.std()/np.sqrt(NS)):+6.1f}")
