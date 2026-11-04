"""Sorting-error experiments on MSR (own simulations).
Ground-truth unit U: log-normal multi-timescale rate (OU at 0.3, 3, 30 s, sigma .6), mean 5 Hz,
2 ms dead time, 20 min.  Each error is applied to the same U realisation (paired), 8 seeds."""
import numpy as np
from msr import msr, poisson, inhom, refractory, multiscale_rate, gamma_renewal, ou

dur = 1200.0
NS = 8
rng0 = np.random.default_rng(10)

# Poisson baseline lookup MSR_P(M)
Mg = np.unique(np.logspace(2, 5, 16).astype(int))
base = np.array([np.mean([msr(np.sort(rng0.uniform(0, dur, m)), dur) for _ in range(12)]) for m in Mg])
def pois_base(M):
    return np.interp(np.log(M), np.log(Mg), base)

def subsample(t, n, rng):
    n = min(n, t.size)
    return np.sort(rng.choice(t, n, replace=False))

def thin(t, p, rng):
    return t[rng.uniform(size=t.size) < p]

def report(label, rows):
    rows = np.array(rows)  # each row: (msr, M)
    m, M = rows[:, 0], rows[:, 1]
    ex = m - pois_base(M)
    print(f"{label:58s} M {M.mean():7.0f}  MSR {m.mean():.4f}+-{m.std():.4f}  excess {ex.mean():+.4f}")

seeds = list(range(NS))
U, V, I, Rg, B = {}, {}, {}, {}, {}
for s in seeds:
    rng = np.random.default_rng(100 + s)
    U[s] = refractory(inhom(multiscale_rate(dur, 5, rng), dur, rng), 0.002)
    V[s] = refractory(inhom(multiscale_rate(dur, 5, rng, taus_s=(0.1, 1.0, 10.0), sigma=0.8), dur, rng), 0.002)
    I[s] = refractory(inhom(multiscale_rate(dur, 30, rng, sigma=0.2), dur, rng), 0.001)
    Rg[s] = gamma_renewal(5, 4.0, dur, rng)
    ev = poisson(1.25, dur, rng); n = 1 + rng.poisson(3, ev.size)
    b = np.concatenate([e + 0.004 * np.arange(m) for e, m in zip(ev, n)]); B[s] = np.sort(b[b < dur])

def run(label, fn):
    rows = []
    for s in seeds:
        rng = np.random.default_rng(1000 + s)
        t = fn(s, rng)
        rows.append((msr(t, dur), t.size))
    report(label, rows)

print("Reference units")
run("U  ground truth (multiscale, 5 Hz)", lambda s, r: U[s])
run("V  other modulated neuron (5 Hz)", lambda s, r: V[s])
run("I  interneuron-like (30 Hz, weak modulation)", lambda s, r: I[s])
run("Rg regular gamma-4 neuron (5 Hz)", lambda s, r: Rg[s])
run("B  bursty neuron", lambda s, r: B[s])

print("\n(a) contamination: U + extra spikes, fraction f of M_U")
for src, name in [("P", "Poisson noise"), ("V", "neuron V"), ("I", "interneuron I"), ("Rg", "regular Rg"), ("B", "bursty B")]:
    for f in [0.05, 0.1, 0.2, 0.5, 1.0]:
        def fn(s, r, f=f, src=src):
            n = int(f * U[s].size)
            if src == "P":
                c = np.sort(r.uniform(0, dur, n))
            else:
                c = subsample({"V": V, "I": I, "Rg": Rg, "B": B}[src][s], n, r)
            return np.sort(np.concatenate([U[s], c]))
        run(f"  + {name} f={f}", fn)

print("\n(b) thinning (missed spikes)")
for p in [0.9, 0.7, 0.5, 0.3, 0.1]:
    run(f"  random keep p={p}", lambda s, r, p=p: thin(U[s], p, r))
for pmin in [0.7, 0.4, 0.1]:
    def fn(s, r, pmin=pmin):
        # amplitude-dependent loss under slow drift: keep prob follows slow sinus (period 20 min) + OU(200 s)
        tt = U[s]
        drift = 0.5 * (1 + np.sin(2 * np.pi * tt / 1200 + r.uniform(0, 6.28)))
        keep = pmin + (1 - pmin) * drift
        return tt[r.uniform(size=tt.size) < keep]
    run(f"  drift-dependent keep prob in [{pmin},1] (20-min cycle)", fn)
for pmin in [0.7, 0.4]:
    def fn(s, r, pmin=pmin):
        tt = U[s]
        d = ou(int(dur), 200, r)  # 1 s grid, tau 200 s
        keep = pmin + (1 - pmin) / (1 + np.exp(-2 * d[np.minimum((tt).astype(int), int(dur) - 1)]))
        return tt[r.uniform(size=tt.size) < keep]
    run(f"  drift(OU 200 s) keep prob in [{pmin},1]", fn)
# fast amplitude-dependent loss: spikes lost preferentially during bursts / high rate (amplitude attenuation)
def fn(s, r):
    tt = U[s]
    isi = np.diff(np.concatenate([[-1.0], tt]))
    keep = np.where(isi < 0.010, 0.4, 0.95)
    return tt[r.uniform(size=tt.size) < keep]
run("  burst-dependent loss (ISI<10 ms kept 40%, else 95%)", fn)

print("\n(c) merges (full spike trains)")
run("  U + V", lambda s, r: np.sort(np.concatenate([U[s], V[s]])))
run("  U + I", lambda s, r: np.sort(np.concatenate([U[s], I[s]])))
run("  U + Rg", lambda s, r: np.sort(np.concatenate([U[s], Rg[s]])))
run("  U + B", lambda s, r: np.sort(np.concatenate([U[s], B[s]])))
run("  U + Poisson(5 Hz)", lambda s, r: np.sort(np.concatenate([U[s], poisson(5, dur, r)])))
run("  Rg + Rg' (two regular neurons)", lambda s, r: np.sort(np.concatenate([Rg[s], Rg[(s + 1) % NS]])))

print("\n(d) splits of U: MSR of each fragment")
def split_rand(s, r, frac=0.5):
    a = r.uniform(size=U[s].size) < frac
    return U[s][a], U[s][~a]
def split_drift(s, r, sharp):
    tt = U[s]
    d = np.sin(2 * np.pi * tt / 1200 + 1.0)  # slow drift coordinate, 20-min cycle
    pa = 1 / (1 + np.exp(-sharp * d))
    a = r.uniform(size=tt.size) < pa
    return tt[a], tt[~a]
def split_time(s, r):
    tt = U[s]
    return tt[tt < 600], tt[tt >= 600]
def split_burst(s, r):
    tt = U[s]
    isi = np.diff(np.concatenate([[-1.0], tt]))
    a = isi < 0.02  # 'second spikes' of bursts with attenuated amplitude go to fragment A
    return tt[a], tt[~a]
for label, fn in [("random 50/50", split_rand), ("random 90/10", lambda s, r: split_rand(s, r, 0.9)),
                  ("drift-boundary soft (sharp=1)", lambda s, r: split_drift(s, r, 1.0)),
                  ("drift-boundary hard (sharp=5)", lambda s, r: split_drift(s, r, 5.0)),
                  ("time split at 10 min", split_time), ("burst 2nd-spike split (ISI<20ms)", split_burst)]:
    ra, rb = [], []
    for s in seeds:
        r = np.random.default_rng(2000 + s)
        a, b = fn(s, r)
        ra.append((msr(a, dur), a.size)); rb.append((msr(b, dur), b.size))
    report(f"  {label}: fragment A", ra)
    report(f"  {label}: fragment B", rb)
    ra, rb = np.array(ra), np.array(rb)
    print(f"{'':58s} mean(A,B) {0.5*(ra[:,0]+rb[:,0]).mean():.4f}  sum {(ra[:,0]+rb[:,0]).mean():.4f}")

print("\n(e) duplicates: fraction d of spikes get an extra copy at lag ~U(0.1,0.5) ms (1 ms base grid)")
for d in [0.01, 0.05, 0.1, 0.2, 0.5]:
    def fn(s, r, d=d):
        tt = U[s]; sel = tt[r.uniform(size=tt.size) < d]
        return np.sort(np.concatenate([tt, sel + r.uniform(1e-4, 5e-4, sel.size)]))
    run(f"  duplicates d={d}", fn)
for d in [0.05, 0.2]:
    def fn(s, r, d=d):
        tt = U[s]; sel = tt[r.uniform(size=tt.size) < d]
        return np.sort(np.concatenate([tt, sel + r.uniform(1e-3, 3e-3, sel.size)]))
    run(f"  near-duplicates at 1-3 ms lag d={d}", fn)
