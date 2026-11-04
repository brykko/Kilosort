"""Re-implementation of multiscale relevance (MSR) following the logic of
rcubero/MSR relevance.py (Cubero, Marsili & Roudi 2020), but vectorised.

Reference-code conventions reproduced:
  * input: spike counts on a base grid of `total_time` points (1 ms in the paper's data)
  * number of bins T taken from unique(int(logspace(0.4, round(log10(0.99*total_time),2), 100)))
    plus T = total_time (the base grid itself)
  * bins are contiguous "balanced baskets": the first (total_time % T) bins have floor+1 points,
    the rest floor points
  * H[s] = -sum_k (k m_k/M) log(k/M) / log M ;  H[K] = -sum_k (k m_k/M) log(k m_k/M) / log M
  * the points (H[s]=0,H[K]=0) and (H[s]=1,H[K]=0) are appended; points sorted by H[s];
    MSR = trapezoid area under H[K](H[s]).
"""
import numpy as np


def partitions(total_time, n=100):
    nmax = np.round(np.log(total_time - 0.01 * total_time) / np.log(10), 2)
    parts = np.unique(np.logspace(0.4, nmax, n).astype(int))
    return np.append(parts, total_time)


def hs_hk_from_binidx(binidx, M):
    _, ks = np.unique(binidx, return_counts=True)
    kv, mk = np.unique(ks, return_counts=True)
    p = kv * mk / M
    lm = np.log(M)
    hk = -np.sum(p * np.log(p)) / lm
    hs = -np.sum(p * np.log(kv / M)) / lm
    return hs, hk


def curve(spike_ms, total_time, n=100):
    """spike_ms: integer base-grid indices of spikes (repeats allowed = counts>1)."""
    j = np.sort(np.asarray(spike_ms, dtype=np.int64))
    j = j[(j >= 0) & (j < total_time)]
    M = j.size
    out = []
    for T in partitions(total_time, n):
        fl = total_time // T
        sd = total_time % T
        cut = sd * (fl + 1)
        b = np.where(j < cut, j // (fl + 1), sd + (j - cut) // fl)
        out.append(hs_hk_from_binidx(b, M))
    return np.array(out)  # columns: H[s], H[K]


def msr_from_curve(c):
    d = np.vstack([c, [[0.0, 0.0], [1.0, 0.0]]])
    d = d[np.lexsort((d[:, 1], d[:, 0]))]
    return np.sum(0.5 * (d[:-1, 1] + d[1:, 1]) * np.abs(d[1:, 0] - d[:-1, 0]))


def msr(spike_times_s, duration_s, base_ms=1.0, n=100):
    total_time = int(round(duration_s * 1000 / base_ms))
    idx = np.floor(np.asarray(spike_times_s) * 1000 / base_ms).astype(np.int64)
    return msr_from_curve(curve(idx, total_time, n))


# ---------------------------------------------------------------- generators
def poisson(rate, dur, rng):
    n = rng.poisson(rate * dur)
    return np.sort(rng.uniform(0, dur, n))


def inhom(rate_fn_ms, dur, rng):
    """rate_fn_ms: array of rates (Hz) on 1 ms grid. Piecewise-constant inhomogeneous Poisson."""
    lam = rate_fn_ms * 1e-3
    n = rng.poisson(lam)
    t0 = np.repeat(np.arange(lam.size), n)
    return np.sort((t0 + rng.uniform(0, 1, t0.size)) * 1e-3)


def refractory(t, tref):
    if tref <= 0 or t.size == 0:
        return t
    keep = np.empty(t.size, bool)
    last = -np.inf
    for i, x in enumerate(t):
        if x - last >= tref:
            keep[i] = True
            last = x
        else:
            keep[i] = False
    return t[keep]


def ou(nsteps, tau_steps, rng):
    """unit-variance AR(1) with correlation time tau_steps."""
    from scipy.signal import lfilter
    a = np.exp(-1.0 / tau_steps)
    x = lfilter([np.sqrt(1 - a * a)], [1, -a], rng.standard_normal(nsteps))
    x[: int(min(nsteps, 5 * tau_steps))] *= 1.0  # burn-in ignored (stationary init approx)
    return x


def multiscale_rate(dur, mean_rate, rng, taus_s=(0.3, 3.0, 30.0), sigma=0.6):
    """log-normal rate with sum of OU components at several timescales, normalised to mean_rate."""
    n = int(round(dur * 1000))
    z = np.zeros(n)
    for tau in taus_s:
        z += ou(n, tau * 1000, rng) * sigma / np.sqrt(len(taus_s))
    r = np.exp(z)
    return r * mean_rate / r.mean()


def gamma_renewal(rate, shape, dur, rng):
    scale = 1.0 / (rate * shape)
    n = int(rate * dur * 1.5 + 100)
    isi = rng.gamma(shape, scale, n)
    t = np.cumsum(isi) + rng.uniform(0, isi[0])
    return t[t < dur]
