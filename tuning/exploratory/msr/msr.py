"""Exploratory: multiscale relevance (MSR) of a spike train (Cubero, Marsili & Roudi 2020, J Comput
Neurosci 48:85-102, doi 10.1007/s10827-020-00740-x; reference code github.com/rcubero/MSR,
relevance.py).

Time is discretised into DT_S = 1 ms points (the paper's choice for the mEC data, section 5.7).
For a partition of the T points into N contiguous bins (lengths differing by at most one), with k_s
spikes in bin s, m_k bins holding k spikes and M spikes in all:
  resolution H[s] = -sum_s (k_s/M) log(k_s/M) / log M = -sum_k (k m_k/M) log(k/M) / log M
  relevance  H[K] = -sum_k (k m_k/M) log(k m_k/M) / log M
Partitions as in relevance.py: N = unique(int(logspace(0.4, round(log10(0.99 T), 2), 100))), plus
N = T (1 ms bins). The anchors (H[K], H[s]) = (0, 0) and (0, 1) are added, the points sorted by
H[s], and MSR = trapezoid area of H[K] over H[s] (units: "Mats^2").

Choices (noted in the research notes):
- counts: by default bins hold spike counts (relevance.py sums whatever array it is given).
  binarize=True first reduces each 1 ms point to 0/1 (the paper binarised in its Boolean-function
  section), which hides sub-millisecond duplicates entirely.
- masked time (harness §6.1-6.2) is not removed; its spikes are already gone from load_clean_units'
  trains. On the kobe excerpt the masked time is well under 1% of 1200 s.
"""
import numpy as np
from numba import njit

DT_S = 1e-3


def partitions(T):
    n_max = round(np.log10(T - 0.01 * T), 2)
    N = np.unique(np.logspace(0.4, n_max, 100).astype(np.int64))
    return np.unique(np.append(N, T)).astype(np.int64)


@njit(cache=True)
def _curve(ticks, T, Ns):
    """ticks: sorted int64 time points in [0, T). Returns H[K], H[s] per partition (normalised)."""
    M = ticks.size
    hk = np.zeros(Ns.size)
    hs = np.zeros(Ns.size)
    mk = np.zeros(M + 1, np.int64)
    logM = np.log(M)
    for j in range(Ns.size):
        N = Ns[j]
        mk[:] = 0
        prev = (ticks[0] * N) // T
        run = 0
        for i in range(M):
            b = (ticks[i] * N) // T
            if b == prev:
                run += 1
            else:
                mk[run] += 1
                prev = b
                run = 1
        mk[run] += 1
        a, c = 0.0, 0.0
        for k in range(1, M + 1):
            if mk[k] > 0:
                p = k * mk[k] / M
                a -= p * np.log(p)
                c -= p * np.log(k / M)
        hk[j] = a / logM
        hs[j] = c / logM
    return hk, hs


def curve(t_s, duration_s, binarize=False, dt_s=DT_S):
    """(H[s], H[K]) points of spike times t_s (seconds, in [0, duration_s))."""
    T = int(round(duration_s / dt_s))
    ticks = np.sort(np.clip(np.floor(np.asarray(t_s) / dt_s).astype(np.int64), 0, T - 1))
    if binarize:
        ticks = np.unique(ticks)
    if ticks.size < 2:
        return np.array([]), np.array([])
    hk, hs = _curve(ticks, T, partitions(T))
    return hs, hk


def area(hs, hk):
    x = np.concatenate([hk, [0.0, 0.0]])
    y = np.concatenate([hs, [0.0, 1.0]])
    o = np.lexsort((x, y))
    x, y = x[o], y[o]
    return float(np.sum(0.5 * np.abs(x[:-1] + x[1:]) * np.abs(y[:-1] - y[1:])))


def msr(t_s, duration_s, binarize=False, dt_s=DT_S):
    hs, hk = curve(t_s, duration_s, binarize, dt_s)
    return area(hs, hk) if hs.size else np.nan


# ---- homogeneous-Poisson baseline as a function of spike count (for a rate-corrected MSR) ----

def poisson_baseline(duration_s, counts, reps=8, seed=0, binarize=False):
    """Mean and SD of MSR of homogeneous Poisson trains with exactly M spikes, M in counts."""
    rng = np.random.default_rng(seed)
    mu, sd = [], []
    for M in counts:
        v = [msr(np.sort(rng.uniform(0, duration_s, int(M))), duration_s, binarize) for _ in range(reps)]
        mu.append(np.mean(v))
        sd.append(np.std(v))
    return np.asarray(mu), np.asarray(sd)


class Baseline:
    """Interpolates the Poisson MSR baseline in log M."""

    def __init__(self, duration_s, m_min=50, m_max=400000, n=40, reps=8, seed=0):
        self.M = np.unique(np.round(np.logspace(np.log10(m_min), np.log10(m_max), n)).astype(int))
        self.mu, self.sd = poisson_baseline(duration_s, self.M, reps, seed)

    def __call__(self, M):
        return np.interp(np.log(M), np.log(self.M), self.mu)

    def as_dict(self):
        return {'M': self.M.tolist(), 'mu': self.mu.tolist(), 'sd': self.sd.tolist()}
