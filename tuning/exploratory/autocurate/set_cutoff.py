"""Python port of the lab pipeline's KS2.5 set_cutoff (neuropixels, ExternalToolboxes/KilosortToolbox/Versions/
kilosort_v2.5/postProcess: set_cutoff.m, calcCcg.m, ccg_rg.m; the autocorrelogram of OldHelpers/aCorrPointProcess.c).

Per unit, the score cut-off starts at th[0] and is lowered in steps of 0.5 down to th[1] while the spikes above
it stay refractory: Q (refractory-zone rate over the larger of the outer-fringe and shoulder rates, minimum over
windows of 1-10 ms) <= the acceptable contamination (0.1; in strict mode, for a unit refractory at th[0] with
Q < 0.05, min(0.05, max(0.01, 2Q))) and R (Poisson probability of the refractory-zone count given the shoulder,
minimum over windows; 0 when that window holds no events) <= 0.05. The cut-off ends at the last level that
passed (th[0] + 0.5 if none did) and spikes with score <= cut-off are deleted. Floating-point operations follow
the MATLAB order where it matters (bin edges, normalisations).
"""
import numpy as np
from scipy.special import erf

NBINS, TBIN, MAX_REFRACT = 500, 1e-3, 10e-3
_IB = np.arange(-NBINS, NBINS + 1)
_FRINGE = np.r_[1:NBINS // 2, 3 * NBINS // 2:2 * NBINS]          # MATLAB [2:nbins/2 (3/2*nbins+1):(2*nbins)]
_LEFT = np.arange(NBINS - 50, NBINS - 10)                         # [nbins+1-50 : nbins-10]
_RIGHT = np.arange(NBINS + 11, NBINS + 50)                        # [nbins+12 : nbins+50]


def acg(t):
    """aCorrPointProcess(t, TBIN, 2*NBINS+1, 0): counts of lags t[b] - t[a] (a != b) in bins centred on
    multiples of TBIN, lags within +-(NBINS + 0.5) TBIN; t sorted, in seconds."""
    nb, mid, w = 2 * NBINS + 1, NBINS, (2 * NBINS + 1) / 2 * TBIN
    K = np.zeros(nb)
    for j in range(1, t.size):
        a, b = t[:-j], t[j:]
        fwd = b <= a + w                       # b seen from a: included while t[b] <= t[a] + window
        bwd = a > b - w                        # a seen from b: skipped while t[a] <= t[b] - window
        if not (fwd.any() or bwd.any()):
            break
        for d, m in (((b - a) / TBIN, fwd), ((a - b) / TBIN, bwd)):
            # a lag of exactly the window's half-width (e.g. 15015 samples) can round to bin -1 or nb: clamp both
            idx = np.clip(np.floor(d[m] + 0.5).astype(np.int64) + mid, 0, nb - 1)
            K += np.bincount(idx, minlength=nb)
    return K


def ccg_stats(t):
    """calcCcg(t, [], true, MAX_REFRACT): Q, R and the event count N of the window where R is smallest."""
    t = np.sort(t)
    n1 = n2 = t.size
    T = t[-1] - t[0]
    K = acg(t)
    with np.errstate(divide='ignore', invalid='ignore'):
        Q00 = K[_FRINGE].sum() / (_FRINGE.size * TBIN * n1 * n2 / T)
        R00 = max(K[_LEFT].mean(), K[_RIGHT].mean())
        R00 = max(R00, K[_FRINGE].mean())
        Q01 = max(K[_LEFT].sum() / (_LEFT.size * TBIN * n1 * n2 / T),
                  K[_RIGHT].sum() / (_RIGHT.size * TBIN * n1 * n2 / T))
        K[NBINS] = 0
        nwin = int(np.ceil(MAX_REFRACT / TBIN))
        Qi, Ri, Ni = np.zeros(nwin), np.zeros(nwin), np.zeros(nwin)
        for i in range(1, nwin + 1):
            s = K[np.abs(_IB) <= i].sum()
            Qi[i - 1] = s / (2 * i * TBIN * n1 * n2 / T)
            n, lam = s / 2, R00 * i
            Ri[i - 1] = 0.5 * (1 + erf((n - lam) / np.sqrt(2 * lam)))
            Ni[i - 1] = n
        Q_all = Qi / _mmax(Q00, Q01)
    Q = np.nan if np.all(np.isnan(Q_all)) else np.nanmin(Q_all)
    iR = 0 if np.all(np.isnan(Ri)) else int(np.nanargmin(Ri))
    return Q, Ri[iR], Ni[iR]


def _mmax(a, b):
    """MATLAB max(a, b) of scalars: NaN ignored."""
    return b if np.isnan(a) else a if np.isnan(b) else max(a, b)


def unit_cutoff(ss, vexp, th, strict):
    """Cut-off and 'good' flag of one unit (spike times ss in seconds, scores vexp)."""
    Th, fcont, good = th[0], 0.1, False
    while Th >= th[1]:
        st = ss[vexp > Th]
        if st.size == 0:
            Th -= 0.5
            continue
        Q, R, N = ccg_stats(st)
        if N == 0:
            R = 0.0
        if Q > fcont or R > 0.05:
            break
        if Th == th[0] and Q < 0.05 and strict:
            fcont = min(0.05, _mmax(0.01, Q * 2))
        good = True
        Th -= 0.5
    return Th + 0.5, good


def set_cutoff(t, clu, score, fs, th, strict):
    """Keep mask over spikes (t in samples), and per unit {cluster: (cut-off, good)}."""
    keep = np.ones(t.size, bool)
    o = np.argsort(clu, kind='stable')
    ids, start = np.unique(clu[o], return_index=True)
    out = {}
    for c, ix in zip(ids, np.split(o, start[1:])):
        ss = t[ix] / fs
        Th, good = unit_cutoff(ss, score[ix], th, strict)
        keep[ix[score[ix] <= Th]] = False
        out[int(c)] = (Th, good)
    return keep, out
