"""One event detected twice, or two events? A joint least-squares refit of short-lag within-unit pairs on raw data.

Question (decides between deleting one member, merging the pair, or treating it as contamination):
for KS4 within-unit spike pairs at 8-15 samples (0.27-0.5 ms; kept by KS4's 0.25 ms window), does the
raw signal contain one spike or two? For each of the units with the most such pairs, the unit's mean
waveform W is estimated from isolated spikes (high-pass 300 Hz, common median reference, the unit's
12 largest channels). Each snippet is fitted as a1*W(t1+j1) + a2*W(t2+j2) with free amplitudes and
times re-optimised within +-2 samples (j1, j2), keeping t1 < t2. One event detected twice gives
a1 ~ 1, a2 ~ 0; two overlapping spikes give a1 ~ a2 ~ 1.
Controls: 'phantom' = an isolated spike t1 with a fake t2 = t1 + L (L from the real pairs' lags):
the null for a2; 'lag 31-45' = within-unit pairs at 1-1.5 ms (refractory violations beyond the bump).
CPU only; reads short snippets of the int16 binary.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

SORT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r'C:\data\ks4_tuning\exploratory\autocurate\sorts\kobe_H8_s1')
RAW = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(r'C:\data\ks4_tuning\26648_kobe_2020-08-21\search\hybrid\ap_1668-2868s_hybrid.bin')
OUT = Path(__file__).parent / f'pair_raw_refit_{SORT.name}.json'
NCH_BIN = 385
FS = 30000
N_UNITS = 25
N_PAIRS = 60
N_MEAN = 300
PRE, POST = 30, 50          # waveform window around the spike time
PADF = 300                  # filter padding
NCH_FIT = 12
J = 2                        # time re-optimisation +- samples
rng = np.random.default_rng(0)
sos = butter(3, 300, 'highpass', fs=FS, output='sos')
raw = np.memmap(RAW, np.int16, 'r').reshape(-1, NCH_BIN)
chan_map = np.load(SORT / 'channel_map.npy').ravel()


def snippet(t0, t1, chans):
    """Filtered, CMR'd data [t0, t1) on chans (indices into chan_map)."""
    a, b = t0 - PADF, t1 + PADF
    x = raw[a:b, chan_map].astype(np.float32)
    x = sosfiltfilt(sos, x, axis=0)
    x -= np.median(x, axis=1, keepdims=True)
    return x[PADF:PADF + (t1 - t0), chans]


def fit_pair(x, W, L, single=False):
    """x: (n, ch) snippet starting at t1 - PRE - J; W: (PRE+POST, ch). Returns best (a1, a2, rss)."""
    n = x.shape[0]
    best = (np.nan, np.nan, np.inf)
    y = x.ravel()
    for j1 in range(-J, J + 1):
        B1 = np.zeros_like(x); s = J + j1; B1[s:s + W.shape[0]] = W
        if single:
            A = B1.ravel()[:, None]
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            r = y - A @ coef; rss = float(r @ r)
            if rss < best[2]:
                best = (float(coef[0]), 0.0, rss)
            continue
        for j2 in range(-J, J + 1):
            if L + j2 - j1 < 1:
                continue
            B2 = np.zeros_like(x); s2 = J + L + j2
            if s2 + W.shape[0] > n:
                continue
            B2[s2:s2 + W.shape[0]] = W
            A = np.stack([B1.ravel(), B2.ravel()], 1)
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            r = y - A @ coef; rss = float(r @ r)
            if rss < best[2]:
                best = (float(coef[0]), float(coef[1]), rss)
    return best


def main():
    st = np.load(SORT / 'spike_times.npy').ravel().astype(np.int64)
    cl = np.load(SORT / 'spike_clusters.npy').ravel().astype(np.int64)
    tmpl = np.load(SORT / 'templates.npy')
    o = np.lexsort((st, cl)); st, cl = st[o], cl[o]
    same = cl[1:] == cl[:-1]
    d = np.diff(st)
    prev_gap = np.r_[np.inf, np.where(same, d, np.inf)]
    next_gap = np.r_[np.where(same, d, np.inf), np.inf]
    short = np.where(same & (d >= 8) & (d <= 15))[0]
    units, counts = np.unique(cl[short], return_counts=True)
    nsp = np.bincount(cl)
    ok = nsp[units] >= 2000
    units, counts = units[ok], counts[ok]
    units = units[np.argsort(counts)[::-1][:N_UNITS]]
    nmax = raw.shape[0]
    res = {'units': []}
    agg = {'pair_8_15': [], 'phantom': [], 'lag_31_45': [], 'single_isolated': []}
    for u in units:
        a, b = np.searchsorted(cl, [u, u + 1])
        ptp = np.ptp(tmpl[u], axis=0)
        chans = np.argsort(ptp)[::-1][:NCH_FIT]
        iso = np.arange(a, b)[(prev_gap[a:b] > 90) & (next_gap[a:b] > 90)]
        iso = iso[(st[iso] > 2 * PADF) & (st[iso] < nmax - 2 * PADF)]
        pick = rng.choice(iso, min(N_MEAN, iso.size), replace=False)
        W = np.mean([snippet(st[i] - PRE, st[i] + POST, chans) for i in pick], axis=0)
        W /= 1.0  # amplitude 1 = mean spike
        pr = short[(cl[short] == u)]
        pr = pr[(st[pr] > 2 * PADF) & (st[pr + 1] < nmax - 2 * PADF)]
        pr = rng.choice(pr, min(N_PAIRS, pr.size), replace=False)
        lags = st[pr + 1] - st[pr]
        far = np.where(same & (d >= 31) & (d <= 45) & (cl[:-1] == u))[0]
        far = far[(st[far] > 2 * PADF) & (st[far + 1] < nmax - 2 * PADF)]
        far = rng.choice(far, min(N_PAIRS, far.size), replace=False) if far.size else far
        out = {'unit': int(u), 'n_spikes': int(b - a), 'n_pairs_8_15': int(np.sum(cl[short] == u)),
               'peak_ptp': float(ptp.max())}
        for name, idx, lagv in (('pair_8_15', pr, lags), ('lag_31_45', far, st[far + 1] - st[far] if far.size else [])):
            vals = []
            for i, L in zip(idx, lagv):
                x = snippet(st[i] - PRE - J, st[i] + L + POST + J, chans)
                vals.append(fit_pair(x, W, int(L))[:2])
            agg[name] += vals
            out[name] = np.median(np.array(vals), 0).tolist() if vals else None
        vals, vals1 = [], []
        for i, L in zip(rng.choice(iso, min(N_PAIRS, iso.size), replace=False), rng.choice(lags, N_PAIRS)):
            x = snippet(st[i] - PRE - J, st[i] + L + POST + J, chans)
            vals.append(fit_pair(x, W, int(L))[:2])
            vals1.append(fit_pair(x, W, int(L), single=True)[0])
        agg['phantom'] += vals
        agg['single_isolated'] += [(v, 0.0) for v in vals1]
        out['phantom'] = np.median(np.array(vals), 0).tolist()
        res['units'].append(out)
        print(out, flush=True)
    summ = {}
    for k, v in agg.items():
        v = np.array(v)
        if not v.size:
            continue
        summ[k] = {'n': int(len(v)), 'a1_median': float(np.median(v[:, 0])), 'a2_median': float(np.median(v[:, 1])),
                   'a2_quartiles': np.percentile(v[:, 1], [10, 25, 75, 90]).tolist(),
                   'frac_a2_gt_0.5': float(np.mean(v[:, 1] > 0.5)), 'frac_a2_lt_0.25': float(np.mean(v[:, 1] < 0.25))}
    res['summary'] = summ
    print(json.dumps(summ, indent=1))
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
