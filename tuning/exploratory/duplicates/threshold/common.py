"""Shared loaders and within-unit lag counting for the duplicate-threshold analysis.

All lags are in samples (30 kHz). `pairs_within(t, clu, maxlag)` returns every ordered within-unit pair
(i earlier, j later) with 0 <= t[j]-t[i] <= maxlag, as index arrays into the input arrays.
"""
from pathlib import Path

import numpy as np

FS = 30000.0
T_SEC = 1200.0
KOBE = Path(r'C:/data/ks4_tuning/26648_kobe_2020-08-21')
MEC = Path(r'C:/data/ks4_tuning/25843_roger_2019-09-11')
AUTOC = Path(r'C:/data/ks4_tuning/exploratory/autocurate/sorts')
TRACED = Path(r'C:/data/ks4_tuning/exploratory/ks25_traced')
TRUTH = KOBE / 'search/hybrid/hybrid_truth.npz'


def pairs_within(t, clu, maxlag):
    t = np.asarray(t, np.int64)
    clu = np.asarray(clu, np.int64)
    o = np.lexsort((t, clu))
    ts, cs = t[o], clu[o]
    I, J = [], []
    d = 1
    while True:
        if d >= ts.size:
            break
        ok = (cs[d:] == cs[:-d]) & (ts[d:] - ts[:-d] <= maxlag)
        if not ok.any():
            break
        idx = np.where(ok)[0]
        I.append(o[idx]); J.append(o[idx + d])
        d += 1
    if not I:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(I), np.concatenate(J)


def load_ks4_full(name):
    """Pre-dedup KS4 sort: time, extraction template, score, cluster, amp, kept mask."""
    s = AUTOC / name
    st = np.load(s / 'full_st.npy')
    return dict(t=st[:, 0].astype(np.int64), tmpl=st[:, 1].astype(np.int64), score=st[:, 2],
                clu=np.load(s / 'full_clu.npy').astype(np.int64), amp=np.load(s / 'full_amp.npy').ravel(),
                kept=np.load(s / 'kept_spikes.npy').astype(bool))


def load_phy(folder):
    folder = Path(folder)
    d = dict(t=np.load(folder / 'spike_times.npy').ravel().astype(np.int64),
             clu=np.load(folder / 'spike_clusters.npy').ravel().astype(np.int64))
    p = folder / 'spike_detection_templates.npy'
    d['tmpl'] = np.load(p if p.exists() else folder / 'spike_templates.npy').ravel().astype(np.int64)
    if (folder / 'amplitudes.npy').exists():
        d['amp'] = np.load(folder / 'amplitudes.npy').ravel()
    return d


def load_traced(name, stage):
    z = np.load(TRACED / name / 'stages.npz')
    return dict(t=z[f'{stage}_t'].astype(np.int64), clu=z[f'{stage}_clu'].astype(np.int64),
                amp=z[f'{stage}_amp'], score=z[f'{stage}_score'])


def lag_profile(t, clu, maxlag=600, base=(150, 600)):
    """Pooled within-unit lag counts per lag sample 0..maxlag, and the expected count per lag sample
    from each unit's own 5-20 ms level. Returns counts, expected (pooled), n spikes."""
    I, J = pairs_within(t, clu, maxlag)
    lag = (np.asarray(t)[J] - np.asarray(t)[I]).astype(np.int64)
    cnt = np.bincount(lag, minlength=maxlag + 1)
    # per-unit baseline level
    u = np.asarray(clu)[I]
    inb = (lag >= base[0]) & (lag < base[1])
    ids, uinv = np.unique(np.asarray(clu), return_inverse=True)
    nb = np.bincount(np.searchsorted(ids, u[inb]), minlength=ids.size)
    level = nb / (base[1] - base[0])        # per unit, pairs per lag sample in the baseline
    return cnt, level.sum(), lag, I, J
