"""Shared loaders and lag-histogram kernels for the duplicate-spike empirics (KS4 vs KS2.5).

Question these scripts answer: does KS4 produce more duplicated spikes (within a unit and across units,
at sub-millisecond lags) than the lab's KS2.5, before and after each sorter's own removal and the harness's
cleaning (spec §6.3), and how much of the refractory-violation contamination gap do duplicates explain?

Sort variants:
- 'raw'   : the sorter's saved output (KS4: after its 0.25 ms within-unit dedup; KS2.5: as saved).
- 'clean' : the same after the harness's cleaning (cleaning.npz: artefact events + cross-unit duplicates
            removed; masked intervals dropped), as tuning/cleaning.py load_clean_units.
- 'pre'   : KS4 seed 1 only, every spike before KS4's dedup (autocurate/sorts/*/full_st.npy, full_clu.npy).
Hybrid copies (sorted units >= 50% hits on inserted spikes, tuning/hybrid.py) are excluded on kobe.
"""
import sys
from pathlib import Path

import numpy as np
from numba import njit

REPO = Path('C:/code/github/Kilosort/tuning')
sys.path.insert(0, str(REPO))
import hybrid  # noqa: E402
from cleaning import mask_intervals  # noqa: E402

FS = 30000
K = Path('C:/data/ks4_tuning/26648_kobe_2020-08-21')
M = Path('C:/data/ks4_tuning/25843_roger_2019-09-11')
AC = Path('C:/data/ks4_tuning/exploratory/autocurate/sorts')
TR = Path('C:/data/ks4_tuning/exploratory/ks25_traced')
BASE = {'kobe': K, 'mec': M}
TRUTH = K / 'search/hybrid/hybrid_truth.npz'
CACHE = Path(__file__).parent / 'cache'

SORTS = {
    'kobe': {'ks4_default': [K / f'runs/hybrid_default_seed{k}' for k in (1, 2, 3, 4, 5)],
             'ks4_H8': [K / f'runs/search/{r}/sort' for r in ('198_H8_cds10_s1', '208_H8_cds10_s2', '218_H8_cds10_s3')],
             'ks25_std': [K / f'runs/ks25_standard_r{k}' for k in (1, 2, 3)],
             'ks25_low': [K / f'runs/ks25_standard_low_r{k}' for k in (1, 2, 3)]},
    'mec': {'ks4_default': [M / f'runs/default_s{k}' for k in (1, 2, 3)],
            'ks4_H8': [M / f'runs/H8_s{k}' for k in (1, 2, 3)],
            'ks25_std': [M / f'runs/ks25_standard_r{k}' for k in (1, 2, 3)],
            'ks25_low': [M / f'runs/ks25_standard_low_r{k}' for k in (1, 2, 3)]},
}
PRE = {('kobe', 'ks4_default'): AC / 'kobe_default_s1', ('kobe', 'ks4_H8'): AC / 'kobe_H8_s1',
       ('mec', 'ks4_default'): AC / 'mec_default_s1', ('mec', 'ks4_H8'): AC / 'mec_H8_s1'}

MAXW = 600     # within-unit lags 0..600 samples (20 ms)
MAXX = 300     # cross-unit |lags| 0..300 samples (10 ms)
DIST_EDGES = np.array([0, 20, 40, 60, 100, 150, 300, 600, 1e9])
SIM_EDGES = np.array([-1, 0.3, 0.5, 0.7, 0.9, 1.01])


@njit(cache=True)
def within_hist(t, u, n_units, maxlag):
    """t sorted by (u, t). H[unit, lag] of forward pairs of the unit's own spikes, lag in samples."""
    H = np.zeros((n_units, maxlag + 1), np.int64)
    for i in range(t.size):
        for j in range(i + 1, t.size):
            if u[j] != u[i]:
                break
            d = t[j] - t[i]
            if d > maxlag:
                break
            H[u[i], d] += 1
    return H


@njit(cache=True)
def cross_hist(t, u, B, n_cat, maxlag):
    """t sorted by time. H[category, |lag|] over pairs of spikes of different units; category B[u_i, u_j]."""
    H = np.zeros((n_cat, maxlag + 1), np.int64)
    for i in range(t.size):
        for j in range(i + 1, t.size):
            d = t[j] - t[i]
            if d > maxlag:
                break
            if u[j] == u[i]:
                continue
            H[B[u[i], u[j]], d] += 1
    return H


@njit(cache=True)
def ks4_dedup(t, u, dt):
    """KS4's remove_duplicates (postprocessing.py:9-29) on time-sorted spikes: drop a spike within dt samples
    of the unit's last kept spike."""
    last = np.full(u.max() + 1, -(1 << 40), np.int64)
    keep = np.zeros(t.size, np.bool_)
    for i in range(t.size):
        if t[i] >= last[u[i]] + dt:
            last[u[i]] = t[i]
            keep[i] = True
    return keep


def template_similarity(sort, ids):
    """Zero-lag cosine similarity of the sort's (whitened) templates, rows/cols in the order of ids."""
    W = np.load(Path(sort) / 'templates.npy').astype(np.float32)
    W = W[ids].reshape(len(ids), -1)
    W /= np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-12)
    return W @ W.T


def load_variant(sort, ex, variant, pre_dir=None, drop_copies=True):
    """Spike times (samples, int64, time-sorted) and cluster ids, plus per-cluster median xy and the
    unmasked duration (s). variant in raw / clean / pre."""
    sort = Path(sort)
    if variant == 'pre':
        f = np.load(pre_dir / 'full_st.npy')
        st = f[:, 0].astype(np.int64)
        clu = np.load(pre_dir / 'full_clu.npy').astype(np.int64)
        kept = np.load(pre_dir / 'kept_spikes.npy')
        assert np.array_equal(st[kept], np.load(sort / 'spike_times.npy').ravel().astype(np.int64)), 'pre != saved sort'
    else:
        st = np.load(sort / 'spike_times.npy').ravel().astype(np.int64)
        clu = np.load(sort / 'spike_clusters.npy').ravel().astype(np.int64)
    xy_all = np.load(sort / 'spike_positions.npy')
    clu_saved = np.load(sort / 'spike_clusters.npy').ravel().astype(np.int64)
    ids = np.unique(clu_saved)
    xy = _median_xy(xy_all, clu_saved, ids)
    T = 1200.0
    o = np.argsort(st, kind='stable')
    st, clu = st[o], clu[o]
    keep = np.ones(st.size, bool)
    if variant == 'clean':
        keep &= ~_remap_removed(sort, o)
        iv = mask_intervals(sort, BASE[ex] / 'search/raw_artefacts_car.npz')
        ts = st / FS
        i = np.searchsorted(iv[:, 0], ts, 'right') - 1
        if iv.size:
            keep &= ~((i >= 0) & (ts < iv[np.maximum(i, 0), 1]))
            T = 1200.0 - float(np.sum(iv[:, 1] - iv[:, 0]))
    copies = set()
    if ex == 'kobe' and drop_copies:
        copies = set(hybrid.recovery(sort, np.load(TRUTH))['hybrid_copies'])
        keep &= ~np.isin(clu, list(copies))
    return st[keep], clu[keep], ids, xy, T, copies


def _remap_removed(sort, o):
    """cleaning.npz flags are in time-sorted (stable argsort) order of the saved spike_times; return them in
    the order given by o (the stable argsort of the saved times, so identical)."""
    c = np.load(Path(sort) / 'cleaning.npz')
    rm_sorted = c['removed_event'] | c['removed_duplicate']   # index k = k-th spike in stable time order
    assert np.array_equal(c['spike_order'], o), 'cleaning spike_order differs from stable argsort'
    return rm_sorted


def _median_xy(xy_all, clu, ids):
    o = np.argsort(clu, kind='stable')
    cs = clu[o]
    b = np.searchsorted(cs, ids, 'left'); e = np.searchsorted(cs, ids, 'right')
    return np.array([np.median(xy_all[o[s:t]], axis=0) for s, t in zip(b, e)])


def by_unit(st, clu, ids):
    """Index units 0..n-1 in the order of ids; return (t, u) sorted by (u, t)."""
    u = np.searchsorted(ids, clu)
    o = np.lexsort((st, u))
    return st[o], u[o]
