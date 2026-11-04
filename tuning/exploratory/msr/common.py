"""Shared loaders for the MSR analyses (exploratory).

clean_units(sort) returns the same spike trains as cleaning.load_clean_units (spec §6.1-6.3: spikes in
multi-unit events, duplicates and masked time removed), without the per-cluster median depth, which
is the slow part of that function and is not needed here; check_loader() asserts equality on a sort.
Eligible units and per-unit scores (S_i, S_track_i) are read from the sort's evaluation.json (written
by evaluate.py; S_i holds exactly the eligible units, hybrid copies already excluded).
"""
import json
import sys
from pathlib import Path

import numpy as np

TUNING = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TUNING))
from cleaning import FS, load_clean_units, mask_intervals  # noqa: E402

KOBE = Path('C:/data/ks4_tuning/26648_kobe_2020-08-21')
RAW = KOBE / 'search/raw_artefacts_car.npz'
REF = KOBE / 'runs/reference_seed1000/theta_ref.npz'
TRUTH = KOBE / 'search/hybrid/hybrid_truth.npz'
T0, DUR = 1668.0, 1200.0
OUT = Path('C:/data/ks4_tuning/exploratory/msr')
OUT.mkdir(parents=True, exist_ok=True)

SORTS = {
    'default_s1': KOBE / 'runs/hybrid_default_seed1',
    'H8_s1': KOBE / 'runs/search/198_H8_cds10_s1/sort',
    'ks25_standard_r1': KOBE / 'runs/ks25_standard_r1',
    'ks25_standard_low_r1': KOBE / 'runs/ks25_standard_low_r1',
}


def clean_units(sort):
    sort = Path(sort)
    st = np.load(sort / 'spike_times.npy').ravel()
    clu = np.load(sort / 'spike_clusters.npy').ravel()
    c = np.load(sort / 'cleaning.npz')
    removed = np.zeros(st.size, bool)
    removed[c['spike_order']] = c['removed_event'] | c['removed_duplicate']
    iv = mask_intervals(sort, RAW)
    t = st / FS
    i = np.searchsorted(iv[:, 0], t, 'right') - 1
    if iv.size:
        removed |= (i >= 0) & (t < iv[np.maximum(i, 0), 1])
    keep = ~removed
    t, clu = t[keep], clu[keep]
    o = np.argsort(clu, kind='stable')
    t, clu = t[o], clu[o]
    ids, start = np.unique(clu, return_index=True)
    parts = np.split(t, start[1:])
    by = {int(k): np.sort(p) for k, p in zip(ids, parts)}
    for k in np.unique(np.load(sort / 'spike_clusters.npy')):   # clusters emptied by cleaning
        by.setdefault(int(k), np.array([]))
    T = DUR - (float(np.sum(iv[:, 1] - iv[:, 0])) if iv.size else 0.0)
    return by, iv, T


def evaluation(sort):
    return json.loads((Path(sort) / 'evaluation.json').read_text())


def check_loader(sort):
    a, _, _ = load_clean_units(sort, RAW)
    b, _, _ = clean_units(sort)
    assert set(a) == set(b), 'cluster sets differ'
    for k in a:
        assert np.array_equal(np.sort(a[k]), b[k]), k
    return True


def spearman(x, y):
    from scipy.stats import spearmanr
    ok = np.isfinite(x) & np.isfinite(y)
    r, p = spearmanr(x[ok], y[ok])
    return float(r), float(p), int(ok.sum())


def partial_spearman(x, y, z):
    """Spearman correlation of x and y with z (one or more columns) partialled out: Pearson on the
    residuals of the ranks after linear regression on the ranks of z."""
    from scipy.stats import pearsonr, rankdata
    z = np.atleast_2d(np.asarray(z, float))
    if z.shape[0] != np.size(x):
        z = z.T
    ok = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(z), 1)
    rx, ry = rankdata(x[ok]), rankdata(y[ok])
    Z = np.column_stack([np.ones(ok.sum())] + [rankdata(c) for c in z[ok].T])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    r, p = pearsonr(ex, ey)
    return float(r), float(p), int(ok.sum())
