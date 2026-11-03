"""Hybrid ground truth (kobe): are short-lag pairs inside sorted hybrid units duplicates of one true spike?

For each of the 40 inserted units, its best-matching sorted unit (tuning/hybrid.py recovery, agreement >= 0.5)
in every kobe sort (raw output; and KS4 seed 1 before its own dedup). Each sorted spike is assigned its
nearest true spike of that inserted unit; a hit is within +-12 samples (0.4 ms, hybrid.TOL). Within-unit
pairs at lags 0-1.5 ms are classed:
  same_true : both members hit the SAME true spike (a duplicate of one spike),
  two_true  : both hit, different true spikes (a short interval that exists in the inserted train),
  one_hit   : one member hits, the other does not (a non-hit within 1.5 ms of a true spike),
  no_hit    : neither.
Also the inserted trains' own short intervals, and cross-unit duplicates of hybrid spikes: for true spikes
hit by the best unit, the share with a spike of another sorted unit within 100 um at |lag| <= 0.5 ms,
minus the same share at a 5-10 ms offset (chance).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from dup_common import PRE, SORTS, TRUTH, _median_xy  # noqa: E402

REPO = Path('C:/code/github/Kilosort/tuning')
sys.path.insert(0, str(REPO))
import hybrid  # noqa: E402

TOL = hybrid.TOL
BANDS = np.array([0, 8, 15, 23, 30, 45])   # samples: 0-0.25, 0.25-0.5, 0.5-0.75, 0.75-1.0, 1.0-1.5 ms
BAND_NAMES = ['0-.25', '.25-.5', '.5-.75', '.75-1', '1-1.5']
CLASSES = ('same_true', 'two_true', 'one_hit', 'no_hit')


def nearest(s, tt):
    i = np.clip(np.searchsorted(tt, s), 1, tt.size - 1)
    j = np.where(np.abs(tt[i] - s) < np.abs(tt[i - 1] - s), i, i - 1)
    return j, np.abs(tt[j] - s) <= TOL


def classify(s, tt):
    j, hit = nearest(s, tt)
    H = {c: np.zeros(len(BANDS) - 1, int) for c in CLASSES}
    for d in range(1, 46):
        a, b = np.arange(s.size - d), np.arange(d, s.size)
        lag = s[b] - s[a]
        m = lag < 45
        if not m.any():
            break
        a, b, lag = a[m], b[m], lag[m]
        band = np.searchsorted(BANDS, lag, 'right') - 1
        cls = np.where(hit[a] & hit[b], np.where(j[a] == j[b], 0, 1), np.where(hit[a] | hit[b], 2, 3))
        for ci, c in enumerate(CLASSES):
            H[c] += np.bincount(band[cls == ci], minlength=len(BANDS) - 1)[:len(BANDS) - 1]
    return H, hit


def run(sort, truth, pre_dir=None):
    rec = hybrid.recovery(sort, truth)
    st = np.load(sort / 'spike_times.npy').ravel().astype(np.int64)
    clu = np.load(sort / 'spike_clusters.npy').ravel().astype(np.int64)
    ids = np.unique(clu)
    xy = _median_xy(np.load(sort / 'spike_positions.npy'), clu, ids)
    if pre_dir is not None:
        st = np.load(pre_dir / 'full_st.npy')[:, 0].astype(np.int64)
        clu = np.load(pre_dir / 'full_clu.npy').astype(np.int64)
    o = np.argsort(st, kind='stable'); st, clu = st[o], clu[o]
    tt_all, tu = truth['spike_samples'], truth['spike_unit']
    tot = {c: np.zeros(len(BANDS) - 1, int) for c in CLASSES}
    n_sorted = n_hit = 0
    xdup = np.zeros(2); xchance = np.zeros(2)
    for pu in rec['per_unit']:
        if pu['agreement'] < 0.5:
            continue
        c = pu['best_sorted']
        s = st[clu == c]
        tt = np.sort(tt_all[tu == pu['inserted']])
        H, hit = classify(s, tt)
        for k in CLASSES:
            tot[k] += H[k]
        n_sorted += s.size; n_hit += hit.sum()
        # cross-unit duplicates of hit true spikes: other units within 100 um
        ci = np.searchsorted(ids, c)
        near = ids[(np.linalg.norm(xy - xy[ci], axis=1) <= 100) & (ids != c)]
        other = st[np.isin(clu, near)]
        if other.size == 0:
            continue
        sh = s[hit]
        for off, acc in ((0, xdup), (225, xchance)):   # 7.5 ms offset for chance
            q = sh + off
            lo, hi = np.searchsorted(other, q - 15, 'left'), np.searchsorted(other, q + 15, 'right')
            acc += [np.sum(hi > lo), sh.size]
    return tot, n_sorted, n_hit, xdup, xchance


if __name__ == '__main__':
    truth = np.load(TRUTH)
    tt, tu = truth['spike_samples'], truth['spike_unit']
    own = np.zeros(len(BANDS) - 1, int)
    for k in np.unique(tu):
        t = np.sort(tt[tu == k])
        for d in range(1, 10):
            lag = t[d:] - t[:-d]
            lag = lag[lag < 45]
            own += np.bincount(np.searchsorted(BANDS, lag, 'right') - 1, minlength=len(BANDS) - 1)[:len(BANDS) - 1]
    print(f'inserted trains: {tt.size} spikes; own pairs per band (ms) {dict(zip(BAND_NAMES, own.tolist()))}')
    print('\nwithin best-matched sorted units (agreement >= 0.5): pairs per 1000 sorted spikes, by lag band (ms)')
    print(f"{'':26s}{'class':10s}" + ''.join(f'{b:>8s}' for b in BAND_NAMES) + '   cross-unit dup/true spike (chance)')
    sets = [(g, k, s, None) for g, ss in SORTS['kobe'].items() for k, s in enumerate(ss, 1)]
    sets += [(g + '_pre', 1, SORTS['kobe'][g][0], PRE[('kobe', g)]) for g in ('ks4_default', 'ks4_H8')]
    for g, k, s, pre in sets:
        tot, ns, nh, xd, xc = run(s, truth, pre)
        for i, c in enumerate(CLASSES):
            lab = f'{g} r{k} (hit {nh / ns:.0%})' if i == 0 else ''
            extra = f'   {xd[0] / max(xd[1], 1):.3f} ({xc[0] / max(xc[1], 1):.3f})' if i == 0 else ''
            print(f'{lab:26s}{c:10s}' + ''.join(f'{1000 * x / ns:8.2f}' for x in tot[c]) + extra, flush=True)
