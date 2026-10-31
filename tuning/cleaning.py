"""Spike-level cleaning of a sort (spec §6.2-6.3), prototype.

- Multi-unit artefact events (§6.2): 0.2 ms windows in which >= k distinct units whose positions
  span >= 1 mm all spike. k is the smallest value at which an interval-jittered copy of the sort
  (5 ms windows) gives < 1 event per 10 min. All spikes in an event are removed and the window is
  masked as missing time.
- Pairwise duplicates (§6.3), on all unit pairs: lag histogram at sample resolution; for each
  5-sample window centred at lag L in [-1, +1] ms, observed vs expected from that window's own
  flanks (10-30 samples either side). A pair is flagged if, at some L, the conditional binomial p
  (window count given window + flank total, null share 5/47) is below 0.01 / (pairs x lag windows)
  and the excess is >= 3% of the smaller unit's spikes. The
  identical procedure on an interval-jittered copy gives the false-positive count. Flagged pairs
  <= 150 um apart are the same neuron detected twice (coincident spikes removed from the unit with
  fewer spikes); pairs further apart are artefact (coincident spikes removed from both units).
- Raw-data safety net (§6.1): transient events from raw_artefacts.py reaching >= RAW_MASK_MIN_CH
  channels, +-1 ms, are masked.

Masked time is missing data: load_clean_units drops spikes inside masked intervals and returns the
intervals, and masked_fraction gives each scoring cycle's masked share.

Unit positions are median spike positions from KS4's spike_positions.npy.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from numba import njit
from scipy.stats import binom

FS = 30000
EVENT_WIN = 6            # 0.2 ms
EVENT_SPAN_UM = 1000.0
JITTER_WIN = 150         # 5 ms
MAXLAG = 60              # histogram half-width (2 ms) so flanks of L = +-30 fit
L_MAX = 30               # lags tested: +-1 ms
HALF_W = 2               # 5-sample windows
FLANK = (10, 30)
ALPHA = 0.01
MIN_EXCESS = 0.03
NEAR_UM = 150.0
RAW_MASK_MIN_CH = 128
RAW_MASK_PAD = 30        # 1 ms


def load_sort(sort):
    sort = Path(sort)
    t = np.load(sort / 'spike_times.npy').ravel().astype(np.int64)
    clu = np.load(sort / 'spike_clusters.npy').ravel().astype(np.int64)
    pos = np.load(sort / 'spike_positions.npy')
    ids, u = np.unique(clu, return_inverse=True)
    xy = np.stack([np.median(pos[u == k], axis=0) for k in range(ids.size)])
    o = np.argsort(t, kind='stable')
    return t[o], u[o], ids, xy


def interval_jitter(t, u, rng):
    """Move each spike uniformly within its 5 ms interval; returns re-sorted times and units."""
    tj = (t // JITTER_WIN) * JITTER_WIN + rng.integers(0, JITTER_WIN, t.size)
    o = np.argsort(tj, kind='stable')
    return tj[o], u[o]


# ---------------- multi-unit events ----------------

@njit(cache=True)
def _event_windows(t, y, k, span_um):
    """Greedy scan: windows [t_i, t_i + EVENT_WIN) with >= k spikes spanning >= span_um.
    Returns start indices and end indices (exclusive) of events."""
    starts, ends = [], []
    i, n = 0, t.size
    while i < n:
        j = i
        while j < n and t[j] < t[i] + EVENT_WIN:
            j += 1
        if j - i >= k:
            ymin, ymax = y[i], y[i]
            for m in range(i, j):
                ymin = min(ymin, y[m]); ymax = max(ymax, y[m])
            if ymax - ymin >= span_um:
                starts.append(i); ends.append(j)
                i = j
                continue
        i += 1
    return np.array(starts, np.int64), np.array(ends, np.int64)


def multiunit_events(t, u, xy, duration_s, rng, k_range=range(3, 41)):
    y = xy[u, 1]
    tj, uj = interval_jitter(t, u, rng)
    yj = xy[uj, 1]
    per10 = duration_s / 600
    null_rate = {k: _event_windows(tj, yj, k, EVENT_SPAN_UM)[0].size / per10 for k in k_range}
    k = next(k for k in k_range if null_rate[k] < 1)
    s, e = _event_windows(t, y, k, EVENT_SPAN_UM)
    remove = np.zeros(t.size, bool)
    for a, b in zip(s, e):
        remove[a:b] = True
    intervals = np.stack([t[s], t[e - 1] + 1], axis=1) if s.size else np.empty((0, 2), np.int64)
    real_rate = {kk: _event_windows(t, y, kk, EVENT_SPAN_UM)[0].size / per10 for kk in k_range}
    return {'k': k, 'n_events': int(s.size), 'events_per_10min': s.size / per10,
            'null_events_per_10min': null_rate, 'real_events_per_10min': real_rate}, remove, intervals


# ---------------- pairwise duplicates ----------------

@njit(cache=True)
def _pair_index(a, b, n):
    return a * n - a * (a + 1) // 2 + (b - a - 1)


@njit(cache=True)
def _lag_histograms(t, u, n):
    """H[pair, lag + MAXLAG] for pairs a < b, lag = t_b - t_a in samples."""
    H = np.zeros((n * (n - 1) // 2, 2 * MAXLAG + 1), np.int32)
    for i in range(t.size):
        for j in range(i + 1, t.size):
            d = t[j] - t[i]
            if d > MAXLAG:
                break
            a, b = u[i], u[j]
            if a == b:
                continue
            if a < b:
                H[_pair_index(a, b, n), d + MAXLAG] += 1
            else:
                H[_pair_index(b, a, n), MAXLAG - d] += 1
    return H


def _evaluate(H, n_spikes, n):
    """Flag pairs; returns arrays (pair a, pair b, lag L, excess fraction, p)."""
    cs = np.concatenate([np.zeros((H.shape[0], 1), np.int64), np.cumsum(H, axis=1, dtype=np.int64)], axis=1)
    c = np.arange(-L_MAX, L_MAX + 1) + MAXLAG
    obs = cs[:, c + HALF_W + 1] - cs[:, c - HALF_W]
    left = cs[:, c - FLANK[0] + 1] - cs[:, c - FLANK[1]]
    right = cs[:, c + FLANK[1] + 1] - cs[:, c + FLANK[0]]
    n_flank = 2 * (FLANK[1] - FLANK[0] + 1)
    expected = (left + right) / n_flank * (2 * HALF_W + 1)
    a_idx, b_idx = np.triu_indices(n, 1)
    n_small = np.minimum(n_spikes[a_idx], n_spikes[b_idx])[:, None]
    frac = (obs - expected) / np.maximum(n_small, 1)
    cand = frac >= MIN_EXCESS
    p = np.ones_like(frac)
    # the flank count is itself noisy (zero for small units), so test the window's share of
    # window + flanks rather than treating the flank estimate as the exact expected count
    share = (2 * HALF_W + 1) / (2 * HALF_W + 1 + n_flank)
    p[cand] = binom.sf(obs[cand] - 1, obs[cand] + (left + right)[cand], share)
    ok = cand & (p < ALPHA / (H.shape[0] * c.size))
    rows = np.where(ok.any(1))[0]
    best = np.argmax(np.where(ok[rows], frac[rows], -np.inf), axis=1)
    return (a_idx[rows], b_idx[rows], best - L_MAX, frac[rows, best], p[rows, best])


@njit(cache=True)
def _coincident(t, u, n, flag_lag, flag_mode, n_spikes):
    """Mark spikes to remove for flagged pairs. flag_lag[pair] = L + MAXLAG (or -1);
    flag_mode: 0 = remove from the unit with fewer spikes, 1 = remove from both."""
    remove = np.zeros(t.size, np.bool_)
    for i in range(t.size):
        for j in range(i + 1, t.size):
            d = t[j] - t[i]
            if d > MAXLAG:
                break
            a, b = u[i], u[j]
            if a == b:
                continue
            ia, ib, lag = (i, j, d) if a < b else (j, i, -d)
            lo, hi = (a, b) if a < b else (b, a)
            pi = _pair_index(lo, hi, n)
            L = flag_lag[pi]
            if L < 0 or abs(lag + MAXLAG - L) > HALF_W:
                continue
            if flag_mode[pi] == 1:
                remove[ia] = True; remove[ib] = True
            elif n_spikes[lo] <= n_spikes[hi]:
                remove[ia] = True
            else:
                remove[ib] = True
    return remove


def duplicates(t, u, xy, rng):
    n = xy.shape[0]
    n_spikes = np.bincount(u, minlength=n)
    a, b, L, frac, p = _evaluate(_lag_histograms(t, u, n), n_spikes, n)
    null = _evaluate(_lag_histograms(*interval_jitter(t, u, rng), n), n_spikes, n)[0].size

    dist = np.linalg.norm(xy[a] - xy[b], axis=1)
    cls = np.where(dist <= NEAR_UM, 'same_neuron', 'artefact')
    flag_lag = np.full(n * (n - 1) // 2, -1, np.int64)
    flag_mode = np.zeros(n * (n - 1) // 2, np.int64)
    for ai, bi, li, ci in zip(a, b, L, cls):
        pi = ai * n - ai * (ai + 1) // 2 + (bi - ai - 1)
        flag_lag[pi] = li + MAXLAG
        flag_mode[pi] = 1 if ci == 'artefact' else 0
    remove = _coincident(t, u, n, flag_lag, flag_mode, n_spikes)
    pairs = [{'a': int(x), 'b': int(y_), 'lag_samples': int(l), 'excess_fraction': float(f), 'p': float(pp),
              'distance_um': float(d), 'class': str(c)} for x, y_, l, f, pp, d, c in zip(a, b, L, frac, p, dist, cls)]
    return {'n_flagged': len(pairs), 'n_flagged_jittered_null': int(null),
            'by_class': {c: int(np.sum(cls == c)) for c in ('same_neuron', 'artefact')}}, pairs, remove


# ---------------- applying the cleaning ----------------

def _merge(iv):
    iv = iv[np.argsort(iv[:, 0])]
    out = []
    for a, b in iv:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return np.array(out, float).reshape(-1, 2)


def mask_intervals(sort, raw_artefacts):
    """Merged masked intervals (excerpt seconds): multi-unit events of this sort plus the raw safety net."""
    ev = np.load(Path(sort) / 'cleaning.npz')['event_intervals_samples'].astype(float)
    raw = np.load(raw_artefacts)['events']
    raw = raw[raw[:, 2] >= RAW_MASK_MIN_CH][:, :2] + [-RAW_MASK_PAD, RAW_MASK_PAD]
    return _merge(np.concatenate([ev, raw]) / FS)


def masked_fraction(bounds, intervals):
    """Masked share of each interval [bounds[i], bounds[i+1]) given merged intervals."""
    if not intervals.size:
        return np.zeros(bounds.size - 1)
    s, e = intervals[:, 0], intervals[:, 1]
    cum = np.concatenate([[0], np.cumsum(e - s)])

    def F(t):  # masked time before t
        i = np.searchsorted(s, t, 'right') - 1
        return np.where(i >= 0, cum[np.maximum(i, 0)] + np.clip(t - s[np.maximum(i, 0)], 0, (e - s)[np.maximum(i, 0)]), 0)
    return np.diff(F(bounds)) / np.diff(bounds)


def load_clean_units(sort, raw_artefacts):
    """Spike times (s) per cluster after §6.1-6.3, median depth per cluster, and masked intervals (s)."""
    sort = Path(sort)
    st = np.load(sort / 'spike_times.npy').ravel()
    clu = np.load(sort / 'spike_clusters.npy').ravel()
    y = np.load(sort / 'spike_positions.npy')[:, 1]
    c = np.load(sort / 'cleaning.npz')
    removed = np.zeros(st.size, bool)
    removed[c['spike_order']] = c['removed_event'] | c['removed_duplicate']
    iv = mask_intervals(sort, raw_artefacts)
    t = st / FS
    i = np.searchsorted(iv[:, 0], t, 'right') - 1
    removed |= (i >= 0) & (t < iv[np.maximum(i, 0), 1]) if iv.size else False
    ids = np.unique(clu)
    depth = {int(k): float(np.median(y[clu == k])) for k in ids}
    keep = ~removed
    t, clu = t[keep], clu[keep]
    return {int(k): t[clu == k] for k in ids}, depth, iv


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sort', required=True)
    ap.add_argument('--duration', type=float, required=True)
    ap.add_argument('--seed', type=int, default=0, help='seed for the jittered copies')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    t, u, ids, xy = load_sort(args.sort)
    ev, rm_events, intervals = multiunit_events(t, u, xy, args.duration, rng)
    keep = ~rm_events
    dup, pairs, rm_dup = duplicates(t[keep], u[keep], xy, rng)
    removed_dup = np.zeros(t.size, bool)
    removed_dup[np.where(keep)[0][rm_dup]] = True
    for pr in pairs:
        pr['a'], pr['b'] = int(ids[pr['a']]), int(ids[pr['b']])
    report = {'n_spikes': int(t.size), 'n_units': int(ids.size),
              'multiunit_events': ev, 'events_spikes_removed': int(rm_events.sum()),
              'duplicates': dup, 'duplicate_spikes_removed': int(removed_dup.sum()),
              'fraction_spikes_removed': float((rm_events | removed_dup).mean())}
    out = Path(args.sort)
    np.savez(out / 'cleaning.npz', spike_order=np.argsort(np.load(out / 'spike_times.npy').ravel(), kind='stable'),
             removed_event=rm_events, removed_duplicate=removed_dup, event_intervals_samples=intervals)
    (out / 'cleaning.json').write_text(json.dumps({**report, 'flagged_pairs': pairs}, indent=2))
    print(json.dumps({**report, 'multiunit_events': {k: v for k, v in ev.items() if 'per_10min' not in k or k == 'events_per_10min'}}, indent=2))
    print('multi-unit events per 10 min by k (real / jittered null):')
    print(' '.join(f"k={k}:{ev['real_events_per_10min'][k]:.0f}/{ev['null_events_per_10min'][k]:.1f}" for k in list(ev['real_events_per_10min'])[:20]))
