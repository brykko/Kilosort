"""Shared loaders for the empirics of the interleaving audit: saved sorts, cleaned spikes, eligible units, spike-level
matching in time and depth, unit chains between rounds and seeds, rate/amplitude strata.

CPU only. Reads C:\\data\\ks4_tuning\\exploratory\\iteration\\ and writes nothing itself.
"""
import json
import sys
from pathlib import Path

import numpy as np
from numba import njit

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'tuning'))

from cleaning import FS, mask_intervals  # noqa: E402
from compare import TOL_S, agreement  # noqa: E402

IT = Path('C:/data/ks4_tuning/exploratory/iteration')
SORTS = IT / 'sorts'
OUT = Path('C:/data/ks4_tuning/exploratory/interleaving/empirics')
K = Path('C:/data/ks4_tuning/26648_kobe_2020-08-21')
M = Path('C:/data/ks4_tuning/25843_roger_2019-09-11')
BASE = {'kobe': K, 'mec': M}
DUR = 1200.0
SEEDS = (1, 2, 3)
NT, NT0 = 61, 20
BATCH = 60000
# spike-identity tolerance between rounds: same event if within this many samples and microns
TOL_SAMPLES = 5
TOL_UM = 40.0
ROUNDS = {'H8': (2, 3, 4), 'default': (2, 3)}


def sort_dir(ex, setting, seed, rnd):
    """Folder of a round: 'H8' -> {ex}_repeat_s{seed}, default -> {ex}_default_repeat_s{seed}."""
    run = f"{ex}_{'default_' if setting == 'default' else ''}repeat_s{seed}"
    last = ROUNDS[setting][-1]
    return SORTS / run if rnd == last else SORTS / run / f'round{rnd}'


_mec_cache = {}


def scored(d, ex, setting='H8'):
    """{unit: S_i} of a sort, as the harness scored it: its keys are the eligible units."""
    d = Path(d)
    if ex == 'kobe':
        si = json.loads((d / 'evaluation.json').read_text())['S_i']
    else:
        f = IT / ('transfer_mec.json' if setting == 'H8' else f'transfer_mec_{setting}.json')
        if f not in _mec_cache:
            _mec_cache[f] = {Path(w['sort']): w for w in json.loads(f.read_text())['winner']}
        si = _mec_cache[f][d]['S_i']
    return {int(c): float(v) for c, v in si.items()}


def amplitudes(d):
    a = np.loadtxt(Path(d) / 'cluster_Amplitude.tsv', skiprows=1, ndmin=2)
    return dict(zip(a[:, 0].astype(int), a[:, 1]))


class Sort:
    """Spikes of a saved sort. t: samples, sorted; clu; y (um, drift-corrected); keep: cleaned spike."""

    def __init__(self, d, ex, setting='H8', clean=True):
        d = Path(d)
        self.d, self.ex, self.setting = d, ex, setting
        t = np.load(d / 'spike_times.npy').ravel().astype(np.int64)
        o = np.argsort(t, kind='stable')
        self.t = t[o]
        self.clu = np.load(d / 'spike_clusters.npy').ravel().astype(np.int64)[o]
        self.y = np.load(d / 'spike_positions.npy')[:, 1].astype(np.float32)[o]
        keep = np.ones(t.size, bool)
        if clean:
            c = np.load(d / 'cleaning.npz')
            rem = np.zeros(t.size, bool)
            rem[c['spike_order']] = c['removed_event'] | c['removed_duplicate']
            iv = mask_intervals(d, BASE[ex] / 'search/raw_artefacts_car.npz')
            ts = t / FS
            i = np.searchsorted(iv[:, 0], ts, 'right') - 1
            if iv.size:
                rem |= (i >= 0) & (ts < iv[np.maximum(i, 0), 1])
            keep = ~rem[o]
        self.keep = keep
        self.order = o
        self.n_units = int(self.clu.max()) + 1
        self.S = None
        self.amp = amplitudes(d)

    def eligible(self):
        if self.S is None:
            self.S = scored(self.d, self.ex, self.setting)
        return self.S

    def wall(self):
        return np.load(self.d / 'Wall.npy', mmap_mode='r')

    def cleaned(self):
        return self.t[self.keep], self.y[self.keep], self.clu[self.keep]


@njit(cache=True)
def nearest_event(ta, ya, tb, yb, tol, dy):
    """For each a-spike (sorted times) index of the nearest-in-time b spike within tol samples and dy um, or -1."""
    na, nb = ta.size, tb.size
    out = np.full(na, -1, dtype=np.int64)
    j0 = 0
    for i in range(na):
        while j0 < nb and tb[j0] < ta[i] - tol:
            j0 += 1
        best, bd, j = -1, tol + 1, j0
        while j < nb and tb[j] <= ta[i] + tol:
            if abs(ya[i] - yb[j]) <= dy:
                d = abs(tb[j] - ta[i])
                if d < bd:
                    best, bd = j, d
            j += 1
        out[i] = best
    return out


def table(a_lab, b_lab):
    """Sparse contingency of matched spikes: unique (a, b) pairs and counts."""
    k = a_lab.astype(np.int64) * 100000 + b_lab
    ids, n = np.unique(k, return_counts=True)
    return ids // 100000, ids % 100000, n


class Link:
    """Spike-level link between two sorts over cleaned spikes (a -> b)."""

    def __init__(self, a, b, tol=TOL_SAMPLES, dy=TOL_UM, shift=0):
        self.ta, ya, self.ca = a.cleaned()
        tb, yb, cb = b.cleaned()
        self.m = nearest_event(self.ta, ya, tb + shift, yb, tol, dy)
        ok = self.m >= 0
        self.cb = np.where(ok, cb[np.maximum(self.m, 0)], -1)
        self.ok = ok
        ua, ub, n = table(self.ca[ok], self.cb[ok])
        self.ua, self.ub, self.n = ua, ub, n

    def successors(self):
        """{a unit: (b unit holding most of its spikes, shared count)}, any b unit, no threshold."""
        out = {}
        o = np.lexsort((self.n, self.ua))
        ua, ub, n = self.ua[o], self.ub[o], self.n[o]
        last = np.flatnonzero(np.r_[ua[1:] != ua[:-1], True])
        for i in last:
            out[int(ua[i])] = (int(ub[i]), int(n[i]))
        return out

    def candidates(self, units, k=3):
        o = np.lexsort((-self.n, self.ua))
        ua, ub = self.ua[o], self.ub[o]
        out = {u: [] for u in units}
        for x, y in zip(ua, ub):
            x = int(x)
            if x in out and len(out[x]) < k:
                out[x].append(int(y))
        return out


def unit_times(s, units):
    t, _, c = s.cleaned()
    o = np.argsort(c, kind='stable')
    t, c = t[o], c[o]
    b = np.searchsorted(c, np.arange(s.n_units + 1))
    return {u: np.sort(t[b[u]:b[u + 1]]) / FS for u in units}


def best_agreement(link, ta, tb, ua, ub):
    """For each eligible unit of a, its best eligible unit of b by compare.agreement (candidates from the link)."""
    ub = set(ub)
    c = link.candidates(list(ua))
    out = {}
    for u in ua:
        best = (0.0, -1)
        for v in c[u]:
            if v in ub and ta[u].size > 1 and tb[v].size > 1:
                best = max(best, (agreement(ta[u], tb[v]), v))
        out[u] = best
    return out


def mutual_pairs(a, b, link_ab=None, link_ba=None, thr=0.5):
    """One-to-one pairs of eligible units: mutual best at agreement >= thr. Returns {a unit: (b unit, agreement)}."""
    link_ab = link_ab or Link(a, b)
    link_ba = link_ba or Link(b, a)
    ea, eb = list(a.eligible()), list(b.eligible())
    ta, tb = unit_times(a, ea), unit_times(b, eb)
    fwd = best_agreement(link_ab, ta, tb, ea, eb)
    back = best_agreement(link_ba, tb, ta, eb, ea)
    return {u: (v, g) for u, (g, v) in fwd.items() if g >= thr and v in back and back[v][1] == u}


def strata(rate, amp):
    """Rate and amplitude terciles over all given units (dict unit -> value): {unit: (rate tercile, amp tercile)}."""
    us = list(rate)
    r = np.array([rate[u] for u in us])
    a = np.array([amp.get(u, np.nan) for u in us])
    rq, aq = np.quantile(r, [1 / 3, 2 / 3]), np.nanquantile(a, [1 / 3, 2 / 3])
    return {u: (int(np.searchsorted(rq, x)), int(np.searchsorted(aq, y))) for u, x, y in zip(us, r, a)}


def rates(s, units, dur=DUR):
    n = np.bincount(s.clu[s.keep], minlength=s.n_units)
    return {u: n[u] / dur for u in units}


def wall_time(W, wpca):
    """Wall units x chans x pcs -> units x chans x nt (whitened)."""
    return W @ wpca


def norm(x):
    return float(np.sqrt((x * x).sum()))
