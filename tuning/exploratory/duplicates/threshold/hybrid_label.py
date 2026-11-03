"""Q: in sorted units that are copies of inserted hybrid units (kobe), which short-lag within-unit pairs are
duplicates (both members explained by ONE inserted spike) and which are not? Gives the duplicate lag
distribution f_D(k) at sample resolution with ground truth, P(dup | lag) on these units, and how well
non-lag features (same extraction template, score/amp ratio) separate duplicates from the rest.

Inserted trains are clean (22 of 223,576 inserted ISIs <= 1 ms), so a non-duplicate short-lag pair in a
hybrid copy involves at least one background (non-inserted) spike.
Labels for a pair (a earlier, b later), times corrected by the unit's median offset to the inserted train:
  dup   : nearest inserted spike to a and to b is the same one, and one member is within W of it
  hyb+bg: exactly one member within W of an inserted spike (the other is background)
  bg+bg : neither member within W of an inserted spike (and not dup)
  hyb+hyb: members within W of two different inserted spikes (genuine inserted ISI)
Writes hybrid_label_<sort>.npz.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import KOBE, TRUTH, load_ks4_full, load_phy, load_traced, pairs_within  # noqa

TOL, COPY, W, MAXLAG = 12, 0.5, 4, 90
HERE = Path(__file__).parent


def copies(t, clu, tt, tu):
    """Sorted units with >= COPY of their spikes within TOL of one inserted unit's spikes, with offset."""
    o = np.argsort(tt)
    tt, tu = tt[o], tu[o]
    j = np.clip(np.searchsorted(tt, t), 1, tt.size - 1)
    near = np.where(np.abs(tt[j - 1] - t) <= np.abs(tt[j] - t), j - 1, j)
    d = t - tt[near]
    hit = np.abs(d) <= TOL
    out = {}
    ids, n = np.unique(clu, return_counts=True)
    for c, nc in zip(ids, n):
        m = (clu == c) & hit
        if m.sum() < COPY * nc:
            continue
        uu = np.bincount(tu[near[m]]).argmax()
        mm = m & (tu[near] == uu)
        if mm.sum() >= COPY * nc:
            out[int(c)] = (int(uu), int(np.median(d[mm])), int(nc))
    return out


def label(t, clu, extra, tt, tu):
    cp = copies(t, clu, tt, tu)
    I, J = pairs_within(t, clu, MAXLAG)
    keep = np.isin(clu[I], list(cp))
    I, J = I[keep], J[keep]
    lab = np.zeros(I.size, np.int8)       # 1 dup, 2 hyb+bg, 3 bg+bg, 4 hyb+hyb, 0 other
    for c, (uu, off, _) in cp.items():
        m = np.where(clu[I] == c)[0]
        s = np.sort(tt[tu == uu])
        def nearest(x):
            x = x - off
            j = np.clip(np.searchsorted(s, x), 1, s.size - 1)
            k = np.where(np.abs(s[j - 1] - x) <= np.abs(s[j] - x), j - 1, j)
            return k, np.abs(s[k] - x)
        ka, da = nearest(t[I[m]]); kb, db = nearest(t[J[m]])
        same = ka == kb
        l = np.zeros(m.size, np.int8)
        l[same & (np.minimum(da, db) <= W)] = 1
        rest = l == 0
        na, nb = da <= W, db <= W
        l[rest & (na ^ nb)] = 2
        l[rest & ~na & ~nb] = 3
        l[rest & na & nb & ~same] = 4
        lab[m] = l
    lag = t[J] - t[I]
    res = dict(lag=lag, lab=lab, unit=clu[I], n_units=len(cp), n_spikes=sum(v[2] for v in cp.values()))
    for k, v in extra.items():
        res[k + '_a'], res[k + '_b'] = v[I], v[J]
    return res


def runs():
    for v in ('default', 'H8'):
        d = load_ks4_full(f'kobe_{v}_s1')
        yield f'ks4_{v}_pre', d['t'], d['clu'], dict(tmpl=d['tmpl'], score=d['score'], amp=d['amp'])
    for r in (1, 2, 3):
        d = load_phy(KOBE / f'runs/ks25_standard_r{r}')
        yield f'ks25_standard_r{r}', d['t'], d['clu'], dict(tmpl=d['tmpl'], amp=d['amp'])
    d = load_traced('kobe_s1', 'extracted')
    yield 'ks25_traced_extracted', d['t'], d['clu'], dict(score=d['score'], amp=d['amp'])
    for s in (2, 3, 4, 5):
        d = load_phy(KOBE / f'runs/hybrid_default_seed{s}')
        yield f'ks4_default_s{s}_final', d['t'], d['clu'], dict(tmpl=d['tmpl'], amp=d['amp'])
    for f in ('208_H8_cds10_s2', '218_H8_cds10_s3'):
        d = load_phy(KOBE / f'runs/search/{f}/sort')
        yield f'ks4_H8_{f[-2:]}_final', d['t'], d['clu'], dict(tmpl=d['tmpl'], amp=d['amp'])


if __name__ == '__main__':
    z = np.load(TRUTH)
    tt, tu = z['spike_samples'].astype(np.int64), z['spike_unit'].astype(np.int64)
    names = ['other', 'dup', 'hyb+bg', 'bg+bg', 'hyb+hyb']
    for name, t, clu, extra in runs():
        r = label(t, clu, extra, tt, tu)
        np.savez(HERE / f'hybrid_label_{name}.npz', **r)
        lag, lab = r['lag'], r['lab']
        print(f"\n{name}: {r['n_units']} hybrid-copy units, {r['n_spikes']} spikes")
        for a, b in [(0, 6), (7, 9), (10, 12), (13, 15), (16, 20), (21, 30), (31, 45), (46, 60), (61, 90)]:
            m = (lag >= a) & (lag <= b)
            c = np.bincount(lab[m], minlength=5)
            print(f'  lag {a:2d}-{b:2d}: ' + ' '.join(f'{names[i]}={c[i]}' for i in range(5)) +
                  f'  P(dup)={c[1] / max(c.sum(), 1):.2f}', flush=True)
