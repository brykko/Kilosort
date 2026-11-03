"""Q: is KS4's 0.23-0.8 ms within-unit excess algorithmic duplication (driven by the waveform, so a
hybrid copy should show it as much as its source) or absorbed fine-timescale correlation with
neighbours (so only the source, which keeps its natural neighbours, shows it)? For each of the 40
inserted units, take the sorted unit best matching the inserted train (copy, at the target site) and
the sorted unit best matching the source train (inserted times minus the shift, at the source site),
in the same hybrid sort, and compare pairs at lags 7-20 samples with the level at 25-45.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import KOBE, TRUTH, load_ks4_full, load_phy, load_traced, pairs_within  # noqa

TOL = 12
N_SAMP = 1200 * 30000


def best_unit(t, clu, train):
    s = np.sort(train)
    j = np.clip(np.searchsorted(s, t), 1, s.size - 1)
    dmin = np.minimum(np.abs(s[j - 1] - t), np.abs(s[j] - t))
    hit = dmin <= TOL
    ids, n = np.unique(clu, return_counts=True)
    h = np.bincount(np.searchsorted(ids, clu[hit]), minlength=ids.size)
    agree = h / (n + train.size - h)
    k = agree.argmax()
    return ids[k], agree[k]


def counts(t, clu, c):
    tt = np.sort(t[clu == c])
    I, J = pairs_within(tt, np.zeros(tt.size, np.int64), 45)
    lag = tt[J] - tt[I]
    return tt.size, int(((lag >= 7) & (lag <= 20)).sum()), int(((lag >= 25) & (lag <= 45)).sum()), \
        int((lag <= 6).sum())


def run(name, t, clu, tt, tu, shift):
    rows = []
    for u in range(40):
        ins = tt[tu == u]
        src = (ins - int(round(shift[u] * 30000))) % N_SAMP
        cc, ac = best_unit(t, clu, ins)
        cs, as_ = best_unit(t, clu, src)
        rows.append((u, ac, as_) + counts(t, clu, cc) + counts(t, clu, cs))
    R = np.array(rows, float)
    ok = (R[:, 1] >= 0.5) & (R[:, 2] >= 0.5)
    print(f'\n{name}: {ok.sum()} of 40 inserted units with copy and source both matched (agreement >= 0.5)')
    for lab, o in (('copy  ', 3), ('source', 7)):
        n, d, c, z = R[ok, o], R[ok, o + 1], R[ok, o + 2], R[ok, o + 3]
        exp = c.sum() * 14 / 21
        print(f'  {lab}: spikes={n.sum():.0f} pairs 0-6={z.sum():.0f} 7-20={d.sum():.0f} exp(25-45)={exp:.0f} '
              f'ratio={d.sum() / max(exp, 1):.2f} excess/1000 spikes={1000 * (d.sum() - exp) / n.sum():.2f}')
    return R


if __name__ == '__main__':
    z = np.load(TRUTH)
    tt, tu, shift = z['spike_samples'].astype(np.int64), z['spike_unit'], z['shift_s']
    for v in ('default', 'H8'):
        d = load_ks4_full(f'kobe_{v}_s1')
        run(f'KS4 {v} s1 pre-dedup', d['t'], d['clu'], tt, tu, shift)
    d = load_phy(KOBE / 'runs/ks25_standard_r1')
    run('KS2.5 standard r1 final', d['t'], d['clu'], tt, tu, shift)
    d = load_traced('kobe_s1', 'extracted')
    run('KS2.5 traced extracted', d['t'], d['clu'], tt, tu, shift)
