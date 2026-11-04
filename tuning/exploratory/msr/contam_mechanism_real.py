"""Quick diagnostic (report writing): why does Poisson contamination raise MSR on real units but lower
it on the theory note's synthetic OU unit? Compare the contamination response of each real target with
that of (a) a jittered surrogate (+-0.5 s uniform jitter: keeps slow modulation, destroys sub-second
structure such as refractoriness, bursts and in-field regularity) and (b) an ISI-shuffled surrogate
(keeps the ISI distribution, destroys slow modulation). Also attribute the MSR change to bin-width bands.
"""
import sys
from pathlib import Path

import numpy as np

MSRDIR = Path('C:/code/github/Kilosort/tuning/exploratory/msr')
sys.path.insert(0, str(MSRDIR))
sys.path.insert(0, str(MSRDIR.parents[0] / 'isolation'))
import common  # noqa: E402
import msr  # noqa: E402
from comod import unit_estimates  # noqa: E402

rng = np.random.default_rng(0)
DUR = common.DUR
sort = common.SORTS[sys.argv[1] if len(sys.argv) > 1 else 'default_s1']
by, iv, T = common.clean_units(sort)
ev = common.evaluation(sort)
units = [int(k) for k in ev['S_i']]
S0 = np.array([ev['S_i'][str(c)] for c in units])
con = {c: unit_estimates(by[c], T)[2]['plain'] for c in units}
q3 = np.quantile(S0, 0.75)
pool = [c for c in units if con[c] <= 0.05 and ev['S_i'][str(c)] >= q3]
targets = [int(c) for c in rng.choice(pool, min(40, len(pool)), replace=False)]
print(len(units), 'eligible,', len(pool), 'pool,', len(targets), 'targets')

Tg = int(round(DUR / msr.DT_S))
Ns = msr.partitions(Tg)
width = DUR / Ns                       # bin width (s) per partition
bands = [(0, 0.01), (0.01, 0.1), (0.1, 1), (1, 10), (10, 1e9)]


def band_area(t):
    """MSR and its split into bin-width bands (segments between consecutive partitions sorted by H[s])."""
    ticks = np.sort(np.clip(np.floor(t / msr.DT_S).astype(np.int64), 0, Tg - 1))
    hk, hs = msr._curve(ticks, Tg, Ns)
    o = np.argsort(hs)
    x, y, w = hs[o], hk[o], width[o]
    seg = 0.5 * (y[:-1] + y[1:]) * np.diff(x)
    wm = np.sqrt(w[:-1] * w[1:])
    out = [seg[(wm >= a) & (wm < b)].sum() for a, b in bands]
    return msr.area(hs, hk), np.array(out)


def contaminate(t, f):
    return np.sort(np.concatenate([t, rng.uniform(0, DUR, int(f * t.size))]))


def jitter(t, h=0.5):
    return np.sort(np.mod(t + rng.uniform(-h, h, t.size), DUR))


def isi_shuffle(t):
    isi = np.diff(t)
    return np.sort(np.mod(t[0] + np.concatenate([[0], np.cumsum(rng.permutation(isi))]), DUR))


def fano(t, w):
    c = np.bincount((t // w).astype(int), minlength=int(DUR // w) + 1)[: int(DUR // w)]
    return c.var() / c.mean()


def lv(t):
    i = np.diff(t)
    return 3 * np.mean(((i[:-1] - i[1:]) / (i[:-1] + i[1:])) ** 2)


F = 0.3
rows = []
for a in targets:
    t = by[a]
    rec = {}
    for name, make in (('real', lambda: t), ('jitter', lambda: jitter(t)), ('isishuf', lambda: isi_shuffle(t))):
        d_m, d_b = [], []
        for _ in range(3):
            base = make()
            m0, b0 = band_area(base)
            m1, b1 = band_area(contaminate(base, F))
            d_m.append(m1 - m0)
            d_b.append(b1 - b0)
        rec[name] = (np.mean(d_m), np.mean(d_b, 0))
    rec['fano'] = [fano(t, w) for w in (0.005, 0.02, 0.1, 1.0, 10.0)]
    rec['lv'] = lv(t)
    rec['frac_isi_lt_20ms'] = float(np.mean(np.diff(t) < 0.02))
    rows.append(rec)

for name in ('real', 'jitter', 'isishuf'):
    dm = np.array([r[name][0] for r in rows])
    db = np.array([r[name][1] for r in rows])
    print(f'{name:8s} dMSR(+30% Poisson, raw): median {np.median(dm):+.5f}  frac up {np.mean(dm > 0):.2f}')
    print('          by band ' + '  '.join(f'{a}-{b}s {np.median(db[:, i]):+.5f}' for i, (a, b) in enumerate(bands)))
fa = np.array([r['fano'] for r in rows])
print('median Fano at 5ms,20ms,100ms,1s,10s:', np.round(np.median(fa, 0), 3))
print('frac units Fano<1 at 5ms,20ms,100ms:', np.round(np.mean(fa[:, :3] < 1, 0), 2))
print('median LV', round(float(np.median([r['lv'] for r in rows])), 3),
      ' median frac ISI<20ms', round(float(np.median([r['frac_isi_lt_20ms'] for r in rows])), 3))
from scipy.stats import spearmanr
dm = np.array([r['real'][0] for r in rows])
fb = np.array([r['frac_isi_lt_20ms'] for r in rows])
lvs = np.array([r['lv'] for r in rows])
print('Spearman(real dMSR, frac ISI<20ms):', spearmanr(dm, fb))
print('Spearman(real dMSR, LV):', spearmanr(dm, lvs))
lo, hi = fb < np.median(fb), fb >= np.median(fb)
print(f'median dMSR, less bursty half {np.median(dm[lo]):+.5f}, more bursty half {np.median(dm[hi]):+.5f}')
