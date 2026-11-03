"""Can KS4's remove_duplicates (postprocessing.py:9-29: same cluster, within dt samples of the last KEPT
spike, keep the earlier) create a lag profile just beyond dt by itself?

1. Simulation: refractory (1.5 ms dead time) 20 Hz trains + 10% Poisson contamination, plus within-unit
   duplicates of 2% of spikes at lags 1..6 samples (some as chains of 3); lag histogram before/after
   remove_duplicates(dt=7) at sample resolution.
2. Real data (kobe/mec, default/H8 scored sorts, every spike before the removal): how many kept within-unit
   pairs at 7..15 samples had a removed spike between them (the chain effect), against all kept pairs there.
"""
import json
from pathlib import Path

import numpy as np

EXP = Path('C:/data/ks4_tuning/exploratory/autocurate/sorts')
OUT = Path(__file__).resolve().parent / 'dedup_sim.json'


def remove_duplicates(t, c, dt):
    keep = np.zeros(t.size, bool)
    last = {}
    for i in range(t.size):
        t0 = last.get(c[i], t[i] - dt)
        if t[i] >= t0 + dt:
            last[c[i]] = t[i]
            keep[i] = True
    return keep


def hist(t, c, maxl=45):
    key = np.sort(c.astype(np.int64) * (1 << 40) + t)
    H = np.zeros(maxl + 1, np.int64)
    d = 1
    while True:
        dk = key[d:] - key[:-d]
        m = dk <= maxl
        if not m.any():
            break
        H += np.bincount(dk[m], minlength=maxl + 1)[:maxl + 1]
        d += 1
    return H


def simulate(rng, n_units=200, dur_s=600, fs=30000):
    T, C = [], []
    for u in range(n_units):
        isi = rng.exponential(1 / 20, int(20 * dur_s * 1.5)) + 1.5e-3
        t = np.cumsum(isi); t = t[t < dur_s]
        cont = rng.uniform(0, dur_s, int(0.1 * t.size))
        t = np.round(np.r_[t, cont] * fs).astype(np.int64)
        dup = rng.choice(t, int(0.02 * t.size), replace=False)
        lag = rng.integers(1, 7, dup.size)
        chain = rng.random(dup.size) < 0.3
        t = np.r_[t, dup + lag, (dup + 2 * lag)[chain]]
        T.append(t); C.append(np.full(t.size, u))
    t, c = np.concatenate(T), np.concatenate(C)
    o = np.argsort(t, kind='stable')
    return t[o], c[o]


def main():
    rng = np.random.default_rng(0)
    t, c = simulate(rng)
    k = remove_duplicates(t, c, 7)
    H0, H1 = hist(t, c), hist(t[k], c[k])
    lvl = H0[30:46].mean()
    res = {'sim_before_ratio': (H0 / lvl).round(3).tolist(), 'sim_after_ratio': (H1 / lvl).round(3).tolist()}
    print('sim before', ' '.join(f'{v:.2f}' for v in H0[:25] / lvl))
    print('sim after ', ' '.join(f'{v:.2f}' for v in H1[:25] / lvl))
    for s in ('kobe_default_s1', 'kobe_H8_s1', 'mec_default_s1', 'mec_H8_s1'):
        f = np.load(EXP / s / 'full_st.npy')
        fc = np.load(EXP / s / 'full_clu.npy').ravel().astype(np.int64)
        kept = np.load(EXP / s / 'kept_spikes.npy')
        tt = f[:, 0].astype(np.int64)
        key = fc * (1 << 40) + tt
        o = np.argsort(key, kind='stable')
        ko, kk = key[o], kept[o]
        # for each kept spike, the next kept spike of the same cluster; was a removed spike in between?
        ki = np.where(kk)[0]
        nxt = ki[1:]; cur = ki[:-1]
        lag = ko[nxt] - ko[cur]
        between = (nxt - cur) > 1
        m = (lag >= 7) & (lag <= 15)
        res[s] = {'kept_consecutive_pairs_7to15': int(m.sum()), 'with_removed_between': int((m & between).sum()),
                  'removed_spikes': int((~kept).sum()), 'n': int(kept.size)}
        print(s, res[s])
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
