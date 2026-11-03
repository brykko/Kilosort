"""Within-unit short-lag spike pairs, per sorter, before/after each sorter's own handling.

Question: does the lab's KS2.5 pipeline leave within-unit pairs closer than KS4's 0.25 ms window
(i.e. does it remove nothing)? And how do the within-unit lag profiles of KS2.5 and KS4 compare
when normalised by the 2-5 ms level (observed / expected per sample of lag)?

For every sort: all within-unit spike pairs (not only consecutive spikes) at lags 0..150 samples
(30 kHz). Density per lag = pairs / (n_spikes) per sample of lag; ratio = density / mean density
at 60-150 samples (2-5 ms). Reported in lag bands. Variants: KS2.5 traced stages (extracted =
before autocurate, cut1, final), KS4 full_st (before KS4's dedup), final Phy output, and final
output after the harness cleaning (cleaning.npz: artefact events + cross-unit duplicates).
"""
import json
import sys
from pathlib import Path

import numpy as np
from numba import njit

KOBE = Path(r'C:\data\ks4_tuning\26648_kobe_2020-08-21\runs')
MEC = Path(r'C:\data\ks4_tuning\25843_roger_2019-09-11\runs')
AUTO = Path(r'C:\data\ks4_tuning\exploratory\autocurate\sorts')
TRACED = Path(r'C:\data\ks4_tuning\exploratory\ks25_traced')
OUT = Path(__file__).parent / 'within_unit_lags.json'
MAXLAG = 150
BANDS = {'0': (0, 0), '1-6 (<=0.2ms)': (1, 6), '7 (0.233ms)': (7, 7), '8-15 (0.27-0.5ms)': (8, 15),
         '16-30 (0.53-1ms)': (16, 30), '31-45 (1-1.5ms)': (31, 45), '46-59 (1.5-2ms)': (46, 59)}


@njit(cache=True)
def lag_hist(t, c):
    """t, c sorted by (c, t). H[lag] = within-unit pairs at that lag (all pairs)."""
    H = np.zeros(MAXLAG + 1, np.int64)
    n = t.size
    for i in range(n):
        j = i + 1
        while j < n and c[j] == c[i]:
            d = t[j] - t[i]
            if d > MAXLAG:
                break
            H[d] += 1
            j += 1
    return H


def summarise(t, c):
    t = np.asarray(t, np.int64).ravel()
    c = np.asarray(c, np.int64).ravel()
    o = np.lexsort((t, c))
    H = lag_hist(t[o], c[o])
    n = t.size
    base = H[60:151].mean()
    out = {'n_spikes': int(n), 'n_units': int(np.unique(c).size)}
    for k, (a, b) in BANDS.items():
        cnt = int(H[a:b + 1].sum())
        out[k] = {'pairs': cnt, 'per_1k_spikes': 1e3 * cnt / n,
                  'obs_over_exp': float(H[a:b + 1].mean() / base) if base else None}
    out['H_0_45'] = H[:46].tolist()
    out['base_per_lag'] = float(base)
    return out


def phy(sort, clean=False):
    t = np.load(sort / 'spike_times.npy').ravel().astype(np.int64)
    c = np.load(sort / 'spike_clusters.npy').ravel().astype(np.int64)
    if clean:
        z = np.load(sort / 'cleaning.npz')
        rm = np.zeros(t.size, bool)
        rm[z['spike_order']] = z['removed_event'] | z['removed_duplicate']
        t, c = t[~rm], c[~rm]
    return t, c


def main():
    res = {}
    sorts = {}
    for r in (1, 2, 3):
        sorts[f'kobe_ks25_standard_r{r}'] = KOBE / f'ks25_standard_r{r}'
        sorts[f'kobe_ks25_low_r{r}'] = KOBE / f'ks25_standard_low_r{r}'
        sorts[f'mec_ks25_standard_r{r}'] = MEC / f'ks25_standard_r{r}'
        sorts[f'mec_ks25_low_r{r}'] = MEC / f'ks25_standard_low_r{r}'
        sorts[f'mec_ks4_default_s{r}'] = MEC / f'default_s{r}'
        sorts[f'mec_ks4_H8_s{r}'] = MEC / f'H8_s{r}'
    for s in (1, 2, 3, 4, 5):
        sorts[f'kobe_ks4_default_s{s}'] = KOBE / f'hybrid_default_seed{s}'
    for s, k in ((1, 198), (2, 208), (3, 218)):
        sorts[f'kobe_ks4_H8_s{s}'] = KOBE / 'search' / f'{k}_H8_cds10_s{s}' / 'sort'
    for name, p in sorts.items():
        if not (p / 'spike_times.npy').exists():
            print('missing', name, p); continue
        res[name + '|final'] = summarise(*phy(p))
        if (p / 'cleaning.npz').exists():
            res[name + '|final_cleaned'] = summarise(*phy(p, True))
        print(name, flush=True)
    for name in ('kobe_default_s1', 'kobe_H8_s1', 'mec_default_s1', 'mec_H8_s1'):
        p = AUTO / name
        fs = np.load(p / 'full_st.npy')
        res[f'auto_ks4_{name}|before_dedup'] = summarise(fs[:, 0], np.load(p / 'full_clu.npy'))
        res[f'auto_ks4_{name}|final'] = summarise(*phy(p))
        print(name, flush=True)
    for ex in ('kobe_s1', 'mec_s1'):
        z = np.load(TRACED / ex / 'stages.npz')
        for st in ('extracted', 'split1', 'cut1', 'merge3'):
            res[f'traced_ks25_{ex}|{st}'] = summarise(z[f'{st}_t'], z[f'{st}_clu'])
        res[f'traced_ks25_{ex}|final_cleaned'] = summarise(*phy(TRACED / ex, True))
        print(ex, flush=True)
    OUT.write_text(json.dumps(res, indent=1))
    keys = list(BANDS)
    print('name'.ljust(46), 'nsp(M)', *[k.split(' ')[0].rjust(16) for k in keys])
    for name, r in res.items():
        print(name.ljust(46), f"{r['n_spikes']/1e6:6.2f}",
              *[f"{r[k]['per_1k_spikes']:7.2f}/{r[k]['obs_over_exp']:6.2f}x".rjust(16) for k in keys])


if __name__ == '__main__':
    main()
