"""Within-unit lag profiles at sample resolution, per sorter, before/after each sorter's own duplicate
handling and after the harness's cleaning (spec 6.2-6.3 masks from cleaning.npz).

Question: where (in lag) do within-unit short-interval pairs sit, how big is the 0.3-0.5 ms bump relative
to the 2-5 ms level, and is it the same in KS4 and KS2.5?

For each spike set: H[L] = number of same-unit pairs with t_j - t_i = L samples (L = 0..150). Reported per
sort: ratio[L] = H[L] / mean(H[60..150]) (2-5 ms level), pairs per 1000 spikes in bands, and the bump
excess = sum over 9..15 samples of (H - floor), floor = mean H over 30..39 samples (1.0-1.3 ms).
Writes lag_profiles.json.
"""
import json
from pathlib import Path

import numpy as np

KOBE = Path('C:/data/ks4_tuning/26648_kobe_2020-08-21/runs')
MEC = Path('C:/data/ks4_tuning/25843_roger_2019-09-11/runs')
EXP = Path('C:/data/ks4_tuning/exploratory')
OUT = Path(__file__).resolve().parent / 'lag_profiles.json'
MAXL = 150


def hist(t, c):
    t = np.asarray(t, np.int64).ravel()
    c = np.asarray(c, np.int64).ravel()
    key = c * (1 << 40) + t
    key.sort()
    H = np.zeros(MAXL + 1, np.int64)
    d = 1
    while True:
        dk = key[d:] - key[:-d]
        m = dk <= MAXL
        if not m.any():
            break
        H += np.bincount(dk[m], minlength=MAXL + 1)[:MAXL + 1]
        d += 1
    return H


def summary(H, n):
    lvl = H[60:151].mean()
    floor = H[30:40].mean()
    return {'n_spikes': int(n),
            'ratio': (H / max(lvl, 1e-9)).round(4).tolist(),
            'H': H.tolist(),
            'pairs_per_1k_0to6': float(H[0:7].sum() / n * 1e3),
            'pairs_per_1k_7to8': float(H[7:9].sum() / n * 1e3),
            'pairs_per_1k_9to15': float(H[9:16].sum() / n * 1e3),
            'pairs_per_1k_16to29': float(H[16:30].sum() / n * 1e3),
            'pairs_per_1k_8to44_hill': float(H[8:45].sum() / n * 1e3),
            'ratio_9to15': float(H[9:16].mean() / lvl), 'ratio_30to39': float(floor / lvl),
            'ratio_60to90': float(H[60:91].mean() / lvl),
            'bump_excess_pairs': float((H[9:16] - floor).sum()),
            'bump_excess_per_1k': float((H[9:16] - floor).sum() / n * 1e3),
            'bump_share_of_hill_violations': float((H[9:16] - floor).sum() / max(H[8:45].sum(), 1))}


def cleaned(sort):
    st = np.load(sort / 'spike_times.npy').ravel()
    cl = np.load(sort / 'spike_clusters.npy').ravel()
    c = np.load(sort / 'cleaning.npz')
    rm = np.zeros(st.size, bool)
    rm[c['spike_order']] = c['removed_event'] | c['removed_duplicate']
    return st, cl, rm


def main():
    res = {}
    sets = {}
    for ex, base, ks4, h8, ks25 in [
            ('kobe', KOBE, [f'hybrid_default_seed{i}' for i in range(1, 6)],
             ['search/198_H8_cds10_s1/sort', 'search/208_H8_cds10_s2/sort', 'search/218_H8_cds10_s3/sort'],
             [f'ks25_standard_r{i}' for i in (1, 2, 3)] + [f'ks25_standard_low_r{i}' for i in (1, 2, 3)]),
            ('mec', MEC, [f'default_s{i}' for i in (1, 2, 3)], [f'H8_s{i}' for i in (1, 2, 3)],
             [f'ks25_standard_r{i}' for i in (1, 2, 3)] + [f'ks25_standard_low_r{i}' for i in (1, 2, 3)])]:
        for name in ks4 + h8 + ks25:
            sets[f'{ex}/{name}'] = base / name
        for v in ('default', 'H8'):
            sets[f'{ex}/scored_{v}_s1'] = EXP / f'autocurate/sorts/{ex}_{v}_s1'
        sets[f'{ex}/ks25_traced_s1'] = EXP / f'ks25_traced/{ex}_s1'
    for key, sort in sets.items():
        if not sort.exists():
            print('missing', sort)
            continue
        st, cl, rm = cleaned(sort)
        r = {'final': summary(hist(st, cl), st.size),
             'final_cleaned': summary(hist(st[~rm], cl[~rm]), int((~rm).sum()))}
        if (sort / 'full_st.npy').exists():
            f = np.load(sort / 'full_st.npy')
            fc = np.load(sort / 'full_clu.npy').ravel()
            r['before_ks4_dedup'] = summary(hist(f[:, 0], fc), f.shape[0])
        if (sort / 'stages.npz').exists():
            s = np.load(sort / 'stages.npz')
            r['ks25_extracted'] = summary(hist(s['extracted_t'], s['extracted_clu']), s['extracted_t'].size)
        res[key] = r
        print(key, {k: (round(v['ratio_9to15'], 3), round(v['ratio_30to39'], 3),
                        round(v['pairs_per_1k_0to6'], 2), round(v['bump_excess_per_1k'], 2),
                        round(v['bump_share_of_hill_violations'], 3)) for k, v in r.items()}, flush=True)
    OUT.write_text(json.dumps(res))


if __name__ == '__main__':
    main()
