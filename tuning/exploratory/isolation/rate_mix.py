"""How much of the difference in % misassigned spikes between sorters comes from their firing-rate mix.

% misassigned = sum_i n_i c_i / sum_i n_i (all units of the cleaned sort, hybrid copies out), i.e. the
spike-weighted mean contamination. Units are binned by firing rate; then
    % misassigned = sum_b w_b cbar_b,
w_b the share of spikes in rate bin b and cbar_b the spike-weighted contamination within it. A difference
between two sorters splits exactly (Kitagawa) into
    rate mix:     sum_b (w2_b - w1_b) (cbar1_b + cbar2_b) / 2   (same contamination at each rate, different rates)
    within rate:  sum_b (cbar2_b - cbar1_b) (w1_b + w2_b) / 2   (same rates, different contamination at each rate)
Seeds of a sorter are pooled.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).parent))
import hybrid  # noqa: E402
from cleaning import load_clean_units  # noqa: E402
from isolation import contamination, violations  # noqa: E402

TAU = 1.5e-3
EDGES = [0, 0.5, 1, 2, 5, 10, 20, np.inf]
K = Path('C:/data/ks4_tuning/26648_kobe_2020-08-21')
M = Path('C:/data/ks4_tuning/25843_roger_2019-09-11')
SETS = {
    'kobe': {'default': [K / f'runs/hybrid_default_seed{k}' for k in (1, 2, 3, 4, 5)],
             'ks25': [K / f'runs/ks25_standard_r{k}' for k in (1, 2, 3)],
             'H8': [K / f'runs/search/{r}/sort' for r in ('198_H8_cds10_s1', '208_H8_cds10_s2', '218_H8_cds10_s3')]},
    'mec': {'default': [M / f'runs/default_s{k}' for k in (1, 2, 3)], 'ks25': [M / f'runs/ks25_standard_r{k}' for k in (1, 2, 3)],
            'H8': [M / f'runs/H8_s{k}' for k in (1, 2, 3)]},
}


def units(sort, ex):
    base = K if ex == 'kobe' else M
    by, _, iv = load_clean_units(sort, base / 'search/raw_artefacts_car.npz')
    if ex == 'kobe':
        copies = set(hybrid.recovery(sort, np.load(base / 'search/hybrid/hybrid_truth.npz'))['hybrid_copies'])
        by = {c: t for c, t in by.items() if c not in copies}
    T = 1200 - float(np.sum(iv[:, 1] - iv[:, 0])) if iv.size else 1200.0
    n = np.array([t.size for t in by.values() if t.size])
    c = np.array([contamination(t.size, violations(t, TAU), T, TAU) for t in by.values() if t.size])
    return n, c, T


if __name__ == '__main__':
    for ex, groups in SETS.items():
        data = {}
        for g, sorts in groups.items():
            n, c, rate = [], [], []
            for s in sorts:
                nn, cc, T = units(s, ex)
                n.append(nn); c.append(cc); rate.append(nn / T)
            n, c, rate = np.concatenate(n), np.concatenate(c), np.concatenate(rate)
            b = np.digitize(rate, EDGES) - 1
            w = np.array([n[b == k].sum() for k in range(len(EDGES) - 1)]) / n.sum()
            cb = np.array([(n * c)[b == k].sum() / max(n[b == k].sum(), 1) for k in range(len(EDGES) - 1)])
            data[g] = (w, cb, (n * c).sum() / n.sum(), np.bincount(b, minlength=len(EDGES) - 1) / len(sorts))
        print(f'\n{ex}: by firing-rate band (Hz): share of spikes / % misassigned within band / units per sort')
        print(f"{'':8s}" + ''.join(f'{f"{a}-{bb}":>22s}' for a, bb in zip(EDGES[:-1], EDGES[1:])) + '   overall')
        for g, (w, cb, tot, nu) in data.items():
            print(f'{g:8s}' + ''.join(f'{w[k]:7.1%} /{cb[k]:6.1%} /{nu[k]:5.0f}' for k in range(len(w))) + f'   {tot:6.1%}')
        for g1, g2 in (('default', 'H8'), ('default', 'ks25'), ('H8', 'ks25')):
            (w1, c1, t1, _), (w2, c2, t2, _) = data[g1], data[g2]
            mix = np.sum((w2 - w1) * (c1 + c2) / 2)
            within = np.sum((c2 - c1) * (w1 + w2) / 2)
            d = t2 - t1
            print(f'  {g2} - {g1}: {d:+6.1%} = rate mix {mix:+6.1%} ({mix / d:5.0%}) + within rate {within:+6.1%} ({within / d:5.0%})')
