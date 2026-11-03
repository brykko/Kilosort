"""Post-hoc on raw_pairs_*.json: are bump pairs two spikes also in well-isolated units? Splits the raw
test's events by the unit's Hill contamination (tau_c 0.25, tau_r 1.5 ms, all spikes of the sort)."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = {'ks4_H8_kobe': 'C:/data/ks4_tuning/exploratory/autocurate/sorts/kobe_H8_s1',
        'ks4_def_kobe': 'C:/data/ks4_tuning/exploratory/autocurate/sorts/kobe_default_s1',
        'ks4_H8_mec': 'C:/data/ks4_tuning/exploratory/autocurate/sorts/mec_H8_s1',
        'ks25_kobe': 'C:/data/ks4_tuning/26648_kobe_2020-08-21/runs/ks25_standard_r1'}
for name, sort in RUNS.items():
    rows = json.loads((HERE / f'raw_pairs_{name}.json').read_text())['rows']
    t = np.load(Path(sort) / 'spike_times.npy').ravel() / 30000
    c = np.load(Path(sort) / 'spike_clusters.npy').ravel()
    cont = {}
    for u in {r['unit'] for r in rows}:
        s = np.sort(t[c == u]); n = s.size
        v = np.sum(np.searchsorted(s, s + 1.5e-3, 'left') - np.searchsorted(s, s + 0.25e-3, 'right'))
        a = v * 1200 / (2 * 1.25e-3 * n * n)
        cont[u] = 1 - np.sqrt(1 - 2 * a) if a <= 0.5 else 1.0
    for lab, f in (('c<=0.1', lambda x: x <= 0.1), ('c>0.1', lambda x: x > 0.1)):
        out = []
        for cls in ('bump', 'ctrl', 'fake'):
            a2 = np.array([r['a2'] for r in rows if r['cls'] == cls and f(cont[r['unit']])])
            out.append(f"{cls} n={a2.size} P(a2>0.5)={np.mean(a2 > 0.5) if a2.size else np.nan:.2f}")
        nu = sum(f(v) for v in cont.values())
        print(name, lab, f'units={nu}', ' | '.join(out))
