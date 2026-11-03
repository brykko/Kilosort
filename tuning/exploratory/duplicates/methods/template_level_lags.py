"""Like-with-like at the matching stage: within-TEMPLATE short-lag pairs, KS4 vs KS2.5, before any curation.

Question: KS2.5's 'extracted' stage labels spikes by the template that matched them (no reclustering),
while KS4's final units are re-clustered. To compare the template matchers themselves, label KS4's
spikes (full_st, before KS4's dedup, after merging_function) by their extraction template and compute
the same lag profile as within_unit_lags.py. Also KS4 by final unit for reference.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from within_unit_lags import summarise, BANDS  # noqa: E402

AUTO = Path(r'C:\data\ks4_tuning\exploratory\autocurate\sorts')
TRACED = Path(r'C:\data\ks4_tuning\exploratory\ks25_traced')

rows = {}
for name in ('kobe_default_s1', 'kobe_H8_s1', 'mec_default_s1', 'mec_H8_s1'):
    fs = np.load(AUTO / name / 'full_st.npy')
    rows[f'ks4 {name} by extraction template'] = summarise(fs[:, 0], fs[:, 1])
    rows[f'ks4 {name} by final unit'] = summarise(fs[:, 0], np.load(AUTO / name / 'full_clu.npy'))
for ex in ('kobe_s1', 'mec_s1'):
    z = np.load(TRACED / ex / 'stages.npz')
    rows[f'ks25 {ex} extracted (by template)'] = summarise(z['extracted_t'], z['extracted_clu'])
keys = list(BANDS)
print('name'.ljust(46), 'nsp(M)', *[k.split(' ')[0].rjust(16) for k in keys])
for name, r in rows.items():
    print(name.ljust(46), f"{r['n_spikes']/1e6:6.2f}",
          *[f"{r[k]['per_1k_spikes']:7.2f}/{r[k]['obs_over_exp']:6.2f}x".rjust(16) for k in keys])
