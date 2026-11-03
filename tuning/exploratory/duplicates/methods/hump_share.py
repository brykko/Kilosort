"""Share of the within-unit near-coincidence hump that KS4's 7-sample window removes.

Question: with the floor estimated as the mean obs/exp ratio at lags 31-45 samples (1-1.5 ms), how
much of the excess (ratio above floor, lags 0-30) lies at lags 0-6 (removed by KS4) vs 7-30 (kept)?
Reads within_unit_lags.json (run within_unit_lags.py first). Excess in pairs per 1k spikes.
"""
import json
from pathlib import Path
import numpy as np
r = json.loads((Path(__file__).parent / 'within_unit_lags.json').read_text())
for k in [k for k in r if 'before_dedup' in k or 'traced' in k or k.endswith('_s1|final') or k.endswith('_r1|final')]:
    v = r[k]; H = np.array(v['H_0_45'], float); base = v['base_per_lag']; n = v['n_spikes']
    ratio = H / base; floor = ratio[31:46].mean()
    ex = np.clip(ratio - floor, 0, None) * base * 1e3 / n
    print(f"{k:45s} floor {floor:.3f}  excess/1k: lags0-6 {ex[:7].sum():.2f}  lags7-30 {ex[7:31].sum():.2f}  "
          f"share removable at <=6: {ex[:7].sum() / max(ex[:31].sum(), 1e-9):.2f}  peak lag {int(np.argmax(ratio[:31]))}")
