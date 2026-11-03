"""Q: can a non-lag feature separate duplicates from contamination pairs? Uses KS4's per-spike position
(spike_positions.npy, kept spikes only) and amplitude: for within-unit pairs in the duplicate band
(lags 8-20 samples) vs the control band (25-45), the distribution of the members' position difference
|dy|, |dx| and amplitude ratio; the feature-conditional excess obs(8-20)/exp(from 25-45) gives
P(dup | band, feature) by bin. Also: is the excess carried by pairs whose second member sits at a
different place (a propagating or second-site detection) or at the same place (a re-detection)?
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import AUTOC, pairs_within  # noqa

HERE = Path(__file__).parent

if __name__ == '__main__':
    for name in ('kobe_default_s1', 'kobe_H8_s1', 'mec_default_s1', 'mec_H8_s1'):
        s = AUTOC / name
        t = np.load(s / 'spike_times.npy').ravel().astype(np.int64)
        clu = np.load(s / 'spike_clusters.npy').ravel().astype(np.int64)
        pos = np.load(s / 'spike_positions.npy')
        amp = np.load(s / 'amplitudes.npy').ravel()
        I, J = pairs_within(t, clu, 45)
        lag = t[J] - t[I]
        dy = np.abs(pos[J, 1] - pos[I, 1]); dx = np.abs(pos[J, 0] - pos[I, 0])
        dr = np.hypot(dx, dy)
        ar = np.minimum(amp[I], amp[J]) / np.maximum(amp[I], amp[J])
        D = (lag >= 8) & (lag <= 20); C = (lag >= 25) & (lag <= 45)
        f = 13 / 21
        # within-unit spread reference: distance between random same-unit spike pairs at 5-20 ms
        print(f'\n== {name}: pairs band {D.sum()} ctrl {C.sum()}; median |dr| band {np.median(dr[D]):.1f} um, '
              f'ctrl {np.median(dr[C]):.1f} um')
        for lab, x, edges in (('|dr| um', dr, [0, 2, 5, 10, 20, 40, 1e9]), ('amp ratio', ar, [0, .5, .7, .85, .95, 1.01])):
            print(f'  {lab}:')
            tot_exc = D.sum() - f * C.sum()
            for a, b in zip(edges[:-1], edges[1:]):
                m = (x >= a) & (x < b)
                o, e = (D & m).sum(), f * (C & m).sum()
                print(f'    [{a:5.2f},{b:7.2f}): band={o:6d} exp={e:8.0f} ratio={o / max(e, 1):5.2f} '
                      f'P(dup)~{max(0, 1 - e / max(o, 1)):.2f}  share of excess={(o - e) / tot_exc:5.2f}  '
                      f'share of band pairs={o / D.sum():.2f}')
