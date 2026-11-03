"""Two follow-ups on the cached histograms (dup_lags.py).

1. Lag distribution of within-unit duplicates at sample resolution, from sorts with NO within-unit removal
   (KS4 seed 1 'pre'; KS2.5 traced 'extracted' and final). Pooled excess over the 1.0-1.5 ms floor per lag;
   the lag below which 50/80/90/95% of the excess (lags 0-1 ms) lies; the largest lag where pooled obs >= 2x
   floor (where most pairs in that bin are duplicates). Gives an empirical duplicate window.
2. Cross-unit excess specific to nearby units: pairs at a distance d have a broad short-lag excess that is
   also present 300-600 um apart (population synchrony or shared artefact, not duplication). Near-specific
   excess = H_d(L) - base_d * ratio_far(L), summed over |lag| < 1 ms, per 1000 spikes; by distance and,
   for pairs < 60 um, by template similarity. raw and clean (§6.3) for replicate 1.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from dup_common import CACHE, DIST_EDGES, SIM_EDGES  # noqa: E402

ND, NS = DIST_EDGES.size - 1, SIM_EDGES.size - 1


def lagdist(Hw, n, a0=0):
    H = Hw[n >= 100].astype(float)
    F = H[:, 30:45].mean(1).sum(); E = H[:, 150:601].mean(1).sum()
    h = H[:, :30].sum(0)
    exc = np.clip(h - F, 0, None); exc[:a0] = 0
    c = np.cumsum(exc) / exc.sum()
    q = {p: int(np.searchsorted(c, p)) for p in (0.5, 0.8, 0.9, 0.95)}
    two = np.where(h >= 2 * F)[0]
    return h / E, F / E, q, (int(two.max()) if two.size else None)


if __name__ == '__main__':
    print('1. within-unit duplicate lag distribution (lags in samples; 30 samples = 1 ms)')
    for ex in ('kobe', 'mec'):
        for name, a0 in ((f'{ex}_ks4_default_1_pre', 0), (f'{ex}_ks4_H8_1_pre', 0), (f'{ex}_ks4_default_1_raw', 7),
                         (f'{ex}_ks25traced_1_extracted', 0), (f'{ex}_ks25traced_1_merge3', 0)):
            z = np.load(CACHE / f'{name}.npz')
            r, f, q, two = lagdist(z['Hw'], z['n'], a0)
            print(f'{name:30s} floor {f:.3f}; excess quantiles (samples) {q}; last lag with obs >= 2x floor: {two}')
            print('    obs/exp lag 0..29: ' + ' '.join(f'{x:.2f}' for x in r))

    print('\n2. cross-unit near-specific excess (|lag| < 1 ms) beyond the 300-600 um shape, per 1000 spikes; '
          'and its ratio at .25-.5 ms (obs / far-shape expectation)')
    for ex in ('kobe', 'mec'):
        for g in ('ks4_default', 'ks4_H8', 'ks25_std', 'ks25_low'):
            for v in ('raw', 'clean'):
                z = np.load(CACHE / f'{ex}_{g}_1_{v}.npz')
                H = z['Hx'].reshape(ND, NS, -1).astype(float); N = z['n'].sum()
                base = H[..., 150:301].mean(-1)
                Hd, bd = H.sum(1), base.sum(1)
                rf = Hd[6] / bd[6]                       # 300-600 um profile per |lag|
                rf[0] = rf[0]                            # lag 0 counted once in both, ratio unaffected
                spec_d = [(Hd[d, :30] - bd[d] * rf[:30]).sum() / N * 1000 for d in range(6)]
                rat_d = [Hd[d, 8:15].sum() / (bd[d] * rf[8:15]).sum() for d in range(6)]
                Hn, bn = H[:3].sum(0), base[:3].sum(0)
                spec_s = [(Hn[s, :30] - bn[s] * rf[:30]).sum() / N * 1000 for s in range(NS)]
                rat_s = [Hn[s, 8:15].sum() / max((bn[s] * rf[8:15]).sum(), 1e-9) for s in range(NS)]
                print(f'{ex:5s}{g:12s}{v:6s} by distance (0-20,20-40,40-60,60-100,100-150,150-300 um): '
                      + ' '.join(f'{e:6.2f}({r:.2f})' for e, r in zip(spec_d, rat_d))
                      + f' | total<150um {sum(spec_d[:5]):6.2f} | <60um by sim (<.3,.3-.5,.5-.7,.7-.9,>.9): '
                      + ' '.join(f'{e:5.2f}({r:.2f})' for e, r in zip(spec_s, rat_s)))
