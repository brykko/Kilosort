"""Q: what does the within-unit lag profile (0-3 ms, sample resolution) look like per sorter, before and
after each sorter's own duplicate handling? Pooled observed/expected, expected from each unit's own
5-20 ms pair level. Writes profiles.npz (counts and expected per sort) and prints a binned table."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import AUTOC, KOBE, MEC, TRACED, load_ks4_full, load_phy, load_traced, lag_profile  # noqa

OUT = Path(__file__).parent / 'profiles.npz'
BINS = [(0, 6), (7, 7), (8, 9), (10, 12), (13, 15), (16, 18), (19, 24), (25, 30), (31, 45), (46, 60), (61, 90)]


def sources():
    for site in ('kobe', 'mec'):
        for v in ('default', 'H8'):
            d = load_ks4_full(f'{site}_{v}_s1')
            yield f'{site} KS4 {v} s1 pre-dedup', d['t'], d['clu']
            yield f'{site} KS4 {v} s1 final', d['t'][d['kept']], d['clu'][d['kept']]
    for s in (2, 3):
        d = load_phy(KOBE / f'runs/hybrid_default_seed{s}')
        yield f'kobe KS4 default s{s} final', d['t'], d['clu']
    for f in ('208_H8_cds10_s2', '218_H8_cds10_s3'):
        d = load_phy(KOBE / f'runs/search/{f}/sort')
        yield f'kobe KS4 H8 {f[-2:]} final', d['t'], d['clu']
    for site, root in (('kobe', KOBE / 'runs'), ('mec', MEC / 'runs')):
        for p in ('standard', 'standard_low'):
            for r in (1, 2, 3):
                d = load_phy(root / f'ks25_{p}_r{r}')
                yield f'{site} KS2.5 {p} r{r} final', d['t'], d['clu']
    for site in ('kobe', 'mec'):
        for stage in ('extracted', 'merge1', 'cut1', 'merge3'):
            d = load_traced(f'{site}_s1', stage)
            yield f'{site} KS2.5 traced {stage}', d['t'], d['clu']


if __name__ == '__main__':
    res = {}
    print('ratio obs/expected by lag bin (samples); excess = pairs per 1000 spikes above the 1.0-1.5 ms level, lags 0-30')
    print(f"{'sort':42s} " + ' '.join(f'{a}-{b}'.rjust(6) for a, b in BINS) + '   exc0-6 exc7-15 exc16-30  N(M)')
    for name, t, clu in sources():
        cnt, lev, *_ = lag_profile(t, clu)
        res[name] = np.concatenate([[lev, t.size], cnt])
        r = [cnt[a:b + 1].sum() / (lev * (b - a + 1)) for a, b in BINS]
        floor = cnt[31:46].sum() / 15
        ex = lambda a, b: 1000 * (cnt[a:b + 1] - floor).sum() / t.size  # noqa
        print(f'{name:42s} ' + ' '.join(f'{x:6.3f}' for x in r) +
              f'   {ex(0, 6):6.2f} {ex(7, 15):6.2f} {ex(16, 30):6.2f}  {t.size / 1e6:.2f}', flush=True)
    np.savez(OUT, **{k.replace(' ', '|'): v for k, v in res.items()})
