"""Q: does a directional score rule identify true duplicates without a tight lag threshold?
Rule: within-unit pair, lag 7..L samples (L = 21, 0.7 ms), LATER member's score < r x EARLIER member's.
(True duplicates in the units raw_classify.py found one-spike: later lower 97-99%, ratio ~0.24-0.38.)
Per sort (pre-dedup KS4): flagged pairs in the band vs the same rule in the control band 25-45 scaled
to the band's width (expected false flags if the rule were lag-independent), so P(dup | flagged) and
excess flagged. Then a raw check on a random sample of flagged pairs across ALL units: fraction judged
one-spike (raw_classify's test) vs unflagged band pairs and control pairs of the same units.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import AUTOC, load_ks4_full, pairs_within  # noqa
from raw_classify import BIN, NCH, PAD, PRE, snippets, trough  # noqa

HERE = Path(__file__).parent
rng = np.random.default_rng(1)


def raw_test(site, name, tt, cc, pairs_first, pairs_lag, nmax=300):
    """Fraction of pairs judged one-spike, per pair using its unit's isolated-spike reference."""
    mm = np.memmap(BIN[site], np.int16, 'r').reshape(-1, NCH)
    W = np.load(AUTOC / name / 'templates.npy'); cmap = np.load(AUTOC / name / 'channel_map.npy').ravel()
    sel = rng.choice(pairs_first.size, min(nmax, pairs_first.size), replace=False)
    ones = []
    cache = {}
    for i in sel:
        c = cc[pairs_first[i]]; t1 = tt[pairs_first[i]]; L = pairs_lag[i]
        if not (PAD + 200 < t1 < mm.shape[0] - PAD - 200):
            continue
        if c not in cache:
            tc = np.sort(tt[cc == c])
            ch = cmap[np.argmax(np.abs(W[c]).max(0))]
            dp = np.r_[10 ** 9, np.diff(tc)]; dn = np.r_[np.diff(tc), 10 ** 9]
            iso = np.where((dp > 90) & (dn > 90) & (tc > PAD + 200) & (tc < mm.shape[0] - PAD - 200))[0]
            iso = rng.choice(iso, min(60, iso.size), replace=False)
            Si = snippets(mm, tc[iso], ch); w = Si.mean(0)
            k0 = PRE - 10 + np.argmin(w[PRE - 10:PRE + 10])
            cache[c] = (ch, Si, k0, w[k0])
        ch, Si, k0, a0 = cache[c]
        S = snippets(mm, [t1], ch)
        tr2 = trough(S, np.array([k0 + L])) [0] / a0
        ref = trough(Si, np.full(Si.shape[0], k0 + L)) / a0
        ones.append(tr2 < np.median(ref) + 0.5)
    return np.mean(ones), len(ones)


if __name__ == '__main__':
    for site, v in (('kobe', 'default'), ('kobe', 'H8'), ('mec', 'default'), ('mec', 'H8')):
        name = f'{site}_{v}_s1'
        d = load_ks4_full(name)
        t, clu, sc = d['t'], d['clu'], d['score']
        I, J = pairs_within(t, clu, 45)
        lag = t[J] - t[I]
        q = sc[J] / sc[I]
        band = (lag >= 7) & (lag <= 21); ctrl = (lag >= 25) & (lag <= 45)
        f = 15 / 21
        print(f'\n== {name}')
        for r in (0.4, 0.5, 0.6):
            fl = q < r
            o, e = (band & fl).sum(), f * (ctrl & fl).sum()
            # spikes removed if each flagged band pair's later member is dropped
            nrm = np.unique(J[band & fl]).size
            print(f'  later/earlier score < {r}: flagged band pairs {o:6d}, expected from ctrl {e:7.0f}, '
                  f'P(dup|flag) ~{1 - e / max(o, 1):.2f}; spikes removed {nrm} ({100 * nrm / t.size:.3f}%)')
        fl = q < 0.5
        a = raw_test(site, name, t, clu, I[band & fl], lag[band & fl])
        b = raw_test(site, name, t, clu, I[band & ~fl], lag[band & ~fl])
        c = raw_test(site, name, t, clu, I[ctrl], lag[ctrl])
        print(f'  raw one-spike fraction: flagged band pairs {a[0]:.2f} (n={a[1]}), unflagged band pairs {b[0]:.2f} '
              f'(n={b[1]}), control pairs {c[0]:.2f} (n={c[1]})', flush=True)
