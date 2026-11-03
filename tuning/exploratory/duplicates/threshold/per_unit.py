"""Q: is KS4's 0.23-0.7 ms within-unit excess spread over all units (so one global lag threshold fits) or
concentrated in a few, and what predicts it? Per unit (pre-dedup KS4 sorts): pairs at lags 7-20 samples
vs the unit's own level at 25-45 samples (0.83-1.5 ms), excess E = n(7-20) - 14 * level(25-45)/21.
Also pair-level features in the 7-20 band vs a control band 25-45: same extraction template, score
min/max, amplitude min/max, whether the later member has the higher score; and the template
autocorrelation width as a predictor of the excess lag (mechanism check). Writes per_unit_<sort>.npz.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import AUTOC, load_ks4_full, pairs_within  # noqa

HERE = Path(__file__).parent
DUP, CTRL = (7, 20), (25, 45)


def tmpl_acf_width(name):
    """Per cluster: lag (samples) where the template's autocorrelation (summed over channels) first drops
    below 0.5, and the lag of its most negative value (side lobe)."""
    W = np.load(AUTOC / name / 'templates.npy')     # (n_clu, nt, nchan)
    out = {}
    for c in range(W.shape[0]):
        w = W[c]
        if not np.any(w):
            continue
        nt = w.shape[0]
        acf = np.array([np.sum(w[:nt - k] * w[k:]) for k in range(30)])
        acf = acf / acf[0]
        half = int(np.argmax(acf < 0.5)) if np.any(acf < 0.5) else 30
        out[c] = (half, int(np.argmin(acf)))
    return out


def main(name):
    d = load_ks4_full(name)
    t, clu = d['t'], d['clu']
    I, J = pairs_within(t, clu, 45)
    lag = t[J] - t[I]
    u = clu[I]
    ids, ns = np.unique(clu, return_counts=True)
    ui = np.searchsorted(ids, u)
    inD = (lag >= DUP[0]) & (lag <= DUP[1])
    inC = (lag >= CTRL[0]) & (lag <= CTRL[1])
    nD = np.bincount(ui[inD], minlength=ids.size)
    nC = np.bincount(ui[inC], minlength=ids.size)
    lenD, lenC = DUP[1] - DUP[0] + 1, CTRL[1] - CTRL[0] + 1
    E = nD - nC * lenD / lenC
    order = np.argsort(-E)
    tot = E[E > 0].sum()
    cum = np.cumsum(E[order]) / E.sum()
    print(f'\n== {name}: {ids.size} units, {t.size} spikes; pooled excess 7-20 = {E.sum():.0f} pairs '
          f'({1000 * E.sum() / t.size:.2f}/1000 spikes)')
    for k in (5, 10, 20, 50, 100):
        print(f'  top {k:3d} units carry {cum[k - 1]:.2f} of the net excess; '
              f'their spikes are {ns[order[:k]].sum() / t.size:.3f} of all')
    # Poisson test of each unit's excess (one-sided): nD vs expected nC*lenD/lenC (ratio of counts)
    from scipy.stats import binomtest
    p = np.array([binomtest(int(a), int(a + b), lenD / (lenD + lenC), 'greater').pvalue if a + b > 0 else 1
                  for a, b in zip(nD, nC)])
    # Benjamini-Hochberg at 5%
    ps = np.sort(p)
    k = np.where(ps <= 0.05 * np.arange(1, p.size + 1) / p.size)[0]
    thr = ps[k.max()] if k.size else 0
    sig = p <= thr
    print(f'  units with significant excess (BH 5%): {sig.sum()} of {ids.size}; they hold '
          f'{E[sig].sum() / E.sum():.2f} of the net excess and {ns[sig].sum() / t.size:.3f} of spikes')
    # excess per spike vs unit size
    for lo, hi in [(0, 1e3), (1e3, 1e4), (1e4, 5e4), (5e4, 1e9)]:
        m = (ns >= lo) & (ns < hi)
        print(f'  units {int(lo):>6d}-{int(min(hi, 1e9)):>9d} spikes: n={m.sum():4d}  excess/1000 spikes='
              f'{1000 * E[m].sum() / max(ns[m].sum(), 1):6.2f}  ratio nD/exp={nD[m].sum() / max(nC[m].sum() * lenD / lenC, 1):.2f}')
    # pair features: dup band vs control band
    same = d['tmpl'][I] == d['tmpl'][J]
    sr = np.minimum(d['score'][I], d['score'][J]) / np.maximum(d['score'][I], d['score'][J])
    ar = np.minimum(d['amp'][I], d['amp'][J]) / np.maximum(d['amp'][I], d['amp'][J])
    later_big = d['score'][J] > d['score'][I]
    for lab, m in (('dup band 7-20', inD), ('ctrl 25-45', inC), ('sig units, 7-20', inD & sig[ui]),
                   ('sig units, 25-45', inC & sig[ui])):
        print(f'  {lab:18s}: n={m.sum():7d} same tmpl={same[m].mean():.3f} score min/max med={np.median(sr[m]):.3f} '
              f'amp min/max med={np.median(ar[m]):.3f} later higher score={later_big[m].mean():.3f}')
    # feature-conditional excess: within sig units, per same/diff template and score-ratio tercile
    print('  feature-conditional ratio obs(7-20)/exp(from 25-45), all units:')
    for fl, f in (('same tmpl', same), ('diff tmpl', ~same), ('score ratio<0.5', sr < 0.5),
                  ('0.5-0.8', (sr >= 0.5) & (sr < 0.8)), ('>=0.8', sr >= 0.8),
                  ('amp ratio<0.5', ar < 0.5), ('amp 0.5-0.8', (ar >= 0.5) & (ar < 0.8)), ('amp>=0.8', ar >= 0.8)):
        a, b = (inD & f).sum(), (inC & f).sum() * lenD / lenC
        print(f'    {fl:16s} obs={a:7d} exp={b:9.0f} ratio={a / max(b, 1):.2f}  P(dup|band,feature)~{max(0, 1 - b / max(a, 1)):.2f}')
    # mechanism: template ACF half-width vs where the excess sits
    acf = tmpl_acf_width(name)
    hw = np.array([acf.get(int(c), (np.nan, np.nan))[0] for c in ids], float)
    print('  template ACF half-width (samples) quartiles:', np.nanpercentile(hw, [25, 50, 75]))
    qs = np.nanpercentile(hw, [33, 67])
    for lab, m in (('narrow', hw <= qs[0]), ('mid', (hw > qs[0]) & (hw <= qs[1])), ('wide', hw > qs[1])):
        mm = m[ui]
        h = np.bincount(lag[mm], minlength=46).astype(float)
        lev = h[25:46].mean()
        r = h / max(lev, 1)
        pk = 7 + int(np.argmax(h[7:21]))
        exc = h[7:25] - lev
        cen = np.sum(np.arange(7, 25) * np.clip(exc, 0, None)) / max(np.clip(exc, 0, None).sum(), 1)
        print(f'    {lab:6s} (hw {np.nanmin(hw[m]):.0f}-{np.nanmax(hw[m]):.0f}): units={m.sum()} peak lag={pk} '
              f'centroid of excess={cen:.1f}  ratio at 7,10,13,16,20,24: ' +
              ' '.join(f'{r[k]:.2f}' for k in (7, 10, 13, 16, 20, 24)))
    np.savez(HERE / f'per_unit_{name}.npz', ids=ids, ns=ns, nD=nD, nC=nC, E=E, p=p, sig=sig, hw=hw)


if __name__ == '__main__':
    for n in sys.argv[1:] or ['kobe_default_s1', 'kobe_H8_s1', 'mec_default_s1', 'mec_H8_s1']:
        main(n)
