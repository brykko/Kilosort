"""Q: what fraction of KS4's within-unit 0.3-0.5 ms excess is true duplication (one physical spike detected
twice) vs two physical spikes (a second neuron, or a real near-synchronous spike)? A waveform-based
pair test that needs no lag threshold.

For the 40 units with the largest excess E (per_unit.py) in each pre-dedup KS4 sort: raw data (CAR,
300 Hz high-pass), unit's best channel. Pair test: trough at the second member's time (min within +-3
samples), divided by the unit's isolated-spike trough a0. The empirical one-spike reference is the same
measure taken at the same lag after ISOLATED spikes (noise included); a pair is 'one spike' if its
trough2 is below that reference's median + 0.5. Applied to band pairs (lags 9-15) and control pairs
(25-45, two spikes by construction, giving the false 'one spike' rate). Duplicates in the band are
estimated as (f1_band - f1_ctrl) / (1 - f1_ctrl) * n_band_pairs, compared with the unit's excess E.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, str(Path(__file__).parent))
from common import AUTOC  # noqa

HERE = Path(__file__).parent
BIN = {'kobe': r'C:/data/ks4_tuning/26648_kobe_2020-08-21/search/hybrid/ap_1668-2868s_hybrid.bin',
       'mec': r'C:/data/ks4_tuning/25843_roger_2019-09-11/search/ap_13575-14775s.bin'}
NCH, PAD, PRE, POST = 385, 200, 30, 60
SOS = butter(3, 300, 'highpass', fs=30000, output='sos')
NB, NI = 120, 150


def snippets(mm, times, ch):
    out = []
    for t in times:
        a = mm[t - PRE - PAD: t + POST + PAD].astype(np.float32)
        a = a[:, ch] - np.median(a[:, :384], 1)
        out.append(sosfiltfilt(SOS, a)[PAD:PAD + PRE + POST])
    return np.array(out)


def trough(S, k):
    return np.array([S[i, k[i] - 3:k[i] + 4].min() for i in range(S.shape[0])])


def main(site, v, nunits=40, seed=0):
    rng = np.random.default_rng(seed)
    name = f'{site}_{v}_s1'
    s = AUTOC / name
    mm = np.memmap(BIN[site], np.int16, 'r').reshape(-1, NCH)
    t = np.load(s / 'spike_times.npy').ravel().astype(np.int64)
    clu = np.load(s / 'spike_clusters.npy').ravel().astype(np.int64)
    W = np.load(s / 'templates.npy'); cmap = np.load(s / 'channel_map.npy').ravel()
    pu = np.load(HERE / f'per_unit_{name}.npz')
    order = np.argsort(-pu['E'])[:nunits]
    rows = []
    for c, E in zip(pu['ids'][order], pu['E'][order]):
        tc = np.sort(t[clu == c])
        ch = cmap[np.argmax(np.abs(W[c]).max(0))]
        ok = (tc > PAD + 100) & (tc < mm.shape[0] - PAD - 100)
        dp = np.r_[10 ** 9, np.diff(tc)]; dn = np.r_[np.diff(tc), 10 ** 9]; dn2 = np.r_[dn[1:], 10 ** 9]
        iso = np.where(ok & (dp > 90) & (dn > 90))[0]
        iso = rng.choice(iso, min(NI, iso.size), replace=False)
        Si = snippets(mm, tc[iso], ch)
        w = Si.mean(0)
        k0 = PRE - 10 + np.argmin(w[PRE - 10:PRE + 10]); a0 = w[k0]
        out = [int(c), tc.size, a0, E]
        for lo, hi in ((9, 15), (25, 45)):
            allf = np.where(ok & (dn >= lo) & (dn <= hi) & (dp > 90) & (dn2 > 60))[0]
            nall = int(np.sum((dn >= lo) & (dn <= hi)))
            f = rng.choice(allf, min(NB, allf.size), replace=False) if allf.size else allf
            if f.size < 10:
                out += [nall, np.nan]; continue
            S = snippets(mm, tc[f], ch)
            lag = dn[f]
            tr2 = trough(S, k0 + lag) / a0
            Lr = rng.choice(lag, Si.shape[0])                       # one-spike reference at matched lags
            ref = trough(Si, k0 + Lr) / a0
            thr = np.median(ref) + 0.5
            out += [nall, np.mean(tr2 < thr)]
        rows.append(out)
    R = np.array(rows, float)
    c, n, a0, E, nb, f1b, nc, f1c = R.T
    fc = np.nanmedian(f1c)
    dups = np.clip((f1b - fc) / (1 - fc), 0, 1) * nb
    big = a0 < np.median(a0)   # a0 is negative: more negative = bigger
    print(f'\n== {name}: top {len(R)} excess units (hold {E.sum() / pu["E"].sum():.2f} of the net excess)')
    print(f'  false one-spike rate on control pairs (median over units) {fc:.3f}')
    print(f'  band pairs judged one-spike: median over units {np.nanmedian(f1b):.2f}; '
          f'estimated true duplicates {np.nansum(dups):.0f} vs net excess E {E.sum():.0f} '
          f'-> duplicate share of the excess ~{np.nansum(dups) / E.sum():.2f}')
    for lab, m in (('larger-amplitude half', big), ('smaller-amplitude half', ~big)):
        print(f'  {lab}: a0 median {np.median(a0[m]):.0f}, rate median {np.median(n[m]) / 1200:.1f} Hz, '
              f'one-spike frac median {np.nanmedian(f1b[m]):.2f}, dups/E {np.nansum(dups[m]) / E[m].sum():.2f}')
    print('  per unit: id n a0 E band_pairs f1_band f1_ctrl')
    for r in R[:12]:
        print('   ', ' '.join(f'{x:.2f}' if isinstance(x, float) and abs(x) < 10 else f'{x:.0f}' for x in r))
    np.save(HERE / f'raw_classify_{name}.npy', R)


if __name__ == '__main__':
    for site, v in (('kobe', 'default'), ('kobe', 'H8'), ('mec', 'default'), ('mec', 'H8')):
        main(site, v)
