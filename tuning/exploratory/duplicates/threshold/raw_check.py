"""Q: are KS4's within-unit pairs at 0.3-0.5 ms two detections of ONE spike (duplicates) or TWO spikes?
In the raw (CAR, 300 Hz high-passed) data on the unit's best channel: for pairs at lags 9-15 samples in
the units with the largest excess, the trough at the second member's time, relative to the unit's
isolated-spike trough a0. One spike predicts trough2 ~ w(lag)/a0 (the single waveform's own value at
that lag, usually small or positive); two spikes predict trough2 ~ 1 + w(lag)/a0. Control pairs at
lags 25-45 (which are two spikes by construction) are shown for reference.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, str(Path(__file__).parent))
from common import AUTOC, pairs_within  # noqa

HERE = Path(__file__).parent
BIN = {'kobe': r'C:/data/ks4_tuning/26648_kobe_2020-08-21/search/hybrid/ap_1668-2868s_hybrid.bin',
       'mec': r'C:/data/ks4_tuning/25843_roger_2019-09-11/search/ap_13575-14775s.bin'}
NCH, PAD, PRE, POST = 385, 300, 30, 60
SOS = butter(3, 300, 'highpass', fs=30000, output='sos')
rng = np.random.default_rng(0)


def snippets(mm, times, ch):
    out = []
    for t in times:
        a = mm[t - PRE - PAD: t + POST + PAD].astype(np.float32)
        a = a - np.median(a[:, :384], 1, keepdims=True)
        x = sosfiltfilt(SOS, a[:, ch], axis=0)[PAD:PAD + PRE + POST]
        out.append(x)
    return np.array(out)


def main(site, v, nunits=6):
    name = f'{site}_{v}_s1'
    s = AUTOC / name
    mm = np.memmap(BIN[site], np.int16, 'r').reshape(-1, NCH)
    t = np.load(s / 'spike_times.npy').ravel().astype(np.int64)
    clu = np.load(s / 'spike_clusters.npy').ravel().astype(np.int64)
    W = np.load(s / 'templates.npy'); cmap = np.load(s / 'channel_map.npy').ravel()
    pu = np.load(HERE / f'per_unit_{name}.npz')
    top = pu['ids'][np.argsort(-pu['E'])[:nunits]]
    print(f'\n== {name}: top {nunits} excess units')
    for c in top:
        tc = np.sort(t[clu == c])
        ch = cmap[np.argmax(np.abs(W[c]).max(0))]
        ok = (tc > PAD + 100) & (tc < mm.shape[0] - PAD - 100)
        d_prev = np.r_[10 ** 9, np.diff(tc)]; d_next = np.r_[np.diff(tc), 10 ** 9]
        iso = np.where(ok & (d_prev > 90) & (d_next > 90))[0]
        iso = rng.choice(iso, min(400, iso.size), replace=False)
        w = snippets(mm, tc[iso], ch).mean(0)
        k0 = PRE - 10 + np.argmin(w[PRE - 10:PRE + 10]); a0 = w[k0]
        res = {}
        for lab, lo, hi in (('dup band 9-15', 9, 15), ('ctrl 25-45', 25, 45)):
            first = np.where(ok & (d_next >= lo) & (d_next <= hi) & (d_prev > 90))[0]
            first = first[(np.r_[d_next[1:], 10 ** 9][first] > 60)]      # no third spike right after
            first = rng.choice(first, min(200, first.size), replace=False)
            if first.size < 10:
                res[lab] = None; continue
            S = snippets(mm, tc[first], ch)
            lag = d_next[first]
            tr1 = np.array([S[i, k0 - 3:k0 + 4].min() for i in range(len(first))]) / a0
            k2 = k0 + lag
            tr2 = np.array([S[i, k2[i] - 3:k2[i] + 4].min() if k2[i] + 4 <= S.shape[1] else np.nan
                            for i in range(len(first))]) / a0
            pred1 = np.array([w[k - 3:k + 4].min() if k + 4 <= w.size else np.nan for k in k2]) / a0
            res[lab] = (len(first), np.median(tr1), np.nanmedian(tr2), np.nanmedian(pred1), np.nanmedian(1 + pred1),
                        np.nanmean(tr2 > 0.5 + 0.5 * pred1))
        print(f'  unit {c} (n={tc.size}, ch {ch}, a0={a0:.0f} uV-ish, excess E={pu["E"][pu["ids"] == c][0]:.0f})')
        for lab, r in res.items():
            if r is None:
                print(f'    {lab}: too few'); continue
            print(f'    {lab:14s}: n={r[0]:3d} trough1/a0={r[1]:.2f} trough2/a0={r[2]:.2f}  '
                  f'one-spike pred={r[3]:.2f} two-spike pred={r[4]:.2f}  frac closer to two-spike={r[5]:.2f}')


if __name__ == '__main__':
    main('kobe', 'default')
    main('mec', 'default')
    main('kobe', 'H8', 4)
