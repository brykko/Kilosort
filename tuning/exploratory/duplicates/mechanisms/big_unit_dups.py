"""The clean, large KS4 units with a sharp same-template within-unit peak at 7-15 samples (kobe default
units 765 and 750, found by bump_units.py): lag histogram at sample resolution, scores of earlier/later
member, extraction templates, and the template's own temporal autocorrelation (whitened, from Wall and
wPCA) - does the duplicate lag sit where the template's autocorrelation has a positive side lobe, or
where the subtraction of a slightly mistimed copy leaves the largest positive residual projection?
Also the per-unit duplicate peak for every KS4 unit: units whose 7-15-sample count is >= 5x their own
1.0-1.5 ms level and >= 50 pairs (sharp duplicate peaks), with their size (median score).
"""
from pathlib import Path

import numpy as np

D = Path(r'C:\data\ks4_tuning\exploratory\autocurate\sorts')


def within_pairs(t, c, maxlag):
    o = np.lexsort((t, c))
    ts, cs = t[o], c[o]
    I, J = [], []
    for k in range(1, 400):
        a, b = np.arange(ts.size - k), np.arange(k, ts.size)
        m = (cs[a] == cs[b]) & (ts[b] - ts[a] <= maxlag)
        if not m.any():
            break
        I.append(a[m]); J.append(b[m])
    return o[np.concatenate(I)], o[np.concatenate(J)]


def acf_and_shift_residual(Wpc, wPCA):
    """Wpc: (Nchan, n_pcs) template in PC space; returns normalised autocorrelation over lags -30..30
    and, for a copy shifted by 0.5 sample, the positive residual projection after a least-squares
    subtraction at lag 0, as a fraction of the original peak projection."""
    w = Wpc @ wPCA                          # (Nchan, nt)
    nt = w.shape[1]
    lags = np.arange(-30, 31)
    full = np.array([np.sum(w[:, max(0, -L):nt - max(0, L)] * w[:, max(0, L):nt - max(0, -L)]) for L in lags])
    acf = full / full[30]
    # half-sample shifted copy via Fourier interpolation
    f = np.fft.rfft(w, n=4 * nt, axis=1)
    k = np.fft.rfftfreq(4 * nt)
    ws = np.fft.irfft(f * np.exp(-2j * np.pi * k * 0.5), n=4 * nt, axis=1)[:, :nt]
    proj = lambda x, L: np.sum(w[:, max(0, -L):nt - max(0, L)] * x[:, max(0, L):nt - max(0, -L)])
    a = proj(ws, 0) / full[30]
    res = np.array([proj(ws, L) - a * full[30] * acf[i] for i, L in enumerate(lags)]) / full[30]
    return lags, acf, res


if __name__ == '__main__':
    for name, units in [('kobe_default_s1', [765, 750]), ('kobe_H8_s1', [])]:
        st = np.load(D / name / 'full_st.npy')
        clu = np.load(D / name / 'full_clu.npy').astype(np.int64)
        Wall = np.load(D / name / 'Wall.npy')
        ops = np.load(D / name / 'ops.npy', allow_pickle=True).item()
        wPCA = np.asarray(ops['wPCA'])
        t, tm, sc = st[:, 0].astype(np.int64), st[:, 1].astype(np.int64), st[:, 2]
        i, j = within_pairs(t, clu, 60)
        lag = t[j] - t[i]
        # sharp duplicate peaks in any unit
        nu = clu.max() + 1
        b7 = np.bincount(clu[i[(lag >= 7) & (lag <= 15)]], minlength=nu)
        f31 = np.bincount(clu[i[(lag >= 31) & (lag <= 45)]], minlength=nu) * 9 / 15
        med = np.array([np.median(sc[clu == u]) if np.any(clu == u) else 0 for u in range(nu)])
        sharp = np.where((b7 >= 5 * np.maximum(f31, 1)) & (b7 >= 50))[0]
        print(f"{name}: units with a sharp 7-15 peak (>=5x own 1-1.5 ms level, >=50 pairs): {sharp.size};"
              f" their pairs {b7[sharp].sum()} = {b7[sharp].sum()/t.size*1e3:.2f}/1000 spikes;"
              f" median scores {np.round(med[sharp],1).tolist()}")
        for u in list(units) + [x for x in sharp if x not in units][:4]:
            m = clu[i] == u
            h = np.bincount(lag[m], minlength=61)
            first, second = i[m], j[m]
            mm = (lag[m] >= 7) & (lag[m] <= 15)
            print(f"  unit {u}: n {np.sum(clu==u)}, templates {np.unique(tm[clu==u], return_counts=True)}")
            print(f"    lag hist 0..30: {h[:31].tolist()}")
            r = sc[second[mm]] / sc[first[mm]]
            print(f"    7-15 pairs: score earlier med {np.median(sc[first[mm]]):.1f}, later med {np.median(sc[second[mm]]):.1f},"
                  f" later/earlier p10/50/90 {np.percentile(r,[10,50,90]).round(2).tolist()}, unit median score {med[u]:.1f};"
                  f" same template {np.mean(tm[first[mm]]==tm[second[mm]]):.2f}")
            # pairs vs spike score: fraction of the unit's spikes in a 7-15 pair, by score quintile
            su = np.where(clu == u)[0]
            inpair = np.zeros(t.size, bool)
            inpair[first[mm]] = True
            inpair[second[mm]] = True
            q = np.percentile(sc[su], [0, 20, 40, 60, 80, 100])
            fr = [np.mean(inpair[su[(sc[su] >= q[k]) & (sc[su] <= q[k + 1])]]) for k in range(5)]
            print(f"    frac of spikes in a 7-15 pair by score quintile: {np.round(fr,3).tolist()} (edges {np.round(q,1).tolist()})")
            ut, uc = np.unique(tm[clu == u], return_counts=True)
            w = Wall[u]
            lags, acf, res = acf_and_shift_residual(w, wPCA)
            pk = lags[(lags > 3)][np.argmax(acf[lags > 3])]
            print(f"    template ACF at lags 0..20: {np.round(acf[30:51],2).tolist()}")
            print(f"    first positive ACF side lobe after lag 3: lag {pk}, value {acf[30+pk]:.2f}")
            print(f"    residual projection of a 0.5-sample-shifted copy after LS subtraction, lags -15..15: "
                  f"{np.round(res[15:46],3).tolist()}")
