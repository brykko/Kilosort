"""Which KS4 units carry the within-unit bump at 7-15 samples (0.23-0.5 ms)? Per unit, bump = pairs at
7-15 samples minus the unit's own 31-45-sample (1.0-1.5 ms) level scaled to 9 samples. Reports how
concentrated the bump is, and for the top units: spikes, rate, number of extraction templates (and the
share of the largest), median score, the 1.0-1.5 ms floor relative to 5-20 ms, and the share of bump
pairs whose members come from different extraction templates. Uses full_st (before KS4's removal).
Same for KS2.5 (final) with no templates.
"""
from pathlib import Path

import numpy as np

KS4 = Path(r'C:\data\ks4_tuning\exploratory\autocurate\sorts')
KS25T = Path(r'C:\data\ks4_tuning\exploratory\ks25_traced')
T_S = 1200.0


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


def report(name, t, c, tmpl, score):
    i, j = within_pairs(t, c, 600)
    lag = t[j] - t[i]
    nu = c.max() + 1
    b7 = np.bincount(c[i[(lag >= 7) & (lag <= 15)]], minlength=nu).astype(float)
    f31 = np.bincount(c[i[(lag >= 31) & (lag <= 45)]], minlength=nu) * 9 / 15
    base = np.bincount(c[i[lag >= 150]], minlength=nu) * 9 / 451
    bump = b7 - f31
    n = np.bincount(c, minlength=nu)
    order = np.argsort(bump)[::-1]
    pos = bump[bump > 0].sum()
    cs = np.cumsum(bump[order])
    print(f"{name}: total bump {bump.sum():.0f} pairs ({bump.sum()/t.size*1e3:.2f} per 1000 spikes); "
          f"positive part {pos:.0f}; top10 {cs[9]/bump.sum():.2f}, top50 {cs[49]/bump.sum():.2f} of the total; "
          f"units >= half: {np.searchsorted(cs, bump.sum()/2)+1} of {np.sum(n>0)}")
    if tmpl is not None:
        diff = tmpl[i] != tmpl[j]
        m7 = (lag >= 7) & (lag <= 15)
        m31 = (lag >= 31) & (lag <= 45)
    for u in order[:12]:
        s = f"  unit {u:5d}: n {n[u]:7d} ({n[u]/T_S:6.1f} Hz) bump {bump[u]:6.0f} ({bump[u]/n[u]*1e3:5.1f}/1000 sp)" \
            f" 7-15/5-20ms {b7[u]/max(base[u],1):.2f} floor(1-1.5ms)/5-20ms {f31[u]/max(base[u],1):.2f}"
        mu = c == u
        if score is not None:
            s += f" med score {np.median(score[mu]):6.1f}"
        if tmpl is not None:
            tu, tc = np.unique(tmpl[mu], return_counts=True)
            mi = c[i] == u
            s += f" | templates {tu.size} (top share {tc.max()/tc.sum():.2f})" \
                 f" diff-tmpl share: 7-15 {diff[mi & m7].mean():.2f} vs 1-1.5ms {diff[mi & m31].mean():.2f}"
        print(s)
    return bump, n


if __name__ == '__main__':
    for name in ['kobe_H8_s1', 'kobe_default_s1', 'mec_H8_s1']:
        st = np.load(KS4 / name / 'full_st.npy')
        report('KS4 ' + name, st[:, 0].astype(np.int64), np.load(KS4 / name / 'full_clu.npy').astype(np.int64),
               st[:, 1].astype(np.int64), st[:, 2])
    for ex in ['kobe_s1', 'mec_s1']:
        z = np.load(KS25T / ex / 'stages.npz')
        report('KS2.5 extracted ' + ex, z['extracted_t'].astype(np.int64), z['extracted_clu'].astype(np.int64),
               None, z['extracted_score'].astype(float))
        report('KS2.5 final ' + ex, z['merge3_t'].astype(np.int64), z['merge3_clu'].astype(np.int64),
               None, z['merge3_score'].astype(float))
