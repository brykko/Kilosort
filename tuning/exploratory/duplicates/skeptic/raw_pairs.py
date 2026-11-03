"""Raw-waveform test: do within-unit spike pairs at 0.3-0.5 ms contain ONE spike in the data or TWO?

For a sort, sample units weighted by their 0.3-0.5 ms (9-15 sample) pair count, then per unit:
- 'bump': same-unit pairs at 9..15 samples;
- 'ctrl': same-unit pairs at 60..90 samples (2-3 ms; both should be real spikes if the unit is good);
- 'fake': isolated spikes of the unit (no same-unit spike within +-3 ms) with a FAKE second detection at
  +12 samples (the null for 'one spike').
Raw data: int16 binary, per-channel mean removed, median across the sort's channels subtracted (CAR),
300 Hz 3rd-order Butterworth high-pass (filtfilt). Unit template = mean over up to 40 isolated spikes,
on the unit's 12 highest-energy channels, samples -30..+50.
Per event, least squares on a window t1-25 .. t2+45:
  model 1: a1 T(t - t1 - d1), d1 in -3..3;  model 2: + a2 T(t - t2 - d2), d2 in -3..3 (joint).
Reported: a1, a2 (template units), dE = (R1 - R2)/|T|^2 (energy the second spike explains, in spikes),
R2/|T|^2, and the spatial check: peak channel of each detection's extraction template (KS4) or
template (KS2.5). Also the mean peak-channel trace of each class aligned on t1 (saved for plotting).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

BIN_KOBE = Path('C:/data/ks4_tuning/26648_kobe_2020-08-21/search/hybrid/ap_1668-2868s_hybrid.bin')
BIN_MEC = Path('C:/data/ks4_tuning/25843_roger_2019-09-11/search/ap_13575-14775s.bin')
EXP = Path('C:/data/ks4_tuning/exploratory')
RUNS_KOBE = Path('C:/data/ks4_tuning/26648_kobe_2020-08-21/runs')
RUNS_MEC = Path('C:/data/ks4_tuning/25843_roger_2019-09-11/runs')
OUTDIR = Path(__file__).resolve().parent
NCH = 385
PAD = 200
TW = np.arange(-30, 51)
NSEL = 12
SOS = butter(3, 300, 'highpass', fs=30000, output='sos')


def load(kind, sort):
    """Spike times, clusters, per-spike template id (for peak channel), templates (for channels), score."""
    sort = Path(sort)
    if (sort / 'full_st.npy').exists():
        f = np.load(sort / 'full_st.npy')
        t = f[:, 0].astype(np.int64)
        c = np.load(sort / 'full_clu.npy').ravel().astype(np.int64)
        tmpl = f[:, 1].astype(np.int64)
        score = f[:, 2]
        keep = np.load(sort / 'kept_spikes.npy')
        t, c, tmpl, score = t[keep], c[keep], tmpl[keep], score[keep]
        xy = np.load(sort / 'spike_positions.npy')   # per-spike positions (KS4), same order as kept
        tpk = None
    else:
        t = np.load(sort / 'spike_times.npy').ravel().astype(np.int64)
        c = np.load(sort / 'spike_clusters.npy').ravel().astype(np.int64)
        tmpl = np.load(sort / 'spike_templates.npy').ravel().astype(np.int64)
        score = np.load(sort / 'amplitudes.npy').ravel()
        T = np.load(sort / 'templates_rezToPhy.npy')   # original templates (spike_templates indexes these)
        tpk = np.argmax((T ** 2).sum(1), 1)
        xy = np.load(sort / 'channel_positions.npy')[tpk[tmpl]]   # KS2.5: original template's peak channel
    o = np.argsort(t, kind='stable')
    T = np.load(sort / 'templates.npy')
    chm = np.load(sort / 'channel_map.npy').ravel()
    pos = np.load(sort / 'channel_positions.npy')
    return t[o], c[o], tmpl[o], score[o], T, xy[o], chm, pos


def same_unit_pairs(t, c, lo, hi):
    key = c * (1 << 40) + t
    o = np.argsort(key, kind='stable')
    k = key[o]
    I, J = [], []
    d = 1
    while True:
        dk = k[d:] - k[:-d]
        if not (dk <= hi).any():
            break
        m = (dk >= lo) & (dk <= hi)
        I.append(o[:-d][m]); J.append(o[d:][m])
        d += 1
    return np.concatenate(I), np.concatenate(J)


def isolated(t, c, w=90):
    key = c * (1 << 40) + t
    o = np.argsort(key, kind='stable')
    k = key[o]
    gap = np.minimum(np.diff(k, prepend=-(1 << 62)), np.diff(k, append=(1 << 62)))
    iso = np.zeros(t.size, bool)
    iso[o] = gap > w
    return iso


class Raw:
    def __init__(self, path, chm):
        self.mm = np.memmap(path, np.int16, 'r').reshape(-1, NCH)
        self.chm = chm

    def snip(self, t0, t1, chans):
        x = np.asarray(self.mm[t0 - PAD:t1 + PAD, :], np.float32)[:, self.chm]
        x -= x.mean(0)
        x -= np.median(x, 1, keepdims=True)
        x = sosfiltfilt(SOS, x, axis=0)
        return x[PAD:-PAD, chans]


def shifted(Tm, t_rel, d, L):
    """Template (len(TW) x C) placed at window position t_rel + d, in a window of length L."""
    out = np.zeros((L, Tm.shape[1]))
    pos = t_rel + d + TW
    ok = (pos >= 0) & (pos < L)
    out[pos[ok]] = Tm[ok]
    return out.ravel()


def fit(x, Tm, r1, r2):
    L = x.shape[0]
    y = x.ravel()
    nT = (Tm ** 2).sum()
    cols1 = {d: shifted(Tm, r1, d, L) for d in range(-3, 4)}
    cols2 = {d: shifted(Tm, r2, d, L) for d in range(-3, 4)}
    best1 = None
    for d, v in cols1.items():
        a = v @ y / (v @ v)
        R = ((y - a * v) ** 2).sum()
        if best1 is None or R < best1[0]:
            best1 = (R, a, d)
    best2 = None
    for d1, v1 in cols1.items():
        for d2, v2 in cols2.items():
            A = np.stack([v1, v2], 1)
            G = A.T @ A
            if np.linalg.cond(G) > 1e8:
                continue
            a = np.linalg.solve(G, A.T @ y)
            R = ((y - A @ a) ** 2).sum()
            if best2 is None or R < best2[0]:
                best2 = (R, a[0], a[1], d1, d2)
    v2 = cols2[best2[4]]
    res1 = y - best1[1] * cols1[best1[2]]
    cos2 = float(res1 @ v2 / np.sqrt((res1 @ res1) * (v2 @ v2)))
    # spatial check: each spike isolated by subtracting the other's fitted contribution (model 2), then
    # its per-channel trough (min over -4..+8 samples around its fitted position) vs the template's
    Lc = x.shape[1]
    X = x.copy()
    s1 = X - best2[2] * cols2[best2[4]].reshape(L, Lc)
    s2 = X - best2[1] * cols1[best2[3]].reshape(L, Lc)
    tmin = Tm[30 - 4:30 + 9].min(0)

    def trough(z, p):
        lo, hi = max(p - 4, 0), min(p + 9, L)
        return z[lo:hi].min(0)
    v1 = trough(s1, r1 + best2[3]); v2 = trough(s2, r2 + best2[4])
    sc = lambda v: float(v @ tmin / np.sqrt((v @ v) * (tmin @ tmin)))
    return {'sp_cos1': sc(v1), 'sp_cos2': sc(v2), 'pk1': int(np.argmin(v1)), 'pk2': int(np.argmin(v2)),
            'pkT': int(np.argmin(tmin)), 'cos2': cos2, 'n_el': y.size, 'a1_m1': best1[1], 'd1_m1': best1[2], 'R1': best1[0] / nT, 'R2': best2[0] / nT,
            'a1': best2[1], 'a2': best2[2], 'd1': best2[3], 'd2': best2[4],
            'dE': (best1[0] - best2[0]) / nT, 'E': (y ** 2).sum() / nT}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--sort', required=True)
    ap.add_argument('--bin', default=str(BIN_KOBE))
    ap.add_argument('--units', type=int, default=40)
    ap.add_argument('--per', type=int, default=15)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    t, c, tmpl, score, T, xy, chm, pos = load(None, a.sort)
    nmax = np.memmap(a.bin, np.int16, 'r').size // NCH
    raw = Raw(a.bin, chm)
    bi, bj = same_unit_pairs(t, c, 9, 15)
    ci, cj = same_unit_pairs(t, c, 60, 90)
    iso = isolated(t, c)
    ids, nb = np.unique(c[bi], return_counts=True)
    elig = nb >= 5
    ids, nb = ids[elig], nb[elig]
    pick = rng.choice(ids, size=min(a.units, ids.size), replace=False, p=nb / nb.sum())
    rows = []
    traces = {'bump': [], 'ctrl': [], 'fake': []}
    for u in pick:
        chans = np.argsort(-(T[u] ** 2).sum(0))[:NSEL]
        su = np.where((c == u) & iso & (t > 2000) & (t < nmax - 2000))[0]
        if su.size < 20:
            continue
        tsp = rng.choice(su, size=min(40, su.size), replace=False)
        Tm = np.mean([raw.snip(t[i] + TW[0], t[i] + TW[-1] + 1, chans) for i in tsp], 0)
        pk = int(np.argmin(Tm.min(0)))  # peak channel within the selection (most negative)
        events = []
        for cls, I, J in (('bump', bi, bj), ('ctrl', ci, cj)):
            m = np.where(c[I] == u)[0]
            m = m[(t[I[m]] > 2000) & (t[J[m]] < nmax - 2000)]
            n = a.per if cls == 'bump' else max(a.per // 2, 5)
            for k in rng.choice(m, size=min(n, m.size), replace=False):
                events.append((cls, I[k], J[k]))
        rest = np.setdiff1d(su, tsp)
        for i in rng.choice(rest, size=min(max(a.per // 2, 5), rest.size), replace=False):
            events.append(('fake', i, -1))
        for cls, i, j in events:
            t1 = t[i]
            t2 = t[j] if j >= 0 else t1 + 12
            w0, w1 = t1 - 25, t2 + 46
            x = raw.snip(w0, w1, chans)
            r = fit(x, Tm, t1 - w0, t2 - w0)
            nz = raw.snip(t1 - 400, t1 - 300, chans)
            r['R2_over_noise'] = r['R2'] * (Tm ** 2).sum() / ((nz ** 2).mean() * r['n_el'])
            r['R1_over_noise'] = r['R1'] * (Tm ** 2).sum() / ((nz ** 2).mean() * r['n_el'])
            r.update({'cls': cls, 'unit': int(u), 'lag': int(t2 - t1),
                      's1': float(score[i]), 's2': float(score[j]) if j >= 0 else np.nan,
                      'same_tmpl': bool(tmpl[i] == tmpl[j]) if j >= 0 else None,
                      'pos_dist_um': float(np.linalg.norm(xy[i] - xy[j])) if j >= 0 else np.nan})
            rows.append(r)
            tr = raw.snip(t1 - 30, t1 + 60, chans)[:, pk]
            traces[cls].append(tr / max(-Tm[:, pk].min(), 1e-6))
            r['trace_pk'] = (tr / max(-Tm[:, pk].min(), 1e-6)).round(3).tolist()
    out = {'rows': rows, 'traces': {k: np.mean(v, 0).tolist() if v else [] for k, v in traces.items()}}
    (OUTDIR / f'raw_pairs_{a.name}.json').write_text(json.dumps(out, default=float))
    print(a.name, 'units', len(pick), 'events', len(rows))
    for cls in ('bump', 'ctrl', 'fake'):
        R = [r for r in rows if r['cls'] == cls]
        if not R:
            continue
        g = lambda k: np.array([r[k] for r in R], float)
        s1, s2 = g('s1'), g('s2')
        st = [r['same_tmpl'] for r in R if r['same_tmpl'] is not None]
        print(f"{cls:5s} n={len(R):4d} med a1={np.median(g('a1')):.2f} a2={np.median(g('a2')):.2f} "
              f"dE={np.median(g('dE')):.2f} R1={np.median(g('R1')):.2f} R2={np.median(g('R2')):.2f} E={np.median(g('E')):.2f} | "
              f"spcos1={np.median(g('sp_cos1')):.2f} spcos2={np.median(g('sp_cos2')):.2f} "
              f"pk2!=pkT={np.mean(g('pk2') != g('pkT')):.2f} pk1!=pkT={np.mean(g('pk1') != g('pkT')):.2f} "
              f"cos2={np.median(g('cos2')):.2f} R1/N={np.median(g('R1_over_noise')):.2f} R2/N={np.median(g('R2_over_noise')):.2f} "
              f"P(a2>0.5)={np.mean(g('a2') > 0.5):.2f} P(dE>0.3)={np.mean(g('dE') > 0.3):.2f} | "
              f"score min/max={np.nanmedian(np.minimum(s1, s2) / np.maximum(s1, s2)):.2f} "
              f"same_tmpl={np.mean(st) if st else float('nan'):.2f} pos_dist={np.nanmedian(g('pos_dist_um')):.1f}")


if __name__ == '__main__':
    main()
