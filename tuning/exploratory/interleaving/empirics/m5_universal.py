"""Measurement 5: the premise. Were the spikes of the final units seen by the universal-template pass, and do units whose
waveform fits the universal templates poorly get less of a template?

st0 (extract_st0.py) holds every universal-pass detection of the H8 sorts; default sorts have no cached st0 in
iteration\\stage_cache, so this is H8 only. All eligible units of each sort (no other selection).

- Alignment: st0's time is the peak of the wTEMP convolution, offset from KS4's spike time; offset and depth error are
  taken from the histogram (printed), not assumed.
- Seen: a universal detection within TOL samples and UM um of the (cleaned) spike. Chance: same with st0 shifted by 1000
  samples. Per unit f_seen; corrected (s - c) / (1 - c). Tolerances varied.
- Fit rho: response of the unit's own template to the universal pass (conv1d with wTEMP, weighted over the 10 nearest
  channels by each size, best centre, shape and lag, exactly the quantity thresholded by Th_universal) over the template's
  norm (the same spike's learned-matching score). A spike of amplitude a x template clears Th_universal if a rho |T| > Th.
  rho is the cosine with the nearest universal element.
- Owner: a unit owns an extraction template if it holds the plurality of that template's spikes (as borrowed_templates.py).
- Round-1 cluster: the first clustering's cluster holding most of the unit's seen spikes; its share of those spikes,
  the unit's share of the cluster, and the cosine of the cluster's template (Wall of clustering pass 1) with the unit's.

Writes m5_{ex}_s{seed}_r{round}.npz (per unit) and m5_{ex}.json.

    python m5_universal.py
"""
import gc
import json

import numpy as np
import torch
from numba import njit
from torch.nn.functional import conv1d

from common import OUT, SEEDS, Sort, nearest_event, rates, sort_dir, strata

TOLS = ((5, 40.0), (10, 50.0), (20, 60.0))
TOL = 1                                # index in TOLS of the main tolerance
FS = 30000
BINS = [0, 7, 8, 9, 10, 12, 15, 20, 30, 1e9]


@njit(cache=True)
def lag_hist(ta, ya, tb, yb, maxlag, dy):
    h = np.zeros(2 * maxlag + 1, np.int64)
    hy = np.zeros(161, np.int64)
    j0 = 0
    for i in range(0, ta.size, 7):
        while j0 < tb.size and tb[j0] < ta[i] - maxlag:
            j0 += 1
        j = j0
        while j < tb.size and tb[j] <= ta[i] + maxlag:
            d = ya[i] - yb[j]
            if abs(d) <= dy:
                h[tb[j] - ta[i] + maxlag] += 1
            if abs(d) < 80:
                hy[int(d + 80)] += 1
            j += 1
    return h, hy


def fit_rho(T, ops):
    """Universal response (the thresholded quantity) of units x chans x nt templates, and norm."""
    wT = torch.as_tensor(np.asarray(ops['wTEMP'])).float()
    iC = torch.as_tensor(np.asarray(ops['iC'])).long()
    weigh = torch.as_tensor(np.asarray(ops['weigh'])).float()
    ycup, xcup = np.asarray(ops['ycup']), np.asarray(ops['xcup'])
    yc, xc = np.asarray(ops['yc']), np.asarray(ops['xc'])
    out = np.zeros(T.shape[0])
    peak = (T ** 2).sum(2).argmax(1)
    for u in range(T.shape[0]):
        B = conv1d(torch.from_numpy(T[u].copy()).float().unsqueeze(1), wT.unsqueeze(1), padding=30)   # chans x 6 x 61
        near = np.flatnonzero(np.abs(ycup - yc[peak[u]]) + np.abs(xcup - xc[peak[u]]) < 120)
        A = torch.einsum('ijk, jklm -> iklm', weigh[:, :, near], B[iC[:, near]])          # sizes x centres x 6 x nt
        out[u] = float(A.abs().max())
    return out, np.sqrt((T ** 2).sum((1, 2)))


def owners(s):
    tpl = np.load(s.d / 'spike_detection_templates.npy').ravel().astype(np.int64)[s.order]
    c = s.clu
    n_t = tpl.max() + 1
    k = tpl * s.n_units + c
    ids, n = np.unique(k, return_counts=True)
    t, u = ids // s.n_units, ids % s.n_units
    o = np.lexsort((n, t))
    last = np.flatnonzero(np.r_[t[o][1:] != t[o][:-1], True])
    own = np.bincount(u[o][last], minlength=s.n_units)
    return own, n_t


def unit_table(s, keyv, n):
    return np.bincount(s, keyv, minlength=n)


def run(ex, seed, rnd, st0, clu1, wall1, ops1, summary):
    s = Sort(sort_dir(ex, 'H8', seed, rnd), ex, 'H8')
    t, y, c = s.cleaned()
    score = np.load(s.d / 'score.npy')[s.order][s.keep]
    o0 = np.argsort(st0[:, 0], kind='stable')
    # st0 time: seconds; first pass only to find the offset
    t0 = np.rint(st0[o0, 0] * FS).astype(np.int64)
    y0 = st0[o0, 1].astype(np.float32)
    amp0 = st0[o0, 2].astype(np.float32)
    lab1 = clu1[o0]
    h, hy = lag_hist(t, y, t0, y0, 40, 40.0)
    off = int(np.argmax(h)) - 40
    t0 = t0 - off
    res = {}
    info = {'offset_samples': off, 'lag_hist_peak_share': float(h[off + 40 - 3:off + 40 + 4].sum() / h.sum()),
            'dy_hist': hy.tolist(), 'lag_hist': h.tolist(), 'n_st0': int(t0.size), 'n_cleaned': int(t.size)}
    for k, (tol, um) in enumerate(TOLS):
        m = nearest_event(t, y, t0, y0, tol, um)
        mc = nearest_event(t, y, t0 + 1000, y0, tol, um)
        info[f'seen_{tol}_{um:g}'] = float((m >= 0).mean())
        info[f'chance_{tol}_{um:g}'] = float((mc >= 0).mean())
        if k == TOL:
            nu = s.n_units
            n = np.bincount(c, minlength=nu).astype(float)
            seen = np.bincount(c[m >= 0], minlength=nu)
            chance = np.bincount(c[mc >= 0], minlength=nu)
            ok = m >= 0
            b = np.digitize(score, BINS) - 1
            info['score_bins'] = [dict(lo=BINS[i], n=int((b == i).sum()), seen=float(ok[b == i].mean()),
                                       chance=float((mc[b == i] >= 0).mean())) for i in range(len(BINS) - 1) if (b == i).any()]
            a = amp0[m[ok]]
            med = np.full(nu, np.nan)
            nearth = np.full(nu, np.nan)
            o = np.argsort(c[ok], kind='stable')
            cs, a2 = c[ok][o], a[o]
            lim = np.searchsorted(cs, np.arange(nu + 1))
            th = 7.0
            lab = lab1[m[ok]][o]
            best_c, share_c, purity_c = np.full(nu, -1), np.full(nu, np.nan), np.full(nu, np.nan)
            n1 = np.bincount(lab[lab >= 0], minlength=int(lab1.max()) + 1).astype(float)   # seen-final spikes per cluster, any unit
            for u in range(nu):
                x = a2[lim[u]:lim[u + 1]]
                if x.size:
                    med[u] = np.median(x)
                    nearth[u] = (x < 1.15 * th).mean()
                    l = lab[lim[u]:lim[u + 1]]
                    q = np.bincount(l[l >= 0])
                    if q.size:
                        best_c[u] = q.argmax()
                        share_c[u] = q.max() / max(l.size, 1)
                        purity_c[u] = q.max() / n1[q.argmax()]
    elig = np.array(sorted(s.eligible()))
    rate = rates(s, list(elig))
    st = strata(rate, s.amp)
    oi = np.load(s.d / 'ops.npy', allow_pickle=True).item()
    T = np.load(s.d / 'Wall.npy') @ np.asarray(oi['wPCA'])
    resp, nrm = fit_rho(T[elig].astype(np.float32), oi)
    own, nt = owners(s)
    w1t = wall1 @ np.asarray(oi['wPCA'])
    c1 = np.full(elig.size, np.nan)
    for i, u in enumerate(elig):
        if best_c[u] >= 0:
            a_, b_ = T[u].ravel(), w1t[best_c[u]].ravel()
            c1[i] = a_ @ b_ / np.sqrt((a_ @ a_) * (b_ @ b_) + 1e-12)
    # clusters of the first pass owned: plurality of seen-final spikes in the eligible unit
    res.update(unit=elig, n=n[elig], seen=seen[elig], chance=chance[elig], med_amp0=med[elig], near_th=nearth[elig],
               resp=resp, norm=nrm, rate=np.array([rate[u] for u in elig]), amp=np.array([s.amp.get(u, np.nan) for u in elig]),
               rq=np.array([st[u][0] for u in elig]), aq=np.array([st[u][1] for u in elig]), own=own[elig],
               c1_share=share_c[elig], c1_purity=purity_c[elig], c1_cos=c1, c1_id=best_c[elig],
               S=np.array([s.eligible()[u] for u in elig]))
    np.savez(OUT / f'm5_{ex}_s{seed}_r{rnd}.npz', **res)
    summary[f's{seed}_r{rnd}'] = info
    print(ex, seed, rnd, {k: v for k, v in info.items() if not isinstance(v, list)}, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)            # small convolutions, one per unit: more threads only contend
    for ex in ('kobe', 'mec'):
        f = OUT / f'm5_{ex}.json'
        summary = json.loads(f.read_text()) if f.exists() else {}
        for seed in SEEDS:
            todo = [r for r in (2, 4) if f's{seed}_r{r}' not in summary]
            if not todo:
                continue
            st0 = np.load(OUT / f'st0_{ex}_s{seed}.npy')
            clu1 = np.load(OUT / f'clu1_{ex}_s{seed}.npy')
            wall1 = np.load(OUT / f'wall1_{ex}_s{seed}.npy')
            for rnd in todo:
                run(ex, seed, rnd, st0, clu1, wall1, None, summary)
                gc.collect()
                f.write_text(json.dumps(summary, indent=1))
