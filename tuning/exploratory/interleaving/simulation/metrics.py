"""Metrics against truth: template error, per-unit accuracy, splits, merges, false units, pair resolution."""
import numpy as np
import torch

from primitives import universal_score
from simdata import NT, NT0

WIN = 4       # matching half-window, samples
SEARCH = 9    # search half-window before the per-pair lag is removed
N_MIN = 12


def truth_pc(ctx, ds):
    """True templates in PC space: (n_units, pc, chan)."""
    T = torch.from_numpy(ds.T_true)                       # (u, chan, NT)
    return (T @ ctx.W.T).permute(0, 2, 1)


def universal_match(ctx, ds):
    """Best universal score of the noiseless mean waveform over its norm: how well the dictionary sees the unit."""
    out = np.zeros(ds.T_true.shape[0])
    for u in range(len(out)):
        X = torch.zeros(ctx.nchan, 4 * NT)
        X[:, NT + NT // 2:NT + NT // 2 + NT] = torch.from_numpy(ds.T_true[u])
        A = universal_score(ctx, X)
        n = float((ds.T_true[u] ** 2).sum() ** .5)
        out[u] = float(A.max()) / max(n, 1e-6)
    return out


def template_error(ctx, D, Ttrue_pc, lags=6):
    """For each true unit: min over dictionary and lags of ||T - truth|| / ||truth||, capped at 1. Also the winner."""
    nu = Ttrue_pc.shape[0]
    if D.shape[0] == 0:
        return np.ones(nu), -np.ones(nu, 'int64')
    W = ctx.W
    Ttime = torch.einsum('ipc, pt -> ict', Ttrue_pc, W)    # (u, chan, NT)
    Dtime = torch.einsum('ipc, pt -> ict', D, W)
    nT = (Ttime ** 2).sum((1, 2))
    best = torch.full((nu,), float('inf'))
    arg = torch.zeros(nu, dtype=torch.int64)
    for lag in range(-lags, lags + 1):
        Dr = torch.roll(Dtime, lag, -1)
        # ||D - T||^2 for all pairs
        d2 = (Dr ** 2).sum((1, 2))[None, :] + nT[:, None] - 2 * torch.einsum('uct, kct -> uk', Ttime, Dr)
        v, i = d2.min(1)
        m = v < best
        best[m] = v[m]; arg[m] = i[m]
    e = (best.clamp(min=0) / nT.clamp(min=1e-6)).sqrt().numpy()
    return np.minimum(e, 1.0), arg.numpy()


def confusion(t_true, u_true, t_out, l_out, nu, nl):
    """Per (true, output) pair TP after removing the pair's median lag; a true spike counts once per output."""
    o = np.argsort(t_out)
    t_out = t_out[o]; l_out = l_out[o]
    lo = np.searchsorted(t_out, t_true - SEARCH); hi = np.searchsorted(t_out, t_true + SEARCH, 'right')
    cnt = hi - lo
    n = cnt.sum()
    if n == 0:
        return np.zeros((nu, nl), 'int64')
    i_true = np.repeat(np.arange(len(t_true)), cnt)
    start = np.repeat(lo, cnt)
    offs = np.arange(n) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    j = start + offs
    dt = t_out[j] - t_true[i_true]
    pair = u_true[i_true].astype(np.int64) * nl + l_out[j]
    # per-pair median lag
    order = np.argsort(pair, kind='stable')
    pair_s = pair[order]; dt_s = dt[order]
    uniq, st, c = np.unique(pair_s, return_index=True, return_counts=True)
    C = np.zeros((nu, nl), 'int64')
    for pr_, s, k in zip(uniq, st, c):
        if k < 3:
            continue
        d = dt_s[s:s + k]
        med = np.median(d)
        ok = np.abs(d - med) <= WIN
        if not ok.any():
            continue
        # one count per true spike
        idx = i_true[order[s:s + k]][ok]
        C[pr_ // nl, pr_ % nl] = len(np.unique(idx))
    return C


def score_output(t_out, l_out, ds, extra=None):
    """All unit-level metrics for a labelled spike train."""
    nu = ds.T_true.shape[0]
    lab_u, l = np.unique(l_out, return_inverse=True)
    nl = len(lab_u)
    n_l = np.bincount(l, minlength=nl).astype(float)
    keep_l = n_l >= N_MIN
    n_u = np.bincount(ds.u, minlength=nu).astype(float)
    C = confusion(ds.t, ds.u, t_out, l, nu, nl).astype(float)
    C = np.minimum(C, n_l[None, :])
    acc = C / np.maximum(n_u[:, None] + n_l[None, :] - C, 1)
    acc[:, ~keep_l] = 0
    best = acc.argmax(1) if nl else np.zeros(nu, 'int64')
    a = acc[np.arange(nu), best]
    rec = C[np.arange(nu), best] / np.maximum(n_u, 1)
    prec = C[np.arange(nu), best] / np.maximum(n_l[best], 1)
    found = a >= 0.5
    pur = C / np.maximum(n_l[None, :], 1)
    frac_u = C / np.maximum(n_u[:, None], 1)
    split = ((pur >= 0.5) & (frac_u >= 0.1) & keep_l[None, :]).sum(1) >= 2
    merged_l = ((frac_u >= 0.2) & (pur >= 0.05)).sum(0) >= 2
    merged_u = (((frac_u >= 0.2) & (pur >= 0.05)) & merged_l[None, :]).any(1)
    false_l = keep_l & (pur.max(0) < 0.5)
    pairs = ds.info['pairs']
    res = [bool(found[a_] and found[b_] and best[a_] != best[b_]) for a_, b_ in pairs]
    return dict(acc=a, rec=rec, prec=prec, found=found, split=split, merged_u=merged_u, n_merged_outputs=int(merged_l.sum()),
                n_false=int(false_l.sum()), n_outputs=int(keep_l.sum()), pair_res=res,
                n_spikes_out=int(len(t_out)))


def summary(sc, ds, um, poor_th=0.75):
    """Scalars and strata for a score dict."""
    r = ds.info['rates']
    poor = um < poor_th
    strata = dict(lt05=r < .5, r05_3=(r >= .5) & (r < 3), r3_15=(r >= 3) & (r < 15), gt15=r >= 15)
    s = dict(acc=float(sc['acc'].mean()), found=float(sc['found'].mean()), rec=float(sc['rec'].mean()),
             prec=float(sc['prec'].mean()), n_false=sc['n_false'], n_out=sc['n_outputs'],
             split=float(sc['split'].mean()), merged=float(sc['merged_u'].mean()), n_merged_out=sc['n_merged_outputs'],
             pair_res=float(np.mean(sc['pair_res'])) if len(sc['pair_res']) else float('nan'))
    pm = np.zeros(len(r), bool)
    for a, b in ds.info['pairs']:
        pm[a] = pm[b] = True
    s['acc_nonpair'] = float(sc['acc'][~pm].mean())
    s['found_poor'] = float(sc['found'][poor].mean()) if poor.any() else float('nan')
    s['found_reg'] = float(sc['found'][~poor].mean())
    s['acc_poor'] = float(sc['acc'][poor].mean()) if poor.any() else float('nan')
    s['acc_reg'] = float(sc['acc'][~poor].mean())
    for k, m in strata.items():
        s['found_' + k] = float(sc['found'][m].mean()) if m.any() else float('nan')
        s['acc_' + k] = float(sc['acc'][m].mean()) if m.any() else float('nan')
        s['rec_' + k] = float(sc['rec'][m].mean()) if m.any() else float('nan')
        s['prec_' + k] = float(sc['prec'][m].mean()) if m.any() else float('nan')
    return s
