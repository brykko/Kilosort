"""Toy: time-ordered against shuffled batches for online template learning, on a non-stationary recording.

Same feature model as toy_online.py, warm-started at the true time-averaged templates so that only the order matters.
Three classes of unit: stationary (60%); epochal, present in one contiguous 25% of the recording (20%); state-dependent
(20%): two states with dwell ~300 s, rate x3 and x0.3, and the waveform moves by a vector of norm 1 (noise SD units) in
the fast state (a stand-in for burst attenuation). KS2.5 'standard' schedule; triage as triageTemplates2: nsp is an EMA
over batches with p1 and a template is dropped if nsp < minFR * batch length, tested every 5 batches.
"""
import numpy as np
import torch
from toy_online import assign, D, TH

NB = 549
BL = 1200 / NB


def make(seed, sites=40):
    r = np.random.default_rng(seed)
    Tu, site = [], []
    for s in range(sites):
        m = r.standard_normal(D); m /= np.linalg.norm(m)
        for _ in range(r.integers(3, 7)):
            g = r.standard_normal(D); g -= g @ m * m; g /= np.linalg.norm(g)
            v = m + .42 * g; v /= np.linalg.norm(v)
            Tu.append(v * np.clip(r.lognormal(np.log(10), .3), 7.5, 30)); site.append(s)
    Tu = np.array(Tu, 'float32')
    K = len(Tu)
    rate = r.lognormal(np.log(2.5), 1.2, K)
    cls = r.choice(3, K, p=[.6, .2, .2])
    # shared state sequence per batch: dwell ~ 300 s, about 137 batches
    st = np.zeros(NB, int); b = 0; s = 0
    while b < NB:
        L = max(1, int(r.exponential(137))); st[b:b + L] = s; s = 1 - s; b += L
    X, lab, bat = [], [], []
    dirs = r.standard_normal((K, D)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    Tmean = np.zeros_like(Tu); Ncount = np.zeros(K)
    active = np.ones((K, NB), bool)
    for u in range(K):
        if cls[u] == 1:
            L = NB // 4; b0 = r.integers(0, NB - L)
            active[u] = False; active[u, b0:b0 + L] = True
        rb = rate[u] * BL * np.where(active[u], 1., 0.)
        if cls[u] == 1:
            rb *= 4                              # same total spikes
        if cls[u] == 2:
            rb = rb * np.where(st == 1, 3., .3) / (np.mean(np.where(st == 1, 3., .3)))
        n = r.poisson(rb)
        ib = np.repeat(np.arange(NB), n)
        a = np.clip(1 + .15 * r.standard_normal(ib.size), .4, 1.8)
        T = np.tile(Tu[u], (ib.size, 1))
        if cls[u] == 2:
            T = T + (st[ib] == 1)[:, None] * dirs[u]
        x = a[:, None] * T + r.standard_normal((ib.size, D))
        X.append(x.astype('float32')); lab.append(np.full(ib.size, u)); bat.append(ib)
        Tmean[u] = x.mean(0) if ib.size else Tu[u]; Ncount[u] = ib.size
    X = np.vstack(X); lab = np.concatenate(lab); bat = np.concatenate(bat)
    o = np.argsort(bat, kind='stable')
    return torch.from_numpy(X[o]), torch.from_numpy(lab[o]), torch.from_numpy(bat[o]), torch.from_numpy(Tmean), cls, Ncount


def learn(X, bat, W0, order, triage=None):
    W = W0.clone(); K = len(W)
    alive = torch.ones(K, dtype=torch.bool)
    cnt = torch.full((K,), 400.)
    idx = torch.arange(K)
    edges = torch.searchsorted(bat, torch.arange(NB + 1))
    nsp = torch.full((K,), 2.)
    for ib, b in enumerate(order):
        x = X[edges[b]:edges[b + 1]]
        c = assign(x, W)
        keep = c >= 0
        k = len(W)
        n = torch.bincount(c[keep], minlength=k).float()
        s = torch.zeros(k, D).index_add_(0, c[keep], x[keep])
        ok = n > 0
        cnt += n
        f = torch.exp(-n / (cnt * .1).clamp(5, 400))
        W[ok] = W[ok] * f[ok, None] + s[ok] / n[ok, None] * (1 - f[ok, None])
        if triage:
            p1, minfr = triage
            nsp = nsp * p1 + (1 - p1) * n
            if ib % 5 == 0:
                d = nsp < minfr * BL
                if d.any():
                    W, cnt, nsp, idx = W[~d], cnt[~d], nsp[~d], idx[~d]
    return W, idx


def score(W, idx, X, lab, Tm, cls, N):
    c = assign(X, W)
    keep = c >= 0
    good = keep & (idx[c.clamp(min=0)] == lab)
    K = len(Tm)
    acc = torch.bincount(lab[good], minlength=K).float() / torch.from_numpy(N).float().clamp(min=1)
    cs = (torch.nn.functional.normalize(W, dim=1) * torch.nn.functional.normalize(Tm[idx], dim=1)).sum(1)
    sn = (1 - cs**2).clamp(min=0).sqrt()
    out = {}
    for k, nm in enumerate(['stationary', 'epochal', 'state']):
        m = torch.from_numpy(cls == k)
        live = m[idx]
        out[nm] = (float(acc[m].mean()), float(sn[live].median()) if live.any() else float('nan'), int(m.sum() - live.sum()), int(m.sum()))
    out['all spikes'] = float(good.float().mean())
    return out


for seed in [1, 2, 3]:
    X, lab, bat, Tm, cls, N = make(seed)
    print(f'seed {seed}: {len(X)/1e6:.2f} M spikes, {len(Tm)} units; classes {np.bincount(cls)}', flush=True)
    r = np.random.default_rng(seed + 5)
    W0 = Tm.clone()
    for name, order in [('time order', np.arange(NB)), ('shuffled', r.permutation(NB))]:
        for tname, tr in [('no triage', None), ('triage p1 .99 minFR .05 (standard)', (.99, .05)), ('triage p1 .95 minFR 1/50 (original)', (.95, .02))]:
            W, idx = learn(X, bat, W0, order, tr)
            o = score(W, idx, X, lab, Tm, cls, N)
            print(f'  {name:11s} {tname:36s} acc all {o["all spikes"]:.3f} | ' + ' | '.join(
                f'{k}: acc {v[0]:.3f} sin {v[1]:.3f} dropped {v[2]}/{v[3]}' for k, v in o.items() if k != 'all spikes'), flush=True)
