"""Toy: rounds against online learning on a mixture with KS-like assignment (no convolution, no overlaps).

d = 60 features (as KS4's 10 channels x 6 PCs), noise N(0,I), about 200 units in sites of similar units (cos ~0.85),
heavy-tailed rates, spikes x = a T_u + n, a ~ N(1, 0.15). Assignment is KS4's: argmax over c of (x.T_c)^2/|T_c|^2 with
threshold Th; no amplitude or rate prior. Update: plain mean of the spikes assigned.

Start: a round-2-like dictionary: 25% of units have no template, their spikes lumped into the most similar unit's
template (the lump's mean); 20% of the fast units have a near-duplicate. So the dictionary has the two faults that the
real one has (orphans and redundancy) and tests which scheme repairs which. The split test is a stand-in for
re-clustering (no graph here): project out the template, 2-means on PC1, accept if d' > 4 and each side >= 30 spikes.

Toy limits: no overlaps, so residual and add-back are absent; no hash; no time structure (see toy_order.py); the
numbers show the logic of each mechanism and no real sort will match them. In online(), `res` caps the spikes kept per
template in a window that is reset every `split_every` batches.
"""
import sys, time
import numpy as np
import torch

torch.set_num_threads(8)
TH = 7.0
D = 60
NB = 549


def make(seed, sites=40, T=1200, nu=(3, 7)):
    r = np.random.default_rng(seed)
    Tu, site = [], []
    for s in range(sites):
        m = r.standard_normal(D); m /= np.linalg.norm(m)
        for _ in range(r.integers(*nu)):
            g = r.standard_normal(D); g -= g @ m * m; g /= np.linalg.norm(g)
            v = m + 0.42 * g; v /= np.linalg.norm(v)
            Tu.append(v * np.clip(r.lognormal(np.log(10), .3), 7.5, 30)); site.append(s)
    Tu = np.array(Tu, 'float32'); site = np.array(site)
    K = len(Tu)
    rate = r.lognormal(np.log(2.5), 1.2, K)
    N = r.poisson(rate * T)
    N = np.maximum(N, 20)
    lab = np.repeat(np.arange(K), N)
    a = np.clip(1 + .15 * r.standard_normal(lab.size), .4, 1.8).astype('float32')
    X = a[:, None] * Tu[lab] + r.standard_normal((lab.size, D)).astype('float32')
    p = r.permutation(lab.size)
    return torch.from_numpy(X[p]), torch.from_numpy(lab[p]), torch.from_numpy(Tu), site, N


def start(Tu, site, N, seed):
    r = np.random.default_rng(seed + 100)
    Tn = Tu.numpy(); K = len(Tn)
    U = Tn / np.linalg.norm(Tn, axis=1, keepdims=True)
    C = U @ U.T - 2 * np.eye(K)
    orphan = np.zeros(K, bool)
    D0 = Tn.copy()
    for u in r.permutation(K):
        if orphan[u] or r.random() > .35:
            continue
        cand = np.where((site == site[u]) & ~orphan)[0]
        cand = cand[cand != u]
        if cand.size == 0:
            continue
        v = cand[C[u, cand].argmax()]
        w = N[u] / (N[u] + N[v])
        D0[v] = w * D0[u] + (1 - w) * D0[v]
        orphan[u] = True
    keep = ~orphan
    D1 = D0[keep]
    fast = np.where(keep & (N > np.percentile(N, 60)))[0]
    fast = r.permutation(fast)[:int(.2 * K)]
    dup = []
    for u in fast:
        g = r.standard_normal(D); g /= np.linalg.norm(g)
        dup.append(Tn[u] + 1.5 * g)
    D1 = np.vstack((D1, np.array(dup, 'float32')))
    D1 = D1 + r.standard_normal(D1.shape).astype('float32') * np.sqrt(1 / 30)
    return torch.from_numpy(D1.astype('float32')), orphan


def assign(X, W, bs=200000):
    nm = (W**2).sum(1, keepdim=True).sqrt().T
    out = []
    for i in range(0, len(X), bs):
        s = (X[i:i + bs] @ W.T) / nm
        v, c = s.max(1)
        c[v < TH] = -1
        out.append(c)
    return torch.cat(out)


def means(X, c, W, nmin=5):
    keep = c >= 0
    K = len(W)
    S = torch.zeros(K, D).index_add_(0, c[keep], X[keep])
    n = torch.bincount(c[keep], minlength=K).float()
    Wn = W.clone()
    ok = n >= nmin
    Wn[ok] = S[ok] / n[ok, None]
    return Wn[ok], ok


def score(X, lab, W, Tu, N):
    c = assign(X, W)
    K = len(Tu); M = len(W)
    keep = c >= 0
    ct = torch.zeros((M, K)).index_put_((c[keep], lab[keep]), torch.ones(int(keep.sum())), accumulate=True)
    owner = ct.argmax(1)
    owned = torch.zeros(K, dtype=torch.bool); owned[owner] = True
    good = keep.clone()
    good[keep] = owner[c[keep]] == lab[keep]
    Nt = torch.from_numpy(N).float()
    acc_u = torch.bincount(lab[good], minlength=K).float() / Nt.clamp(min=1)
    o = torch.argsort(Nt)
    q = [o[:K // 3], o[K // 3:2 * K // 3], o[2 * K // 3:]]
    # template error for owned units, against the best template of each
    cs = torch.nn.functional.normalize(W, dim=1) @ torch.nn.functional.normalize(Tu, dim=1).T
    best = cs.max(0)[0]
    return dict(acc=float(good.float().mean()), slow=float(acc_u[q[0]].mean()), mid=float(acc_u[q[1]].mean()), fast=float(acc_u[q[2]].mean()),
                K=M, recovered=float(owned.float().mean()), sinerr=float(((1 - best[owned]**2).clamp(min=0).sqrt()).median()))


def split_merge(X, c, W, nmin=30, dth=4.0, merge=.975):
    """one global structure step: split each template's spikes if bimodal off the template axis; merge near-duplicates"""
    new, nsplit = [], 0
    for k in range(len(W)):
        x = X[c == k]
        if len(x) < 2 * nmin:
            new.append(W[k][None]); continue
        if len(x) > 20000:
            x = x[torch.randperm(len(x))[:20000]]
        t = W[k] / W[k].norm()
        z = x - (x @ t)[:, None] * t[None]
        z = z - z.mean(0)
        _, _, v = torch.pca_lowrank(z, q=1, center=False, niter=3)
        p = (z @ v[:, 0])
        cut = torch.tensor(0.)
        for _ in range(10):
            a, b = p[p > cut], p[p <= cut]
            if len(a) == 0 or len(b) == 0:
                break
            cut = (a.mean() + b.mean()) / 2
        a, b = p > cut, p <= cut
        if a.sum() >= nmin and b.sum() >= nmin:
            s = (((p[a] - p[a].mean())**2).sum() + ((p[b] - p[b].mean())**2).sum()) / (len(p) - 2)
            dp = (p[a].mean() - p[b].mean()) / s.sqrt()
            if dp > dth:
                new += [x[a].mean(0, keepdim=True), x[b].mean(0, keepdim=True)]
                nsplit += 1
                continue
        new.append(W[k][None])
    W2 = torch.cat(new)
    return dedup(W2, merge), nsplit


def dedup(W, th=.975):
    U = torch.nn.functional.normalize(W, dim=1)
    C = U @ U.T
    C.fill_diagonal_(0)
    nrm = W.norm(dim=1)
    drop = torch.zeros(len(W), dtype=torch.bool)
    for i in torch.argsort(nrm):
        if drop[i]:
            continue
        j = torch.where((C[i] > th) & ~drop)[0]
        j = j[(nrm[j] - nrm[i]).abs() < .2 * nrm[i]]
        if len(j):
            drop[i] = True
    return W[~drop]


def online(X, W, passes=1.0, sched='ks', c0=100., order=None, tracker=None, every=None, split_every=None, res=400, triage=None):
    """minibatch hard EM: sched 'ks' = per-spike exponential, memory clip(0.1 n, 5, 400) (KS2.5 standard, warm count c0);
    'inv' = 1/n per centre (Sculley); 'mem' = memoized sufficient statistics (Neal-Hinton incremental)."""
    M = len(X)
    B = NB
    edges = np.linspace(0, M, B + 1).astype(int)
    W = W.clone()
    cnt = torch.full((len(W),), c0)
    S = None
    if sched == 'mem':
        S = {}
        tot = W * c0
        cnt0 = torch.full((len(W),), c0)
    ordr = np.arange(B) if order is None else order
    nb = int(B * passes)
    resv = {}
    nspt = None
    for ib in range(nb):
        b = ordr[ib % B]
        x = X[edges[b]:edges[b + 1]]
        c = assign(x, W)
        keep = c >= 0
        K = len(W)
        n = torch.bincount(c[keep], minlength=K).float()
        s = torch.zeros(K, D).index_add_(0, c[keep], x[keep])
        if sched == 'mem':
            if b in S:
                tot = tot - S[b][0]; cnt0 = cnt0 - S[b][1]
            S[b] = (s, n)
            tot = tot + s; cnt0 = cnt0 + n
            ok = cnt0 > 1
            W[ok] = tot[ok] / cnt0[ok, None]
        else:
            ok = n > 0
            cnt += n
            if sched == 'ks':
                m = (cnt * .1).clamp(5, 400)
                f = torch.exp(-n / m)
            else:
                f = (1 - n / cnt).clamp(min=0)
            W[ok] = (W[ok] * f[ok, None]) + (s[ok] / n[ok, None]) * (1 - f[ok, None])
        if split_every:
            for k in torch.unique(c[keep]).tolist():
                resv.setdefault(k, []).append(x[c == k])
            if (ib + 1) % split_every == 0:
                # reservoir: the last res spikes of each template
                news = []
                for k in range(len(W)):
                    if k not in resv:
                        news.append(W[k][None]); continue
                    z = torch.cat(resv[k])[-res:]
                    resv[k] = [z]
                    if len(z) < 60:
                        news.append(W[k][None]); continue
                    w2, ns = split_merge(z, torch.zeros(len(z), dtype=torch.long), W[k][None], merge=2.)
                    news.append(w2)
                Wn = torch.cat(news)
                Wn = dedup(Wn)
                W = Wn
                cnt = torch.full((len(W),), 400.)
                resv = {}
        if tracker is not None and every and (ib + 1) % every == 0:
            tracker((ib + 1) / B, W)
    return W


def run(seed):
    t0 = time.time()
    X, lab, Tu, site, N = make(seed)
    W0, orphan = start(Tu, site, N, seed)
    print(f'seed {seed}: {len(X)/1e6:.2f} M spikes, {len(Tu)} units, median N {np.median(N):.0f}, {orphan.sum()} orphans, start dictionary {len(W0)}', flush=True)
    rows = []
    def rec(name, W, cost):
        d = score(X, lab, W, Tu, N)
        rows.append((name, cost, d))
        print(f"  {name:34s} cost {cost:5.2f}  acc {d['acc']:.3f}  slow {d['slow']:.3f} mid {d['mid']:.3f} fast {d['fast']:.3f}  "
              f"K {d['K']:4d}  units owning {d['recovered']:.3f}  median sin(err) {d['sinerr']:.3f}", flush=True)
    rec('ideal (true templates)', Tu, 0)
    rec('start', W0, 0)
    # Lloyd rounds
    W = W0
    for r in range(1, 6):
        c = assign(X, W)
        W, ok = means(X, c, W)
        rec(f'rounds, mean only, r{r}', W, r)
    Wlloyd = W
    # online from the start dictionary
    for sched in ['ks', 'inv', 'mem']:
        for p in [.05, .1, .25, .5, 1, 3]:
            W = online(X, W0, passes=p, sched=sched)
            rec(f'online {sched}, {p} pass', W, p)
    # structure
    W = W0
    for r in range(1, 6):
        c = assign(X, W)
        W, ok = means(X, c, W)
        c = assign(X, W)
        W, ns = split_merge(X, c, W)
        rec(f'rounds + global split/merge, r{r} ({ns} splits)', W, r)
    Wr = W
    # online with reservoir splits
    for res, ev in [(400, 50), (2000, 50)]:
        for p in [1, 3]:
            W = online(X, W0, passes=p, sched='ks', split_every=ev, res=res)
            rec(f'online ks + reservoir {res} split, {p} pass', W, p)
    # online one pass, then one global
    W = online(X, W0, passes=1, sched='ks')
    c = assign(X, W)
    W, ns = split_merge(X, c, W)
    W = dedup(W)
    rec(f'online 1 pass, then 1 global ({ns} splits)', W, 2)
    c = assign(X, W)
    W, ok = means(X, c, W)
    c = assign(X, W)
    W, ns = split_merge(X, c, W)
    rec(f'  ... then a second global ({ns} splits)', W, 3)
    print(f'  time {time.time()-t0:.0f} s', flush=True)
    return rows


if __name__ == '__main__':
    for s in [int(a) for a in sys.argv[1:]] or [1, 2]:
        run(s)
