"""Shared primitives: universal detection, matching pursuit, clustering, template building, merges.

Every schedule (R, Rd, S, I) calls these and nothing else, so differences between arms are scheduling.
Index order follows KS4: templates D are (n_templates, n_pcs, n_chan).
"""
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.nn.functional import conv1d, max_pool1d, max_pool2d
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import KMeans

from simdata import NT, NT0, FS, universal_bases

TSHIFT = NT // 2 - NT0        # window-centre time minus trough time


@dataclass
class Ctx:
    xc: np.ndarray
    yc: np.ndarray
    Th_u: float = 7.0
    Th_l: float = 7.0
    nC: int = 10
    max_peels: int = 60
    mf_ops: float = 0.0
    fpe: float = 0.0
    clock: dict = field(default_factory=dict)

    def __post_init__(self):
        wT, wP = universal_bases()
        self.wT = torch.from_numpy(wT)
        self.W = torch.from_numpy(wP)
        self.npc = wP.shape[0]
        self.nchan = len(self.xc)
        d2 = (self.xc[:, None] - self.xc) ** 2 + (self.yc[:, None] - self.yc) ** 2
        self.iCC = torch.from_numpy(np.argsort(d2, 1)[:, :self.nC].copy())        # (chan, nC) nearest channels
        # universal spatial filters on a (rows, columns) grid of centres
        ys = np.unique(self.yc)
        xs = np.unique(self.xc)
        cy, cx = np.meshgrid(ys, xs, indexing='ij')
        self.grid = cy.shape
        cy, cx = cy.ravel(), cx.ravel()
        self.ncen = len(cy)
        w = []
        self.sigs = (15.0, 30.0)
        for sig in self.sigs:
            ds = (cx[:, None] - self.xc) ** 2 + (cy[:, None] - self.yc) ** 2
            g = np.exp(-ds / (2 * sig ** 2))
            g = g / np.sqrt((g ** 2).sum(1, keepdims=True))
            w.append(g)
        self.weigh = torch.tensor(np.concatenate(w, 0), dtype=torch.float32)      # (ncen*nsig, chan)
        dch = (cx[:, None] - self.xc) ** 2 + (cy[:, None] - self.yc) ** 2
        self.cen_chan = torch.from_numpy(dch.argmin(1))                           # nearest channel of each centre
        W = self.W
        ww = conv1d(W.reshape(-1, 1, NT), W.reshape(-1, 1, NT), padding=NT)
        self.WtW = torch.flip(ww, [2])                                            # (pc, pc, 2NT+1)

    def tick(self, name, t0):
        self.clock[name] = self.clock.get(name, 0.0) + time.perf_counter() - t0


# ---------------------------------------------------------------- universal detection

def universal_score(ctx, X):
    """(ncen, NT) absolute score: best over temporal shapes and spatial widths, as template_match."""
    B = conv1d(X.unsqueeze(1), ctx.wT.unsqueeze(1), padding=NT // 2)          # (chan, 6, NT)
    nch, nk, nt = B.shape
    A = ctx.weigh @ B.reshape(nch, nk * nt)                                   # (ncen*nsig, 6*NT)
    A = A.reshape(-1, nk, nt).abs().max(1)[0]                                 # (ncen*nsig, NT)
    A = A.reshape(len(ctx.sigs), ctx.ncen, nt).max(0)[0]
    return A


def window_max_table(x, v):
    """Sparse table over sorted positions x for range-max queries."""
    tab = [v]
    j = 1
    while (1 << j) <= len(v):
        prev = tab[-1]
        tab.append(np.maximum(prev[:len(prev) - (1 << (j - 1))], prev[(1 << (j - 1)):]))
        j += 1
    return tab


def window_max_query(x, tab, q, w):
    """Max of v over positions within q +- w; 0 where the window is empty."""
    lo = np.searchsorted(x, q - w); hi = np.searchsorted(x, q + w, 'right')
    out = np.zeros(len(q), tab[0].dtype)
    m = hi > lo
    if not m.any():
        return out
    span = hi[m] - lo[m]
    k = np.floor(np.log2(span)).astype(int)
    res = np.zeros(m.sum(), tab[0].dtype)
    for kk in np.unique(k):
        i = k == kk
        l, h = lo[m][i], hi[m][i]
        res[i] = np.maximum(tab[kk][l], tab[kk][h - (1 << kk)])
    out[m] = res
    return out


def detect(ctx, Xin, lo_keep, hi_keep):
    """Local maxima of the universal score over neighbouring centres and time, as template_match, computed
    on the entries above threshold (the same set as max-pooling the whole array).
    Returns window-centre times (relative to Xin) and the channel nearest each spike's centre."""
    t0 = time.perf_counter()
    A = universal_score(ctx, Xin)
    nt = A.shape[1]
    ctx.mf_ops += (ctx.ncen * len(ctx.sigs) * ctx.wT.shape[0]) * nt
    A[:, :NT] = 0; A[:, -NT:] = 0
    ny, nx = ctx.grid
    Arow, col = A.reshape(ny, nx, nt).max(1)                                 # best column in each row
    r, tt = (Arow > ctx.Th_u).nonzero(as_tuple=True)
    r = r.numpy(); tt = tt.numpy()
    if len(r) == 0:
        ctx.tick('detect', t0)
        return torch.zeros(0, dtype=torch.int64), torch.zeros(0, dtype=torch.int64)
    v = Arow.numpy()[r, tt]
    cc = col.numpy()[r, tt]
    tabs = {}
    urows = np.unique(r)
    for q in urows:
        m = r == q
        o = np.argsort(tt[m], kind='stable')
        x = tt[m][o]
        tabs[q] = (x, window_max_table(x, v[m][o]))
    is_peak = np.ones(len(r), bool)
    for q in urows:
        m = np.flatnonzero(r == q)
        for q2 in range(q - 2, q + 3):
            if q2 in tabs:
                mx = window_max_query(tabs[q2][0], tabs[q2][1], tt[m], NT0)
                is_peak[m[mx > v[m]]] = False
    keep = is_peak & (tt >= lo_keep) & (tt < hi_keep)
    cen = r[keep] * nx + cc[keep]
    ctx.tick('detect', t0)
    return torch.from_numpy(tt[keep].astype('int64')), ctx.cen_chan[torch.from_numpy(cen.astype('int64'))]


def features_at(ctx, Xs, tt, main_ch):
    """PC features on the nC channels nearest main_ch at window-centre times tt."""
    chs = ctx.iCC[main_ch]                                                    # (n, nC)
    tiw = torch.arange(-(NT // 2), NT // 2 + 1)
    clips = Xs[chs.unsqueeze(-1), (tt.unsqueeze(-1) + tiw).unsqueeze(1)]      # (n, nC, NT)
    return (clips @ ctx.W.T).numpy().astype('float32'), chs.numpy()


# ---------------------------------------------------------------- matching pursuit

def prep_ctc(ctx, D):
    """Sparse cross-correlations, as template_matching.prepare_matching: for each template y the rows u
    whose channels overlap it, and their (2NT+1) lag vectors."""
    Dn = D.reshape(D.shape[0], ctx.npc, ctx.nchan)
    K = Dn.shape[0]
    UtU = torch.einsum('ikl, jml -> ijkm', Dn, Dn)
    ctc = torch.einsum('ijkm, kml -> ijl', UtU, ctx.WtW)
    u, y = torch.nonzero((ctc != 0).any(-1), as_tuple=True)
    vals = ctc[u, y]
    order = torch.sort(y, stable=True)[1]
    counts = torch.bincount(y, minlength=K)
    ptr = torch.zeros(K + 1, dtype=torch.int64)
    ptr[1:] = torch.cumsum(counts, 0)
    return dict(ptr=ptr, rows=u[order], vals=vals[order])


def _bupdate(B, ctc, iX, iY, amp, trange):
    lo, hi = ctc['ptr'][iY], ctc['ptr'][iY + 1]
    cnt = hi - lo
    s = torch.repeat_interleave(torch.arange(len(iY)), cnt)
    e = lo[s] + torch.arange(int(cnt.sum())) - (torch.cumsum(cnt, 0) - cnt)[s]
    idx = ctc['rows'][e].unsqueeze(-1) * B.shape[1] + iX[s].unsqueeze(-1) + trange
    B.view(-1).index_add_(0, idx.reshape(-1), -(amp[s].unsqueeze(-1) * ctc['vals'][e]).reshape(-1))


def project_templates(ctx, P, D):
    """B = D @ P."""
    return D.reshape(D.shape[0], -1) @ P.reshape(ctx.npc * ctx.nchan, -1)


def peak_candidates(Cfmax, Th2):
    idx = torch.nonzero(Cfmax > Th2)[:, 0].numpy()
    if len(idx) == 0:
        return idx
    v = Cfmax.numpy()[idx]
    tab = window_max_table(idx, v)
    mx = window_max_query(idx, tab, idx, NT)
    return idx[np.abs(mx - v) < 1e-9]


def matching(ctx, Xb, D, ctc, lo_keep, hi_keep):
    """KS4 run_matching on CPU. Returns times (relative to Xb), template ids, amps, residual."""
    t0 = time.perf_counter()
    Th = ctx.Th_l
    K = D.shape[0]
    nm = (D ** 2).sum((1, 2))
    P = conv1d(Xb.unsqueeze(1), ctx.W.unsqueeze(1), padding=NT // 2).permute(1, 0, 2)    # (pc, chan, NT)
    B = project_templates(ctx, P, D)
    del P
    NTb = B.shape[1]
    ctx.mf_ops += K * NTb
    trange = torch.arange(-NT, NT + 1)
    Cf = torch.relu(B) ** 2 / nm.unsqueeze(-1)
    Cf[:, :NT] = 0; Cf[:, -NT:] = 0
    Cfmax, imax = torch.max(Cf, 0)
    del Cf
    Sx, Sy, Sa = [], [], []
    for _ in range(ctx.max_peels):
        pk = peak_candidates(Cfmax, Th ** 2)
        if len(pk) == 0:
            break
        iX = torch.from_numpy(pk.astype('int64'))
        iY = imax[iX]
        amp = B[iY, iX] / nm[iY]
        Sx.append(iX); Sy.append(iY); Sa.append(amp)
        nsp = len(iX)
        n = 2
        while nsp > n and bool((iX[n:] - iX[:-n] <= 2 * NT).any()):
            n += 1
        for j in range(n):
            ix = iX[j::n]
            if len(ix):
                _bupdate(B, ctc, ix, iY[j::n], amp[j::n], trange)
        cols = torch.unique(iX.unsqueeze(-1) + trange)
        cols = cols[(cols >= NT) & (cols < NTb - NT)]
        Cfmax[cols], imax[cols] = torch.max(torch.relu(B[:, cols]) ** 2 / nm.unsqueeze(-1), 0)
    if Sx:
        tt = torch.cat(Sx); ids = torch.cat(Sy); amps = torch.cat(Sa)
    else:
        tt = torch.zeros(0, dtype=torch.int64); ids = tt.clone(); amps = torch.zeros(0)
    # residual in one go, on each template's own channels
    Xres = Xb.clone()
    if len(tt):
        E = torch.einsum('ijk, jl -> ikl', D, ctx.W)                         # (K, chan, NT)
        iU = main_chans(ctx, D)
        chs = ctx.iCC[iU[ids]]                                               # (n, nC)
        tw = torch.arange(-(NT // 2), NT // 2 + 1)
        idx = (tt[:, None] + tw)[:, None, :] + (chs * NTb)[:, :, None]
        wv = E[ids.unsqueeze(-1), chs]                                       # (n, nC, NT)
        Xres.view(-1).index_add_(0, idx.reshape(-1), -(amps[:, None, None] * wv).reshape(-1))
    keep = (tt >= lo_keep) & (tt < hi_keep)
    ctx.tick('match', t0)
    return tt[keep], ids[keep], amps[keep], Xres


def main_chans(ctx, D):
    return ((D.reshape(D.shape[0], ctx.npc, ctx.nchan) ** 2).sum(1)).argmax(1)


def match_features(ctx, Xres, tt, ids, amps, D, iU):
    """Residual + amp * template on the template's nC channels (KS4 xfeat)."""
    t0 = time.perf_counter()
    f, chs = features_at(ctx, Xres, tt, iU[ids])
    Dn = D.reshape(D.shape[0], ctx.npc, ctx.nchan)
    chs_t = torch.from_numpy(chs)
    add = Dn[ids.unsqueeze(-1), :, chs_t] * amps[:, None, None]               # (n, nC, pc)
    ctx.tick('match', t0)
    return f + add.numpy()


# ---------------------------------------------------------------- templates

def mean_templates(ctx, labels, feats, chs, nlab):
    """Per-channel mean of features by label: (nlab, pc, chan) and counts of spikes per label."""
    t0 = time.perf_counter()
    n, nc, npc = feats.shape
    idx = (labels[:, None] * ctx.nchan + chs).ravel()
    cnt = np.bincount(idx, minlength=nlab * ctx.nchan).reshape(nlab, ctx.nchan).astype('float64')
    D = np.zeros((nlab, npc, ctx.nchan), 'float32')
    for p in range(npc):
        s = np.bincount(idx, weights=feats[:, :, p].ravel(), minlength=nlab * ctx.nchan)
        D[:, p, :] = (s.reshape(nlab, ctx.nchan) / np.maximum(cnt, 1)).astype('float32')
    # keep the template's own nC channels
    keep = torch.zeros(nlab, ctx.nchan, dtype=torch.bool)
    Dt = torch.from_numpy(D)
    iU = main_chans(ctx, Dt)
    keep[torch.arange(nlab)[:, None], ctx.iCC[iU]] = True
    Dt = Dt * keep[:, None, :]
    ctx.tick('templates', t0)
    return Dt, np.bincount(labels, minlength=nlab)


def recenter(ctx, D):
    keep = torch.zeros(D.shape[0], ctx.nchan, dtype=torch.bool)
    iU = main_chans(ctx, D)
    keep[torch.arange(D.shape[0])[:, None], ctx.iCC[iU]] = True
    return D * keep[:, None, :]


def align(ctx, D):
    """align_U: roll so the best universal response sits at the window centre."""
    if D.shape[0] == 0:
        return D
    K = D.shape[0]
    Uex = torch.einsum('xpc, pt -> xtc', D, ctx.W)          # (K, NT, chan)
    X = Uex.reshape(-1, ctx.nchan).T
    X = conv1d(X.unsqueeze(1), ctx.wT.unsqueeze(1), padding=NT // 2)
    Xmax = X.abs().max(0)[0].max(0)[0].reshape(-1, NT)
    imax = torch.argmax(Xmax, 1)
    Unew = Uex.clone()
    for j in range(NT):
        ix = imax == j
        if ix.any():
            Unew[ix] = torch.roll(Unew[ix], NT // 2 - j, -2)
    return torch.einsum('xtc, pt -> xpc', Unew, ctx.W)


def similarity(ctx, D):
    """Max-lag normalised cross-correlation between templates (CCG.similarity)."""
    mu = (D ** 2).sum((1, 2), keepdim=True) ** .5
    Wn = D / (1e-6 + mu)
    UtU = torch.einsum('ikl, jml -> ijkm', Wn, Wn)
    cc = torch.einsum('ijkm, kml -> ijl', UtU, ctx.WtW)
    return cc.max(-1)[0].numpy(), mu.squeeze().reshape(-1).numpy()


def merge_shape(ctx, D, ns, rho):
    """merging_function mode 'mu': correlation > rho and norms within 20%, biggest first, weighted mean."""
    D = D.clone()
    ns = ns.astype('float64').copy()
    for _ in range(8):
        K = D.shape[0]
        if K < 2:
            break
        cc, mu = similarity(ctx, D)
        np.fill_diagonal(cc, 0)
        gone = np.zeros(K, bool)
        n_merge = 0
        for kk in np.argsort(-ns):
            if gone[kk]:
                continue
            for jj in np.argsort(-cc[kk]):
                if cc[kk, jj] < rho:
                    break
                if gone[jj] or jj == kk:
                    continue
                dmu = 2 * (mu[kk] - mu[jj]) / (mu[kk] + mu[jj] + 1e-9)
                if abs(dmu) < 0.2:
                    wk = ns[kk] / (ns[kk] + ns[jj] + 1e-9)
                    D[kk] = wk * D[kk] + (1 - wk) * D[jj]
                    ns[kk] += ns[jj]; ns[jj] = 0
                    gone[jj] = True; n_merge += 1
                    break
        D = D[torch.from_numpy(~gone)]
        ns = ns[~gone]
        if n_merge == 0:
            break
    return D, ns


# ---------------------------------------------------------------- clustering

def bimod_score(x, n):
    """KS4 swarmsplitter's bimodality on a projection scaled so the two groups sit at about -1, +1, with a KDE
    width that follows n (the KS4 histogram smoothing assumes many spikes)."""
    h = max(0.06, 1.06 * np.std(x[np.abs(x) < 3] if (np.abs(x) < 3).any() else x) * n ** (-0.2) * 0.5)
    edges = np.linspace(-2.5, 2.5, 251)
    c = np.histogram(x, edges)[0].astype(float)
    grid = (edges[:-1] + edges[1:]) / 2
    sd = h / (edges[1] - edges[0])
    r = int(4 * sd) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sd) ** 2)
    y = np.convolve(c, k / k.sum(), mode='same')
    mid = (grid > -0.6) & (grid < 0.6)
    i0 = np.flatnonzero(mid)
    imin = i0[np.argmin(y[i0])]
    xmin = y[imin]
    xm1 = y[:imin + 1].max(); xm2 = y[imin:].max()
    if xm1 <= 0 or xm2 <= 0:
        return 0.0
    return 1 - max(xmin / xm1, xmin / xm2)


def cluster_group(F, rng, K_init, bim, n_min):
    """Top-down splits along a Ward tree over k-means centroids. Returns labels 0..L-1 in this group."""
    n = len(F)
    if n < 2 * n_min:
        return np.zeros(n, 'int64')
    k = min(K_init, max(2, n // max(4, n_min // 2)))
    F2 = F - F.mean(0)
    u, s, vt = np.linalg.svd(F2[rng.choice(n, min(n, 3000), replace=False)], full_matrices=False)
    Z = F2 @ vt[:8].T
    km = KMeans(n_clusters=k, n_init=1, max_iter=12, random_state=int(rng.integers(1 << 31)), init='k-means++')
    lab0 = km.fit_predict(Z)
    cen = km.cluster_centers_
    w = np.bincount(lab0, minlength=k)
    # Ward tree on centroids, repeated by weight is overkill; plain Ward
    if k == 2:
        L = np.array([[0, 1, 0, 2.]])
    else:
        L = linkage(cen, 'ward')
    # children lists
    members = {i: [i] for i in range(k)}
    for i, row in enumerate(L):
        members[k + i] = members[int(row[0])] + members[int(row[1])]
    children = {k + i: (int(r[0]), int(r[1])) for i, r in enumerate(L)}

    out = -np.ones(n, 'int64')
    next_id = [0]

    def rec(node):
        if node not in children:
            m = np.isin(lab0, members[node])
            if m.sum() >= n_min:
                out[m] = next_id[0]; next_id[0] += 1
            return
        a, b = children[node]
        ma = np.isin(lab0, members[a]); mb = np.isin(lab0, members[b])
        na, nb = ma.sum(), mb.sum()
        if na < n_min and nb < n_min:
            return
        if na < n_min:      # outlier branch: leave it unassigned and go on
            return rec(b)
        if nb < n_min:
            return rec(a)
        Za, Zb = Z[ma], Z[mb]
        mu_d = Za.mean(0) - Zb.mean(0)
        Sw = (np.cov(Za.T) * (na - 1) + np.cov(Zb.T) * (nb - 1)) / (na + nb - 2) + 1e-3 * np.eye(Z.shape[1])
        w_ = np.linalg.solve(Sw, mu_d)
        mid = (Za.mean(0) + Zb.mean(0)) / 2
        pr = (Z[ma | mb] - mid) @ w_
        half = (mu_d @ w_) / 2 + 1e-9
        x = pr / half
        if bimod_score(x, na + nb) > bim:
            rec(a); rec(b)
        else:
            m = ma | mb
            out[m] = next_id[0]; next_id[0] += 1

    rec(int(L.shape[0] + k - 1))
    if next_id[0] == 0:
        return np.zeros(n, 'int64')
    # unassigned outliers go to the nearest cluster
    un = out < 0
    if un.any():
        cents = np.stack([Z[out == c].mean(0) for c in range(next_id[0])])
        out[un] = ((Z[un, None] - cents[None]) ** 2).sum(-1).argmin(1)
    return out


def cluster_all(ctx, feats, chs, group_ch, rng, K_init, bim, n_min):
    """Group by depth row of group_ch, cluster each group on the union of channels. Returns labels."""
    t0 = time.perf_counter()
    rows = np.round(ctx.yc[group_ch] / 20.0).astype(int)
    labels = -np.ones(len(rows), 'int64')
    nxt = 0
    for r in np.unique(rows):
        idx = np.flatnonzero(rows == r)
        Fg = np.zeros((len(idx), ctx.nchan, ctx.npc), 'float32')
        Fg[np.arange(len(idx))[:, None], chs[idx]] = feats[idx]
        use = np.abs(Fg).sum((0, 2)) > 0
        Fg = Fg[:, use].reshape(len(idx), -1)
        lab = cluster_group(Fg, rng, K_init, bim, n_min)
        labels[idx] = lab + nxt
        nxt += lab.max() + 1
    ctx.tick('cluster', t0)
    return labels, nxt


def build_dictionary(ctx, labels, nlab, feats, chs, n_min, rho1):
    """Cluster means → drop small → align → shape merge (postprocess_templates)."""
    D, ns = mean_templates(ctx, labels, feats, chs, nlab)
    keep = ns >= n_min
    D = D[torch.from_numpy(keep)]; ns = ns[keep]
    D = align(ctx, D)
    D, ns = merge_shape(ctx, D, ns, rho1)
    return recenter(ctx, D), ns


def refractory_ratio(ta, tb, fs=FS, win=0.0015, T=None):
    """Coincidences of two trains within +-win over the Poisson expectation."""
    lo = np.searchsorted(tb, ta - win * fs); hi = np.searchsorted(tb, ta + win * fs, 'right')
    c = (hi - lo).sum()
    e = len(ta) * len(tb) * 2 * win * fs / max(T, 1)
    return c, e


def ccg_merge(ctx, t, lab, D, T_samples, rho=0.9):
    """Final merge: template correlation above rho and a refractory dip in the CCG. D rows match labels."""
    t0 = time.perf_counter()
    K = D.shape[0]
    if K < 2:
        return lab
    cc, _ = similarity(ctx, D)
    np.fill_diagonal(cc, 0)
    lab = lab.copy()
    root = np.arange(K)

    def find(a):
        while root[a] != a:
            a = root[a]
        return a

    tt = {k: np.sort(t[lab == k]) for k in range(K)}
    pairs = np.argwhere(np.triu(cc > rho))
    pairs = pairs[np.argsort(-cc[pairs[:, 0], pairs[:, 1]])]
    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        ta, tb = tt[ra], tt[rb]
        c, e = refractory_ratio(ta, tb, T=T_samples)
        if e >= 5 and c / e < 0.25:
            root[rb] = ra
            tt[ra] = np.sort(np.concatenate((ta, tb)))
    rt = np.array([find(k) for k in range(K)])
    ctx.tick('ccg', t0)
    return rt[lab]


def mean_templates_raw(ctx, labels, feats, chs, nlab):
    """Per-channel mean of features by label without re-restricting channels (for running updates)."""
    n, nc, npc = feats.shape
    idx = (labels[:, None] * ctx.nchan + chs).ravel()
    cnt = np.bincount(idx, minlength=nlab * ctx.nchan).reshape(nlab, ctx.nchan).astype('float64')
    D = np.zeros((nlab, npc, ctx.nchan), 'float32')
    for p in range(npc):
        s = np.bincount(idx, weights=feats[:, :, p].ravel(), minlength=nlab * ctx.nchan)
        D[:, p, :] = (s.reshape(nlab, ctx.nchan) / np.maximum(cnt, 1)).astype('float32')
    return torch.from_numpy(D), np.bincount(labels, minlength=nlab)
