"""Cost of the template cross-correlations with a changing dictionary, on the real dictionary (cached first-pass
clusters of kobe H8 seed 1, aligned; random subsets for smaller n): stock prepare_matching; recomputing k templates'
rows and columns (stock dense einsum, and restricted to the channels and templates that can overlap); writing them
into the sparse form in place; number of templates firing per batch. Output: ctc_bench.json."""
from common import *
from torch.nn.functional import conv1d

ops, bfile, folder = load_sort()
c = torch.load(next(CACHE.glob('B_f42d66d3d2b13925*')) / 'cluster1.pt', weights_only=False)
_, Wall1 = c['out']
Uall, _ = tm.align_U(Wall1[(Wall1**2).sum((1, 2)) > 0], ops, device=DEVICE)
Uall = Uall.contiguous()
nt = ops['nt']
W = ops['wPCA'].contiguous()
WtW = torch.flip(conv1d(W.reshape(-1, 1, nt), W.reshape(-1, 1, nt), padding=nt), [2, ])
torch.manual_seed(0)
perm = torch.randperm(Uall.shape[0], device=DEVICE)


def med(f, rep=7):
    ts = []
    for _ in range(rep):
        t = now(); r = f(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    return float(np.median(ts)), float(np.min(ts)), float(np.max(ts)), r


def dense_block(U, S):
    """ctc[S, all, :] as prepare_matching computes it."""
    UtU = torch.einsum('ikl, jml -> ijkm', U[S], U)
    return torch.einsum('ijkm, kml -> ijl', UtU, WtW)


def local_block(U, S, supp, ov):
    """Same, only over the channels of S and the templates that overlap S."""
    chans = supp[S].any(0).nonzero()[:, 0]
    J = ov[S].any(0).nonzero()[:, 0]
    Ua, Ub = U[S][:, :, chans], U[J][:, :, chans]
    UtU = torch.einsum('ikl, jml -> ijkm', Ua, Ub)
    return J, torch.einsum('ijkm, kml -> ijl', UtU, WtW)


res = {'nt': nt, 'sizes': {}}
for n in [250, 500, 1000, 1767, 2504, 3933]:
    idx = perm[:n] if n < Uall.shape[0] else perm
    U = Uall[idx].contiguous()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    tm_med, tm_min, tm_max, ctc = med(lambda: tm.prepare_matching(ops, U))
    peak = (torch.cuda.max_memory_allocated() - base) / 2**20
    nnz = len(ctc['rows'])
    bytes_ctc = sum(ctc[k].nbytes for k in ('ptr', 'rows', 'vals'))
    supp = (U**2).sum(1) > 0
    ov = (supp.float() @ supp.float().T) > 0
    r = {'n': n, 'prepare_matching_ms': tm_med * 1e3, 'min_ms': tm_min * 1e3, 'max_ms': tm_max * 1e3, 'nnz': nnz,
         'density': nnz / n**2, 'kmax': ctc['kmax'], 'mean_col': nnz / n, 'ctc_MB': bytes_ctc / 2**20, 'dense_MB': n**2 * (2*nt+1) * 4 / 2**20,
         'peak_MB': peak, 'support_chans_median': float(supp.sum(1).float().median()), 'overlap_nnz': int(ov.sum())}
    # the sparse pattern is the channel-overlap pattern?
    # update of k templates
    pos = torch.full((n, n), -1, dtype=torch.int64, device=DEVICE)
    ys = torch.repeat_interleave(torch.arange(n, device=DEVICE), ctc['ptr'][1:] - ctc['ptr'][:-1])
    pos[ctc['rows'], ys] = torch.arange(nnz, device=DEVICE)
    vals = ctc['vals'].clone()
    ups = {}
    for k in [1, 10, 50, 100, 300, 1000, n]:
        if k > n:
            continue
        S = torch.randperm(n, device=DEVICE)[:k]

        def stock():
            out = []
            for a in range(0, k, tm.CTC_BLOCK):
                s = S[a:a+tm.CTC_BLOCK]
                b = dense_block(U, s)
                u, y = torch.nonzero((b != 0).any(-1), as_tuple=True)
                e = pos[s[u], y]
                vals[e] = b[u, y]
                e2 = pos[y, s[u]]
                vals[e2] = torch.flip(b[u, y], [1])
            return vals

        def local():
            # blocks of spatially close templates, so that the channel restriction helps
            order = S[torch.argsort(torch.argmax(supp[S].float().cumsum(1), 1))]
            for a in range(0, k, 64):
                s = order[a:a+64]
                J, b = local_block(U, s, supp, ov)
                u, y = torch.nonzero((b != 0).any(-1), as_tuple=True)
                yy = J[y]
                e = pos[s[u], yy]
                vals[e] = b[u, y]
                e2 = pos[yy, s[u]]
                vals[e2] = torch.flip(b[u, y], [1])
            return vals
        a = med(stock, 5)
        b = med(local, 5)
        ups[k] = {'stock_block_ms': a[0] * 1e3, 'local_ms': b[0] * 1e3}
    r['update'] = ups
    # check of the in-place update and of the symmetry
    vals[:] = 0
    S = torch.arange(n, device=DEVICE)
    for a in range(0, n, 64):
        s = S[a:a+64]
        J, b = local_block(U, s, supp, ov)
        u, y = torch.nonzero((b != 0).any(-1), as_tuple=True)
        vals[pos[s[u], J[y]]] = b[u, y]
    r['local_vs_stock_max_rel'] = float(((vals - ctc['vals']).abs().max() / ctc['vals'].abs().max()).cpu())
    # symmetry: ctc[u,y,l] = ctc[y,u,-l]
    e = torch.randint(0, nnz, (min(nnz, 200000),), device=DEVICE)
    u, y = ctc['rows'][e], ys[e]
    r['symmetry_max_rel'] = float((ctc['vals'][e] - torch.flip(ctc['vals'][pos[y, u]], [1])).abs().max().cpu() / ctc['vals'].abs().max().cpu())
    res['sizes'][n] = r
    print(json.dumps(r), flush=True)
    del ctc, pos, vals, ov, supp
    torch.cuda.empty_cache()

# templates firing per batch, on the real spikes
st = np.load(SORTS / 'kobe_repeat_s1/full_st.npy', mmap_mode='r')[:, :2]
b = (st[:, 0] // ops['batch_size']).astype(np.int64)
t = st[:, 1].astype(np.int64)
fire = {}
for R in [1, 10, 100]:
    nb = 600 // R
    un = [len(np.unique(t[(b // R) == i])) for i in range(0, nb, max(1, nb // 40))]
    cnt = [np.bincount(t[(b // R) == i]) for i in range(0, nb, max(1, nb // 40))]
    c10 = [int((x >= 10).sum()) for x in cnt]
    fire[R] = {'templates_with_spike': float(np.median(un)), 'templates_with_10_spikes': float(np.median(c10)), 'n_templates': int(t.max() + 1)}
res['firing'] = fire
print(fire)
save_json('ctc_bench.json', res)
