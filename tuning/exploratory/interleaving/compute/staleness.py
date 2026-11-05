"""What a stale ctc costs. A reference path evolves the dictionary batch by batch on 120 consecutive batches of kobe's
20-min excerpt (running mean of each template towards its spikes' features, momentum m spikes, as KS2.5's
updateTemplates, ctc refreshed every batch). Then test batches are rematched with the same U_t but the ctc of U_(t-R),
for R up to 100 batches, and compared with the fresh match: spikes, spikes matched (same template, +-5 samples), extra
ones, repeated-template ghosts (same template within nt samples of a shared spike), and residual power. Also the size
of the template change per batch, the running-mean's cost, and the self-subtraction error. Output: staleness.json."""
from common import *

ops, bfile, folder = load_sort()
U0 = templates_from_wall(ops, SORTS / 'kobe_repeat_s1/round3/Wall.npy')
nt = ops['nt']; nC = ops['settings']['nearest_chans']
iCC, _, iU, _ = tm.prepare_extract(ops['xc'], ops['yc'], U0, nC, ops['settings']['position_limit'], device=DEVICE)
tiwave = torch.arange(-(nt//2), nt//2+1, device=DEVICE)
n = U0.shape[0]; npc = U0.shape[1]
chans = iCC[:, iU]                       # (nC, n)
mask = ((U0**2).sum(1) > 0)
mask[torch.arange(n, device=DEVICE).repeat(nC), chans.flatten()] = True
midx = mask.nonzero(as_tuple=True)
FIRST, NB = 300, 120
batches = list(range(FIRST, FIRST + NB))


def feats(X, U, ctc):
    stt, amps, th, Xres = tm.run_matching(ops, X, U, ctc, device=DEVICE)
    xfeat = Xres[iCC[:, iU[stt[:, 1:2]]], stt[:, :1] + tiwave] @ ops['wPCA'].T
    Ucc = U[torch.arange(n, device=DEVICE), :, chans]        # (nC, n, npc) with this indexing order
    xfeat += amps * Ucc[:, stt[:, 1]]
    return stt, amps, th, Xres, xfeat


def run_path(m):
    p = np.exp(-1 / m)
    U = U0.clone()
    path = [U[:, :, :].permute(0, 2, 1)[midx[0], midx[1]].cpu().clone()]
    tupd, tctc, stats, scores = [], [], [], []
    ctc = tm.prepare_matching(ops, U)
    for t, ib in enumerate(batches):
        X = bfile.padded_batch_to_torch(ib, ops)
        stt, amps, th, Xres, xfeat = feats(X, U, ctc)
        t0 = now()
        y = stt[:, 1]
        cnt = torch.bincount(y, minlength=n).float()
        s = torch.zeros((n, nC, npc), device=DEVICE)
        s.index_add_(0, y, xfeat.transpose(0, 1))
        hit = cnt > 0
        lam = 1 - p ** cnt
        mean = s[hit] / cnt[hit, None, None]
        Ucc = U[torch.arange(n, device=DEVICE), :, chans].transpose(0, 1).clone()      # (n, nC, npc) ?
        Ucc[hit] = (1 - lam[hit, None, None]) * Ucc[hit] + lam[hit, None, None] * mean
        U = U.clone()
        U[torch.arange(n, device=DEVICE), :, chans] = Ucc.transpose(0, 1)
        t1 = now()
        ctc = tm.prepare_matching(ops, U)
        t2 = now()
        tupd.append(t1 - t0); tctc.append(t2 - t1)
        path.append(U.permute(0, 2, 1)[midx[0], midx[1]].cpu().clone())
        scores.append(th[:, 0].cpu().numpy())
        del X, Xres, xfeat
    return path, np.array(tupd), np.array(tctc), np.concatenate(scores)


def unpack(v):
    U = torch.zeros((n, 374, npc), device=DEVICE)
    U[midx[0], midx[1]] = v.to(DEVICE)
    return U.permute(0, 2, 1).contiguous()


def match(a, b, tol=5):
    """Spikes of a (time, template) with a spike of the same template within tol samples in b."""
    ka = a[:, 1] * 100000 + a[:, 0]; kb = np.sort(b[:, 1] * 100000 + b[:, 0])
    i = np.searchsorted(kb, ka - tol)
    i = np.minimum(i, len(kb) - 1)
    return (kb[i] >= ka - tol) & (kb[i] <= ka + tol)


res = {}
for m in (100, 400):
    t0 = now()
    path, tupd, tctc, scores = run_path(m)
    tpath = now() - t0
    norm0 = (path[0]**2).sum()**.5
    dU = {R: float(((path[R] - path[0])**2).sum()**.5 / norm0) for R in (1, 5, 10, 20, 50, 100, 120)}
    # per-template relative change, and the self-subtraction error ctc_stale[y,y,0] / nm_fresh - 1
    per = {}
    for R in (1, 10, 50, 100):
        Ua, Ub = unpack(path[-1 - R]), unpack(path[-1])
        rel = ((Ub - Ua)**2).sum((1, 2))**.5 / (Ua**2).sum((1, 2))**.5
        # zero-lag autocorrelation stale template against fresh norm: the two norms
        e = ((Ua**2).sum((1, 2)) / (Ub**2).sum((1, 2)) - 1).abs()
        per[R] = {'rel_median': float(rel.median()), 'rel_p95': float(rel.quantile(.95)), 'self_err_median': float(e.median()), 'self_err_p95': float(e.quantile(.95))}
    out = {}
    for R in (1, 5, 10, 20, 50, 100):
        T = {k: [] for k in ('n_fresh', 'n_stale', 'lost', 'extra', 'ghost', 'res_ratio')}
        for t in range(max(R, 1), NB, max(4, (NB - R) // 12)):
            ib = batches[t]
            X = bfile.padded_batch_to_torch(ib, ops)
            U = unpack(path[t])
            f = tm.run_matching(ops, X, U, tm.prepare_matching(ops, U), device=DEVICE)
            s = tm.run_matching(ops, X, U, tm.prepare_matching(ops, unpack(path[t - R])), device=DEVICE)
            a, b = f[0].cpu().numpy().astype(np.int64), s[0].cpu().numpy().astype(np.int64)
            T['n_fresh'].append(len(a)); T['n_stale'].append(len(b))
            T['lost'].append(int((~match(a, b)).sum())); ex = ~match(b, a); T['extra'].append(int(ex.sum()))
            # ghosts: an extra spike with a same-template spike of the fresh match within nt samples
            T['ghost'].append(int((match(b[ex], a, tol=nt)).sum()))
            T['res_ratio'].append(float((s[3]**2).sum() / (f[3]**2).sum()))
            del X, U, f, s
        out[R] = {k: float(np.mean(v)) for k, v in T.items()}
        print(m, R, out[R], flush=True)
    res[m] = {'U_rel_change_vs_start': dU, 'per_template': per, 'path_s_per_batch': tpath / NB, 'update_ms': float(np.median(tupd) * 1e3), 'ctc_ms': float(np.median(tctc) * 1e3),
              'score_pct': {str(q): float(np.percentile(scores, q)) for q in (50, 90, 99, 99.9)}, 'stale': out}
    print(m, json.dumps({k: v for k, v in res[m].items() if k != 'stale'}), flush=True)
    del path
    torch.cuda.empty_cache()
save_json('staleness.json', res)
