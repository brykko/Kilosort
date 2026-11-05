"""How much a fresh match changes when the dictionary changes (the yardstick for staleness.py): the same running-mean
path (m = 400), matching batch t with fresh U_t and its ctc against fresh U_(t-R) and its ctc. Also the size of the
ctc change, and matching repeated with identical inputs (GPU determinism). Output: sensitivity.json."""
from common import *

ops, bfile, folder = load_sort()
U0 = templates_from_wall(ops, SORTS / 'kobe_repeat_s1/round3/Wall.npy')
nt = ops['nt']; nC = ops['settings']['nearest_chans']
iCC, _, iU, _ = tm.prepare_extract(ops['xc'], ops['yc'], U0, nC, ops['settings']['position_limit'], device=DEVICE)
tiwave = torch.arange(-(nt//2), nt//2+1, device=DEVICE)
n, npc = U0.shape[0], U0.shape[1]
chans = iCC[:, iU]; ar = torch.arange(n, device=DEVICE)
FIRST, NB = 300, 120; p = np.exp(-1 / 400)
U = U0.clone(); path = [U.cpu()]
ctc = tm.prepare_matching(ops, U)
for t in range(NB):
    X = bfile.padded_batch_to_torch(FIRST + t, ops)
    stt, amps, th, Xres = tm.run_matching(ops, X, U, ctc, device=DEVICE)
    xfeat = Xres[iCC[:, iU[stt[:, 1:2]]], stt[:, :1] + tiwave] @ ops['wPCA'].T
    Ucc = U[ar, :, chans].transpose(0, 1).clone()   # (n, nC, npc): advanced indices first, then the slice
    xfeat = xfeat + amps * Ucc[stt[:, 1]].transpose(0, 1)
    y = stt[:, 1]; cnt = torch.bincount(y, minlength=n).float()
    s = torch.zeros((n, nC, npc), device=DEVICE); s.index_add_(0, y, xfeat.transpose(0, 1))
    hit = cnt > 0; lam = 1 - p ** cnt
    Ucc[hit] = (1 - lam[hit, None, None]) * Ucc[hit] + lam[hit, None, None] * s[hit] / cnt[hit, None, None]
    U = U.clone(); U[ar, :, chans] = Ucc.transpose(0, 1)
    ctc = tm.prepare_matching(ops, U)
    path.append(U.cpu())


def match(a, b, tol=5):
    ka = a[:, 1] * 100000 + a[:, 0]; kb = np.sort(b[:, 1] * 100000 + b[:, 0])
    i = np.minimum(np.searchsorted(kb, ka - tol), len(kb) - 1)
    return (kb[i] >= ka - tol) & (kb[i] <= ka + tol)


res = {}
for R in (0, 1, 10, 50, 100):
    L = []
    for t in range(max(R, 1), NB, 10):
        X = bfile.padded_batch_to_torch(FIRST + t, ops)
        Ua, Ub = path[t].to(DEVICE), path[t - R].to(DEVICE)
        f = tm.run_matching(ops, X, Ua, tm.prepare_matching(ops, Ua), device=DEVICE)
        s = tm.run_matching(ops, X, Ub, tm.prepare_matching(ops, Ub), device=DEVICE)
        a, b = f[0].cpu().numpy().astype(np.int64), s[0].cpu().numpy().astype(np.int64)
        L.append((len(a), len(b), int((~match(a, b)).sum()), int((~match(b, a)).sum()), float((s[3]**2).sum() / (f[3]**2).sum())))
    L = np.array(L)
    res[R] = {'n_a': L[:, 0].mean(), 'n_b': L[:, 1].mean(), 'only_a': L[:, 2].mean(), 'only_b': L[:, 3].mean(), 'res_ratio': L[:, 4].mean()}
    print(R, res[R], flush=True)
save_json('sensitivity.json', res)
