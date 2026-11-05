"""Is the per-template sum of an online update reproducible on the GPU? torch lists index_add_ on CUDA as
non-deterministic: repeated sums over a real batch's spikes, compared bit for bit, against a sorted cumulative sum and
a CPU sum. Output: update_determinism.json."""
from common import *

ops, bfile, folder = load_sort()
U = templates_from_wall(ops, SORTS / 'kobe_repeat_s1/round3/Wall.npy')
nt = ops['nt']; nC = 10; n, npc = U.shape[0], U.shape[1]
iCC, _, iU, Ucc = tm.prepare_extract(ops['xc'], ops['yc'], U, nC, ops['settings']['position_limit'], device=DEVICE)
ctc = tm.prepare_matching(ops, U)
tiwave = torch.arange(-(nt//2), nt//2+1, device=DEVICE)
X = bfile.padded_batch_to_torch(300, ops)
stt, amps, th, Xres = tm.run_matching(ops, X, U, ctc, device=DEVICE)
xfeat = (Xres[iCC[:, iU[stt[:, 1:2]]], stt[:, :1] + tiwave] @ ops['wPCA'].T + amps * Ucc[:, stt[:, 1]]).transpose(0, 1)
y = stt[:, 1]


def ia():
    s = torch.zeros((n, nC, npc), device=DEVICE); s.index_add_(0, y, xfeat); return s


def sorted_sum():
    o = torch.sort(y, stable=True)[1]
    yy, xx = y[o], xfeat[o].double()
    c = torch.cat((torch.zeros_like(xx[:1]), torch.cumsum(xx, 0)))
    last = torch.searchsorted(yy, torch.arange(n, device=DEVICE), right=True)
    first = torch.searchsorted(yy, torch.arange(n, device=DEVICE))
    return (c[last] - c[first]).float()


t = {}
res = {'spikes': len(y)}
for name, f in (('index_add_gpu', ia), ('sorted_cumsum_gpu', sorted_sum)):
    r = [f() for _ in range(30)]
    t0 = now(); f(); t[name] = now() - t0
    res[name] = {'distinct_of_30': len({x.cpu().numpy().tobytes() for x in r}), 'ms': t[name] * 1e3, 'max_rel_diff': float(max(((a - r[0]).abs().max() / r[0].abs().max()).cpu() for a in r))}
print(res)
save_json('update_determinism.json', res)
