"""Per-batch cost of learned extraction, split into read, preprocessing, projection, peel and feature steps, for
the round-2 dictionary (2504 templates) and a round-3-sized one, on 60 consecutive batches of kobe's 20-min
excerpt. Sections are synchronised (torch.cuda.synchronize), so the sum runs a little above an unsynchronised
pass, which is timed too. Also times postprocess_templates. Output: batch_breakdown.json."""
from common import *
import common
import torch.nn.functional as F
from kilosort import io as kio
from kilosort.io import BinaryRWFile

ops, bfile, folder = load_sort()
c = torch.load(next(CACHE.glob('B_f42d66d3d2b13925*')) / 'cluster1.pt', weights_only=False)
clu, Wall1 = c['out']
print('cluster1', Wall1.shape, flush=True)
t0 = now()
U1 = tm.postprocess_templates(Wall1, ops, np.asarray(clu), np.zeros((len(clu), 3)), torch.empty((len(clu), 0, 0)), device=DEVICE).contiguous()
t_post = now() - t0
print('postprocess_templates', U1.shape, t_post, flush=True)
U2 = templates_from_wall(ops, SORTS / 'kobe_repeat_s1/round3/Wall.npy')
print('U2', U2.shape, flush=True)

nt = ops['nt']; W = ops['wPCA'].contiguous(); Th = ops['Th_learned']; max_peels = ops['max_peels']
batches = list(range(270, 330))


def setup(U):
    iCC, iCC_mask, iU, Ucc = tm.prepare_extract(ops['xc'], ops['yc'], U, ops['settings']['nearest_chans'],
                                                ops['settings']['position_limit'], device=DEVICE)
    t = now(); ctc = tm.prepare_matching(ops, U); tp = now() - t
    return iCC, iU, Ucc, ctc, tp


def timed_pass(U, batches):
    iCC, iU, Ucc, ctc, tp = setup(U)
    tiwave = torch.arange(-(nt//2), nt//2+1, device=DEVICE)
    keys = ['read', 'filter', 'conv', 'project', 'cf', 'peel', 'feat', 'host']
    T = {k: [] for k in keys}; ns = []
    nm = (U**2).sum(-1).sum(-1).contiguous()
    fast = False
    for ib in batches:
        t = now()
        X = BinaryRWFile.padded_batch_to_torch(bfile, ib); torch.cuda.synchronize(); t1 = time.perf_counter(); T['read'].append(t1 - t)
        X = bfile.filter(X, ops, ib); torch.cuda.synchronize(); t2 = time.perf_counter(); T['filter'].append(t2 - t1)
        B = F.conv1d(X.unsqueeze(1), W.unsqueeze(1), padding=nt//2); torch.cuda.synchronize(); t3 = time.perf_counter(); T['conv'].append(t3 - t2)
        B = torch.einsum('ijk, kjl -> il', U, B); torch.cuda.synchronize(); t4 = time.perf_counter(); T['project'].append(t4 - t3)
        st = torch.zeros((100000, 2), dtype=torch.int64, device=DEVICE); amps = torch.zeros((100000, 1), device=DEVICE); th = torch.zeros((100000, 1), device=DEVICE)
        Xres = X.clone()
        Cf = torch.relu(B)**2 / nm.unsqueeze(-1); Cf[:, :nt] = 0; Cf[:, -nt:] = 0
        Cfmax, imax = torch.max(Cf, 0); del Cf
        torch.cuda.synchronize(); t5 = time.perf_counter(); T['cf'].append(t5 - t4)
        k = tm._peel_fused(B.contiguous(), nm, ctc, U, W, Xres, Cfmax, imax, st, amps, th, Th, nt, max_peels, fast)
        torch.cuda.synchronize(); t6 = time.perf_counter(); T['peel'].append(t6 - t5)
        stt, a = st[:k], amps[:k]
        xfeat = Xres[iCC[:, iU[stt[:, 1:2]]], stt[:, :1] + tiwave] @ ops['wPCA'].T
        xfeat += a * Ucc[:, stt[:, 1]]
        torch.cuda.synchronize(); t7 = time.perf_counter(); T['feat'].append(t7 - t6)
        s = stt.double().cpu().numpy(); tb = xfeat.transpose(0, 1).cpu().numpy().copy()
        t8 = time.perf_counter(); T['host'].append(t8 - t7)
        ns.append(int(k))
        del B, X, Xres, xfeat, stt, st, amps, th
    return T, ns, tp


def plain_pass(U, batches):
    iCC, iU, Ucc, ctc, tp = setup(U)
    tiwave = torch.arange(-(nt//2), nt//2+1, device=DEVICE)
    t0 = now()
    for ib in batches:
        X = bfile.padded_batch_to_torch(ib, ops)
        stt, amps, th, Xres = tm.run_matching(ops, X, U, ctc, device=DEVICE)
        xfeat = Xres[iCC[:, iU[stt[:, 1:2]]], stt[:, :1] + tiwave] @ ops['wPCA'].T
        xfeat += amps * Ucc[:, stt[:, 1]]
        _ = stt.double().cpu().numpy(), xfeat.transpose(0, 1).cpu().numpy().copy()
    return now() - t0


res = {'postprocess_templates_s': t_post, 'n_clusters_in': int(Wall1.shape[0]), 'batches': [batches[0], batches[-1]]}
for name, U in (('U_round2', U1), ('U_round3', U2)):
    timed_pass(U, batches[:5])                       # warm up
    T, ns, tp = timed_pass(U, batches)
    plain = plain_pass(U, batches)
    res[name] = {'n_templates': U.shape[0], 'prepare_matching_s': tp, 'plain_pass_s': plain, 'plain_s_per_batch': plain / len(batches),
                 'spikes_per_batch': float(np.mean(ns)), 'ms_per_batch_median': {k: float(np.median(v) * 1e3) for k, v in T.items()},
                 'ms_per_batch_mean': {k: float(np.mean(v) * 1e3) for k, v in T.items()}}
    print(json.dumps(res[name], indent=1), flush=True)
    torch.cuda.empty_cache()
res['gpu_peak_GB'] = torch.cuda.max_memory_allocated() / 2**30
save_json('batch_breakdown.json', res)
print(res['gpu_peak_GB'])
