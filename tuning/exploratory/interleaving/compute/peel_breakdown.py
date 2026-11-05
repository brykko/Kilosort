"""Where the peel loop's time goes (a copy of template_matching._peel_fused with a synchronised timer around each
kernel, which inflates the total a little): 30 batches, 1795 and 2504 templates, exact and fast_matching.
Output: peel_breakdown.json."""
from common import *
import torch.nn.functional as F

ops, bfile, folder = load_sort()
c = torch.load(next(CACHE.glob('B_f42d66d3d2b13925*')) / 'cluster1.pt', weights_only=False)
clu, Wall1 = c['out']
U1 = tm.postprocess_templates(Wall1, ops, np.asarray(clu), np.zeros((len(clu), 3)), torch.empty((len(clu), 0, 0)), device=DEVICE).contiguous()
U2 = templates_from_wall(ops, SORTS / 'kobe_repeat_s1/round3/Wall.npy')
nt = ops['nt']; W = ops['wPCA'].contiguous(); Th = ops['Th_learned']; max_peels = ops['max_peels']
ck = cuda_kernels


def peel(B, nm, ctc, U, Xres, Cfmax, imax, st, amps, th, fast, T):
    NT = B.shape[1]
    flags = torch.empty(NT, dtype=torch.uint8, device=DEVICE)
    peaks = torch.empty(NT + 1, dtype=torch.int64, device=DEVICE)
    iYbuf = torch.empty(NT, dtype=torch.int64, device=DEVICE)
    head = 1025
    host = torch.empty(head, dtype=torch.int64, pin_memory=True)
    k = 0; peels = []
    for t in range(max_peels):
        t0 = now()
        ck.matching_peaks(Cfmax, nt, Th**2, flags, peaks); host.copy_(peaks[:head]); nsp = int(host[0])
        t1 = now(); T['peaks'] += t1 - t0
        if nsp == 0:
            break
        iX_host = host[1:1+nsp].numpy() if nsp < head else peaks[1:1+nsp].cpu().numpy()
        peels.append(nsp)
        iX = peaks[1:1+nsp]
        t2 = now()
        ck.matching_record(B, nm, imax, Cfmax, iX, nt, st[k:], amps[k:], th[k:], iYbuf)
        t3 = now(); T['record'] += t3 - t2
        iY = iYbuf[:nsp]; amp = amps[k:k+nsp, 0]
        n = 2
        if nsp > 2:
            close = np.searchsorted(iX_host, iX_host + 2*nt, side='right') - np.arange(nsp) - 1
            n = max(2, min(nsp, int(close.max()) + 1))
        for j in range(n):
            t4 = now()
            if fast:
                ck.matching_xres_fast(Xres, U, W, iX[j::n], iY[j::n], amp[j::n])
            else:
                E = torch.einsum('ijk, jl -> kil', U[iY[j::n]], W).contiguous()
                ck.matching_xres(Xres, E, iX[j::n], amp[j::n])
            t5 = now(); T['xres'] += t5 - t4
            ck.matching_bupdate(B, ctc, iX[j::n], iY[j::n], amp[j::n], nt)
            t6 = now(); T['bupdate'] += t6 - t5
        t7 = now()
        ck.matching_colmax(B, nm, iX, nt, Cfmax, imax)
        t8 = now(); T['colmax'] += t8 - t7
        k += nsp
    return k, peels


res = {}
batches = list(range(300, 330))
for name, U, fast in (('n1795_exact', U2, False), ('n1795_fast', U2, True), ('n2504_exact', U1, False)):
    ctc = tm.prepare_matching(ops, U); nm = (U**2).sum(-1).sum(-1).contiguous()
    T = dict.fromkeys(['peaks', 'record', 'xres', 'bupdate', 'colmax'], 0.0); ks = []; npeel = []; first3 = []
    t00 = now()
    for ib in batches:
        X = bfile.padded_batch_to_torch(ib, ops)
        B = torch.einsum('ijk, kjl -> il', U, F.conv1d(X.unsqueeze(1), W.unsqueeze(1), padding=nt//2))
        st = torch.zeros((100000, 2), dtype=torch.int64, device=DEVICE); amps = torch.zeros((100000, 1), device=DEVICE); th = torch.zeros((100000, 1), device=DEVICE)
        Xres = X.clone()
        Cf = torch.relu(B)**2 / nm.unsqueeze(-1); Cf[:, :nt] = 0; Cf[:, -nt:] = 0
        Cfmax, imax = torch.max(Cf, 0); del Cf
        k, peels = peel(B.contiguous(), nm, ctc, U, Xres, Cfmax, imax, st, amps, th, fast, T)
        ks.append(k); npeel.append(len(peels)); first3.append(sum(peels[:3]) / k)
    tot = now() - t00
    res[name] = {'ms_per_batch': {k: v / len(batches) * 1e3 for k, v in T.items()}, 'wall_ms_per_batch': tot / len(batches) * 1e3,
                 'spikes': float(np.mean(ks)), 'peels': float(np.mean(npeel)), 'peels_hit_cap': float(np.mean(np.array(npeel) >= max_peels)),
                 'spikes_in_first3_peels': float(np.mean(first3))}
    print(name, json.dumps(res[name]), flush=True)
save_json('peel_breakdown.json', res)
