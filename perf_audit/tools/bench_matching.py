"""Prototype: run_matching that recomputes Cf only on the columns each peel changed.
Checks outputs are bit-identical to KS4's run_matching and times both."""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import conv1d, max_pool1d

run = Path(sys.argv[1])
sys.path.insert(0, r'C:\code\github\Kilosort')
from kilosort import io, template_matching  # noqa: E402


def run_matching_inc(ops, X, U, ctc, device=torch.device('cuda')):
    Th = ops['Th_learned']
    nt = ops['nt']
    max_peels = ops['max_peels']
    W = ops['wPCA'].contiguous()
    nm = (U**2).sum(-1).sum(-1)

    B = conv1d(X.unsqueeze(1), W.unsqueeze(1), padding=nt//2)
    B = torch.einsum('ijk, kjl -> il', U, B)
    NT = B.shape[1]

    trange = torch.arange(-nt, nt+1, device=device)
    tiwave = torch.arange(-(nt//2), nt//2+1, device=device)
    st = torch.zeros((100000, 2), dtype=torch.int64, device=device)
    amps = torch.zeros((100000, 1), dtype=torch.float, device=device)
    th_amps = torch.zeros((100000, 1), dtype=torch.float, device=device)
    k = 0
    Xres = X.clone()

    # full computation once; afterwards only on columns that B changed in
    Cf = torch.relu(B)**2 / nm.unsqueeze(-1)
    Cf[:, :nt] = 0
    Cf[:, -nt:] = 0
    Cfmax, imax = torch.max(Cf, 0)
    del Cf
    edge = torch.zeros(NT, dtype=torch.bool, device=device)
    edge[:nt] = True
    edge[-nt:] = True

    for t in range(max_peels):
        Cmax = max_pool1d(Cfmax.unsqueeze(0).unsqueeze(0), (2*nt+1), stride=1, padding=(nt))
        cnd1 = Cmax[0, 0] > Th**2
        cnd2 = torch.abs(Cmax[0, 0] - Cfmax) < 1e-9
        xs = torch.nonzero(cnd1 * cnd2)
        if len(xs) == 0:
            break
        iX = xs[:, :1]
        iY = imax[iX]
        nsp = len(iX)
        st[k:k+nsp, 0] = iX[:, 0]
        st[k:k+nsp, 1] = iY[:, 0]
        amps[k:k+nsp] = B[iY, iX] / nm[iY]
        amp = amps[k:k+nsp]
        th_amps[k:k+nsp] = Cmax[0, 0, iX[:, 0], None]**.5
        k += nsp

        n = 2
        for j in range(n):
            Xres[:, iX[j::n] + tiwave] -= amp[j::n] * torch.einsum('ijk, jl -> kil', U[iY[j::n, 0]], W)
            B[:, iX[j::n] + trange] -= amp[j::n] * ctc[:, iY[j::n, 0], :]

        cols = torch.unique(iX + trange)           # every column B changed in
        Cs = torch.relu(B[:, cols])**2 / nm.unsqueeze(-1)
        Cs[:, edge[cols]] = 0
        m, i = torch.max(Cs, 0)
        Cfmax[cols] = m
        imax[cols] = i

    return st[:k], amps[:k], th_amps[:k], Xres


import ctypes
_lib = ctypes.CDLL(str(Path(__file__).resolve().parents[1] / 'fused_match.dll'))
_vp, _ci = ctypes.c_void_p, ctypes.c_int
_lib.fused_bupdate.argtypes = [_vp] * 5 + [_ci] * 4
_lib.fused_colmax.argtypes = [_vp] * 3 + [_ci] * 4 + [_vp] * 2


def run_matching_fused(ops, X, U, ctc, device=torch.device('cuda')):
    Th = ops['Th_learned']
    nt = ops['nt']
    max_peels = ops['max_peels']
    W = ops['wPCA'].contiguous()
    nm = (U**2).sum(-1).sum(-1).contiguous()
    ctc = ctc.contiguous()
    NU = U.shape[0]

    B = conv1d(X.unsqueeze(1), W.unsqueeze(1), padding=nt//2)
    B = torch.einsum('ijk, kjl -> il', U, B).contiguous()
    NT = B.shape[1]

    trange = torch.arange(-nt, nt+1, device=device)
    tiwave = torch.arange(-(nt//2), nt//2+1, device=device)
    st = torch.zeros((100000, 2), dtype=torch.int64, device=device)
    amps = torch.zeros((100000, 1), dtype=torch.float, device=device)
    th_amps = torch.zeros((100000, 1), dtype=torch.float, device=device)
    k = 0
    Xres = X.clone()

    Cf = torch.relu(B)**2 / nm.unsqueeze(-1)
    Cf[:, :nt] = 0
    Cf[:, -nt:] = 0
    Cfmax, imax = torch.max(Cf, 0)
    del Cf

    for t in range(max_peels):
        Cmax = max_pool1d(Cfmax.unsqueeze(0).unsqueeze(0), (2*nt+1), stride=1, padding=(nt))
        cnd1 = Cmax[0, 0] > Th**2
        cnd2 = torch.abs(Cmax[0, 0] - Cfmax) < 1e-9
        xs = torch.nonzero(cnd1 * cnd2)
        if len(xs) == 0:
            break
        iX = xs[:, :1]
        iY = imax[iX]
        nsp = len(iX)
        st[k:k+nsp, 0] = iX[:, 0]
        st[k:k+nsp, 1] = iY[:, 0]
        amps[k:k+nsp] = B[iY, iX] / nm[iY]
        amp = amps[k:k+nsp]
        th_amps[k:k+nsp] = Cmax[0, 0, iX[:, 0], None]**.5
        k += nsp

        n = 2
        for j in range(n):
            Xres[:, iX[j::n] + tiwave] -= amp[j::n] * torch.einsum('ijk, jl -> kil', U[iY[j::n, 0]], W)
            ix_j = iX[j::n, 0].contiguous(); iy_j = iY[j::n, 0].contiguous(); a_j = amp[j::n, 0].contiguous()
            err = _lib.fused_bupdate(B.data_ptr(), ctc.data_ptr(), ix_j.data_ptr(), iy_j.data_ptr(),
                                     a_j.data_ptr(), len(ix_j), NU, NT, nt)
            assert err == 0, err

        cols = torch.unique(iX + trange)
        err = _lib.fused_colmax(B.data_ptr(), nm.data_ptr(), cols.data_ptr(), len(cols), NU, NT, nt,
                                Cfmax.data_ptr(), imax.data_ptr())
        assert err == 0, err

    return st[:k], amps[:k], th_amps[:k], Xres


dev = torch.device('cuda')
ops = io.load_ops(run / 'ops.npy', device=dev)
bfile = io.bfile_from_ops(ops, device=dev)
U = torch.load(run / 'snapshot' / 'U_first.pt').to(dev)
template_matching.prepare_extract(ops['xc'], ops['yc'], U, ops['settings']['nearest_chans'],
                                  ops['settings']['position_limit'], device=dev)
ctc = template_matching.prepare_matching(ops, U)

FUSED = len(sys.argv) > 3 and sys.argv[3] == 'fused'
batches = np.linspace(1, bfile.n_batches - 2, int(sys.argv[2]) if len(sys.argv) > 2 else 30).astype(int)
t_old, t_new, same = [], [], []
for ib in batches:
    X = bfile.padded_batch_to_torch(int(ib), ops)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    a = template_matching.run_matching(ops, X, U, ctc, device=dev)
    torch.cuda.synchronize(); t1 = time.perf_counter()
    b = (run_matching_fused if FUSED else run_matching_inc)(ops, X, U, ctc, device=dev)
    torch.cuda.synchronize(); t2 = time.perf_counter()
    t_old.append(t1 - t0); t_new.append(t2 - t1)
    same.append(all(x.shape == y.shape and torch.equal(x, y) for x, y in zip(a, b)))
    if not same[-1]:
        print(f'batch {ib}: spikes {len(a[0])} vs {len(b[0])}; ',
              [(tuple(x.shape), (x.float() - y.float()).abs().max().item() if x.shape == y.shape else None)
               for x, y in zip(a, b)])
print(f'{len(batches)} batches  KS4 {1e3*np.median(t_old):.0f} ms  {"fused" if FUSED else "incremental"} {1e3*np.median(t_new):.0f} ms  '
      f'speedup x{np.median(t_old)/np.median(t_new):.2f}  bit-identical {sum(same)}/{len(same)}')
