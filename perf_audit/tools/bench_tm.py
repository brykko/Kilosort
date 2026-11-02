"""Compare the fused CUDA template_match with KS4's: outputs and speed."""
import ctypes
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import conv1d, max_pool1d

run = Path(sys.argv[1])
order = int(sys.argv[2]) if len(sys.argv) > 2 else 0
sys.path.insert(0, r'C:\code\github\Kilosort')
from kilosort import io, spikedetect  # noqa: E402

lib = ctypes.CDLL(str(Path(__file__).resolve().parents[1] / 'fused_tm.dll'))
vp, ci = ctypes.c_void_p, ctypes.c_int
lib.fused_template_match.argtypes = [vp] * 7 + [ci] * 6 + [ctypes.c_float, ci]


def template_match_fused(X, ops, iC, iC2, weigh, device=torch.device('cuda'), keep=None):
    nt = ops['nt']
    nt0 = ops['settings']['nt0min']
    nk = ops['settings']['n_templates']
    NT = X.shape[-1]
    Nfilt = iC.shape[1]
    W = ops['wTEMP'].unsqueeze(1)
    B = conv1d(X.unsqueeze(1), W, padding=nt//2).contiguous()
    As = torch.empty((Nfilt, NT), device=device)
    imaxs = torch.empty((Nfilt, NT), dtype=torch.int64, device=device)
    Amaxs = torch.empty((Nfilt, NT), device=device)
    iC, iC2, weigh = iC.contiguous(), iC2.contiguous(), weigh.contiguous()
    err = lib.fused_template_match(B.data_ptr(), iC.data_ptr(), weigh.data_ptr(), As.data_ptr(),
                                   imaxs.data_ptr(), iC2.data_ptr(), Amaxs.data_ptr(),
                                   B.shape[1], NT, iC.shape[0], Nfilt, weigh.shape[0], iC2.shape[0],
                                   float(ops['Th_universal']), nt0)
    assert err == 0, err
    if keep is not None:
        keep.update(As=As.clone(), imaxs=imaxs.clone(), Amaxs=Amaxs.clone())
    Amaxs[:, :nt] = 0
    Amaxs[:, -nt:] = 0
    Amaxs = max_pool1d(Amaxs.unsqueeze(0), (2*nt0+1), stride=1, padding=nt0).squeeze(0)
    xy = torch.logical_and(Amaxs == As, As > ops['Th_universal']).nonzero()
    imax = imaxs[xy[:, 0], xy[:, 1]]
    amp = As[xy[:, 0], xy[:, 1]]
    ssign = imax.sign()
    imax = imax.abs() - 1
    adist = B[iC[:, xy[:, 0]], imax % nk, xy[:, 1]] * ssign
    return xy, imax, amp, adist


if __name__ == '__main__':
    dev = torch.device('cuda')
    ops = io.load_ops(run / 'ops.npy', device=dev)
    bfile = io.bfile_from_ops(ops, device=dev)
    batches = np.linspace(1, bfile.n_batches - 2, 30).astype(int)
    t_old, t_new, same, nsp = [], [], [], []
    for ib in batches:
        X = bfile.padded_batch_to_torch(int(ib), ops)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        a = spikedetect.template_match(X, ops, ops['iC'], ops['iC2'], ops['weigh'], device=dev)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        b = template_match_fused(X, ops, ops['iC'], ops['iC2'], ops['weigh'], device=dev)
        torch.cuda.synchronize(); t2 = time.perf_counter()
        t_old.append(t1 - t0); t_new.append(t2 - t1); nsp.append((len(a[0]), len(b[0])))
        same.append(all(x.shape == y.shape and torch.equal(x, y) for x, y in zip(a, b)))
    print(f'order={order}: KS4 {1e3*np.median(t_old):.0f} ms  fused {1e3*np.median(t_new):.1f} ms  '
          f'speedup x{np.median(t_old)/np.median(t_new):.1f}  bit-identical {sum(same)}/{len(same)}')
    if not all(same):
        d = np.array(nsp); print('spikes per batch (KS4, fused), first 5:', d[:5].tolist(),
                                 ' total diff', int(d[:, 1].sum() - d[:, 0].sum()), 'of', int(d[:, 0].sum()))
