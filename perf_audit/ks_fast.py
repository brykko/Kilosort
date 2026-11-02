"""Monkeypatches for KS4 tested in the performance audit. Nothing here edits KS4's files.

apply(['det', 'h2d', 'tm', 'rm', 'noplots']):
  det     - single-threaded KMeans in extract_wPCA_wTEMP (makes wTEMP reproducible)
  h2d     - copy int16 batches to the GPU contiguous and transpose there
  tm      - fused CUDA kernel for spikedetect.template_match
  rm      - incremental peeling with fused CUDA kernels for template_matching.run_matching
  race    - like rm, plus no overlapping writes when peaks tie (fixes a KS4 race)
  clu     - (not recommended: slower) dense cluster-assignment variants
  noplots - skip the four PNG plots
"""
import ctypes
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import conv1d, max_pool1d

_here = Path(__file__).parent
_vp, _ci = ctypes.c_void_p, ctypes.c_int
_tm = ctypes.CDLL(str(_here / 'fused_tm.dll'))
_tm.fused_template_match.argtypes = [_vp] * 7 + [_ci] * 6 + [ctypes.c_float, _ci]
_rm = ctypes.CDLL(str(_here / 'fused_match.dll'))
_rm.fused_bupdate.argtypes = [_vp] * 5 + [_ci] * 4
_rm.fused_colmax.argtypes = [_vp] * 3 + [_ci] * 4 + [_vp] * 2


def template_match_fused(X, ops, iC, iC2, weigh, device=torch.device('cuda')):
    nt = ops['nt']
    nt0 = ops['settings']['nt0min']
    nk = ops['settings']['n_templates']
    NT = X.shape[-1]
    Nfilt = iC.shape[1]
    B = conv1d(X.unsqueeze(1), ops['wTEMP'].unsqueeze(1), padding=nt//2).contiguous()
    As = torch.empty((Nfilt, NT), device=device)
    imaxs = torch.empty((Nfilt, NT), dtype=torch.int64, device=device)
    Amaxs = torch.empty((Nfilt, NT), device=device)
    iC, iC2, weigh = iC.contiguous(), iC2.contiguous(), weigh.contiguous()
    err = _tm.fused_template_match(B.data_ptr(), iC.data_ptr(), weigh.data_ptr(), As.data_ptr(),
                                   imaxs.data_ptr(), iC2.data_ptr(), Amaxs.data_ptr(),
                                   B.shape[1], NT, iC.shape[0], Nfilt, weigh.shape[0], iC2.shape[0],
                                   float(ops['Th_universal']), nt0)
    assert err == 0, err
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


def run_matching_fused(ops, X, U, ctc, device=torch.device('cuda'), race_free=False):
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

    # full computation once; afterwards only on the columns each peel changed
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

        # KS4 updates spikes in two interleaved groups, assuming windows within a group never
        # overlap. Tied peaks within nt samples break that and make the writes race; use more
        # groups for those (rare) peels. n == 2 whenever KS4's assumption holds.
        n = 2
        if race_free:
            while nsp > n and bool((iX[n:, 0] - iX[:-n, 0] <= 2*nt).any()):
                n += 1
        for j in range(n):
            Xres[:, iX[j::n] + tiwave] -= amp[j::n] * torch.einsum('ijk, jl -> kil', U[iY[j::n, 0]], W)
            ix_j = iX[j::n, 0].contiguous()
            iy_j = iY[j::n, 0].contiguous()
            a_j = amp[j::n, 0].contiguous()
            err = _rm.fused_bupdate(B.data_ptr(), ctc.data_ptr(), ix_j.data_ptr(), iy_j.data_ptr(),
                                    a_j.data_ptr(), len(ix_j), NU, NT, nt)
            assert err == 0, err

        cols = torch.unique(iX + trange)
        err = _rm.fused_colmax(B.data_ptr(), nm.data_ptr(), cols.data_ptr(), len(cols), NU, NT, nt,
                               Cfmax.data_ptr(), imax.data_ptr())
        assert err == 0, err

    return st[:k], amps[:k], th_amps[:k], Xres


def kmeans_plusplus_nosync(Xg, niter=200, seed=1, device=torch.device('cuda'), verbose=False):
    """clustering_qr.kmeans_plusplus with the boolean-mask writes replaced by torch.where
    (same values, same RNG calls, no host sync per iteration)."""
    from kilosort.clustering_qr import subsample_idx
    vtot = torch.norm(Xg, 2, dim=1)**2
    n1 = vtot.shape[0]
    if n1 > 2**24:
        n2 = n1 - 2**24
        idx, rev_idx = subsample_idx(n1, n2)
        rev_idx = torch.from_numpy(rev_idx).to(device)
        subsample = True
    else:
        subsample = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    ntry = 100
    n_spikes, n_features = Xg.shape
    mu = torch.zeros((niter, n_features), device=device)
    vexp0 = torch.zeros(n_spikes, device=device)
    iclust = torch.zeros((n_spikes,), dtype=torch.int, device=device)
    for j in range(niter):
        v2 = torch.relu(vtot - vexp0)
        if subsample:
            isamp = rev_idx[torch.multinomial(v2[idx], ntry)]
        else:
            isamp = torch.multinomial(v2, ntry)
        Xc = Xg[isamp]
        vexp = 2 * Xg @ Xc.T - (Xc**2).sum(1)
        dexp = torch.relu(vexp - vexp0.unsqueeze(1))
        vsum = dexp.sum(0)
        imax = torch.argmax(vsum)
        ix = dexp[:, imax] > 0
        iclust = torch.where(ix, torch.tensor(j, dtype=iclust.dtype, device=device), iclust)
        mu[j] = Xc[imax]
        vexp0 = torch.where(ix, vexp[:, imax], vexp0)
        del vexp, dexp
    return iclust


def assign_iclust_dense(rows_neigh, isub, kn, tones2, nclust, lam, m, ki, kj, device=torch.device('cuda')):
    n_spikes = kn.shape[0]
    xN = torch.zeros((n_spikes, int(nclust)), device=device)
    xN.index_put_((rows_neigh.flatten(), isub[kn].flatten()), tones2.flatten(), accumulate=True)
    if lam > 0:
        kN = torch.bincount(isub, minlength=int(nclust)).float().unsqueeze(0)
        xN = xN - lam/m * (ki.unsqueeze(-1) * kN)
    return torch.argmax(xN, 1)


def assign_isub_dense(iclust, kn, tones2, nclust, nsub, lam, m, ki, kj, device=torch.device('cuda')):
    n_neigh = kn.shape[1]
    cols = iclust.unsqueeze(-1).tile((1, n_neigh))
    xS = torch.zeros((nsub, int(nclust)), device=device)
    xS.index_put_((kn.flatten(), cols.flatten()), tones2.flatten(), accumulate=True)
    if lam > 0:
        kN = torch.bincount(iclust, minlength=int(nclust)).float().unsqueeze(0)
        xS = xS - lam / m * (kj.unsqueeze(-1) * kN)
    return torch.argmax(xS, 1)


def apply(patches):
    from kilosort import io, spikedetect, template_matching
    import kilosort.plots as kplots

    if 'det' in patches:
        from threadpoolctl import threadpool_limits
        _orig = spikedetect.extract_wPCA_wTEMP

        def extract_wPCA_wTEMP_det(*a, **k):
            with threadpool_limits(limits=1):
                return _orig(*a, **k)
        spikedetect.extract_wPCA_wTEMP = extract_wPCA_wTEMP_det

    if 'h2d' in patches:
        _orig_pb = io.BinaryRWFile.padded_batch_to_torch

        def padded_batch_to_torch(self, ibatch, return_inds=False):
            if self.dtype == 'uint16' or self.scale is not None or self.shift is not None:
                return _orig_pb(self, ibatch, return_inds=return_inds)
            ibatch *= self.batch_downsampling
            bstart, bend = self._get_batch_edges(ibatch)
            data = torch.from_numpy(np.asarray(self.file[bstart:bend])).to(self.device).T.float()
            nsamp = data.shape[-1]
            X = torch.zeros((self.n_chan_bin, self.NT + 2*self.nt), device=self.device)
            if ibatch == 0:
                X[:, self.nt: self.nt+nsamp] = data
                X[:, :self.nt] = X[:, self.nt: self.nt+1]
                bstart = self.imin - self.nt
            elif ibatch == self.n_batches-1:
                X[:, :nsamp] = data
                X[:, nsamp:] = X[:, nsamp-1:nsamp]
                bend += self.nt
            else:
                X[:] = data
            if return_inds:
                return X, [bstart, bend]
            return X
        io.BinaryRWFile.padded_batch_to_torch = padded_batch_to_torch

    if 'tm' in patches:
        spikedetect.template_match = template_match_fused
    if 'rm' in patches:
        template_matching.run_matching = run_matching_fused
    if 'race' in patches:
        template_matching.run_matching = lambda *a, **k: run_matching_fused(*a, race_free=True, **k)
    if 'clu' in patches:
        from kilosort import clustering_qr
        clustering_qr.kmeans_plusplus = kmeans_plusplus_nosync
        clustering_qr.assign_iclust = assign_iclust_dense
        clustering_qr.assign_isub = assign_isub_dense
    if 'noplots' in patches:
        for n in ['plot_drift_amount', 'plot_drift_scatter', 'plot_diagnostics', 'plot_spike_positions']:
            setattr(kplots, n, lambda *a, **k: None)
