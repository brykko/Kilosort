"""The schedules: R (rounds), Rd (rounds + residual discovery), S (subsampled rounds), I (interleaved)."""
import time
from collections import deque

import numpy as np
import torch

import primitives as pr
from primitives import TSHIFT
from simdata import NT

R_DEFAULT = dict(rho1=0.95, bim=0.5, K_init=12, n_min=12)
M_TRACK = 200
I_DEFAULT = dict(schedule=('spikes', 20, 2000), n_birth=15, rho_merge=0.95, buffer_batches=20, birth_every=5,
                 triage_every=5, n_keep=50, minFR=0.02, nsp_forget=0.8, rho_dup=0.95, K_init=12, bim=0.5, n_min=12,
                 mscale=1.0)


def bpad(ds, b):
    Xb, lo, s, e = ds.batch(b)
    return Xb, lo, s - lo, e - lo


def full_extract(ctx, ds, D, batches, discover=False):
    """Matching over batches. Returns dict of spikes (times in trough convention, global) with features."""
    iU = pr.main_chans(ctx, D)
    ctc = pr.prep_ctc(ctx, D)
    T, I, A, F, C = [], [], [], [], []
    dT, dF, dC, dG = [], [], [], []
    for b in batches:
        Xb, lo, k0, k1 = bpad(ds, b)
        tt, ids, amps, Xres = pr.matching(ctx, Xb, D, ctc, k0 + TSHIFT, k1 + TSHIFT)
        f = pr.match_features(ctx, Xres, tt, ids, amps, D, iU)
        T.append(tt.numpy() + lo - TSHIFT); I.append(ids.numpy()); A.append(amps.numpy()); F.append(f)
        C.append(ctx.iCC[iU[ids]].numpy())
        if discover:
            t2, ch2 = pr.detect(ctx, Xres, k0 + TSHIFT, k1 + TSHIFT)
            f2, c2 = pr.features_at(ctx, Xres, t2, ch2)
            dT.append(t2.numpy() + lo - TSHIFT); dF.append(f2); dC.append(c2); dG.append(ch2.numpy())
    ctx.fpe += len(batches) / ds.n_batches
    out = dict(t=np.concatenate(T), id=np.concatenate(I), amp=np.concatenate(A), feats=np.concatenate(F),
               chs=np.concatenate(C), group=iU[torch.from_numpy(np.concatenate(I))].numpy())
    if discover and dT:
        out['d_t'] = np.concatenate(dT); out['d_f'] = np.concatenate(dF)
        out['d_c'] = np.concatenate(dC); out['d_g'] = np.concatenate(dG)
    return out


def detect_pass(ctx, ds, batches):
    T, F, C, G = [], [], [], []
    for b in batches:
        Xb, lo, k0, k1 = bpad(ds, b)
        tt, ch = pr.detect(ctx, Xb, k0 + TSHIFT, k1 + TSHIFT)
        f, c = pr.features_at(ctx, Xb, tt, ch)
        T.append(tt.numpy() + lo - TSHIFT); F.append(f); C.append(c); G.append(ch.numpy())
    ctx.fpe += len(batches) / ds.n_batches
    return np.concatenate(T), np.concatenate(F), np.concatenate(C), np.concatenate(G)


def c_labels(ctx, ds, spk, par, rng, D_from=None):
    """Global clustering of an extraction + CCG merge → (times, labels). Also the cluster templates."""
    lab, nl = pr.cluster_all(ctx, spk['feats'], spk['chs'], spk['group'], rng, par['K_init'], par['bim'], par['n_min'])
    Dc, ns = pr.mean_templates(ctx, lab, spk['feats'], spk['chs'], nl)
    final = pr.ccg_merge(ctx, spk['t'], lab, Dc, ds.X.shape[1])
    return lab, nl, final


def run_rounds(ctx, ds, par, rng, batches, K, discover=False, checkpoint=None):
    """R / Rd on `batches` for K rounds. checkpoint(k, D, spk, labels) at each round from 2."""
    t, f, c, g = detect_pass(ctx, ds, batches)
    lab, nl = pr.cluster_all(ctx, f, c, g, rng, par['K_init'], par['bim'], par['n_min'])
    D, ns = pr.build_dictionary(ctx, lab, nl, f, c, par['n_min'], par['rho1'])
    if checkpoint:
        checkpoint(1, D, None, None)
    for k in range(2, K + 1):
        if D.shape[0] == 0:
            break
        spk = full_extract(ctx, ds, D, batches, discover=discover)
        if discover and 'd_t' in spk:
            cl = dict(t=np.concatenate((spk['t'], spk['d_t'])),
                      feats=np.concatenate((spk['feats'], spk['d_f'])),
                      chs=np.concatenate((spk['chs'], spk['d_c'])),
                      group=np.concatenate((spk['group'], spk['d_g'])))
        else:
            cl = spk
        lab, nl = pr.cluster_all(ctx, cl['feats'], cl['chs'], cl['group'], rng, par['K_init'], par['bim'],
                                 par['n_min'])
        Dc, _ = pr.mean_templates(ctx, lab, cl['feats'], cl['chs'], nl)
        final = pr.ccg_merge(ctx, cl['t'], lab, Dc, ds.X.shape[1])
        D_next, ns = pr.build_dictionary(ctx, lab, nl, cl['feats'], cl['chs'], par['n_min'], par['rho1'])
        if checkpoint:
            checkpoint(k, D_next, spk, (cl['t'], final))
        D = D_next
    return D


def run_S(ctx, ds, par, rng, f, K, checkpoint=None):
    n = max(1, int(round(f * ds.n_batches)))
    sub = np.unique(np.round(np.linspace(0, ds.n_batches - 1, n)).astype(int))
    D = run_rounds(ctx, ds, par, rng, sub, K, checkpoint=None)
    if checkpoint:
        checkpoint(0, D, None, None)
    spk = full_extract(ctx, ds, D, np.arange(ds.n_batches))
    lab, nl, final = c_labels(ctx, ds, spk, par, rng)
    if checkpoint:
        checkpoint(1, D, spk, (spk['t'], final))
    return D


def run_R_tracked(ctx, ds, par, rng, K, track, checkpoint=None):
    D = run_rounds(ctx, ds, par, rng, np.arange(ds.n_batches), K)
    D, spk = tracked_extract(ctx, ds, D, track)
    lab, nl, final = c_labels(ctx, ds, spk, par, rng)
    if checkpoint:
        checkpoint(1, D, spk, (spk['t'], final))
    return D


def momentum(par, it, niter, n_tot):
    kind = par['schedule'][0]
    if kind == 'batches':
        m0, m1 = par['schedule'][1:3]
        return np.full(len(n_tot), m0 + (m1 - m0) * it / max(1, niter - 1))
    _, mmin, mmax = par['schedule']
    return np.clip(par['mscale'] * n_tot, mmin, mmax)


class Buffer:
    """Rolling store of unexplained spikes; each spike keeps the iteration it arrived in."""
    def __init__(self, nbatches):
        self.nb = nbatches
        self.parts = []          # (its, f, c, g, t)

    def add(self, it, f, c, g, t):
        if len(f):
            self.parts.append((np.full(len(f), it), f, c, g, t))

    def expire(self, it):
        self.parts = [p for p in self.parts if p[0].max() > it - self.nb]

    def get(self):
        if not self.parts:
            return None
        return tuple(np.concatenate([x[i] for x in self.parts]) for i in range(5))

    def replace(self, arrays):
        self.parts = [arrays] if len(arrays[0]) else []


def tracked_extract(ctx, ds, D, m_track):
    """Final pass in time order with templates kept up to date (KS2.5 trackAndSort without the middle-out order)."""
    D = D.clone()
    T, I, A, F, C, G = [], [], [], [], [], []
    for b in range(ds.n_batches):
        Xb, lo, k0, k1 = bpad(ds, b)
        iU = pr.main_chans(ctx, D)
        ctc = pr.prep_ctc(ctx, D)
        tt, ids, amps, Xres = pr.matching(ctx, Xb, D, ctc, k0 + TSHIFT, k1 + TSHIFT)
        f = pr.match_features(ctx, Xres, tt, ids, amps, D, iU)
        chs = ctx.iCC[iU[ids]].numpy()
        T.append(tt.numpy() + lo - TSHIFT); I.append(ids.numpy()); A.append(amps.numpy()); F.append(f); C.append(chs)
        G.append(iU[ids].numpy())
        if len(tt):
            K = D.shape[0]
            nb_k = np.bincount(ids.numpy(), minlength=K)
            mt, _ = _means(ctx, ids.numpy(), f, chs, K)
            phi = torch.tensor(np.exp(-nb_k / m_track), dtype=torch.float32)[:, None, None]
            upd = torch.from_numpy(nb_k > 0)
            D[upd] = (phi * D + (1 - phi) * mt)[upd]
            D = pr.recenter(ctx, D)
    ctx.fpe += 1.0
    return D, dict(t=np.concatenate(T), id=np.concatenate(I), amp=np.concatenate(A), feats=np.concatenate(F),
                   chs=np.concatenate(C), group=np.concatenate(G))


def run_I(ctx, ds, par, rng, p=1.0, order='shuffled', checkpoint=None, ck_every=0.05, fc_par=None, track=0):
    Nb = ds.n_batches
    niter = max(1, int(round(p * Nb)))
    if order == 'shuffled':
        seq = np.concatenate([rng.permutation(Nb) for _ in range(int(np.ceil(p)) + 1)])[:niter]
    else:
        seq = np.arange(niter) % Nb
    npc, nch = ctx.npc, ctx.nchan
    D = torch.zeros((0, npc, nch))
    n_tot = np.zeros(0); nsp = np.zeros(0)
    buf = Buffer(par['buffer_batches'])
    m0 = par['minFR'] * ds.batch_len / pr.FS
    next_ck = 0.0
    ctc = None; dirty = True; dict_trace = []
    for it, b in enumerate(seq):
        Xb, lo, k0, k1 = bpad(ds, b)
        K = D.shape[0]
        if K > 0:
            ctc = pr.prep_ctc(ctx, D)
            iU = pr.main_chans(ctx, D)
            tt, ids, amps, Xres = pr.matching(ctx, Xb, D, ctc, k0 + TSHIFT, k1 + TSHIFT)
            f = pr.match_features(ctx, Xres, tt, ids, amps, D, iU)
            chs = ctx.iCC[iU[ids]].numpy()
            nb_k = np.bincount(ids.numpy(), minlength=K)
            if len(tt):
                mt, _ = _means(ctx, ids.numpy(), f, chs, K)
                n_tot = n_tot + nb_k
                m = momentum(par, it, niter, n_tot)
                phi = torch.tensor(np.exp(-nb_k / m), dtype=torch.float32)[:, None, None]
                upd = torch.from_numpy(nb_k > 0)
                D[upd] = (phi * D + (1 - phi) * mt)[upd]
                D = pr.recenter(ctx, D)
            nsp = par['nsp_forget'] * nsp + (1 - par['nsp_forget']) * nb_k
        else:
            Xres = Xb
        t2, ch2 = pr.detect(ctx, Xres, k0 + TSHIFT, k1 + TSHIFT)
        f2, c2 = pr.features_at(ctx, Xres, t2, ch2)
        buf.add(it, f2, c2, ch2.numpy(), t2.numpy() + lo - TSHIFT)
        buf.expire(it)
        # births
        if (it + 1) % par['birth_every'] == 0 and buf.parts:
            it3, f3, c3, g3, t3 = buf.get()
            lab, nl = pr.cluster_all(ctx, f3, c3, g3, rng, par['K_init'], par['bim'], par['n_min'])
            Dn, ns = pr.mean_templates(ctx, lab, f3, c3, nl)
            ok = ns >= par['n_birth']
            if ok.any():
                idx_ok = np.flatnonzero(ok)
                Dn = pr.align(ctx, Dn[torch.from_numpy(ok)])
                ns_ok = ns[ok]
                if D.shape[0] > 0:
                    cc = _cross_sim(ctx, Dn, D)
                    fresh = cc.max(1) < par['rho_dup']
                else:
                    fresh = np.ones(len(idx_ok), bool)
                # remove all consumed spikes from the buffer (those of accepted clusters and of duplicates)
                consumed = np.isin(lab, idx_ok)
                k_ = ~consumed
                buf.replace((it3[k_], f3[k_], c3[k_], g3[k_], t3[k_]))
                if fresh.any():
                    Dnew = pr.recenter(ctx, Dn[torch.from_numpy(fresh)])
                    D = torch.cat((D, Dnew), 0)
                    n_tot = np.concatenate((n_tot, ns_ok[fresh].astype(float)))
                    nsp = np.concatenate((nsp, np.full(fresh.sum(), max(m0 * 2, 0.2))))
        # triage
        if (it + 1) % par['triage_every'] == 0 and D.shape[0] > 0:
            drop = (nsp < m0) & (n_tot < par['n_keep'])
            if drop.any():
                k = torch.from_numpy(~drop)
                D = D[k]; n_tot = n_tot[k.numpy()]; nsp = nsp[k.numpy()]
            if D.shape[0] > 1:
                D, n_tot, nsp = _merge_state(ctx, D, n_tot, nsp, par['rho_merge'])
            D = pr.recenter(ctx, D)
        ctx.fpe += 1.0 / Nb
        if checkpoint and (it + 1) / Nb >= next_ck:
            checkpoint((it + 1) / Nb, D.clone(), None, None)
            next_ck += ck_every
    D = pr.align(ctx, D)
    D = pr.recenter(ctx, D)
    if track:
        D, spk = tracked_extract(ctx, ds, D, track)
    else:
        spk = full_extract(ctx, ds, D, np.arange(Nb))
    lab, nl, final = c_labels(ctx, ds, spk, fc_par or par, rng)
    if checkpoint:
        # the dictionary rebuilt from this extraction's clusters, as R rebuilds at each round (FPE p + 1)
        fp = fc_par or par
        Dref, _ = pr.build_dictionary(ctx, lab, nl, spk['feats'], spk['chs'], fp['n_min'], fp['rho1'])
        checkpoint(-2, Dref, None, None)
        checkpoint(-1, D, spk, (spk['t'], final))
    return D


def _means(ctx, ids, f, chs, K):
    return pr.mean_templates_raw(ctx, ids, f, chs, K)


def _cross_sim(ctx, A, B):
    C = torch.cat((A, B), 0)
    cc, _ = pr.similarity(ctx, C)
    return cc[:len(A), len(A):]


def _merge_state(ctx, D, n_tot, nsp, rho):
    K = D.shape[0]
    cc, mu = pr.similarity(ctx, D)
    np.fill_diagonal(cc, 0)
    gone = np.zeros(K, bool)
    D = D.clone(); n_tot = n_tot.copy(); nsp = nsp.copy()
    for kk in np.argsort(-n_tot):
        if gone[kk]:
            continue
        for jj in np.argsort(-cc[kk]):
            if cc[kk, jj] < rho:
                break
            if gone[jj]:
                continue
            dmu = 2 * (mu[kk] - mu[jj]) / (mu[kk] + mu[jj] + 1e-9)
            if abs(dmu) < 0.2:
                wk = n_tot[kk] / (n_tot[kk] + n_tot[jj] + 1e-9)
                D[kk] = wk * D[kk] + (1 - wk) * D[jj]
                n_tot[kk] += n_tot[jj]; nsp[kk] += nsp[jj]
                gone[jj] = True
    keep = ~gone
    return D[torch.from_numpy(keep)], n_tot[keep], nsp[keep]
