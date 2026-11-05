"""Run one arm on one dataset and score every checkpoint."""
import copy
import os
import time

for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_k, '1')

import numpy as np
import torch

torch.set_num_threads(1)

import metrics as me
import primitives as pr
import schedulers as sc
import simdata

POOR_TH = 0.75


def merged_par(base, over):
    p = dict(base)
    if over:
        p.update(over)
    return p


class Scorer:
    def __init__(self, ctx, ds):
        self.ctx = ctx
        self.ds = ds
        self.um = me.universal_match(ctx, ds)
        self.Tpc = me.truth_pc(ctx, ds)
        self.records = []

    def record(self, arm, tag, D, spk, labs, t_start, per_unit=False):
        ctx, ds = self.ctx, self.ds
        te, _ = me.template_error(ctx, D, self.Tpc)
        rec = dict(arm=arm, tag=tag, fpe=float(ctx.fpe), mf_ops=float(ctx.mf_ops), clock=time.perf_counter() - t_start,
                   K=int(D.shape[0]), te_med=float(np.median(te)), te_mean=float(te.mean()),
                   te_poor=float(np.median(te[self.um < POOR_TH])) if (self.um < POOR_TH).any() else float('nan'),
                   te_reg=float(np.median(te[self.um >= POOR_TH])))
        if per_unit or spk is not None:
            rec['te'] = te.round(4).tolist()
        if spk is not None:
            sT = me.score_output(spk['t'], spk['id'], ds)
            rec['T'] = me.summary(sT, ds, self.um, POOR_TH)
            rec['T_acc'] = sT['acc'].round(3).tolist()
            rec['T_found'] = sT['found'].astype(int).tolist()
        if labs is not None:
            sC = me.score_output(labs[0], labs[1], ds)
            rec['C'] = me.summary(sC, ds, self.um, POOR_TH)
            rec['C_acc'] = sC['acc'].round(3).tolist()
            rec['C_found'] = sC['found'].astype(int).tolist()
        self.records.append(rec)
        return rec


def run_arm(arm, ds, par_r=None, par_i=None, algo_seed=0):
    """arm: 'O' | 'R<k>' | 'Rd<k>' | 'S<f>_<K>' | 'I<p>_<order>'. Returns records of every checkpoint."""
    ctx = pr.Ctx(ds.xc, ds.yc)
    sco = Scorer(ctx, ds)
    rng = np.random.default_rng(1000 * ds.info['seed'] + 17 * algo_seed + 5)
    torch.manual_seed(algo_seed)
    pr_ = merged_par(sc.R_DEFAULT, par_r)
    pi_ = merged_par(sc.I_DEFAULT, par_i)
    t0 = time.perf_counter()
    if arm == 'O':
        D = pr.recenter(ctx, sco.Tpc.clone())
        spk = sc.full_extract(ctx, ds, D, np.arange(ds.n_batches))
        lab, nl, final = sc.c_labels(ctx, ds, spk, pr_, rng)
        sco.record('O', 'final', D, spk, (spk['t'], final), t0)
    elif arm.startswith('Tr'):            # R3 then a tracked final pass
        def ck(k, D, spk, labs):
            sco.record(arm, 'final', D, spk, labs, t0)
        sc.run_R_tracked(ctx, ds, pr_, rng, int(arm[2:]), sc.M_TRACK, checkpoint=ck)
    elif arm.startswith('Rd') or arm.startswith('R'):
        disc = arm.startswith('Rd')
        K = int(arm[2:]) if disc else int(arm[1:])

        def ck(k, D, spk, labs):
            sco.record(arm, f'round{k}', D, spk, labs, t0)
        sc.run_rounds(ctx, ds, pr_, rng, np.arange(ds.n_batches), K, discover=disc, checkpoint=ck)
    elif arm.startswith('S'):
        f, K = arm[1:].split('_')

        def ck(k, D, spk, labs):
            sco.record(arm, 'final' if k == 1 else 'learned', D, spk, labs, t0)
        sc.run_S(ctx, ds, pr_, rng, float(f), int(K), checkpoint=ck)
    elif arm.startswith('I'):
        p, order, *tr = arm[1:].split('_')

        def ck(k, D, spk, labs):
            tag = 'final' if k == -1 else ('refit' if k == -2 else f'it{k:.3f}')
            sco.record(arm, tag, D, spk, labs, t0)
        sc.run_I(ctx, ds, pi_, rng, p=float(p), order=order, checkpoint=ck, fc_par=pr_,
                 track=sc.M_TRACK if tr else 0)
    else:
        raise ValueError(arm)
    for r in sco.records:
        r['clock_detail'] = {k: round(v, 2) for k, v in ctx.clock.items()}
    return sco.records, sco
