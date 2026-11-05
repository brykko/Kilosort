"""Job queue. Stages: tune, main, sens, scale, vari. Each job builds one dataset and runs a list of (name, arm, par_r, par_i).

python run_queue.py <stage> [--workers 6]
Results: one JSON per job in OUT/<stage>/ (skipped if present, so a stage can be resumed).
"""
import argparse
import itertools
import json
import multiprocessing as mp
import os
import sys
import time

OUT = r'C:\data\ks4_tuning\exploratory\interleaving\simulation'
HERE = os.path.dirname(os.path.abspath(__file__))

R_GRID = [dict(rho1=a, bim=b, K_init=c) for a, b, c in itertools.product((.90, .95, .98), (.4, .6), (8, 16))]
I_GRID = [dict(schedule=a, n_birth=b, rho_merge=c) for a, b, c in itertools.product(
    (('batches', 20, 400), ('spikes', 20, 200), ('spikes', 20, 2000)), (15, 30), (.93, .97))]
VARS = ('stationary', 'drift', 'nonstat')
MAIN_T = 120.0


def tuned():
    with open(os.path.join(OUT, 'tuned.json')) as f:
        t = json.load(f)
    t['I']['schedule'] = tuple(t['I']['schedule'])
    t['I'] = dict(buffer_batches=20, birth_every=5, triage_every=5, **t['I'])
    return t['R'], t['I']


def jobs_for(stage):
    jobs = []
    if stage == 'tune':
        for seed, v in ((7001, 'stationary'), (7002, 'drift'), (7003, 'nonstat')):
            for i in range(0, 12, 6):
                arms = [(f'Rg{j}', 'R3', R_GRID[j], None) for j in range(i, i + 6)]
                jobs.append(dict(id=f'{seed}_{v}_R{i}', seed=seed, variant=v, over={}, arms=arms))
                arms = [(f'Ig{j}', 'I1_shuffled', None, I_GRID[j]) for j in range(i, i + 6)]
                jobs.append(dict(id=f'{seed}_{v}_I{i}', seed=seed, variant=v, over={}, arms=arms))
    elif stage == 'main':
        pr_, pi_ = tuned()
        for seed in range(1, 11):
            for v in VARS:
                names = ['R4', 'Rd3', 'S0.25_3', 'I1_shuffled', 'I1_time', 'I0.25_shuffled']
                if seed <= 3:
                    names.append('O')
                if v == 'drift':
                    names += ['I1_shuffled_track', 'Tr3']
                # split into two jobs so that workers stay balanced
                arms = [(n, n, pr_, pi_) for n in names]
                a0 = [a for a in arms if a[0][0] != 'I']
                a1 = [a for a in arms if a[0][0] == 'I']
                jobs.append(dict(id=f'{seed}_{v}_a', seed=seed, variant=v, over=dict(T=MAIN_T), arms=a0))
                jobs.append(dict(id=f'{seed}_{v}_b', seed=seed, variant=v, over=dict(T=MAIN_T), arms=a1))
    elif stage == 'vari':
        pr_, pi_ = tuned()
        for seed in (1, 2, 3):
            for alg in (1, 2):
                arms = [(f'R3_alg{alg}', 'R3', pr_, pi_), (f'I1_alg{alg}', 'I1_shuffled', pr_, pi_)]
                jobs.append(dict(id=f'{seed}_alg{alg}', seed=seed, variant='stationary', over=dict(T=MAIN_T), arms=arms, alg=alg))
    elif stage == 'sens':
        pr_, pi_ = tuned()
        scheds = [('batches', 20, 400), ('spikes', 20, 200), ('spikes', 20, 2000)]
        cfg = [(f'sch_{s[0]}{s[2]}', dict(schedule=s), {}) for s in scheds if s != tuple(pi_['schedule'])]
        cfg += [('birth8', dict(n_birth=8), {}), ('birth60', dict(n_birth=60), {}),
                ('buf5', dict(buffer_batches=5), {}), ('buf60', dict(buffer_batches=60), {})]
        for bs in (0.5, 8.0):
            r = 2.0 / bs
            cfg.append((f'batch{bs}', dict(buffer_batches=int(pi_['buffer_batches'] * r),
                                           birth_every=max(1, int(pi_['birth_every'] * r)),
                                           triage_every=max(1, int(pi_['triage_every'] * r))), dict(batch_s=bs)))
        for seed in (1, 2):
            for name, ov, dov in cfg:
                jobs.append(dict(id=f'{seed}_stationary_{name}', seed=seed, variant='stationary', over=dict(T=MAIN_T, **dov),
                                 arms=[(name, 'I1_shuffled', pr_, {**pi_, **ov})]))
                if name.startswith('sch_'):
                    jobs.append(dict(id=f'{seed}_drift_{name}', seed=seed, variant='drift', over=dict(T=MAIN_T, **dov),
                                     arms=[(name, 'I1_shuffled', pr_, {**pi_, **ov})]))
            jobs.append(dict(id=f'{seed}_nonstat_keep0', seed=seed, variant='nonstat', over=dict(T=MAIN_T),
                             arms=[('keep0', 'I1_time', pr_, {**pi_, 'n_keep': 0}),
                                   ('keep0s', 'I1_shuffled', pr_, {**pi_, 'n_keep': 0})]))
    elif stage == 'posthoc':
        # added after seeing main: I's dictionary rebuilt from its final extraction, and I given two passes
        pr_, pi_ = tuned()
        for seed in range(1, 7):
            for v in ('stationary', 'nonstat'):
                jobs.append(dict(id=f'{seed}_{v}', seed=seed, variant=v, over=dict(T=MAIN_T),
                                 arms=[(a, a, pr_, pi_) for a in ('I1_shuffled', 'I2_shuffled')]))
    elif stage == 'ext1':
        # post hoc: R beyond its grid edge, chosen on seeds 1-2 (stationary), reported on 3-8
        pr_, pi_ = tuned()
        for seed in (1, 2):
            for name, ov in (('Rx_b3k8', dict(bim=.3, K_init=8)), ('Rx_b4k4', dict(bim=.4, K_init=4)), ('Rx_b3k4', dict(bim=.3, K_init=4))):
                jobs.append(dict(id=f'{seed}_{name}', seed=seed, variant='stationary', over=dict(T=MAIN_T),
                                 arms=[(name, 'R3', {**pr_, **ov}, pi_)]))
    elif stage == 'ext2':
        # post hoc: best-case I (settings read from the sensitivity table on seeds 1-2) and R's extension, seeds 3-8
        pr_, pi_ = tuned()
        with open(os.path.join(OUT, 'ext_R.json')) as f:
            rx = json.load(f)
        best = {**pi_, 'n_birth': 8, 'schedule': ('spikes', 20, 200), 'buffer_batches': 80, 'birth_every': 20, 'triage_every': 20}
        best2 = {**pi_, 'n_birth': 8, 'schedule': ('spikes', 20, 200)}
        for seed in range(3, 9):
            for v in ('stationary', 'nonstat'):
                jobs.append(dict(id=f'{seed}_{v}_Ib', seed=seed, variant=v, over=dict(T=MAIN_T, batch_s=.5),
                                 arms=[('Ibest', 'I1_shuffled', pr_, best)]))
                jobs.append(dict(id=f'{seed}_{v}_x', seed=seed, variant=v, over=dict(T=MAIN_T),
                                 arms=[('Rx', 'R3', {**pr_, **rx}, pi_), ('Ibest2s', 'I1_shuffled', pr_, best2)]))
    elif stage == 'ext3':
        # post hoc: the crowded conditions with the per-row cluster cap lifted (K_init 8 allows 128 clusters for 192 units)
        pr_, pi_ = tuned()
        for seed in (1, 2, 3):
            for name, nu, np_, k in (('u192', 192, 24, 32), ('u96', 96, 12, 16)):
                rk = {**pr_, 'K_init': k}
                arms = [(f'{name}|R3k', 'R3', rk, pi_), (f'{name}|Ik', 'I1_shuffled', rk, {**pi_, 'K_init': k}), (f'{name}|O', 'O', rk, pi_)]
                jobs.append(dict(id=f'{seed}_{name}', seed=seed, variant='stationary',
                                 over=dict(T=MAIN_T, n_units=nu, n_pairs=np_), arms=arms))
    elif stage == 'scale':
        pr_, pi_ = tuned()
        conds = [('u24', dict(n_units=24, n_pairs=3)), ('u96', dict(n_units=96, n_pairs=12)),
                 ('u192', dict(n_units=192, n_pairs=24)), ('rate1', dict(rate_med=1.0)), ('rate15', dict(rate_med=15.0)),
                 ('batch0.5', dict(batch_s=0.5)), ('batch8', dict(batch_s=8.0)), ('T360', dict(T=360.0))]
        for seed in (1, 2, 3):
            for name, ov in conds:
                over = {'T': MAIN_T, **ov}
                par = dict(pi_)
                bs = over.get('batch_s', 2.0)
                r = 2.0 / bs
                par.update(buffer_batches=int(pi_['buffer_batches'] * r), birth_every=max(1, int(pi_['birth_every'] * r)),
                           triage_every=max(1, int(pi_['triage_every'] * r)))
                arms = [(f'{name}|{a}', a, pr_, par) for a in ('R3', 'I1_shuffled')]
                jobs.append(dict(id=f'{seed}_{name}', seed=seed, variant='stationary', over=over, arms=arms))
            arms = [(f'drift10|{a}', a, pr_, pi_) for a in ('R3', 'I1_shuffled', 'I1_time', 'I1_shuffled_track', 'Tr3')]
            jobs.append(dict(id=f'{seed}_drift10', seed=seed, variant='drift', over=dict(T=MAIN_T, drift_pp=10.0), arms=arms))
    return jobs


def work(args):
    stage, job = args
    path = os.path.join(OUT, stage, job['id'] + '.json')
    if os.path.exists(path):
        return job['id'], 0.0
    sys.path.insert(0, HERE)
    import numpy as np
    import harness as h
    import simdata
    t0 = time.time()
    ds = simdata.make_dataset(job['seed'], job['variant'], **job['over'])
    res = dict(job={k: v for k, v in job.items() if k != 'arms'}, arms=[])
    ctx_um = None
    for name, arm, par_r, par_i in job['arms']:
        ta = time.time()
        recs, sco = h.run_arm(arm, ds, par_r, par_i, algo_seed=job.get('alg', 0))
        if ctx_um is None:
            ctx_um = sco.um.tolist()
        res['arms'].append(dict(name=name, arm=arm, par_r=par_r, par_i=par_i, seconds=time.time() - ta, records=recs))
    info = ds.info
    res['truth'] = dict(um=ctx_um, rates=np.asarray(info['rates']).tolist(), norm=np.asarray(info['norm']).tolist(),
                        kinds=list(info['kinds']), pairs=info['pairs'], silent=np.asarray(info['silent']).astype(int).tolist(),
                        burst=np.asarray(info['burst']).astype(int).tolist(), n_spikes=np.asarray(info['n_spikes_unit']).tolist())
    res['seconds'] = time.time() - t0
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(res, f)
    os.replace(tmp, path)
    return job['id'], res['seconds']


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage')
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    os.makedirs(os.path.join(OUT, a.stage), exist_ok=True)
    jobs = jobs_for(a.stage)
    print(a.stage, len(jobs), 'jobs', flush=True)
    t0 = time.time()
    with mp.get_context('spawn').Pool(a.workers, maxtasksperchild=1) as pool:
        for i, (jid, sec) in enumerate(pool.imap_unordered(work, [(a.stage, j) for j in jobs])):
            print(f'{i + 1}/{len(jobs)} {jid} {sec:.0f}s total {(time.time() - t0) / 60:.1f} min', flush=True)
    print('done', a.stage, (time.time() - t0) / 60, flush=True)
