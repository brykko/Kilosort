"""Duration-vs-nskip test of KS4 pass-2 clustering on one spatial group, CPU only.

usage: run_group.py GROUP_NPZ SEEDS(comma) [CONDITIONS(comma)] [--stock]
Per condition: subset spikes by time window, run clustering_qr.cluster -> hierarchical.maketree
-> swarmsplitter.split -> new_clusters exactly as clustering_qr.run(mode='template') does.
Appends one JSON line per run to results.jsonl (or results_stock.jsonl with --stock).
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys, json, time, math
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, r'C:\code\github\Kilosort')
from kilosort import clustering_qr, hierarchical, swarmsplitter

torch.set_num_threads(int(os.environ.get('NTHREADS', '8')))
DEV = torch.device('cpu')
OUT = Path(__file__).parent
S = json.load(open(OUT / 'settings.json'))['settings']
STOCK = '--stock' in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith('--')]
if STOCK:   # stock KS4 graph/splitter settings (resolution None -> lam=1; bimodality 0.6)
    S = {**S, 'cluster_resolution': None, 'merge_bimodality_threshold': 0.6}

# --- count iterations of the assignment loop: cluster() calls assign_iclust once per loop iteration
_calls = {'n': 0}
_orig_assign_iclust = clustering_qr.assign_iclust
def _counting(*a, **k):
    _calls['n'] += 1
    return _orig_assign_iclust(*a, **k)
clustering_qr.assign_iclust = _counting

g = np.load(args[0])
Xall = torch.from_numpy(g['Xd'])
tall = g['st0']
ii, pctl = int(g['ii']), int(g['pctl'])
seeds = [int(s) for s in args[1].split(',')]
conds = args[2].split(',') if len(args) > 2 else ['i', 'ii', 'iii', 'iv', 'v', 'L10']
T = 1200.0

# condition -> (t0, t1, nskip, max_sub rule)
COND = {
    'v':   (0, 300, 20, 'stock'),     # first 5 min
    'i':   (0, 600, 20, 'stock'),     # first 10 min
    'ii':  (0, T, 20, 'stock'),       # all 20 min
    'iii': (0, 600, 10, 'stock'),     # first 10 min, nskip 10
    'iv':  (0, T, 20, 'cap_i'),       # all 20 min, max_sub = landmarks of (i)
    'L10': (600, T, 20, 'stock'),     # last 10 min
    'L5':  (900, T, 20, 'stock'),     # last 5 min
    'L10s10': (600, T, 10, 'stock'),  # last 10 min, nskip 10
}

def run(cond, seed):
    t0, t1, nskip, rule = COND[cond]
    sel = (tall >= t0) & (tall < t1)
    Xd = Xall[torch.from_numpy(sel)].contiguous()
    st0 = tall[sel]
    max_sub = S['max_cluster_subset']
    if rule == 'cap_i':
        n_i = int(((tall >= 0) & (tall < 600)).sum())
        max_sub = math.ceil(n_i / 20)
    tic = time.time()
    _calls['n'] = 0
    iclust, iclust0, M, _ = clustering_qr.cluster(
        Xd, nskip=nskip, n_neigh=S['cluster_neighbors'], max_sub=max_sub,
        nclust=S['cluster_init_clusters'], lam=1, resolution=S['cluster_resolution'],
        seed=seed, device=DEV)
    t_graph = time.time() - tic
    niter = _calls['n'] - 0   # loop calls only (the final assign_isub after the loop is not counted)
    leaves = int(np.unique(iclust).size)
    xtree, tstat, my_clus = hierarchical.maketree(M, iclust, iclust0)
    rec = []
    xtree2, tstat2 = swarmsplitter.split(
        Xd.numpy(), xtree, tstat, iclust, my_clus, meta=st0,
        bimodality_threshold=S['merge_bimodality_threshold'],
        connectivity_threshold=S['tree_connectivity_threshold'],
        refractory_significance=S['tree_refractory_significance'],
        walk=S['tree_walk'], use_shape_criterion=S['correlogram_shape_criterion'], record=rec)
    final = swarmsplitter.new_clusters(iclust, my_clus, xtree2, tstat2)
    nfinal = int(final.max() + 1)
    # same tree, splitter without the refractory veto (meta=None, as in pass 1)
    xt3, ts3 = swarmsplitter.split(
        Xd.numpy(), xtree, tstat, iclust, my_clus, meta=None,
        bimodality_threshold=S['merge_bimodality_threshold'],
        connectivity_threshold=S['tree_connectivity_threshold'],
        walk=S['tree_walk'], use_shape_criterion=S['correlogram_shape_criterion'])
    nfinal_nometa = int(swarmsplitter.new_clusters(iclust, my_clus, xt3, ts3).max() + 1)
    src = {}
    for r in rec:
        key = f"{r['source']}:{'merge' if r['criterion'] == 1 else 'split'}"
        src[key] = src.get(key, 0) + 1
    out = dict(group=ii, pctl=pctl, cond=cond, seed=seed, stock=STOCK, t0=t0, t1=t1, nskip=nskip,
               max_sub=max_sub, n_spikes=int(Xd.shape[0]), n_landmarks=int(M.shape[1]),
               leaves=leaves, final=nfinal, final_nometa=nfinal_nometa, niter=niter, cap_hit=niter >= 200,
               t_graph=round(t_graph, 1), t_total=round(time.time() - tic, 1), decisions=src)
    print(json.dumps(out), flush=True)
    with open(OUT / ('results_stock.jsonl' if STOCK else 'results.jsonl'), 'a') as f:
        f.write(json.dumps(out) + '\n')

for seed in seeds:
    for c in conds:
        run(c, seed)
