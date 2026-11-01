"""Clustering diagnostic for search round 1 (search_round1.md §2.3). Behaviour-blind: reads no
tracking data or scores.

Runs one default sort (seed 1) with clustering_qr.cluster and swarmsplitter.split wrapped. On every
STRIDE-th clustering centre (both passes), after the stock clustering, it re-clusters the same
features under each variant below and runs the merge tree on the result, recording:
- clusters after cluster() and after the merge tree;
- adjusted Rand index of the post-tree labels against stock;
- for stock, the graph connectivity (tstat[:,0]) of each node at which the tree merges.
The random-number state is saved and restored around the variants, so the sort itself is the stock
sort (its output is checked against the default replicate afterwards).

    python cluster_diagnostic.py --out <sort folder> --json <results .json> [--tree '<json>']

--tree overrides merge-tree settings (e.g. '{"bimodality_threshold": 0}') for the variants only; a
'stock_clustering' variant then gives the stock clustering under the overridden tree.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import kilosort  # noqa: E402
from kilosort import clustering_qr, hierarchical, run_kilosort, swarmsplitter  # noqa: E402
from kilosort.io import load_probe  # noqa: E402

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / 'config.json').read_text())
BASE = Path(CFG['excerpt_dir'])
STRIDE = 4
VARIANTS = {
    'lam0': dict(lam=0),
    **{f'gamma{g}': dict(resolution=g) for g in (0.25, 0.5, 1, 2, 4, 8)},
    'nclust400': dict(nclust=400),
    'nclust400_gamma1': dict(nclust=400, resolution=1),
}

stock_cluster, stock_split = clustering_qr.cluster, swarmsplitter.split
state = {'calls': 0, 'pending': None, 'tree': {}}
records = []


def tree_labels(Xd_np, iclust, isub, M, meta, split_kw):
    xtree, tstat, my_clus = hierarchical.maketree(M, iclust, isub)
    xtree, tstat = stock_split(Xd_np, xtree, tstat, iclust, my_clus, meta=meta, **split_kw)
    return swarmsplitter.new_clusters(iclust, my_clus, xtree, tstat)


def merged_connectivity(xtree0, tstat0, xtree1):
    """Connectivity at the topmost removed nodes, i.e. the nodes the tree merged."""
    kept = set(np.asarray(xtree1)[:, 2].tolist()) if len(xtree1) else set()
    parent = {int(c): int(n) for n, a, b in np.asarray(xtree0) for c in (a, b)}
    out = []
    for row, (_, _, node) in enumerate(np.asarray(xtree0)):
        if node not in kept and (parent.get(int(node)) is None or parent[int(node)] in kept):
            out.append(float(tstat0[row, 0]))
    return out


def cluster(Xd, *a, **kw):
    out = stock_cluster(Xd, *a, **kw)
    state['calls'] += 1
    state['pending'] = (Xd, kw, out) if state['calls'] % STRIDE == 1 else None
    return out


def split(Xd_np, xtree, tstat, iclust, my_clus, meta=None, **split_kw):
    xtree0, tstat0 = np.array(xtree).copy(), np.array(tstat).copy()
    result = stock_split(Xd_np, xtree, tstat, iclust, my_clus, meta=meta, **split_kw)
    if state['pending'] is None:
        return result
    Xd, kw, (ic0, isub0, M0, _) = state['pending']
    state['pending'] = None
    rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all(), np.random.get_state())
    t = time.time()
    stock_final = swarmsplitter.new_clusters(iclust, my_clus, np.array(result[0]).copy(), result[1])
    rec = {'call': state['calls'], 'pass': 'template' if meta is not None else 'spikes',
           'n_spikes': int(Xd.shape[0]),
           'stock': {'clusters': int(ic0.max() + 1), 'after_tree': int(stock_final.max() + 1)},
           'merged_connectivity': merged_connectivity(xtree0, tstat0, result[0]),
           'connectivity_all': tstat0[:, 0].tolist()}
    for name, v in VARIANTS.items():
        ic, isub, M, _ = stock_cluster(Xd, **{**kw, 'lam': 1, 'resolution': None, **v})
        final = tree_labels(Xd_np, ic, isub, M, meta, {**split_kw, **state['tree']}) if ic.max() > 0 else ic
        rec[name] = {'clusters': int(ic.max() + 1), 'after_tree': int(final.max() + 1),
                     'ari_vs_stock': float(adjusted_rand_score(stock_final, final))}
    torch.set_rng_state(rng[0]); torch.cuda.set_rng_state_all(rng[1]); np.random.set_state(rng[2])
    rec['seconds'] = time.time() - t
    records.append(rec)
    print(f"[diag] call {rec['call']} {rec['pass']} n={rec['n_spikes']} stock {rec['stock']} "
          f"gamma1 {rec['gamma1']['after_tree']} gamma4 {rec['gamma4']['after_tree']} ({rec['seconds']:.0f} s)",
          flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True, help='sort folder (must not exist)')
    p.add_argument('--json', required=True)
    p.add_argument('--tree', default='{}', help='JSON of swarmsplitter.split overrides for the variants')
    args = p.parse_args()
    state['tree'] = json.loads(args.tree)
    if state['tree']:
        VARIANTS = {'stock_clustering': {}, **VARIANTS}
    if Path(args.out).exists():
        raise SystemExit(f'{args.out} already exists')
    clustering_qr.cluster, swarmsplitter.split = cluster, split
    settings = {'n_chan_bin': CFG['n_chan_bin'], 'fs': 30000, 'random_seed': 1, 'cluster_init_seed': 5}
    run_kilosort(settings=settings, probe=load_probe(str(BASE / CFG['probe'])), filename=str(BASE / CFG['data']),
                 results_dir=args.out, data_dtype='int16', verbose_console=False, make_plots=False,
                 save_pc_features=False)
    Path(args.json).write_text(json.dumps({'stride': STRIDE, 'tree': state['tree'], 'variants': VARIANTS,
                                           'records': records}))
    print(f'{len(records)} centres recorded -> {args.json}')
