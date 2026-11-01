"""Merge-tree diagnostic for search round 2 (search_round2.md §2.3). Behaviour-blind: reads no
tracking data or scores.

Runs one seed-1 sort with the `standard` preset and records, per clustering group and pass:
- the group's spike count, including groups below the 1000-spike floor (made one cluster);
- per tree node: the test that decided it (connectivity, refractory with a significant dip,
  refractory through the sparse shortcut only, bimodality), both sides' spike counts, and for
  merges the number of nodes the collapse removed untested;
- clusters after the tree for the stock tree and, on the same clustering, for each fix:
  refractory significance, bottom-up walk, both;
- for groups of 200-999 spikes, the clusters that clustering them would give (floor 200).
The random-number state is restored after each re-run, so the sort itself is the stock sort.

    python round2_diagnostic.py --out <sort folder> --json <results .json>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kilosort import clustering_qr, hierarchical, run_kilosort, swarmsplitter  # noqa: E402
from kilosort.io import load_probe  # noqa: E402
from kilosort.parameters import get_preset  # noqa: E402

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / 'config.json').read_text())
BASE = Path(CFG['excerpt_dir'])
FLOOR = 200
VARIANTS = {'significance': dict(refractory_significance=True),
            'bottom_up': dict(walk='bottom_up'),
            'both': dict(refractory_significance=True, walk='bottom_up')}

stock = {'run': clustering_qr.run, 'get_data': clustering_qr.get_data_cpu, 'split': swarmsplitter.split}
state = {'mode': None, 'ops': None, 'st': None}
groups, nodes = [], []


def rng_state():
    return torch.get_rng_state(), torch.cuda.get_rng_state_all(), np.random.get_state()


def set_rng(s):
    torch.set_rng_state(s[0]); torch.cuda.set_rng_state_all(s[1]); np.random.set_state(s[2])


def tree_clusters(Xd_np, iclust, isub, M, meta, kw):
    xtree, tstat, my_clus = hierarchical.maketree(M, iclust, isub)
    xtree, tstat = stock['split'](Xd_np, xtree, tstat, iclust, my_clus, meta=meta, **kw)
    return int(swarmsplitter.new_clusters(iclust, my_clus, np.array(xtree).copy(), tstat).max() + 1)


def split_kwargs(ops):
    s = ops['settings']
    return dict(bimodality_threshold=s['merge_bimodality_threshold'],
                connectivity_threshold=s['tree_connectivity_threshold'],
                use_shape_criterion=s['correlogram_shape_criterion'],
                refractory_significance=s['tree_refractory_significance'], walk=s['tree_walk'])


def run(ops, st, tF, mode='template', **kw):
    state.update(mode=mode, ops=ops, st=st)
    return stock['run'](ops, st, tF, mode=mode, **kw)


def get_data(ops, xy, iC, iclust_template, tF, ycent, xcent, **kw):
    Xd, igood, ichan = stock['get_data'](ops, xy, iC, iclust_template, tF, ycent, xcent, **kw)
    if Xd is None:
        return Xd, igood, ichan
    n = int(Xd.shape[0])
    rec = {'pass': state['mode'], 'n_spikes': n}
    if FLOOR <= n < ops['settings']['cluster_min_group_spikes']:
        s, rs = ops['settings'], rng_state()
        iclust, isub, M, _ = clustering_qr.cluster(
            Xd, nskip=s['cluster_downsampling'], n_neigh=s['cluster_neighbors'], max_sub=s['max_cluster_subset'],
            nclust=s['cluster_init_clusters'], lam=1, resolution=s['cluster_resolution'],
            seed=s['cluster_init_seed'], device=torch.device('cuda'))
        meta = state['st'][igood, 0] / ops['fs'] if state['mode'] == 'template' else None
        rec['clusters_if_clustered'] = tree_clusters(Xd.numpy(), iclust, isub, M, meta, split_kwargs(ops))
        set_rng(rs)
    groups.append(rec)
    return Xd, igood, ichan


def split(Xd_np, xtree, tstat, iclust, my_clus, meta=None, **kw):
    xtree0, tstat0 = np.array(xtree).copy(), np.array(tstat).copy()
    record = []
    result = stock['split'](Xd_np, xtree, tstat, iclust, my_clus, meta=meta, record=record, **kw)
    out = {'pass': state['mode'], 'leaves': int(iclust.max() + 1), 'nodes': record,
           'stock': int(swarmsplitter.new_clusters(iclust, my_clus, np.array(result[0]).copy(), result[1]).max() + 1)}
    for name, v in VARIANTS.items():
        xt, ts = stock['split'](Xd_np, xtree0.copy(), tstat0.copy(), iclust, my_clus, meta=meta, **{**kw, **v})
        out[name] = int(swarmsplitter.new_clusters(iclust, my_clus, np.array(xt).copy(), ts).max() + 1)
    nodes.append(out)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True, help='sort folder (must not exist)')
    p.add_argument('--json', required=True)
    args = p.parse_args()
    if Path(args.out).exists():
        raise SystemExit(f'{args.out} already exists')
    clustering_qr.run, clustering_qr.get_data_cpu, swarmsplitter.split = run, get_data, split
    settings = {'n_chan_bin': CFG['n_chan_bin'], 'fs': 30000, 'random_seed': 1, 'cluster_init_seed': 5,
                **get_preset('standard')}
    run_kilosort(settings=settings, probe=load_probe(str(BASE / CFG['probe'])),
                 filename=str(BASE / CFG["data"]), results_dir=args.out, data_dtype="int16",
                 verbose_console=False, make_plots=False, save_pc_features=False)
    Path(args.json).write_text(json.dumps({'settings': settings, 'groups': groups, 'trees': nodes}))
    print(f'{len(groups)} groups, {len(nodes)} trees -> {args.json}')
