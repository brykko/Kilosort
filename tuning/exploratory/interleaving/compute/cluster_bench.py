"""Clustering cost against the number of spikes: clustering_qr.run on subsets of the round-2 spikes of kobe H8 seed 1
(from the stage cache, not re-extracted), every k-th batch (D3) or a contiguous rolling buffer (D1, D2), with timers
around its parts, and cluster_downsampling varied. Patches are in this process only. Output: cluster_bench.json."""
from common import *
from kilosort import clustering_qr as cq, hierarchical, swarmsplitter

ops, bfile, folder = load_sort()
d = next(CACHE.glob('B_f42d66d3d2b13925*'))
c = torch.load(d / 'extract.pt', weights_only=False)
ops.update({k: v for k, v in c['ops'].items()})
print({k: getattr(v, 'shape', v) for k, v in c['ops'].items() if k in ('iU', 'iCC', 'iCC_mask')})
st, tF = c['out'][0], c['out'][1]
del c
print(type(st), st.shape, type(tF), tF.shape, flush=True)
NT = ops['batch_size']
ib = (st[:, 0] // NT).astype(np.int64)
T = {}


def timed(mod, name):
    f = getattr(mod, name)
    def g(*a, **k):
        torch.cuda.synchronize() if name != 'get_data_cpu' else None
        t = time.perf_counter()
        r = f(*a, **k)
        torch.cuda.synchronize()
        T[name] = T.get(name, 0) + time.perf_counter() - t
        return r
    setattr(mod, name, g)


for mod, name in ((cq, 'get_data_cpu'), (cq, 'neigh_mat'), (cq, 'kmeans_plusplus'), (cq, 'cluster'), (hierarchical, 'maketree'), (swarmsplitter, 'split')):
    timed(mod, name)


def run(label, sel, cds=10):
    T.clear()
    ops['settings']['cluster_downsampling'] = cds
    s, f = st[sel], tF[torch.from_numpy(sel)] if sel.dtype == bool else tF[sel]
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(1); np.random.seed(1)
    t0 = now()
    clu, Wall = cq.run(ops, s, f, mode='template', device=DEVICE)
    t = now() - t0
    r = {'label': label, 'cds': cds, 'spikes': int(sel.sum() if sel.dtype == bool else len(sel)), 'wall_s': t, 'us_per_spike': t / len(s) * 1e6, 'clusters': int(clu.max() + 1),
         'parts_s': {k: float(v) for k, v in T.items()}, 'gpu_peak_GB': torch.cuda.max_memory_allocated() / 2**30}
    # the 'cluster' timer contains neigh_mat and kmeans_plusplus
    print(json.dumps(r), flush=True)
    return r


nb = 600
res = []
res.append(run('all', np.ones(len(st), bool)))
for k in (2, 4, 8, 16, 30):
    res.append(run(f'every_{k}', ib % k == 0))
for L in (30, 60, 120):
    res.append(run(f'rolling_{L}', (ib >= 300) & (ib < 300 + L)))
res.append(run('every_4', ib % 4 == 0, cds=20))
res.append(run('every_4', ib % 4 == 0, cds=5))
res.append(run('every_30', ib % 30 == 0, cds=1))
res.append(run('rolling_30', (ib >= 300) & (ib < 330), cds=1))
res.append(run('all', np.ones(len(st), bool), cds=20))
save_json('cluster_bench.json', res)
