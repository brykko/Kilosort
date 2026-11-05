"""How much of the dictionary shrink could the first pass have had by itself? In clustering_qr.run the refractory test
(meta) is passed only in mode 'template', so at H8 (bimodality 0) the first pass merges no tree node at all, and the
only merge before extraction is 'mu' (r >= 0.9 and norms within 20%). From the saved first clusters and universal spike
times (no sorting): pairs of first clusters that are close (peak channels within 40 um, shape r >= 0.7) and
cross-refractory by the KS4 test, joined by union-find.

This is an upper-bound style estimate: the tree can only merge along its own nodes, and only within one clustering
centre. Both the stock test (sparse pairs pass without significance) and the significance-only test are reported."""
from common import *  # noqa
import torch  # noqa: E402
from kilosort import template_matching as tm  # noqa: E402
from kilosort.swarmsplitter import check_CCG  # noqa: E402

torch.set_num_threads(4)


def find(p, i):
    while p[i] != i:
        p[i] = p[p[i]]
        i = p[i]
    return i


for ex in EXS:
    run = O / 'sorts' / f'{ex}_repeat_s1'
    prov = json.loads((run / 'provenance.json').read_text())
    c = torch.load(O / 'stage_cache' / f"B_{prov['cache']['B']}" / 'cluster1.pt', weights_only=False, map_location='cpu')
    clu0, Wall0 = c['out']
    del c
    a = torch.load(O / 'stage_cache' / f"A_{prov['cache']['A']}" / 'detect1.pt', weights_only=False, map_location='cpu')
    t0 = a['out'][0][:, 0].copy()
    del a
    ops = dict(ops_of(run / 'round2'))
    ops['wPCA'] = torch.as_tensor(tonp(ops['wPCA'])).float()
    xc, yc = tonp(ops['xc']), tonp(ops['yc'])
    Wall0 = Wall0.cpu()
    n = Wall0.shape[0]
    pk = (Wall0**2).sum(2).argmax(1).numpy()
    ns = np.bincount(clu0, minlength=n)
    r = tm.max_lag_correlation(Wall0, Wall0, ops, block=24)
    near = (r >= .7) & (np.hypot(xc[pk][:, None] - xc[pk][None], yc[pk][:, None] - yc[pk][None]) <= 40)
    i, j = np.nonzero(np.triu(near, 1))
    print(ex, 'first clusters', n, 'median spikes', np.median(ns), 'candidate pairs', i.size, flush=True)
    o = np.argsort(clu0, kind='stable')
    b = np.r_[0, np.cumsum(ns)]
    times = [np.sort(t0[o[b[u]:b[u+1]]]) for u in range(n)]
    pa, ps = np.arange(n), np.arange(n)
    n_all, n_sig = 0, 0
    for u, v in zip(i, j):
        if ns[u] < 2 or ns[v] < 2:
            continue
        cr = check_CCG(times[u], times[v])[1]
        if cr:
            n_all += 1
            pa[find(pa, u)] = find(pa, v)
            if check_CCG(times[u], times[v], require_significance=True)[1]:
                n_sig += 1
                ps[find(ps, u)] = find(ps, v)
    ca = len({find(pa, x) for x in range(n)})
    cs = len({find(ps, x) for x in range(n)})
    rounds = ops_of(run)['template_rounds']
    print(f'  cross-refractory pairs {n_all}, significant {n_sig}; clusters after union: stock test {ca}, significance only {cs}; '
          f'for scale: templates at round 2 {rounds[0]["n_templates"]}, round-2 clusters {rounds[0]["n_clusters"]}, round 3 templates {rounds[1]["n_templates"]}', flush=True)
    del times, t0
