"""What the dictionary is at each round. Rebuilds round 2's extraction templates from the first clustering
(stage cache, H8 only) and counts near-duplicate shapes; for every saved round counts templates per unit,
borrowed spikes, and how far each unit's spikes are spread over peak channels, and so over the 10-channel
feature windows that clustering zero-pads (clustering_qr.get_data_cpu)."""
from common import *  # noqa
import torch  # noqa: E402
from kilosort import template_matching as tm  # noqa: E402

torch.set_num_threads(4)
dev = torch.device('cpu')


def prep(ops):
    ops = dict(ops)
    ops['wPCA'] = torch.as_tensor(tonp(ops['wPCA'])).float()
    ops['wTEMP'] = torch.as_tensor(tonp(ops['wTEMP'])).float()
    return ops


def near_dup(U, ops):
    """U (n, pcs, chans). Nearest-neighbour correlation, pairs at r >= thr, greedy shape-only collapse."""
    Ut = U.transpose(1, 2)
    r = tm.max_lag_correlation(Ut, Ut, ops, block=24)
    np.fill_diagonal(r, -1)
    nn = r.max(1)
    mu = (U**2).sum((1, 2)).sqrt().numpy()
    iu = (U**2).sum(1).argmax(-1).numpy()
    out = {'n': len(U), 'median_nn_r': float(np.median(nn))}
    for th in (.9, .95, .975):
        out[f'nn>={th}'] = float((nn >= th).mean())
        i, j = np.nonzero(np.triu(r >= th, 1))
        dmu = np.abs(2 * (mu[i] - mu[j]) / (mu[i] + mu[j]))
        out[f'pairs>={th}'] = int(i.size)
        out[f'pairs>={th}_norm_differs'] = float((dmu >= .2).mean()) if i.size else 0.
        out[f'pairs>={th}_same_peak'] = float((iu[i] == iu[j]).mean()) if i.size else 0.
        alive = np.ones(len(U), bool)
        keep = 0
        for k in np.argsort(-mu):
            if alive[k]:
                keep += 1
                alive[r[k] >= th] = False
                alive[k] = False
        out[f'greedy_n_at_{th}'] = keep
    return out


def spread(d):
    o = ops_of(d)
    tpl = np.load(d / 'spike_detection_templates.npy').ravel().astype(np.int64)
    clu = np.load(d / 'spike_clusters.npy').ravel().astype(np.int64)
    iU, iCC = tonp(o['iU']).astype(int), tonp(o['iCC']).astype(int)
    xc, yc = tonp(o['xc']), tonp(o['yc'])
    owner, t, c, n = owners(tpl, clu)
    ns = np.bincount(clu)
    big = ns >= 100
    owned = np.bincount(owner[owner >= 0], minlength=ns.size)
    p = n / ns[c]
    simpson = np.bincount(c, p**2, minlength=ns.size)
    eff = 1 / np.maximum(simpson, 1e-12)
    borrowed = 1 - (owner[tpl] == clu).mean()
    ch = iU[t]
    nch = iU.max() + 1
    key = c * nch + ch
    uk, inv = np.unique(key, return_inverse=True)
    cnt = np.bincount(inv, n)
    cu, chu = uk // nch, uk % nch
    o2 = np.lexsort((cnt, cu))
    last = np.r_[cu[o2][1:] != cu[o2][:-1], True]
    modal = -np.ones(ns.size, int)
    modal[cu[o2][last]] = chu[o2][last]
    m = modal[cu]
    w = cnt / cnt.sum()
    off_modal = float(w[chu != m].sum())
    dist = np.hypot(xc[chu] - xc[m], yc[chu] - yc[m])
    far = float(w[dist > 25].sum())
    nC = iCC.shape[0]
    a = np.sort(iCC[:, chu].T, 1)
    b = np.sort(iCC[:, m].T, 1)
    ov = np.array([np.intersect1d(x, y, assume_unique=True).size for x, y in zip(a, b)]) / nC
    return {'templates_used': int((owner >= 0).sum()), 'n_clusters': int((ns > 0).sum()),
            'spikes': int(tpl.size), 'templates_per_cluster': float((owner >= 0).sum() / (ns > 0).sum()),
            'owned_per_unit_mean_100sp': float(owned[big].mean()), 'orphans_100sp': float((owned[big] == 0).mean()),
            'eff_templates_median_100sp': float(np.median(eff[big])), 'spikes_borrowed': float(borrowed),
            'spikes_off_modal_peak': off_modal, 'spikes_peak>25um_from_modal': far,
            'window_channels_not_shared': float((w * (1 - ov)).sum())}


if __name__ == '__main__':
    res = {}
    for setting, rounds in (('H8', (2, 3, 4)), ('default', (2, 3)), ('aug', (3,))):
        for ex in EXS:
            for s in SEEDS:
                for k in rounds:
                    d = folder(ex, setting, k, s)
                    res[f'{ex}_{setting}_r{k}_s{s}'] = spread(d)
                    print(ex, setting, k, s, res[f'{ex}_{setting}_r{k}_s{s}'], flush=True)
    (OUT / 's1_dictionary.json').write_text(json.dumps(res, indent=1))
    for ex in EXS:
        for s in (1,):      # one seed: this part is slow on CPU and the counts are reproduced by the logs
            run = O / 'sorts' / f'{ex}_repeat_s{s}'
            prov = json.loads((run / 'provenance.json').read_text())
            ckey = prov['cache']['B']
            c = torch.load(O / 'stage_cache' / f'B_{ckey}' / 'cluster1.pt', weights_only=False, map_location='cpu')
            clu0, Wall0 = c['out']
            ops = prep(ops_of(run / 'round2'))
            n0 = int(clu0.max()) + 1
            st_fake = np.zeros((len(clu0), 3))
            U = tm.postprocess_templates(Wall0.cpu(), ops, clu0, st_fake, torch.empty((len(clu0), 0, 0)),
                                         device=dev).cpu()
            r = {'first_clusters': n0, 'U_r2': near_dup(U, ops)}
            expected = ops_of(run)['template_rounds'][0]['n_templates']
            r['U_r2_matches_saved_count'] = int(len(U)) == int(expected)
            res[f'{ex}_s{s}_first'] = r
            print(ex, s, n0, len(U), expected, r['U_r2'], flush=True)
            torch.save(U, OUT / f'U_r2_{ex}_s{s}.pt')
            del c, clu0, Wall0, U
    (OUT / 's1_dictionary.json').write_text(json.dumps(res, indent=1))
