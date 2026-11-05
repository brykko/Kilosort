"""Direct measurements across rounds on the same spikes and the same units; no ground truth needed.

Same spike: a spike of sort A and one of sort B within 6 samples and 40 um, mutual nearest (time first). Stable
unit pair: the units are each other's best by paired spikes, agreement >= 0.5.

For the same spike in a stable pair, in two sorts:
- score = <x, U> / |U|, the projection of the data on the unit-norm extraction template (KS4 compares it with
  Th_learned). For a fixed spike a higher score means the template that caught it points closer to it. A smaller
  dictionary works against this (a maximum over fewer templates), so a rise is not a selection effect.
- amp = norm of the spike's features on its template's 10 channels. Falls if neighbours are subtracted better.

Template steps: 1 - maximum-lag correlation of the two units' Wall templates, for stable pairs with >= 1000 spikes
on both sides. Between rounds of one seed this is the step size; between seeds at one round it is how far apart two
independent estimates of one unit's template are. If rounds converge on one template, steps shrink and the
cross-seed distance shrinks.

    python s2_pairs.py <job index> <n jobs>
"""
import gc

from common import *  # noqa
import torch  # noqa: E402

torch.set_num_threads(2)


def pair(a, b, dt=6, dy=40.):
    ta, tb = a['t'], b['t']
    ya, yb = a['pos'][:, 1], b['pos'][:, 1]
    lo = np.searchsorted(tb, ta - dt, 'left')
    hi = np.searchsorted(tb, ta + dt, 'right')
    ia, ib = [], []
    block = 1 << 20
    for s in range(0, ta.size, block):
        e = min(s + block, ta.size)
        cnt = hi[s:e] - lo[s:e]
        i = np.repeat(np.arange(s, e), cnt)
        j = np.repeat(lo[s:e], cnt) + np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        ok = np.abs(ya[i] - yb[j]) <= dy
        ia.append(i[ok])
        ib.append(j[ok])
    ia, ib = np.concatenate(ia), np.concatenate(ib)
    cost = np.abs(ta[ia] - tb[ib]) * 1000 + np.abs(ya[ia] - yb[ib])
    o = np.lexsort((cost, ia))
    ia, ib, cost = ia[o], ib[o], cost[o]
    first = np.r_[True, ia[1:] != ia[:-1]]
    ia, ib, cost = ia[first], ib[first], cost[first]
    o = np.lexsort((cost, ib))
    ia, ib = ia[o], ib[o]
    first = np.r_[True, ib[1:] != ib[:-1]]
    return ia[first], ib[first]


def stable(a, b, ia, ib, min_agree=.5):
    na, nb = np.bincount(a['clu']), np.bincount(b['clu'])
    M = int(b['clu'].max()) + 1
    u, n = np.unique(a['clu'][ia] * M + b['clu'][ib], return_counts=True)
    ca, cb = u // M, u % M
    o = np.lexsort((n, ca))
    x, y, m = ca[o], cb[o], n[o]
    last = np.r_[x[1:] != x[:-1], True]
    fwd = dict(zip(x[last], y[last]))
    o = np.lexsort((n, cb))
    x, y, m = cb[o], ca[o], n[o]
    last = np.r_[x[1:] != x[:-1], True]
    back = dict(zip(x[last], y[last]))
    shared = dict(zip(zip(ca, cb), n))
    out = {}
    for u_, v in fwd.items():
        if back.get(v) == u_:
            s = shared[(u_, v)]
            if s / (na[u_] + nb[v] - s) >= min_agree:
                out[int(u_)] = int(v)
    return out, na, nb


def wall_step(da, db, pairs, na, nb, ops):
    W = torch.as_tensor(tonp(ops['wPCA'])).float()
    nt = ops['nt']
    WtW = torch.flip(torch.nn.functional.conv1d(W.reshape(-1, 1, nt), W.reshape(-1, 1, nt), padding=nt), [2, ])
    Wa = torch.from_numpy(np.load(da / 'Wall.npy'))
    Wb = torch.from_numpy(np.load(db / 'Wall.npy'))
    u = np.array([k for k, v in pairs.items() if na[k] >= 1000 and nb[v] >= 1000])
    v = np.array([pairs[k] for k in u])
    A = Wa[u] / (1e-6 + (Wa[u]**2).sum((1, 2), keepdims=True)**.5)
    B = Wb[v] / (1e-6 + (Wb[v]**2).sum((1, 2), keepdims=True)**.5)
    r = []
    for i in range(0, len(u), 512):
        UtU = torch.einsum('ilk, ilm -> ikm', A[i:i+512], B[i:i+512])
        r.append(torch.einsum('ikm, kml -> il', UtU, WtW).max(-1)[0])
    return 1 - torch.cat(r).numpy(), u, v


def analyse(tag, da, db, ex):
    a, b = spikes(da), spikes(db)
    ia, ib = pair(a, b)
    pairs, na, nb = stable(a, b, ia, ib)
    owner_a, _, _, _ = owners(a['tpl'], a['clu'])
    ua = a['clu'][ia]
    keep = np.zeros(int(a['clu'].max()) + 1, np.int64) - 1
    for k, v in pairs.items():
        keep[k] = v
    st = keep[ua] == b['clu'][ib]
    x, y = ia[st], ib[st]
    dls = np.log(b['score'][y] / a['score'][x]) * 100
    dla = np.log(b['amp'][y] / a['amp'][x]) * 100
    same_tpl_owner = owner_a[a['tpl'][x]] == a['clu'][x]
    o = dict(tag=tag, spikes_a=int(a['t'].size), spikes_b=int(b['t'].size), paired=float(ia.size / a['t'].size),
             stable_units=len(pairs), paired_stable=float(st.sum() / a['t'].size),
             dscore_med=float(np.median(dls)), dscore_mean=float(dls.mean()), damp_med=float(np.median(dla)),
             damp_mean=float(dla.mean()), dcos_med=float(np.median(dls - dla)),
             dscore_owned=float(np.median(dls[same_tpl_owner])), dscore_borrowed=float(np.median(dls[~same_tpl_owner])),
             damp_owned=float(np.median(dla[same_tpl_owner])), damp_borrowed=float(np.median(dla[~same_tpl_owner])),
             frac_borrowed=float((~same_tpl_owner).mean()),
             score_a=float(np.median(a['score'][x])), score_b=float(np.median(b['score'][y])),
             frac_score_up=float((dls > 0).mean()))
    step, u, v = wall_step(Path(da), Path(db), pairs, na, nb, ops_of(da))
    o.update(n_wall=int(step.size), wall_step_med=float(np.median(step)), wall_step_q25=float(np.quantile(step, .25)),
             wall_step_q75=float(np.quantile(step, .75)), wall_step_mean=float(step.mean()))
    # per-unit table for s4
    if tag.startswith('H8 r2->r4') or tag.startswith('default r2->r3'):
        ids = np.array(sorted(pairs))
        sel = keep[ua] >= 0
        np.savez_compressed(OUT / f"units_{ex}_{tag.replace(' ', '_').replace('->', '-')}.npz", u=ids,
                            v=np.array([pairs[i] for i in ids]))
    print(json.dumps(o), flush=True)
    return o


def jobs():
    J = []
    for ex in EXS:
        for s in SEEDS:
            J.append((ex, f'H8 r2->r3 s{s}', folder(ex, 'H8', 2, s), folder(ex, 'H8', 3, s)))
            J.append((ex, f'H8 r3->r4 s{s}', folder(ex, 'H8', 3, s), folder(ex, 'H8', 4, s)))
            J.append((ex, f'H8 r2->r4 s{s}', folder(ex, 'H8', 2, s), folder(ex, 'H8', 4, s)))
            J.append((ex, f'default r2->r3 s{s}', folder(ex, 'default', 2, s), folder(ex, 'default', 3, s)))
            J.append((ex, f'H8 r2->aug3 s{s}', folder(ex, 'H8', 2, s), folder(ex, 'aug', 3, s)))
        for s, t in ((1, 2), (2, 3), (3, 1)):
            for k in (2, 3, 4):
                J.append((ex, f'H8 seeds r{k} s{s}-s{t}', folder(ex, 'H8', k, s), folder(ex, 'H8', k, t)))
            for k in (2, 3):
                J.append((ex, f'default seeds r{k} s{s}-s{t}', folder(ex, 'default', k, s), folder(ex, 'default', k, t)))
    return J


if __name__ == '__main__':
    i, n = int(sys.argv[1]), int(sys.argv[2])
    res = []
    f = OUT / f's2_pairs_{i}.jsonl'
    for j, (ex, tag, da, db) in enumerate(jobs()):
        if j % n != i:
            continue
        o = analyse(tag, da, db, ex)
        o['ex'] = ex
        with open(f, 'a') as h:
            h.write(json.dumps(o) + '\n')
        gc.collect()
