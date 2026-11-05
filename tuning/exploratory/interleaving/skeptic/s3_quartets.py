"""Do independent runs converge on one template? The same units followed through rounds in two seeds.

A quartet is (u1, v1, u2, v2): u1 in seed A round 2, v1 its stable partner in seed A round 4, u2 and v2 likewise in seed B,
with u1-u2 a stable pair across seeds at round 2 and v1-v2 a stable pair across seeds at round 4. All four have >= 1000
spikes. So the same cell in all four sorts, by one rule, with no change of composition between rounds.

Reported: 1 - maximum-lag correlation of Wall, between seeds at round 2 and at round 4 on the same quartets, and the
round-to-round step in each seed. A template 'converging' would give a cross-seed distance that falls."""
from common import *  # noqa
import torch  # noqa: E402
from s2_pairs import pair, stable

torch.set_num_threads(2)


def dist(da, db, u, v, ops):
    W = torch.as_tensor(tonp(ops['wPCA'])).float()
    nt = ops['nt']
    WtW = torch.flip(torch.nn.functional.conv1d(W.reshape(-1, 1, nt), W.reshape(-1, 1, nt), padding=nt), [2, ])
    Wa = torch.from_numpy(np.load(da / 'Wall.npy'))[u]
    Wb = torch.from_numpy(np.load(db / 'Wall.npy'))[v]
    A = Wa / (1e-6 + (Wa**2).sum((1, 2), keepdims=True)**.5)
    B = Wb / (1e-6 + (Wb**2).sum((1, 2), keepdims=True)**.5)
    UtU = torch.einsum('ilk, ilm -> ikm', A, B)
    return 1 - torch.einsum('ikm, kml -> il', UtU, WtW).max(-1)[0].numpy()


def sp(d, cache):
    if d not in cache:
        cache[d] = spikes(d)
    return cache[d]


def stab(da, db, cache):
    a, b = sp(da, cache), sp(db, cache)
    ia, ib = pair(a, b)
    return stable(a, b, ia, ib)


for setting, last in (('H8', 4), ('default', 3)):
    for ex in EXS:
        for s, t in ((1, 2), (2, 3), (3, 1)):
            A2, A4 = folder(ex, setting, 2, s), folder(ex, setting, last, s)
            B2, B4 = folder(ex, setting, 2, t), folder(ex, setting, last, t)
            cache = {}
            pA, n2a, n4a = stab(A2, A4, cache)
            pB, n2b, n4b = stab(B2, B4, cache)
            p2, _, _ = stab(A2, B2, cache)
            p4, _, _ = stab(A4, B4, cache)
            q = [(u1, pA[u1], p2[u1], pB[p2[u1]]) for u1 in pA if u1 in p2 and p2[u1] in pB and p4.get(pA[u1]) == pB[p2[u1]]]
            q = np.array([x for x in q if min(n2a[x[0]], n4a[x[1]], n2b[x[2]], n4b[x[3]]) >= 1000])
            d2 = dist(A2, B2, q[:, 0], q[:, 2], ops_of(A2))
            d4 = dist(A4, B4, q[:, 1], q[:, 3], ops_of(A2))
            sa = dist(A2, A4, q[:, 0], q[:, 1], ops_of(A2))
            sb = dist(B2, B4, q[:, 2], q[:, 3], ops_of(A2))
            amp = np.sqrt((np.load(A2 / 'Wall.npy')[q[:, 0]]**2).sum((1, 2)))
            lo = amp < np.median(amp)
            print(f'{ex} {setting} s{s}-s{t}: quartets {len(q)}; 1000*(1-r) median: cross-seed r2 {np.median(d2) * 1e3:.3f}, r{last} {np.median(d4) * 1e3:.3f}; '
                  f'mean {d2.mean() * 1e3:.2f} -> {d4.mean() * 1e3:.2f}; fell in {np.mean(d4 < d2):.2f}; '
                  f'low-amp half {np.median(d2[lo]) * 1e3:.3f} -> {np.median(d4[lo]) * 1e3:.3f}; steps r2->r{last}: {np.median(sa) * 1e3:.3f}, {np.median(sb) * 1e3:.3f}', flush=True)
            del cache
