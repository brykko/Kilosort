"""Null scale for the gain in stable unit pairs, and the guards still available.

For stable pairs (s2_pairs.py rule, both units scored) the sum of S_v - S_u, and how much of that sum comes from the
top pairs. As a null the same sum between round-2 sorts of two seeds (no round between them): if rounds were a re-roll
this is the size of gain one would see. Also the count of 'untuned' units (spec 7.1: S_i below the 25th percentile of
default's S_i) by round, kobe and MEC, since the hybrid guard is gone."""
from common import *  # noqa
from s2_pairs import pair, stable


def run(tag, da, db, ex):
    Sa, Sb = S_i(da, ex), S_i(db, ex)
    a, b = spikes(da), spikes(db)
    ia, ib = pair(a, b)
    pairs, na, nb = stable(a, b, ia, ib)
    u = np.array([k for k, v in pairs.items() if k in Sa and v in Sb])
    v = np.array([pairs[k] for k in u])
    d = np.array([Sb[j] - Sa[i] for i, j in zip(u, v)])
    o = np.sort(d)[::-1]
    print(f'{ex} {tag}: pairs {u.size}, sum dS {d.sum():+.1f}, SD per pair {d.std():.3f}, sum|dS| {np.abs(d).sum():.1f}, '
          f'top 5% pairs {o[:int(.05 * o.size)].sum():+.1f}, bottom 5% {o[-int(.05 * o.size):].sum():+.1f}, '
          f'unmatched scored: {len(Sa) - u.size} ({sum(Sa.values()) - sum(Sa[i] for i in u):.1f}) -> {len(Sb) - u.size} ({sum(Sb.values()) - sum(Sb[j] for j in v):.1f})',
          flush=True)


for ex in EXS:
    for s, t in ((1, 2), (2, 3), (3, 1)):
        run(f'H8 r2 seeds s{s}-s{t}', folder(ex, 'H8', 2, s), folder(ex, 'H8', 2, t), ex)
    for s in SEEDS:
        run(f'H8 r2->r4 s{s}', folder(ex, 'H8', 2, s), folder(ex, 'H8', 4, s), ex)
    for s in SEEDS:
        run(f'H8 r2->r3 s{s}', folder(ex, 'H8', 2, s), folder(ex, 'H8', 3, s), ex)
    q = np.concatenate([np.array(list(S_i(folder(ex, 'default', 2, s), ex).values())) for s in SEEDS])
    th = np.quantile(q, .25)
    for k in (2, 3, 4):
        c = [sum(x < th for x in S_i(folder(ex, 'H8', k, s), ex).values()) for s in SEEDS]
        print(f'{ex} untuned (S_i < {th:.3f}) H8 round {k}: {c}', flush=True)
    for k in (2, 3):
        c = [sum(x < th for x in S_i(folder(ex, 'default', k, s), ex).values()) for s in SEEDS]
        print(f'{ex} untuned default round {k}: {c}', flush=True)
